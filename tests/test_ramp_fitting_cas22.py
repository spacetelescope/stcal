"""
Unit tests for ramp-fitting functions.
"""
import astropy.units as u
import numpy as np
import pytest

from stcal.ramp_fitting import ols_cas22_fit as ramp

# Purposefully set a fixed seed so that the tests in this module are deterministic
RNG = np.random.default_rng(42)

# Read Time in seconds
#   For Roman, the read time of the detectors is a fixed value and is currently
#   backed into code. Will need to refactor to consider the more general case.
#   Used to deconstruct the MultiAccum tables into integration times.
ROMAN_READ_TIME = 3.04


@pytest.mark.parametrize("use_unit", [True, False])
@pytest.mark.parametrize("use_dq", [True, False])
def test_simulated_ramps(use_unit, use_dq):
    # Perfect square like the detector, this is so we can test that the code
    #   reshapes the data correctly for the computation and then reshapes it back
    #   to the original shape.
    ntrial = 320 * 320
    read_pattern, flux, read_noise, resultants = simulate_many_ramps(ntrial=ntrial)

    # So we get a detector-like input shape
    resultants = resultants.reshape((len(read_pattern), 320, 320))

    if use_unit:
        resultants = resultants * u.electron

    dq = np.zeros(resultants.shape, dtype=np.int32)
    read_noise = np.ones(resultants.shape[1:], dtype=np.float32) * read_noise

    # now let's mark a bunch of the ramps as compromised. When using dq flags
    if use_dq:
        bad = RNG.uniform(size=resultants.shape) > 0.7
        dq |= bad

    output = ramp.fit_ramps_casertano(
        resultants, dq, read_noise, ROMAN_READ_TIME, read_pattern, threshold_constant=0, threshold_intercept=0
    )  # set the threshold parameters
    #   to demo the interface. This
    #   will raise an error if
    #   the interface changes, but
    #   does not effect the computation
    #   since jump detection is off in
    #   this case.

    # Check that the output shapes are correct
    assert output.parameters.shape == (320, 320, 2) == resultants.shape[1:] + (2,)
    assert output.variances.shape == (320, 320, 3) == resultants.shape[1:] + (3,)
    assert output.dq.shape == dq.shape

    # check the unit
    if use_unit:
        assert output.parameters.unit == u.electron
        parameters = output.parameters.value
    else:
        parameters = output.parameters

    # Turn into single dimension arrays to make the indexing for the math easier
    parameters = parameters.reshape((320 * 320, 2))
    variances = output.variances.reshape((320 * 320, 3))

    # only use okay ramps
    # ramps passing the below criterion have at least two adjacent valid reads
    # i.e., we can make a measurement from them.
    okay = np.sum((dq[1:, :] == 0) & (dq[:-1, :] == 0), axis=0) != 0
    okay = okay.reshape(320 * 320)

    # Sanity check that when no dq is used, all ramps are used
    if not use_dq:
        assert np.all(okay)

    chi2dof_slope = np.sum((parameters[okay, 1] - flux) ** 2 / variances[okay, 2]) / np.sum(okay)
    assert np.abs(chi2dof_slope - 1) < 0.03
    assert np.all(parameters[~okay, 1] == 0)
    assert np.all(variances[~okay, 1] == 0)


# #########
# Utilities
# #########
def simulate_many_ramps(ntrial=100, flux=100, readnoise=5, read_pattern=None):
    """Simulate many ramps with a particular flux, read noise, and ma_table.

    To test ramp fitting, it's useful to be able to simulate a large number
    of ramps that are identical up to noise.  This function does that.

    Parameters
    ----------
    ntrial : int
        number of ramps to simulate
    flux : float
        flux in electrons / s
    read_noise : float
        read noise in electrons
    read_pattern : list[list] (int)
        An optional read pattern

    Returns
    -------
    ma_table : list[list] (int)
        ma_table used
    flux : float
        flux used
    readnoise : float
        read noise used
    resultants : np.ndarray[n_resultant, ntrial] (float)
        simulated resultants"""
    if read_pattern is None:
        read_pattern = [
            [1, 2, 3, 4],
            [5],
            [6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            [19, 20, 21],
            [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
        ]
    nread = np.array([len(x) for x in read_pattern])
    resultants = np.zeros((len(read_pattern), ntrial), dtype="f4")
    buf = np.zeros(ntrial, dtype="i4")
    for i, reads in enumerate(read_pattern):
        subbuf = np.zeros(ntrial, dtype="i4")
        for _ in reads:
            buf += RNG.poisson(ROMAN_READ_TIME * flux, ntrial)
            subbuf += buf
        resultants[i] = (subbuf / len(reads)).astype("f4")
    resultants += RNG.standard_normal(size=(len(read_pattern), ntrial)) * (
        readnoise / np.sqrt(nread)
    ).reshape(len(read_pattern), 1)
    return (read_pattern, flux, readnoise, resultants)
