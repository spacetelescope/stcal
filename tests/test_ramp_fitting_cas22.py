
"""
Unit tests for ramp-fitting functions.
"""
import numpy as np

from stcal.ramp_fitting import ols_cas22_fit as ramp
from stcal.ramp_fitting import ols_cas22_util

# Read Time in seconds
#   For Roman, the read time of the detectors is a fixed value and is currently
#   backed into code. Will need to refactor to consider the more general case.
#   Used to deconstruct the MultiAccum tables into integration times.
ROMAN_READ_TIME = 3.04


def test_matable_to_readpattern():
    """Test conversion from read pattern to multi-accum table"""
    ma_table = [[1, 1], [2, 2], [4, 1], [5, 4], [9,2], [11,1]]
    expected = [[1], [2, 3], [4], [5, 6, 7, 8], [9, 10], [11]]

    result = ols_cas22_util.matable_to_readpattern(ma_table)

    assert result == expected


def test_readpattern_to_matable():
    """Test conversion from read pattern to multi-accum table"""
    pattern = [[1], [2, 3], [4], [5, 6, 7, 8], [9, 10], [11]]
    expected = [[1, 1], [2, 2], [4, 1], [5, 4], [9,2], [11,1]]

    result = ols_cas22_util.readpattern_to_matable(pattern)

    assert result == expected


def test_simulated_ramps():
    ntrial = 100000
    ma_table, flux, read_noise, resultants = simulate_many_ramps(ntrial=ntrial)

    par, var = ramp.fit_ramps_casertano(
        resultants, resultants * 0, read_noise, ROMAN_READ_TIME, ma_table=ma_table)
    chi2dof_slope = np.sum((par[:, 1] - flux)**2 / var[:, 2]) / ntrial
    assert np.abs(chi2dof_slope - 1) < 0.03

    # now let's mark a bunch of the ramps as compromised.
    bad = np.random.uniform(size=resultants.shape) > 0.7
    dq = resultants * 0 + bad
    par, var = ramp.fit_ramps_casertano(
        resultants, dq, read_noise, ROMAN_READ_TIME, ma_table=ma_table)
    # only use okay ramps
    # ramps passing the below criterion have at least two adjacent valid reads
    # i.e., we can make a measurement from them.
    m = np.sum((dq[1:, :] == 0) & (dq[:-1, :] == 0), axis=0) != 0
    chi2dof_slope = np.sum((par[m, 1] - flux)**2 / var[m, 2]) / np.sum(m)
    assert np.abs(chi2dof_slope - 1) < 0.03
    assert np.all(par[~m, 1] == 0)
    assert np.all(var[~m, 1] == 0)


# #########
# Utilities
# #########
def simulate_many_ramps(ntrial=100, flux=100, readnoise=5, ma_table=None):
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
    ma_table : list[list] (int)
        list of lists indicating first read and number of reads in each
        resultant

    Returns
    -------
    ma_table : list[list] (int)
        ma_table used
    flux : float
        flux used
    readnoise : float
        read noise used
    resultants : np.ndarray[n_resultant, ntrial] (float)
        simulated resultants
"""
    if ma_table is None:
        ma_table = [[1, 4], [5, 1], [6, 3], [9, 10], [19, 3], [22, 15]]
    nread = np.array([x[1] for x in ma_table])
    tij = ols_cas22_util.ma_table_to_tij(ma_table, ROMAN_READ_TIME)
    resultants = np.zeros((len(ma_table), ntrial), dtype='f4')
    buf = np.zeros(ntrial, dtype='i4')
    for i, ti in enumerate(tij):
        subbuf = np.zeros(ntrial, dtype='i4')
        for t0 in ti:
            buf += np.random.poisson(ROMAN_READ_TIME * flux, ntrial)
            subbuf += buf
        resultants[i] = (subbuf / len(ti)).astype('f4')
    resultants += np.random.randn(len(ma_table), ntrial) * (
        readnoise / np.sqrt(nread)).reshape(len(ma_table), 1)
    return (ma_table, flux, readnoise, resultants)
