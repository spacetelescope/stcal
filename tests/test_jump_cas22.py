import numpy as np
import pytest
from numpy.testing import assert_allclose

from stcal.ramp_fitting.ols_cas22 import JUMP_DET, Parameter, Variance, fit_ramps
from stcal.ramp_fitting.ols_cas22._jump import (
    FixedOffsets,
    PixelOffsets,
    _fill_pixel_values,
    fill_fixed_values,
)
from stcal.ramp_fitting.ols_cas22._ramp import from_read_pattern, init_ramps

# Purposefully set a fixed seed so that the tests in this module are deterministic
RNG = np.random.default_rng(619)

# The read time is constant for the given telescope/instrument so we set it here
#    to be the one for Roman as it is known to be a reasonable value
READ_TIME = 3.04

# Choose small read noise relative to the flux to make it extremely unlikely
#    that the random process will "accidentally" generate a set of data, which
#    can trigger jump detection. This makes it easier to cleanly test jump
#    detection is doing what we expect.
FLUX = 100
READ_NOISE = np.float32(5)

# Set a value for jumps which makes them obvious relative to the normal flux
JUMP_VALUE = 1_000

# Choose reasonable values for arbitrary test parameters, these are kept the same
#    across all tests to make it easier to isolate the effects of something using
#    multiple tests.
N_PIXELS = 100_000
CHI2_TOL = 0.03
GOOD_PROB = 0.7


def test_init_ramps():
    """
    Test turning dq flags into initial ramp splits
        Note that because `init_ramps` itself returns a stack, which does not have
        a direct python equivalent, we call the wrapper for `init_ramps` which
        converts that stack into a list ordered in the same fashion as the stack
    """
    # from stcal.ramp_fitting.ols_cas22._core import _init_ramps_list

    dq = np.array(
        [
            [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        ],
        dtype=np.int32,
    )

    n_resultants, n_pixels = dq.shape
    ramps = [init_ramps(dq[:, index], n_resultants) for index in range(n_pixels)]

    assert len(ramps) == dq.shape[1] == 16

    # Check that the ramps are correct

    # No DQ
    assert ramps[0] == [{"start": 0, "end": 3}]

    # 1 DQ
    assert ramps[1] == [{"start": 1, "end": 3}]
    assert ramps[2] == [{"start": 0, "end": 0}, {"start": 2, "end": 3}]
    assert ramps[3] == [{"start": 0, "end": 1}, {"start": 3, "end": 3}]
    assert ramps[4] == [{"start": 0, "end": 2}]

    # 2 DQ
    assert ramps[5] == [{"start": 2, "end": 3}]
    assert ramps[6] == [{"start": 1, "end": 1}, {"start": 3, "end": 3}]
    assert ramps[7] == [{"start": 1, "end": 2}]
    assert ramps[8] == [{"start": 0, "end": 0}, {"start": 3, "end": 3}]
    assert ramps[9] == [{"start": 0, "end": 0}, {"start": 2, "end": 2}]
    assert ramps[10] == [{"start": 0, "end": 1}]

    # 3 DQ
    assert ramps[11] == [{"start": 3, "end": 3}]
    assert ramps[12] == [{"start": 2, "end": 2}]
    assert ramps[13] == [{"start": 1, "end": 1}]
    assert ramps[14] == [{"start": 0, "end": 0}]

    # 4 DQ
    assert ramps[15] == []


@pytest.fixture(scope="module")
def read_pattern():
    """
    Basic data for simulating ramps for testing (not unpacked)

    Returns
    -------
        read_pattern : list[list[int]]
            The example read pattern
        metadata : dict
            The metadata computed from the read pattern
    """
    return [
        [1, 2, 3, 4],
        [5],
        [6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        [19, 20, 21],
        [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
    ]


def test_from_read_pattern(read_pattern):
    """Test turning read_pattern into the time data"""
    metadata = from_read_pattern(read_pattern, READ_TIME, len(read_pattern))._to_dict()  # noqa: SLF001

    t_bar = metadata["t_bar"]
    tau = metadata["tau"]
    n_reads = metadata["n_reads"]

    # Check that the data is correct
    assert_allclose(t_bar, [7.6, 15.2, 21.279999, 41.040001, 60.799999, 88.159996])
    assert_allclose(tau, [5.7, 15.2, 19.928888, 36.023998, 59.448887, 80.593781])
    assert np.all(n_reads == [4, 1, 3, 10, 3, 15])

    # Check datatypes
    assert t_bar.dtype == np.float32
    assert tau.dtype == np.float32
    assert n_reads.dtype == np.int32


@pytest.fixture(scope="module")
def ramp_data(read_pattern):
    """
    Basic data for simulating ramps for testing (not unpacked)

    Returns
    -------
        read_pattern : list[list[int]]
            The example read pattern
        metadata : dict
            The metadata computed from the read pattern
    """
    data = from_read_pattern(read_pattern, READ_TIME, len(read_pattern))._to_dict()  # noqa: SLF001

    return data["t_bar"], data["tau"], data["n_reads"], read_pattern


def test_fill_fixed_values(ramp_data):
    """Test computing the fixed data for all pixels"""
    t_bar, tau, n_reads, _ = ramp_data

    n_resultants = len(t_bar)
    fixed = np.empty((FixedOffsets.n_fixed_offsets, n_resultants - 1), dtype=np.float32)
    fixed = fill_fixed_values(fixed, t_bar, tau, n_reads, n_resultants)

    # Sanity check that the shape of fixed is correct
    assert fixed.shape == (2 * 4, n_resultants - 1)

    # Split into the different types of data
    t_bar_diffs = fixed[FixedOffsets.single_t_bar_diff : FixedOffsets.double_t_bar_diff + 1, :]
    t_bar_diff_sqrs = fixed[FixedOffsets.single_t_bar_diff_sqr : FixedOffsets.double_t_bar_diff_sqr + 1, :]
    read_recip = fixed[FixedOffsets.single_read_recip : FixedOffsets.double_read_recip + 1, :]
    var_slope_vals = fixed[FixedOffsets.single_var_slope_val : FixedOffsets.double_var_slope_val + 1, :]

    # Sanity check that these are all the right shape
    assert t_bar_diffs.shape == (2, n_resultants - 1)
    assert t_bar_diff_sqrs.shape == (2, n_resultants - 1)
    assert read_recip.shape == (2, n_resultants - 1)
    assert var_slope_vals.shape == (2, n_resultants - 1)

    # Check the computed data
    #   These are computed using loop in cython, here we check against numpy
    # Single diffs
    assert np.all(t_bar_diffs[0] == t_bar[1:] - t_bar[:-1])
    assert np.all(t_bar_diff_sqrs[0] == (t_bar[1:] - t_bar[:-1]) ** 2)
    assert np.all(read_recip[0] == np.float32(1 / n_reads[1:]) + np.float32(1 / n_reads[:-1]))
    assert np.all(var_slope_vals[0] == (tau[1:] + tau[:-1] - 2 * np.minimum(t_bar[1:], t_bar[:-1])))

    # Double diffs
    assert np.all(t_bar_diffs[1, :-1] == t_bar[2:] - t_bar[:-2])
    assert np.all(t_bar_diff_sqrs[1, :-1] == (t_bar[2:] - t_bar[:-2]) ** 2)
    assert np.all(read_recip[1, :-1] == np.float32(1 / n_reads[2:]) + np.float32(1 / n_reads[:-2]))
    assert np.all(var_slope_vals[1, :-1] == (tau[2:] + tau[:-2] - 2 * np.minimum(t_bar[2:], t_bar[:-2])))

    # Last double diff should be NaN
    assert np.isnan(t_bar_diffs[1, -1])
    assert np.isnan(t_bar_diff_sqrs[1, -1])
    assert np.isnan(read_recip[1, -1])
    assert np.isnan(var_slope_vals[1, -1])


def _generate_resultants(read_pattern, n_pixels=1):
    """
    Generate a set of resultants for a pixel

    Parameters:
        read_pattern : list[list[int]]
            The read pattern to use
        n_pixels:
            The number of pixels to generate resultants for. Default is 1.

    Returns:
        resultants
            The resultants generated
    """
    resultants = np.zeros((len(read_pattern), n_pixels), dtype=np.float32)

    # Use Poisson process to simulate the accumulation of the ramp
    ramp_value = np.zeros(n_pixels, dtype=np.float32)  # Last value of ramp
    for index, reads in enumerate(read_pattern):
        resultant_total = np.zeros(n_pixels, dtype=np.float32)  # Total of all reads in this resultant
        for _ in reads:
            # Compute the next value of the ramp
            #   Using a Poisson process for the flux
            ramp_value += RNG.poisson(FLUX * READ_TIME, size=n_pixels).astype(np.float32)

            # Add to running total for the resultant
            resultant_total += ramp_value

        # Add read noise to the resultant
        resultant_total += (
            RNG.standard_normal(size=n_pixels, dtype=np.float32) * READ_NOISE / np.sqrt(len(reads))
        )

        # Record the average value for resultant (i.e., the average of the reads)
        resultants[index] = (resultant_total / len(reads)).astype(np.float32)

    if n_pixels == 1:
        resultants = resultants[:, 0]

    return resultants


@pytest.fixture(scope="module")
def pixel_data(ramp_data):
    """
    Create data for a single pixel

    Returns:
        resultants
            Resultants for a single pixel
        t_bar:
            The t_bar values for the read pattern used for the resultants
        tau:
            The tau values for the read pattern used for the resultants
        n_reads:
            The number of reads for the read pattern used for the resultants
    """
    t_bar, tau, n_reads, read_pattern = ramp_data

    n_resultants = len(t_bar)
    fixed = np.empty((FixedOffsets.n_fixed_offsets, n_resultants - 1), dtype=np.float32)
    fixed = fill_fixed_values(fixed, t_bar, tau, n_reads, n_resultants)

    resultants = _generate_resultants(read_pattern)

    return resultants, t_bar, tau, n_reads, fixed


def test__fill_pixel_values(pixel_data):
    """Test computing the initial pixel data"""
    resultants, t_bar, tau, n_reads, fixed = pixel_data

    n_resultants = len(t_bar)
    pixel = np.empty((PixelOffsets.n_pixel_offsets, n_resultants - 1), dtype=np.float32)
    pixel = _fill_pixel_values(pixel, resultants, fixed, READ_NOISE, n_resultants)

    # Sanity check that the shape of pixel is correct
    assert pixel.shape == (2 * 2, n_resultants - 1)

    # Split into the different types of data
    local_slopes = pixel[PixelOffsets.single_local_slope : PixelOffsets.double_local_slope + 1, :]
    var_read_noise = pixel[PixelOffsets.single_var_read_noise : PixelOffsets.double_var_read_noise + 1, :]

    # Sanity check that these are all the right shape
    assert local_slopes.shape == (2, n_resultants - 1)
    assert var_read_noise.shape == (2, n_resultants - 1)

    # Check the computed data
    #   These are computed using loop in cython, here we check against numpy
    # Single diffs
    assert np.all(local_slopes[0] == (resultants[1:] - resultants[:-1]) / (t_bar[1:] - t_bar[:-1]))
    assert np.all(
        var_read_noise[0]
        == np.float32(READ_NOISE**2) * (np.float32(1 / n_reads[1:]) + np.float32(1 / n_reads[:-1]))
    )

    # Double diffs
    assert np.all(local_slopes[1, :-1] == (resultants[2:] - resultants[:-2]) / (t_bar[2:] - t_bar[:-2]))
    assert np.all(
        var_read_noise[1, :-1]
        == np.float32(READ_NOISE**2) * (np.float32(1 / n_reads[2:]) + np.float32(1 / n_reads[:-2]))
    )

    # Last double diff should be NaN
    assert np.isnan(local_slopes[1, -1])
    assert np.isnan(var_read_noise[1, -1])


@pytest.fixture(scope="module")
def detector_data(read_pattern):
    """
    Generate a set of with no jumps data as if for a single detector as it
        would be passed in by the supporting code.

    Returns:
        resultants
            The resultants for a large number of pixels
        read_noise:
            The read noise vector for those pixels
        read_pattern:
            The read pattern used for the resultants
    """
    read_noise = np.ones(N_PIXELS, dtype=np.float32) * READ_NOISE

    resultants = _generate_resultants(read_pattern, n_pixels=N_PIXELS)

    return resultants, read_noise, read_pattern


@pytest.mark.parametrize("use_jump", [True, False])
@pytest.mark.parametrize("use_dq", [True, False])
def test_fit_ramps(detector_data, use_jump, use_dq):
    """
    Test fitting ramps
        Since no jumps are simulated in the data, jump detection shouldn't pick
        up any jumps.
    """
    resultants, read_noise, read_pattern = detector_data
    dq = (
        (RNG.uniform(size=resultants.shape) > GOOD_PROB).astype(np.int32)
        if use_dq
        else np.zeros(resultants.shape, dtype=np.int32)
    )

    # only use okay ramps
    #   ramps passing the below criterion have at least two adjacent valid reads
    #   i.e., we can make a measurement from them.
    okay = np.sum((dq[1:, :] == 0) & (dq[:-1, :] == 0), axis=0) != 0
    assert okay.dtype == bool

    # Note that for use_dq = False, okay == True for all ramps, so we perform
    #    a sanity check that the above criterion is correct
    if not use_dq:
        assert okay.all()

    output = fit_ramps(
        resultants, dq, read_noise, READ_TIME, read_pattern, use_jump=use_jump, include_diagnostic=True
    )
    assert len(output.fits) == N_PIXELS  # sanity check that a fit is output for each pixel

    chi2 = 0
    for fit, use in zip(output.fits, okay):
        if not use_dq and not use_jump:
            ##### The not use_jump makes this NOT test for false positives #####
            # Check that the data generated does not generate any false positives
            #   for jumps as this data is reused for `test_find_jumps` below.
            #   This guarantees that all jumps detected in that test are the
            #   purposefully placed ones which we know about. So the `test_find_jumps`
            #   can focus on checking that the jumps found are the correct ones,
            #   and that all jumps introduced are detected properly.
            assert len(fit["fits"]) == 1

        if use:
            # Add okay ramps to chi2
            total_var = fit["average"]["read_var"] + fit["average"]["poisson_var"]
            if total_var != 0:
                chi2 += (fit["average"]["slope"] - FLUX) ** 2 / total_var
        else:
            # Check no slope fit for bad ramps
            assert fit["average"]["slope"] == 0
            assert fit["average"]["read_var"] == 0
            assert fit["average"]["poisson_var"] == 0

            assert use_dq  # sanity check that this branch is only encountered when use_dq = True

    chi2 /= np.sum(okay)
    assert np.abs(chi2 - 1) < CHI2_TOL


@pytest.mark.parametrize("use_jump", [True, False])
def test_fit_ramps_array_outputs(detector_data, use_jump):
    """
    Test that the array outputs line up with the dictionary output
    """
    resultants, read_noise, read_pattern = detector_data
    dq = np.zeros(resultants.shape, dtype=np.int32)

    output = fit_ramps(
        resultants, dq, read_noise, READ_TIME, read_pattern, use_jump=use_jump, include_diagnostic=True
    )

    for fit, par, var in zip(output.fits, output.parameters, output.variances):
        assert par[Parameter.intercept] == 0
        assert par[Parameter.slope] == fit["average"]["slope"]

        assert var[Variance.read_var] == fit["average"]["read_var"]
        assert var[Variance.poisson_var] == fit["average"]["poisson_var"]
        assert var[Variance.total_var] == np.float32(
            fit["average"]["read_var"] + fit["average"]["poisson_var"]
        )


@pytest.fixture(scope="module")
def jump_data(detector_data):
    """
    Generate resultants with single jumps in them for testing jump detection.
        Note this specifically checks that we can detect jumps in any read, meaning
        it has an insurance check that a jump has been placed in every single
        read position.

    Returns:
        resultants
            The resultants for a large number of pixels
        read_noise:
            The read noise vector for those pixels
        read_pattern:
            The read pattern used for the resultants
        jump_reads:
            Index of read where a jump occurs for each pixel
        jump_resultants:
            Index of resultant where a jump occurs for each pixel
    """
    resultants, read_noise, read_pattern = detector_data

    # Choose read to place a single jump in for each pixel
    num_reads = read_pattern[-1][-1]
    jump_reads = RNG.integers(num_reads - 1, size=N_PIXELS)

    # This shows that a jump as been placed in every single possible
    #   read position. Technically, this check can fail; however,
    #   N_PIXELS >> num_reads so it is very unlikely in practice since
    #   all reads are equally likely to be chosen for a jump.
    # It is a good check that we can detect a jump occurring in any read except
    #    the first read.
    assert set(jump_reads) == set(range(num_reads - 1))

    # Fill out jump reads with jump values
    jump_flux = np.zeros((num_reads, N_PIXELS), dtype=np.float32)
    for index, jump in enumerate(jump_reads):
        jump_flux[jump:, index] = JUMP_VALUE

    # Average the reads into the resultants
    jump_resultants = np.zeros(N_PIXELS, dtype=np.int32)
    for index, reads in enumerate(read_pattern):
        indices = np.array(reads) - 1
        resultants[index, :] += np.mean(jump_flux[indices, :], axis=0)
        for read in reads:
            jump_resultants[np.where(jump_reads == read)] = index

    return resultants, read_noise, read_pattern, jump_reads, jump_resultants


def test_find_jumps(jump_data):
    """
    Full unit tests to demonstrate that we can detect jumps in any read (except
    the first one) and that we correctly remove these reads from the fit to recover
    the correct FLUX/slope.
    """
    resultants, read_noise, read_pattern, jump_reads, jump_resultants = jump_data
    dq = np.zeros(resultants.shape, dtype=np.int32)

    output = fit_ramps(
        resultants, dq, read_noise, READ_TIME, read_pattern, use_jump=True, include_diagnostic=True
    )
    assert len(output.fits) == len(jump_reads)  # sanity check that a fit/jump is set for every pixel

    chi2 = 0
    incorrect_too_few = 0
    incorrect_too_many = 0
    incorrect_does_not_capture = 0
    incorrect_other = 0
    for fit, jump_index, resultant_index in zip(output.fits, jump_reads, jump_resultants):
        # Check that the jumps are detected correctly
        if jump_index == 0:
            # There is no way to detect a jump if it is in the very first read
            # The very first pixel in this case has a jump in the first read
            assert len(fit["jumps"]) == 0
            assert resultant_index == 0  # sanity check that the jump is indeed in the first resultant

            # Test the correct ramp_index was recorded:
            assert len(fit["index"]) == 1
            assert fit["index"][0]["start"] == 0
            assert fit["index"][0]["end"] == len(read_pattern) - 1
        else:
            # There should be a single jump detected; however, this results in
            # two resultants being excluded.
            if resultant_index not in fit["jumps"]:
                incorrect_does_not_capture += 1
                continue
            if len(fit["jumps"]) < 2:
                incorrect_too_few += 1
                continue
            if len(fit["jumps"]) > 2:
                incorrect_too_many += 1
                continue

            # The two resultants excluded should be adjacent
            jump_correct = [
                (jump in (resultant_index, resultant_index - 1, resultant_index + 1)) for jump in fit["jumps"]
            ]
            if not all(jump_correct):
                incorrect_other += 1
                continue

            # Because we do not have a data set with no false positives, we cannot run the below
            # # Test the correct ramp indexes are recorded
            # ramp_indices = []
            # for ramp_index in fit['index']:
            #     # Note start/end of a ramp_index are inclusive meaning that end
            #     #    is an index included in the ramp_index so the range is to end + 1
            #     new_indices = list(range(ramp_index["start"], ramp_index["end"] + 1))

            #     # check that all the ramps are non-overlapping
            #     assert set(ramp_indices).isdisjoint(new_indices)

            #     ramp_indices.extend(new_indices)

            # # check that no ramp_index is a jump
            # assert set(ramp_indices).isdisjoint(fit['jumps'])

            # # check that all resultant indices are either in a ramp or listed as a jump
            # assert set(ramp_indices).union(fit['jumps']) == set(range(len(read_pattern)))

        # Compute the chi2 for the fit and add it to a running "total chi2"
        total_var = fit["average"]["read_var"] + fit["average"]["poisson_var"]
        chi2 += (fit["average"]["slope"] - FLUX) ** 2 / total_var

    # Check that the average chi2 is ~1.
    chi2 /= N_PIXELS - incorrect_too_few - incorrect_too_many - incorrect_does_not_capture - incorrect_other
    assert np.abs(chi2 - 1) < CHI2_TOL


def test_override_default_threshold(jump_data):
    """This tests that we can override the default jump detection threshold constants"""
    resultants, read_noise, read_pattern, jump_reads, jump_resultants = jump_data
    dq = np.zeros(resultants.shape, dtype=np.int32)

    standard = fit_ramps(resultants, dq, read_noise, READ_TIME, read_pattern, use_jump=True)
    override = fit_ramps(
        resultants, dq, read_noise, READ_TIME, read_pattern, use_jump=True, intercept=0, constant=0
    )

    # All this is intended to do is show that with all other things being equal passing non-default
    #    threshold parameters changes the results.
    assert (standard.parameters != override.parameters).any()


def test_jump_dq_set(jump_data):
    # Check the DQ flag value to start
    assert 2**2 == JUMP_DET

    resultants, read_noise, read_pattern, jump_reads, jump_resultants = jump_data
    dq = np.zeros(resultants.shape, dtype=np.int32)

    output = fit_ramps(
        resultants, dq, read_noise, READ_TIME, read_pattern, use_jump=True, include_diagnostic=True
    )

    for fit, pixel_dq in zip(output.fits, output.dq.transpose()):
        # Check that all jumps found get marked
        assert (pixel_dq[fit["jumps"]] == JUMP_DET).all()

        # Check that dq flags for jumps are only set if the jump is marked
        assert set(np.where(pixel_dq == JUMP_DET)[0]) == set(fit["jumps"])
