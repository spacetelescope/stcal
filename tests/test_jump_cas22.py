import numpy as np
import pytest
from numpy.testing import assert_allclose

from stcal.ramp_fitting.ols_cas22._fixed import fixed_values_from_metadata
from stcal.ramp_fitting.ols_cas22._jump import threshold
from stcal.ramp_fitting.ols_cas22._pixel import make_pixel
from stcal.ramp_fitting.ols_cas22._ramp import init_ramps
from stcal.ramp_fitting.ols_cas22._read_pattern import from_read_pattern

from stcal.ramp_fitting.ols_cas22 import fit_ramps, Parameter, Variance, Diff, RampJumpDQ


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


@pytest.fixture(scope="module")
def ramp_data():
    """
    Basic data for simulating ramps for testing (not unpacked)

    Returns
    -------
        read_pattern : list[list[int]]
            The example read pattern
        metadata : dict
            The metadata computed from the read pattern
    """
    read_pattern = [
        [1, 2, 3, 4],
        [5],
        [6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        [19, 20, 21],
        [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    ]

    yield read_pattern, from_read_pattern(read_pattern, READ_TIME)


def test_from_read_pattern(ramp_data):
    """Test turning read_pattern into the time data"""
    read_pattern, data_object = ramp_data
    data = data_object._to_dict()

    # Basic sanity checks (structs become dicts)
    assert isinstance(data, dict)
    assert 't_bar' in data
    assert 'tau' in data
    assert 'n_reads' in data
    assert len(data) == 4

    # Check that the data is correct
    assert data['n_resultants'] == len(read_pattern)
    assert_allclose(data['t_bar'], [7.6, 15.2, 21.279999, 41.040001, 60.799999, 88.159996])
    assert_allclose(data['tau'], [5.7, 15.2, 19.928888, 36.023998, 59.448887, 80.593781])
    assert np.all(data['n_reads'] == [4, 1, 3, 10, 3, 15])

    # Check datatypes
    assert data['t_bar'].dtype == np.float32
    assert data['tau'].dtype == np.float32
    assert data['n_reads'].dtype == np.int32


def test_init_ramps():
    """
    Test turning dq flags into initial ramp splits
        Note that because `init_ramps` itself returns a stack, which does not have
        a direct python equivalent, we call the wrapper for `init_ramps` which
        converts that stack into a list ordered in the same fashion as the stack
    """
    # from stcal.ramp_fitting.ols_cas22._core import _init_ramps_list

    dq = np.array([[0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
                   [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                   [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                   [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]], dtype=np.int32)

    n_resultants, n_pixels = dq.shape
    ramps = [init_ramps(dq, n_resultants, index_pixel) for index_pixel in range(n_pixels)]

    assert len(ramps) == dq.shape[1] == 16

    # Check that the ramps are correct

    # No DQ
    assert ramps[0] == [{'start': 0, 'end': 3}]

    # 1 DQ
    assert ramps[1] == [{'start': 1, 'end': 3}]
    assert ramps[2] == [{'start': 0, 'end': 0}, {'start': 2, 'end': 3}]
    assert ramps[3] == [{'start': 0, 'end': 1}, {'start': 3, 'end': 3}]
    assert ramps[4] == [{'start': 0, 'end': 2}]

    # 2 DQ
    assert ramps[5] == [{'start': 2, 'end': 3}]
    assert ramps[6] == [{'start': 1, 'end': 1}, {'start': 3, 'end': 3}]
    assert ramps[7] == [{'start': 1, 'end': 2}]
    assert ramps[8] == [{'start': 0, 'end': 0}, {'start': 3, 'end': 3}]
    assert ramps[9] == [{'start': 0, 'end': 0}, {'start': 2, 'end': 2}]
    assert ramps[10] == [{'start': 0, 'end': 1}]

    # 3 DQ
    assert ramps[11] == [{'start': 3, 'end': 3}]
    assert ramps[12] == [{'start': 2, 'end': 2}]
    assert ramps[13] == [{'start': 1, 'end': 1}]
    assert ramps[14] == [{'start': 0, 'end': 0}]

    # 4 DQ
    assert ramps[15] == []


def test_threshold():
    """
    Test the threshold object/fucnction
        intercept - constant * log10(slope) = threshold
    """

    # Create the python analog of the Threshold struct
    #    Note that structs get mapped to/from python as dictionary objects with
    #    the keys being the struct members.
    thresh = {
        'intercept': np.float32(5.5),
        'constant': np.float32(1/3)
    }

    # Check the 'intercept' is correctly interpreted.
    #    Since the log of the input slope is taken, log10(1) = 0, meaning that
    #    we should directly recover the intercept value in that case.
    assert thresh['intercept'] == threshold(thresh, 1.0)

    # Check the 'constant' is correctly interpreted.
    #    Since we know that the intercept is correctly identified and that `log10(10) = 1`,
    #    we can use that to check that the constant is correctly interpreted.
    assert np.float32(thresh['intercept'] - thresh['constant']) == threshold(thresh, 10.0)


@pytest.mark.parametrize("use_jump", [True, False])
def test_fixed_values_from_metadata(ramp_data, use_jump):
    """Test computing the fixed data for all pixels"""
    _, data = ramp_data

    data_dict = data._to_dict()
    t_bar = data_dict['t_bar']
    tau = data_dict['tau']
    n_reads = data_dict['n_reads']

    # Create the python analog of the Threshold struct
    #    Note that structs get mapped to/from python as dictionary objects with
    #    the keys being the struct members.
    thresh = {
        'intercept': np.float32(5.5),
        'constant': np.float32(1/3)
    }

    # Note this is converted to a dictionary so we can directly interrogate the
    #   variables in question
    fixed = fixed_values_from_metadata(data, thresh, use_jump)._to_dict()

    # Basic sanity checks that data passed in survives
    assert (fixed['data']['t_bar'] == t_bar).all()
    assert (fixed['data']['tau'] == tau).all()
    assert (fixed['data']['n_reads'] == n_reads).all()
    assert fixed['threshold']["intercept"] == thresh['intercept']
    assert fixed['threshold']["constant"] == thresh['constant']

    # Check the computed data
    # These are computed via vectorized operations in the main code, here we
    #    check using item-by-item operations
    if use_jump:
        single_gen = zip(
            fixed['t_bar_diffs'][Diff.single],
            fixed['t_bar_diff_sqrs'][Diff.single],
            fixed['read_recip_coeffs'][Diff.single],
            fixed['var_slope_coeffs'][Diff.single]
        )
        double_gen = zip(
            fixed['t_bar_diffs'][Diff.double],
            fixed['t_bar_diff_sqrs'][Diff.double],
            fixed['read_recip_coeffs'][Diff.double],
            fixed['var_slope_coeffs'][Diff.double]
        )

        for index, (t_bar_diff_1, t_bar_diff_sqr_1, read_recip_1, var_slope_1) in enumerate(single_gen):
            assert t_bar_diff_1 == t_bar[index + 1] - t_bar[index]
            assert t_bar_diff_sqr_1 == np.float32((t_bar[index + 1] - t_bar[index]) ** 2)
            assert read_recip_1 == np.float32(1 / n_reads[index + 1]) + np.float32(1 / n_reads[index])
            assert var_slope_1 == (tau[index + 1] + tau[index] - 2 * min(t_bar[index], t_bar[index + 1]))

        for index, (t_bar_diff_2, t_bar_diff_sqr_2, read_recip_2, var_slope_2) in enumerate(double_gen):
            if index == len(fixed['t_bar_diffs'][1]) - 1:
                # Last value must be NaN
                assert np.isnan(t_bar_diff_2)
                assert np.isnan(read_recip_2)
                assert np.isnan(var_slope_2)
            else:
                assert t_bar_diff_2 == t_bar[index + 2] - t_bar[index]
                assert t_bar_diff_sqr_2 == np.float32((t_bar[index + 2] - t_bar[index])**2)
                assert read_recip_2 == np.float32(1 / n_reads[index + 2]) + np.float32(1 / n_reads[index])
                assert var_slope_2 == (tau[index + 2] + tau[index] - 2 * min(t_bar[index], t_bar[index + 2]))
    else:
        # If not using jumps, these values should not even exist. However, for wrapping
        #    purposes, they are checked to be non-existent and then set to NaN
        assert np.isnan(fixed['t_bar_diffs']).all()
        assert np.isnan(fixed['t_bar_diff_sqrs']).all()
        assert np.isnan(fixed['read_recip_coeffs']).all()
        assert np.isnan(fixed['var_slope_coeffs']).all()


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

    read_pattern, metadata = ramp_data
    resultants = _generate_resultants(read_pattern)

    yield resultants, metadata


@pytest.mark.parametrize("use_jump", [True, False])
def test_make_pixel(pixel_data, use_jump):
    """Test computing the initial pixel data"""
    resultants, metadata = pixel_data

    data = metadata._to_dict()
    t_bar = data['t_bar']
    tau = data['tau']
    n_reads = data['n_reads']

    thresh = {
        'intercept': np.float32(5.5),
        'constant': np.float32(1/3)
    }
    fixed = fixed_values_from_metadata(metadata, thresh, use_jump)

    # Note this is converted to a dictionary so we can directly interrogate the
    #   variables in question
    pixel = make_pixel(fixed, READ_NOISE, resultants)._to_dict()

    # Basic sanity checks that data passed in survives
    assert (pixel['resultants'] == resultants).all()
    assert READ_NOISE == pixel['read_noise']

    # the "fixed" data is not checked as this is already done above

    # Check the computed data
    # These are computed via vectorized operations in the main code, here we
    #    check using item-by-item operations
    if use_jump:
        single_gen = zip(pixel['local_slopes'][Diff.single], pixel['var_read_noise'][Diff.single])
        double_gen = zip(pixel['local_slopes'][Diff.double], pixel['var_read_noise'][Diff.double])

        for index, (local_slope_1, var_read_noise_1) in enumerate(single_gen):
            assert local_slope_1 == (
                (resultants[index + 1] - resultants[index]) / (t_bar[index + 1] - t_bar[index]))
            assert var_read_noise_1 == np.float32(READ_NOISE ** 2)* (
                np.float32(1 / n_reads[index + 1]) + np.float32(1 / n_reads[index])
            )

        for index, (local_slope_2, var_read_noise_2) in enumerate(double_gen):
            if index == len(pixel['local_slopes'][1]) - 1:
                # Last value must be NaN
                assert np.isnan(local_slope_2)
                assert np.isnan(var_read_noise_2)
            else:
                assert local_slope_2 == (
                    (resultants[index + 2] - resultants[index]) / (t_bar[index + 2] - t_bar[index])
                )
                assert var_read_noise_2 == np.float32(READ_NOISE ** 2) * (
                    np.float32(1 / n_reads[index + 2]) + np.float32(1 / n_reads[index])
                )
    else:
        # If not using jumps, these values should not even exist. However, for wrapping
        #    purposes, they are checked to be non-existent and then set to NaN
        assert np.isnan(pixel['local_slopes']).all()
        assert np.isnan(pixel['var_read_noise']).all()


@pytest.fixture(scope="module")
def detector_data(ramp_data):
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
    read_pattern, _ = ramp_data
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
        (RNG.uniform(size=resultants.shape) > GOOD_PROB).astype(np.int32) if use_dq else
        np.zeros(resultants.shape, dtype=np.int32)
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

    output = fit_ramps(resultants, dq, read_noise, READ_TIME, read_pattern, use_jump=use_jump,
                       include_diagnostic=True)
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
            assert len(fit['fits']) == 1

        if use:
            # Add okay ramps to chi2
            total_var = fit['average']['read_var'] + fit['average']['poisson_var']
            if total_var != 0:
                chi2 += (fit['average']['slope'] - FLUX)**2 / total_var
        else:
            # Check no slope fit for bad ramps
            assert fit['average']['slope'] == 0
            assert fit['average']['read_var'] == 0
            assert fit['average']['poisson_var'] == 0

            assert use_dq # sanity check that this branch is only encountered when use_dq = True

    chi2 /= np.sum(okay)
    assert np.abs(chi2 - 1) < CHI2_TOL


@pytest.mark.parametrize("use_jump", [True, False])
def test_fit_ramps_array_outputs(detector_data, use_jump):
    """
    Test that the array outputs line up with the dictionary output
    """
    resultants, read_noise, read_pattern = detector_data
    dq = np.zeros(resultants.shape, dtype=np.int32)

    output = fit_ramps(resultants, dq, read_noise, READ_TIME, read_pattern, use_jump=use_jump,
                       include_diagnostic=True)

    for fit, par, var in zip(output.fits, output.parameters, output.variances):
        assert par[Parameter.intercept] == 0
        assert par[Parameter.slope] == fit['average']['slope']

        assert var[Variance.read_var] == fit['average']['read_var']
        assert var[Variance.poisson_var] == fit['average']['poisson_var']
        assert var[Variance.total_var] == np.float32(
            fit['average']['read_var'] + fit['average']['poisson_var']
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

    output = fit_ramps(resultants, dq, read_noise, READ_TIME, read_pattern, use_jump=True,
                       include_diagnostic=True)
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
            assert len(fit['jumps']) == 0
            assert resultant_index == 0  # sanity check that the jump is indeed in the first resultant

            # Test the correct ramp_index was recorded:
            assert len(fit['index']) == 1
            assert fit['index'][0]['start'] == 0
            assert fit['index'][0]['end'] == len(read_pattern) - 1
        else:
            # There should be a single jump detected; however, this results in
            # two resultants being excluded.
            if resultant_index not in fit['jumps']:
                incorrect_does_not_capture += 1
                continue
            if len(fit['jumps']) < 2:
                incorrect_too_few += 1
                continue
            if len(fit['jumps']) > 2:
                incorrect_too_many += 1
                continue

            # The two resultants excluded should be adjacent
            jump_correct = []
            for jump in fit['jumps']:
                jump_correct.append(jump == resultant_index or
                                    jump == resultant_index - 1 or
                                    jump == resultant_index + 1)
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
        total_var = fit['average']['read_var'] + fit['average']['poisson_var']
        chi2 += (fit['average']['slope'] - FLUX)**2 / total_var

    # Check that the average chi2 is ~1.
    chi2 /= (N_PIXELS - incorrect_too_few - incorrect_too_many - incorrect_does_not_capture - incorrect_other)
    assert np.abs(chi2 - 1) < CHI2_TOL


def test_override_default_threshold(jump_data):
    """This tests that we can override the default jump detection threshold constants"""
    resultants, read_noise, read_pattern, jump_reads, jump_resultants = jump_data
    dq = np.zeros(resultants.shape, dtype=np.int32)

    standard = fit_ramps(resultants, dq, read_noise, READ_TIME, read_pattern, use_jump=True)
    override = fit_ramps(resultants, dq, read_noise, READ_TIME, read_pattern, use_jump=True,
                         intercept=0, constant=0)

    # All this is intended to do is show that with all other things being equal passing non-default
    #    threshold parameters changes the results.
    assert (standard.parameters != override.parameters).any()


def test_jump_dq_set(jump_data):
    # Check the DQ flag value to start
    assert RampJumpDQ.JUMP_DET == 2**2

    resultants, read_noise, read_pattern, jump_reads, jump_resultants = jump_data
    dq = np.zeros(resultants.shape, dtype=np.int32)

    output = fit_ramps(resultants, dq, read_noise, READ_TIME, read_pattern, use_jump=True,
                       include_diagnostic=True)

    for fit, pixel_dq in zip(output.fits, output.dq.transpose()):
        # Check that all jumps found get marked
        assert (pixel_dq[fit['jumps']] == RampJumpDQ.JUMP_DET).all()

        # Check that dq flags for jumps are only set if the jump is marked
        assert set(np.where(pixel_dq == RampJumpDQ.JUMP_DET)[0]) == set(fit['jumps'])
