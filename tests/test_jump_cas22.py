import numpy as np
import pytest
from numpy.testing import assert_allclose

from stcal.ramp_fitting.ols_cas22._wrappers import metadata_from_read_pattern
from stcal.ramp_fitting.ols_cas22._wrappers import init_ramps
from stcal.ramp_fitting.ols_cas22._wrappers import run_threshold, fixed_values_from_metadata, make_pixel

from stcal.ramp_fitting.ols_cas22 import fit_ramps, Parameter, Variance, Diff


RNG = np.random.default_rng(619)
ROMAN_READ_TIME = 3.04
READ_NOISE = np.float32(5)
N_PIXELS = 100_000
FLUX = 100
JUMP_VALUE = 10_000
CHI2_TOL = 0.03


@pytest.fixture(scope="module")
def base_ramp_data():
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

    yield read_pattern, metadata_from_read_pattern(read_pattern, ROMAN_READ_TIME)


def test_metadata_from_read_pattern(base_ramp_data):
    """Test turning read_pattern into the time data"""
    _, data = base_ramp_data

    # Basic sanity checks (structs become dicts)
    assert isinstance(data, dict)
    assert 't_bar' in data
    assert 'tau' in data
    assert 'n_reads' in data
    assert len(data) == 3

    # Check that the data is correct
    assert_allclose(data['t_bar'], [7.6, 15.2, 21.279999, 41.040001, 60.799999, 88.159996])
    assert_allclose(data['tau'], [5.7, 15.2, 19.928888, 36.023998, 59.448887, 80.593781])
    assert data['n_reads'] == [4, 1, 3, 10, 3, 15]


def test_init_ramps():
    """Test turning dq flags into initial ramp splits"""
    dq = np.array([[0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
                   [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                   [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                   [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]], dtype=np.int32)

    ramps = init_ramps(dq)
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
    """Test the threshold object/fucnction)"""
    intercept = np.float32(5.5)
    constant = np.float32(1/3)

    assert intercept == run_threshold(intercept, constant, 1.0) # check intercept
    assert np.float32(intercept - constant) == run_threshold(intercept, constant, 10.0) # check constant


@pytest.fixture(scope="module")
def ramp_data(base_ramp_data):
    """
    Unpacked metadata for simulating ramps for testing

    Returns
    -------
        read_pattern:
            The read pattern used for testing
        t_bar:
            The t_bar values for the read pattern
        tau:
            The tau values for the read pattern
        n_reads:
            The number of reads for the read pattern
    """
    read_pattern, read_pattern_metadata = base_ramp_data
    t_bar = np.array(read_pattern_metadata['t_bar'], dtype=np.float32)
    tau = np.array(read_pattern_metadata['tau'], dtype=np.float32)
    n_reads = np.array(read_pattern_metadata['n_reads'], dtype=np.int32)

    yield read_pattern, t_bar, tau, n_reads


@pytest.mark.parametrize("use_jump", [True, False])
def test_fixed_values_from_metadata(ramp_data, use_jump):
    """Test computing the fixed data for all pixels"""
    _, t_bar, tau, n_reads = ramp_data

    intercept = np.float32(5.5)
    constant = np.float32(1/3)

    fixed = fixed_values_from_metadata(t_bar, tau, n_reads, intercept, constant, use_jump)

    # Basic sanity checks that data passed in survives
    assert (fixed['data']['t_bar'] == t_bar).all()
    assert (fixed['data']['tau'] == tau).all()
    assert (fixed['data']['n_reads'] == n_reads).all()
    assert fixed["intercept"] == intercept
    assert fixed["constant"] == constant

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
            assert var_slope_1 == (tau[index + 1] + tau[index] - min(t_bar[index], t_bar[index + 1]))

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
                assert var_slope_2 == (tau[index + 2] + tau[index] - min(t_bar[index], t_bar[index + 2]))
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
            ramp_value += RNG.poisson(FLUX * ROMAN_READ_TIME, size=n_pixels).astype(np.float32)

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

    read_pattern, t_bar, tau, n_reads = ramp_data
    resultants = _generate_resultants(read_pattern)

    yield resultants, t_bar, tau, n_reads


@pytest.mark.parametrize("use_jump", [True, False])
def test_make_pixel(pixel_data, use_jump):
    """Test computing the initial pixel data"""
    resultants, t_bar, tau, n_reads = pixel_data

    intercept = np.float32(5.5)
    constant = np.float32(1/3)

    pixel = make_pixel(resultants, t_bar, tau, n_reads, READ_NOISE, intercept, constant, use_jump)

    # Basic sanity checks that data passed in survives
    assert (pixel['resultants'] == resultants).all()
    assert READ_NOISE == pixel['read_noise']

    # Check the computed data
    # These are computed via vectorized operations in the main code, here we
    #    check using item-by-item operations
    if use_jump:
        single_gen = zip(pixel['local_slopes'][Diff.single], pixel['var_read_noise'][Diff.single])
        double_gen = zip(pixel['local_slopes'][Diff.double], pixel['var_read_noise'][Diff.double])

        for index, (local_slope_1, var_read_noise_1) in enumerate(single_gen):
            assert local_slope_1 == (
                (resultants[index + 1] - resultants[index]) / (t_bar[index + 1] - t_bar[index]))
            assert var_read_noise_1 == READ_NOISE * (
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
                assert var_read_noise_2 == READ_NOISE * (
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
    read_pattern, *_ = ramp_data
    read_noise = np.ones(N_PIXELS, dtype=np.float32) * READ_NOISE

    resultants = _generate_resultants(read_pattern, n_pixels=N_PIXELS)

    return resultants, read_noise, read_pattern

@pytest.mark.parametrize("use_jump", [True, False])
def test_fit_ramps_array_outputs(detector_data, use_jump):
    """
    Test that the array outputs line up with the dictionary output
    """
    resultants, read_noise, read_pattern = detector_data
    dq = np.zeros(resultants.shape, dtype=np.int32)

    fits, parameters, variances = fit_ramps(
        resultants, dq, read_noise, ROMAN_READ_TIME, read_pattern, use_jump=use_jump
    )

    for fit, par, var in zip(fits, parameters, variances):
        assert par[Parameter.intercept] == 0
        assert par[Parameter.slope] == fit['average']['slope']

        assert var[Variance.read_var] == fit['average']['read_var']
        assert var[Variance.poisson_var] == fit['average']['poisson_var']
        assert var[Variance.total_var] == np.float32(
            fit['average']['read_var'] + fit['average']['poisson_var']
        )


@pytest.mark.parametrize("use_jump", [True, False])
def test_fit_ramps_no_dq(detector_data, use_jump):
    """
    Test fitting ramps with no dq flags set on data which has no jumps
        Since no jumps are simulated in the data, jump detection shouldn't pick
        up any jumps.
    """
    resultants, read_noise, read_pattern  = detector_data
    dq = np.zeros(resultants.shape, dtype=np.int32)

    fits, _, _ = fit_ramps(resultants, dq, read_noise, ROMAN_READ_TIME, read_pattern, use_jump=use_jump)
    assert len(fits) == N_PIXELS # sanity check that a fit is output for each pixel

    # Check that the chi2 for the resulting fit relative to the assumed flux is ~1
    chi2 = 0
    for fit in fits:
        assert len(fit['fits']) == 1  # only one fit per pixel since no dq/jump

        total_var = fit['average']['read_var'] + fit['average']['poisson_var']
        chi2 += (fit['average']['slope'] - FLUX)**2 / total_var

    chi2 /= N_PIXELS

    assert np.abs(chi2 - 1) < CHI2_TOL


@pytest.mark.parametrize("use_jump", [True, False])
def test_fit_ramps_dq(detector_data, use_jump):
    """
    Test fitting ramps with dq flags set
        Since no jumps are simulated in the data, jump detection shouldn't pick
        up any jumps.
    """
    resultants, read_noise, read_pattern = detector_data
    dq = (RNG.uniform(size=resultants.shape) > 1).astype(np.int32)

    # only use okay ramps
    #   ramps passing the below criterion have at least two adjacent valid reads
    #   i.e., we can make a measurement from them.
    okay = np.sum((dq[1:, :] == 0) & (dq[:-1, :] == 0), axis=0) != 0

    fits, _, _ = fit_ramps(resultants, dq, read_noise, ROMAN_READ_TIME, read_pattern, use_jump=use_jump)
    assert len(fits) == N_PIXELS  # sanity check that a fit is output for each pixel

    chi2 = 0
    for fit, use in zip(fits, okay):
        if use:
            # Add okay ramps to chi2
            total_var = fit['average']['read_var'] + fit['average']['poisson_var']
            chi2 += (fit['average']['slope'] - FLUX)**2 / total_var
        else:
            # Check no slope fit for bad ramps
            assert fit['average']['slope'] == 0
            assert fit['average']['read_var'] == 0
            assert fit['average']['poisson_var'] == 0

    chi2 /= np.sum(okay)
    assert np.abs(chi2 - 1) < CHI2_TOL


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

    fits, _, _ = fit_ramps(resultants, dq, read_noise, ROMAN_READ_TIME, read_pattern, use_jump=True)
    assert len(fits) == len(jump_reads)  # sanity check that a fit/jump is set for every pixel

    chi2 = 0
    for fit, jump_index, resultant_index in zip(fits, jump_reads, jump_resultants):

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
            assert len(fit['jumps']) == 2
            assert resultant_index in fit['jumps']

            # The two resultants excluded should be adjacent
            for jump in fit['jumps']:
                assert jump == resultant_index or jump == resultant_index - 1 or jump == resultant_index + 1

            # Test the correct ramp indexes are recorded
            ramp_indices = []
            for ramp_index in fit['index']:
                # Note start/end of a ramp_index are inclusive meaning that end
                #    is an index included in the ramp_index so the range is to end + 1
                new_indices = list(range(ramp_index["start"], ramp_index["end"] + 1))

                # check that all the ramps are non-overlapping
                assert set(ramp_indices).isdisjoint(new_indices)

                ramp_indices.extend(new_indices)

            # check that no ramp_index is a jump
            assert set(ramp_indices).isdisjoint(fit['jumps'])

            # check that all resultant indices are either in a ramp or listed as a jump
            assert set(ramp_indices).union(fit['jumps']) == set(range(len(read_pattern)))

        # Compute the chi2 for the fit and add it to a running "total chi2"
        total_var = fit['average']['read_var'] + fit['average']['poisson_var']
        chi2 += (fit['average']['slope'] - FLUX)**2 / total_var

    # Check that the average chi2 is ~1.
    chi2 /= N_PIXELS
    assert np.abs(chi2 - 1) < CHI2_TOL


def test_override_default_threshold(jump_data):
    """This tests that we can override the default jump detection threshold constants"""
    resultants, read_noise, read_pattern, jump_reads, jump_resultants = jump_data
    dq = np.zeros(resultants.shape, dtype=np.int32)

    _, standard, _ = fit_ramps(resultants, dq, read_noise, ROMAN_READ_TIME, read_pattern, use_jump=True)
    _, override, _ = fit_ramps(resultants, dq, read_noise, ROMAN_READ_TIME, read_pattern, use_jump=True,
                                    intercept=0, constant=0)

    # All this is intended to do is show that with all other things being equal passing non-default
    #    threshold parameters changes the results.
    assert (standard != override).any()
    
