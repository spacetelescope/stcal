import numpy as np
import pytest
from numpy.testing import assert_allclose

from stcal.ramp_fitting.ols_cas22._wrappers import read_data
from stcal.ramp_fitting.ols_cas22._wrappers import init_ramps
from stcal.ramp_fitting.ols_cas22._wrappers import run_threshold, make_fixed, make_pixel, fit_ramp

from stcal.ramp_fitting.ols_cas22 import fit_ramps


RNG = np.random.default_rng(619)
ROMAN_READ_TIME = 3.04
N_PIXELS = 100_000
FLUX = 100
JUMP_VALUE = 10_000
CHI2_TOL = 0.03


@pytest.fixture(scope="module")
def base_ramp_data():
    """Basic data for simulating ramps for testing (not unpacked)"""
    read_pattern = [
        [1, 2, 3, 4],
        [5],
        [6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        [19, 20, 21],
        [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    ]

    yield read_pattern, read_data(read_pattern, ROMAN_READ_TIME)


def test_read_data(base_ramp_data):
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
    """Unpacked data for simulating ramps for testing"""
    t_bar = np.array(base_ramp_data[1]['t_bar'], dtype=np.float32)
    tau = np.array(base_ramp_data[1]['tau'], dtype=np.float32)
    n_reads = np.array(base_ramp_data[1]['n_reads'], dtype=np.int32)

    yield base_ramp_data[0], t_bar, tau, n_reads


@pytest.mark.parametrize("use_jump", [True, False])
def test_make_fixed(ramp_data, use_jump):
    """Test computing the fixed data for all pixels"""
    _, t_bar, tau, n_reads = ramp_data

    intercept = np.float32(5.5)
    constant = np.float32(1/3)

    fixed = make_fixed(t_bar, tau, n_reads, intercept, constant, use_jump)

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
        single_gen = zip(fixed['t_bar_diff'][0], fixed['recip'][0], fixed['slope_var'][0])
        double_gen = zip(fixed['t_bar_diff'][1], fixed['recip'][1], fixed['slope_var'][1])

        for index, (t_bar_1, recip_1, slope_var_1) in enumerate(single_gen):
            assert t_bar_1 == t_bar[index + 1] - t_bar[index]
            assert recip_1 == np.float32(1 / n_reads[index + 1]) + np.float32(1 / n_reads[index])
            assert slope_var_1 == (tau[index + 1] + tau[index] - min(t_bar[index], t_bar[index + 1]))

        for index, (t_bar_2, recip_2, slope_var_2) in enumerate(double_gen):
            if index == len(fixed['t_bar_diff'][1]) - 1:
                # Last value must be NaN
                assert np.isnan(t_bar_2)
                assert np.isnan(recip_2)
                assert np.isnan(slope_var_2)
            else:
                assert t_bar_2 == t_bar[index + 2] - t_bar[index]
                assert recip_2 == np.float32(1 / n_reads[index + 2]) + np.float32(1 / n_reads[index])
                assert slope_var_2 == (tau[index + 2] + tau[index] - min(t_bar[index], t_bar[index + 2]))
    else:
        # If not using jumps, these values should not even exist. However, for wrapping
        #    purposes, they are checked to be non-existent and then set to NaN
        assert np.isnan(fixed['t_bar_diff']).all()
        assert np.isnan(fixed['recip']).all()
        assert np.isnan(fixed['slope_var']).all()


def _generate_resultants(read_pattern, flux, read_noise, n_pixels=1):
    """Generate a set of resultants for a pixel"""
    resultants = np.zeros((len(read_pattern), n_pixels), dtype=np.float32)

    # Use Poisson process to simulate the accumulation of the ramp
    ramp_value = np.zeros(n_pixels, dtype=np.float32)  # Last value of ramp
    for index, reads in enumerate(read_pattern):
        resultant_total = np.zeros(n_pixels, dtype=np.float32)  # Total of all reads in this resultant
        for _ in reads:
            # Compute the next value of the ramp
            #   - Poisson process for the flux
            #   - Gaussian process for the read noise
            ramp_value += RNG.poisson(flux * ROMAN_READ_TIME, size=n_pixels).astype(np.float32)
            ramp_value += RNG.standard_normal(size=n_pixels, dtype=np.float32) * read_noise / np.sqrt(len(reads))

            # Add to running total for the resultant
            resultant_total += ramp_value

        # Record the average value for resultant (i.e., the average of the reads)
        resultants[index] = (resultant_total / len(reads)).astype(np.float32)

    if n_pixels == 1:
        resultants = resultants[:, 0]

    return resultants


@pytest.fixture(scope="module")
def pixel_data(ramp_data):
    """Create data for a single pixel"""
    read_noise = np.float32(5)

    read_pattern, t_bar, tau, n_reads = ramp_data
    resultants = _generate_resultants(read_pattern, FLUX, read_noise)

    yield resultants, t_bar, tau, n_reads, read_noise, FLUX


@pytest.mark.parametrize("use_jump", [True, False])
def test_make_pixel(pixel_data, use_jump):
    """Test computing the initial pixel data"""
    resultants, t_bar, tau, n_reads, read_noise, _ = pixel_data

    intercept = np.float32(5.5)
    constant = np.float32(1/3)

    pixel = make_pixel(resultants, t_bar, tau, n_reads, read_noise, intercept, constant, use_jump)

    # Basic sanity checks that data passed in survives
    assert (pixel['resultants'] == resultants).all()
    assert read_noise == pixel['read_noise']

    # Check the computed data
    # These are computed via vectorized operations in the main code, here we
    #    check using item-by-item operations
    if use_jump:
        single_gen = zip(pixel['delta'][0], pixel['sigma'][0])
        double_gen = zip(pixel['delta'][1], pixel['sigma'][1])

        for index, (delta_1, sigma_1) in enumerate(single_gen):
            assert delta_1 == (resultants[index + 1] - resultants[index]) / (t_bar[index + 1] - t_bar[index])
            assert sigma_1 == read_noise * (
                np.float32(1 / n_reads[index + 1]) + np.float32(1 / n_reads[index])
            )

        for index, (delta_2, sigma_2) in enumerate(double_gen):
            if index == len(pixel['delta'][1]) - 1:
                # Last value must be NaN
                assert np.isnan(delta_2)
                assert np.isnan(sigma_2)
            else:
                assert delta_2 == (resultants[index + 2] - resultants[index]) / (t_bar[index + 2] - t_bar[index])
                assert sigma_2 == read_noise * (
                    np.float32(1 / n_reads[index + 2]) + np.float32(1 / n_reads[index])
                )
    else:
        # If not using jumps, these values should not even exist. However, for wrapping
        #    purposes, they are checked to be non-existent and then set to NaN
        assert np.isnan(pixel['delta']).all()
        assert np.isnan(pixel['sigma']).all()


@pytest.fixture(scope="module")
def detector_data(ramp_data):
    """
    Generate a set of with no jumps data as if for a single detector as it
        would be passed in by the supporting code.
    """
    read_pattern, *_ = ramp_data
    read_noise = np.ones(N_PIXELS, dtype=np.float32) * 5

    resultants = _generate_resultants(read_pattern, FLUX, read_noise, n_pixels=N_PIXELS)

    return resultants, read_noise, read_pattern, N_PIXELS, FLUX


@pytest.mark.parametrize("use_jump", [True, False])
def test_fit_ramps_no_dq(detector_data, use_jump):
    """
    Test fitting ramps with no dq flags set on data which has no jumps
        Since no jumps are simulated in the data, jump detection shouldn't pick
        up any jumps.
    """
    resultants, read_noise, read_pattern, n_pixels, flux = detector_data
    dq = np.zeros(resultants.shape, dtype=np.int32)

    fits = fit_ramps(resultants, dq, read_noise, ROMAN_READ_TIME, read_pattern, use_jump=use_jump)
    assert len(fits) == n_pixels  # sanity check that a fit is output for each pixel

    # Check that the chi2 for the resulting fit relative to the assumed flux is ~1
    chi2 = 0
    for fit in fits:
        assert len(fit['fits']) == 1  # only one fit per pixel since no dq/jump

        total_var = fit['average']['read_var'] + fit['average']['poisson_var']
        chi2 += (fit['average']['slope'] - flux)**2 / total_var

    chi2 /= n_pixels

    assert np.abs(chi2 - 1) < CHI2_TOL


@pytest.mark.parametrize("use_jump", [True, False])
def test_fit_ramps_dq(detector_data, use_jump):
    """
    Test fitting ramps with dq flags set
        Since no jumps are simulated in the data, jump detection shouldn't pick
        up any jumps.
    """
    resultants, read_noise, read_pattern, n_pixels, flux = detector_data
    dq = np.zeros(resultants.shape, dtype=np.int32) + (RNG.uniform(size=resultants.shape) > 1).astype(np.int32)

    # only use okay ramps
    #   ramps passing the below criterion have at least two adjacent valid reads
    #   i.e., we can make a measurement from them.
    okay = np.sum((dq[1:, :] == 0) & (dq[:-1, :] == 0), axis=0) != 0

    fits = fit_ramps(resultants, dq, read_noise, ROMAN_READ_TIME, read_pattern, use_jump=use_jump)
    assert len(fits) == n_pixels  # sanity check that a fit is output for each pixel

    chi2 = 0
    for fit, use in zip(fits, okay):
        if use:
            # Add okay ramps to chi2
            total_var = fit['average']['read_var'] + fit['average']['poisson_var']
            chi2 += (fit['average']['slope'] - flux)**2 / total_var
        else:
            # Check no slope fit for bad ramps
            assert fit['average']['slope'] == 0
            assert fit['average']['read_var'] == 0
            assert fit['average']['poisson_var'] == 0

    chi2 /= np.sum(okay)
    assert np.abs(chi2 - 1) < CHI2_TOL


@pytest.fixture(scope="module")
def jump_data():
    """
    Generate a set of data were jumps are simulated in each possible read.
        - jumps should occur in read of same index as the pixel index.
    """

    # Generate a read pattern with 8 reads per resultant
    shape = (8, 8)
    read_pattern = np.arange(np.prod(shape)).reshape(shape).tolist()

    resultants = np.zeros((len(read_pattern), np.prod(shape)), dtype=np.float32)
    jumps = np.zeros((len(read_pattern), np.prod(shape)), dtype=bool)
    jump_res = -1
    for jump_index in range(np.prod(shape)):
        read_values = np.zeros(np.prod(shape), dtype=np.float32)
        for index in range(np.prod(shape)):
            if index >= jump_index:
                read_values[index] = JUMP_VALUE

        if jump_index % shape[1] == 0:
            # Start indicating a new resultant
            jump_res += 1
        jumps[jump_res, jump_index] = True
        
        resultants[:, jump_index] = np.mean(read_values.reshape(shape), axis=1).astype(np.float32)

    n_pixels = np.prod(shape)
    read_noise = np.ones(n_pixels, dtype=np.float32) * 5

    # Add actual ramp data in addition to the jump data
    resultants += _generate_resultants(read_pattern, FLUX, read_noise, n_pixels=n_pixels)

    return resultants, read_noise, read_pattern, n_pixels, jumps.transpose()


def test_find_jumps(jump_data):
    """
    Check that we can locate all the jumps in a given ramp
    """
    resultants, read_noise, read_pattern, n_pixels, jumps = jump_data
    dq = np.zeros(resultants.shape, dtype=np.int32)

    fits = fit_ramps(resultants, dq, read_noise, ROMAN_READ_TIME, read_pattern, use_jump=True)

    # Check that all the jumps have been located per the algorithm's constraints
    for index, (fit, jump) in enumerate(zip(fits, jumps)):
        # sanity check that only one jump should have been added
        assert np.where(jump)[0].shape == (1,)
        if index == 0:
            # There is no way to detect a jump if it is in the very first read
            # The very first pixel in this case has a jump in the first read
            assert len(fit['jumps']) == 0
            assert jump[0]
            assert not np.all(jump[1:])

            # Test that the correct index was recorded
            assert len(fit['index']) == 1
            assert fit['index'][0]['start'] == 0
            assert fit['index'][0]['end'] == len(read_pattern) - 1
        else:
            # Select the single jump and check that it is recorded as a jump
            assert np.where(jump)[0][0] in fit['jumps']

            # In all cases here we have to exclude two resultants
            assert len(fit['jumps']) == 2

            # Test that all the jumps recorded are +/- 1 of the real jump
            #    This is due to the need to exclude two resultants
            for jump_index in fit['jumps']:
                assert jump[jump_index] or jump[jump_index + 1] or jump[jump_index - 1]

            # Test that the correct indexes are recorded
            ramp_indicies = []
            for ramp_index in fit["index"]:
                # Note start/end of a ramp_index are inclusive meaning that end
                #    is an index included in the ramp_index so the range is to end + 1
                new_indicies = list(range(ramp_index["start"], ramp_index["end"] + 1))

                # check that all the ramps are non-overlapping
                assert set(ramp_indicies).isdisjoint(new_indicies)

                ramp_indicies.extend(new_indicies)

            # check that no ramp_index is a jump
            assert set(ramp_indicies).isdisjoint(fit['jumps'])

            # check that all resultant indicies are either in a ramp or listed as a jump
            assert set(ramp_indicies).union(fit['jumps']) == set(range(len(read_pattern)))

    # Check that the slopes have been estimated reasonably well
    #   There are not that many pixels to test this against and many resultants
    #   have been thrown out due to the jumps. Thus we only check the slope is
    #   "fairly close" to the expected value. This is purposely a loose check
    #   because the main purpose of this test is to verify that the jumps are
    #   being detected correctly, above.
    chi2 = 0
    for fit in fits:
        assert_allclose(fit['average']['slope'], FLUX, rtol=3)
