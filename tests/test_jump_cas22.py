import numpy as np
import pytest
from numpy.testing import assert_allclose

from stcal.ramp_fitting.ols_cas22._wrappers import read_data
from stcal.ramp_fitting.ols_cas22._wrappers import init_ramps
from stcal.ramp_fitting.ols_cas22._wrappers import make_threshold, run_threshold, make_fixed, make_pixel, fit_ramp

from stcal.ramp_fitting.ols_cas22 import fit_ramps


RNG = np.random.default_rng(619)
ROMAN_READ_TIME = 3.04


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

    thresh = make_threshold(intercept, constant)

    # Parameters are not directly accessible
    assert intercept == run_threshold(thresh, 1.0) # check intercept
    assert np.float32(intercept - constant) == run_threshold(thresh, 10.0) # check constant


@pytest.fixture(scope="module")
def ramp_data(base_ramp_data):
    """Upacked data for simulating ramps for testing"""
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
    if use_jump:
        single_gen = zip(fixed['t_bar_1'], fixed['t_bar_1_sq'], fixed['recip_1'], fixed['slope_var_1'])
        double_gen = zip(fixed['t_bar_2'], fixed['t_bar_2_sq'], fixed['recip_2'], fixed['slope_var_2'])

        for index, (t_bar_1, t_bar_1_sq, recip_1, slope_var_1) in enumerate(single_gen):
            assert t_bar_1 == t_bar[index + 1] - t_bar[index]
            assert t_bar_1_sq == np.float32((t_bar[index + 1] - t_bar[index])**2)
            assert recip_1 == np.float32(1 / n_reads[index + 1]) + np.float32(1 / n_reads[index])
            assert slope_var_1 == (tau[index + 1] + tau[index] - min(t_bar[index], t_bar[index + 1]))

        for index, (t_bar_2, t_bar_2_sq, recip_2, slope_var_2) in enumerate(double_gen):
            assert t_bar_2 == t_bar[index + 2] - t_bar[index]
            assert t_bar_2_sq == np.float32((t_bar[index + 2] - t_bar[index])**2)
            assert recip_2 == np.float32(1 / n_reads[index + 2]) + np.float32(1 / n_reads[index])
            assert slope_var_2 == (tau[index + 2] + tau[index] - min(t_bar[index], t_bar[index + 2]))
    else:
        assert fixed['t_bar_1'] == np.zeros(1, np.float32)
        assert fixed['t_bar_2'] == np.zeros(1, np.float32)
        assert fixed['t_bar_1_sq'] == np.zeros(1, np.float32)
        assert fixed['t_bar_2_sq'] == np.zeros(1, np.float32)
        assert fixed['recip_1'] == np.zeros(1, np.float32)
        assert fixed['recip_2'] == np.zeros(1, np.float32)
        assert fixed['slope_var_1'] == np.zeros(1, np.float32)
        assert fixed['slope_var_2'] == np.zeros(1, np.float32)


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
            ramp_value += RNG.standard_normal(size=n_pixels, dtype=np.float32) * read_noise

            # Add to running total for the resultant
            resultant_total += ramp_value

        # Record the average value for resultant (i.e., the average of the reads)
        resultants[index] = (resultant_total / len(reads)).astype(np.float32)

    if n_pixels == 1:
        resultants = resultants[:, 0]

    return resultants


@pytest.fixture(scope="module")
def pixel_data(ramp_data):
    read_noise = np.float32(5)
    flux = 100

    read_pattern, t_bar, tau, n_reads = ramp_data
    resultants = _generate_resultants(read_pattern, flux, read_noise)

    yield resultants, t_bar, tau, n_reads, read_noise, flux


@pytest.mark.parametrize("use_jump", [True, False])
def test_make_pixel(pixel_data, use_jump):
    """Test computing the pixel data"""
    resultants, t_bar, tau, n_reads, read_noise, _ = pixel_data

    intercept = np.float32(5.5)
    constant = np.float32(1/3)

    pixel = make_pixel(resultants, t_bar, tau, n_reads, read_noise, intercept, constant, use_jump)

    assert (pixel['resultants'] == resultants).all()
    assert read_noise == pixel['read_noise']

    if use_jump:
        single_gen = zip(pixel['delta_1'], pixel['sigma_1'])
        double_gen = zip(pixel['delta_2'], pixel['sigma_2'])

        for index, (delta_1, sigma_1) in enumerate(single_gen):
            assert delta_1 == (resultants[index + 1] - resultants[index]) / (t_bar[index + 1] - t_bar[index])
            assert sigma_1 == read_noise * (
                np.float32(1 / n_reads[index + 1]) + np.float32(1 / n_reads[index])
            )

        for index, (delta_2, sigma_2) in enumerate(double_gen):
            assert delta_2 == (resultants[index + 2] - resultants[index]) / (t_bar[index + 2] - t_bar[index])
            assert sigma_2 == read_noise * (
                np.float32(1 / n_reads[index + 2]) + np.float32(1 / n_reads[index])
            )
    else:
        assert pixel['delta_1'] == np.zeros(1, np.float32)
        assert pixel['delta_2'] == np.zeros(1, np.float32)
        assert pixel['sigma_1'] == np.zeros(1, np.float32)
        assert pixel['sigma_2'] == np.zeros(1, np.float32)


def test_fit_ramp_slope(pixel_data):
    """
    Test fitting the slope of a ramp
    """
    resultants, t_bar, tau, n_reads, read_noise, flux = pixel_data

    fit = fit_ramp(resultants, t_bar, tau, n_reads, read_noise, 0, len(resultants) - 1)

    # check that the fit is correct is enough
    assert_allclose(fit['slope'], flux, atol=1, rtol=1e-2)

    # check that the variances and slope are correct relative to each other
    total_var = fit['read_var'] + fit['poisson_var'] * fit['slope']
    chi2 = (fit['slope'] - flux)**2 / total_var**2
    assert np.abs(chi2 - 1) < 0.03


@pytest.fixture(scope="module")
def detector_data(ramp_data):
    read_pattern, *_ = ramp_data

    n_pixels = 100_000
    read_noise = RNG.lognormal(5, size=n_pixels).astype(np.float32)
    flux = 100

    resultants = _generate_resultants(read_pattern, flux, read_noise, n_pixels=n_pixels)

    return resultants, read_noise, read_pattern


# def _compute_averages(slope, read_var, poisson_var):
#     weights = (read_var != 0) / (read_var + (read_var == 0)) # Avoid divide by zero and map those to 0
#     total_weight = np.sum(weights)

#     average_slope = np.sum(weights * slope) / (total_weight + (total_weight == 0))
#     average_read_var = np.sum(weights**2 * read_var) / (total_weight**2 + (total_weight == 0))
#     average_poisson_var = np.sum(weights**2 * poisson_var) / (total_weight**2 + (total_weight == 0)) * average_slope

#     return average_slope, average_read_var, average_poisson_var


def test_fit_ramps(detector_data):
    """
    Test fitting ramps without jump detection
    """
    resultants, read_noise, read_pattern = detector_data
    dq = np.zeros(resultants.shape, dtype=np.int32)

    fit = fit_ramps(resultants, dq, read_noise, ROMAN_READ_TIME, read_pattern, False)

#     slope = np.array(fit['slope'], dtype=np.float32)
#     read_var = np.array(fit['read_var'], dtype=np.float32)
#     poisson_var = np.array(fit['poisson_var'], dtype=np.float32)

#     # Only one slope per pixel
#     assert slope.shape == (resultants.shape[1], 1)
#     assert read_var.shape == (resultants.shape[1], 1)
#     assert poisson_var.shape == (resultants.shape[1], 1)
