import numpy as np
import pytest
from numpy.testing import assert_allclose

from stcal.ramp_fitting.ols_cas22._wrappers import read_data
from stcal.ramp_fitting.ols_cas22._wrappers import init_ramps
from stcal.ramp_fitting.ols_cas22._wrappers import make_threshold, run_threshold, make_fixed, make_pixel

def test_read_data():
    """Test turning read_pattern into the time data"""
    pattern = [[1, 2], [4, 5, 6], [7], [8, 9, 10, 11]]
    data = read_data(pattern, 3.0)

    # Basic sanity checks (structs become dicts)
    assert isinstance(data, dict)
    assert 't_bar' in data
    assert 'tau' in data
    assert 'n_reads' in data
    assert len(data) == 3

    # Check that the data is correct
    assert_allclose(data['t_bar'], [4.5, 15, 21, 28.5])
    assert_allclose(data['tau'], [3.75, 13.666667, 21, 26.625])
    assert data['n_reads'] == [2, 3, 1, 4]


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
    intercept = np.float32(5.5)
    constant = np.float32(1/3)

    thresh = make_threshold(intercept, constant)

    # Parameters are not directly accessible
    assert intercept == run_threshold(thresh, 1.0) # check intercept
    assert np.float32(intercept - constant) == run_threshold(thresh, 10.0) # check constant


@pytest.mark.parametrize("use_jump", [True, False])
def test_make_fixed(use_jump):
    pattern = [[1, 2], [4, 5, 6], [7], [8, 9, 10, 11]]
    data = read_data(pattern, 3.0)

    t_bar = np.array(data['t_bar'], dtype=np.float32)
    tau = np.array(data['tau'], dtype=np.float32)
    n_reads = np.array(data['n_reads'], dtype=np.int32)
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
            assert t_bar_1_sq == (t_bar[index + 1] - t_bar[index])**2
            assert recip_1 == np.float32(1 / n_reads[index + 1]) + np.float32(1 / n_reads[index])
            assert slope_var_1 == (tau[index + 1] + tau[index] - min(t_bar[index], t_bar[index + 1]))

        for index, (t_bar_2, t_bar_2_sq, recip_2, slope_var_2) in enumerate(double_gen):
            assert t_bar_2 == t_bar[index + 2] - t_bar[index]
            assert t_bar_2_sq == (t_bar[index + 2] - t_bar[index])**2
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


@pytest.mark.parametrize("use_jump", [True, False])
def test_make_pixel(use_jump):
    pattern = [[1, 2], [4, 5, 6], [7], [8, 9, 10, 11]]
    data = read_data(pattern, 3.0)

    resultants = np.random.random(4).astype(np.float32)
    read_noise = np.float32(1.4)
    t_bar = np.array(data['t_bar'], dtype=np.float32)
    tau = np.array(data['tau'], dtype=np.float32)
    n_reads = np.array(data['n_reads'], dtype=np.int32)
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
