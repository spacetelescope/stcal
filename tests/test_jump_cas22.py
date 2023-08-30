import numpy as np
from numpy.testing import assert_allclose

from stcal.ramp_fitting.ols_cas22._wrappers import read_data
from stcal.ramp_fitting.ols_cas22._wrappers import init_ramps
from stcal.ramp_fitting.ols_cas22._wrappers import make_threshold, run_threshold

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
    intercept = 5.5
    constant = 1/3

    thresh = make_threshold(intercept, constant)

    # Parameters are not directly accessible
    assert intercept == run_threshold(thresh, 1.0) # check intercept
    assert_allclose(intercept - constant, run_threshold(thresh, 10.0)) # check constant


def test_make_fixed():
    pass
