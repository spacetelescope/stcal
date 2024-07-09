import warnings

import pytest
import numpy as np
import scipy.signal

from stcal.outlier_detection.utils import (
    _abs_deriv,
    compute_weight_threshold,
    medfilt,
)


@pytest.mark.parametrize("shape,diff", [
    ([5, 5], 100),
    ([7, 7], 200),
])
def test_abs_deriv(shape, diff):
    arr = np.zeros(shape)
    # put diff at the center
    np.put(arr, arr.size // 2, diff)
    # since abs_deriv with a single non-zero value is the same as a
    # convolution with a 3x3 cross kernel use it to test the result
    expected = scipy.signal.convolve2d(arr, [[0, 1, 0], [1, 1, 1], [0, 1, 0]], mode='same')
    result = _abs_deriv(arr)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("shape,mean,maskpt,expected", [
    ([5, 5], 11, 0.5, 5.5),
    ([5, 5], 11, 0.25, 2.75),
    ([3, 3, 3], 17, 0.5, 8.5),
])
def test_compute_weight_threshold(shape, mean, maskpt, expected):
    arr = np.ones(shape, dtype=np.float32) * mean
    result = compute_weight_threshold(arr, maskpt)
    np.testing.assert_allclose(result, expected)


def test_compute_weight_threshold_outlier():
    """
    Test that a large outlier doesn't bias the threshold
    """
    arr = np.ones([7, 7, 7], dtype=np.float32) * 42
    arr[3, 3] = 9000
    result = compute_weight_threshold(arr, 0.5)
    np.testing.assert_allclose(result, 21)


def test_compute_weight_threshold_zeros():
    """
    Test that zeros are ignored
    """
    arr = np.zeros([10, 10], dtype=np.float32)
    arr[:5, :5] = 42
    result = compute_weight_threshold(arr, 0.5)
    np.testing.assert_allclose(result, 21)


@pytest.mark.parametrize("shape,kern_size", [
    ([7, 7], [3, 3]),
    ([7, 7], [3, 1]),
    ([7, 7], [1, 3]),
    ([7, 5], [3, 3]),
    ([5, 7], [3, 3]),
    ([42, 42], [7, 7]),
    ([42, 42], [7, 5]),
    ([42, 42], [5, 7]),
    ([42, 7, 5], [3, 3, 3]),
    ([5, 7, 42], [5, 5, 5]),
])
def test_medfilt_against_scipy(shape, kern_size):
    arr = np.arange(np.prod(shape), dtype='uint32').reshape(shape)
    result = medfilt(arr, kern_size)

    # The use of scipy.signal.medfilt is ok here ONLY because the
    # input has no nans. See the medfilt docstring
    expected = scipy.signal.medfilt(arr, kern_size)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("arr,kern_size,expected", [
    ([2, np.nan, 0], [3], [1, 1, 0]),
    ([np.nan, np.nan, np.nan], [3], [0, np.nan, 0]),
])
def test_medfilt_nan(arr, kern_size, expected):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice",
            category=RuntimeWarning
        )
        result = medfilt(arr, kern_size)
    np.testing.assert_allclose(result, expected)
