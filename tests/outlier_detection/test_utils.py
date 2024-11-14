import warnings

import gwcs
import pytest
import numpy as np
import scipy.signal
from astropy.modeling import models

from stcal.outlier_detection.utils import (
    _abs_deriv,
    compute_weight_threshold,
    flag_crs,
    flag_resampled_crs,
    gwcs_blot,
    calc_gwcs_pixmap,
    reproject,
    medfilt,
)
from stcal.testing_helpers import MemoryThreshold


@pytest.mark.parametrize("shape,diff", [
    ([5, 7], 100),
    ([17, 13], -200),
])
def test_abs_deriv_single_value(shape, diff):
    arr = np.zeros(shape)
    # put diff at the center
    np.put(arr, arr.size // 2, diff)
    # since abs_deriv with a single non-zero value is the same as a
    # convolution with a 3x3 cross kernel use it to test the result
    expected = scipy.signal.convolve2d(np.abs(arr), [[0, 1, 0], [1, 1, 1], [0, 1, 0]], mode='same')
    result = _abs_deriv(arr)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("nrows,ncols", [(5, 5), (7, 11), (17, 13)])
def test_abs_deriv_range(nrows, ncols):
    arr = np.arange(nrows * ncols).reshape(nrows, ncols)
    result = _abs_deriv(arr)
    np.testing.assert_allclose(result, ncols)


def test_abs_deriv_nan():
    arr = np.arange(25, dtype='f4').reshape(5, 5)
    arr[2, 2] = np.nan
    expect_nan = np.zeros_like(arr, dtype=bool)
    expect_nan[2, 2] = True
    result = _abs_deriv(arr)
    assert np.isnan(result[expect_nan])
    assert np.all(np.isfinite(result[~expect_nan]))


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


def test_compute_weight_threshold_memory():
    """Test that weight threshold function modifies
    the weight array in place"""
    arr = np.zeros([500, 500], dtype=np.float32)
    arr[:250, :250] = 42
    arr[10,10] = 0
    arr[-10,-10] = np.nan

    # buffer to account for memory overhead needs to be small enough
    # to ensure that the array was not copied
    fractional_memory_buffer = 0.9
    expected_mem = int(arr.nbytes*fractional_memory_buffer)
    with MemoryThreshold(str(expected_mem) + " B"):
        result = compute_weight_threshold(arr, 0.5)
    np.testing.assert_allclose(result, 21)


def test_flag_crs():
    sci = np.zeros((10, 10), dtype=np.float32)
    err = np.ones_like(sci)
    blot = np.zeros_like(sci)
    # add a cr
    sci[2, 3] = 10
    crs = flag_crs(sci, err, blot, 1)
    ys, xs = np.where(crs)
    np.testing.assert_equal(ys, 2)
    np.testing.assert_equal(xs, 3)


def test_flag_resampled_crs():
    sci = np.zeros((10, 10), dtype=np.float32)
    err = np.ones_like(sci)
    blot = np.zeros_like(sci)
    # add a cr
    sci[2, 3] = 10

    snr1, snr2 = 5, 4
    scale1, scale2 = 1.2, 0.7
    backg = 0.0
    crs = flag_resampled_crs(sci, err, blot, snr1, snr2, scale1, scale2, backg)
    ys, xs = np.where(crs)
    np.testing.assert_equal(ys, 2)
    np.testing.assert_equal(xs, 3)


def test_gwcs_blot():
    # set up a very simple wcs that scales by 1x
    output_frame = gwcs.Frame2D(name="world")
    forward_transform = models.Scale(1) & models.Scale(1)

    median_data = np.arange(100, dtype=np.float32).reshape((10, 10))
    median_wcs = gwcs.WCS(forward_transform, output_frame=output_frame)
    blot_shape = (5, 5)
    blot_wcs = gwcs.WCS(forward_transform, output_frame=output_frame)
    pix_ratio = 1.0

    blotted = gwcs_blot(median_data, median_wcs, blot_shape, blot_wcs, pix_ratio)
    # since the median data is larger and the wcs are equivalent the blot
    # will window the data to the shape of the blot data
    assert blotted.shape == blot_shape
    np.testing.assert_equal(blotted, median_data[:blot_shape[0], :blot_shape[1]])


@pytest.mark.parametrize('fillval', [0.0, np.nan])
def test_gwcs_blot_fillval(fillval):
    # set up a very simple wcs that scales by 1x
    output_frame = gwcs.Frame2D(name="world")
    forward_transform = models.Scale(1) & models.Scale(1)

    median_shape = (10, 10)
    median_data = np.arange(100, dtype=np.float32).reshape((10, 10))
    median_wcs = gwcs.WCS(forward_transform, output_frame=output_frame)
    blot_shape = (20, 20)
    blot_wcs = gwcs.WCS(forward_transform, output_frame=output_frame)
    pix_ratio = 1.0

    blotted = gwcs_blot(median_data, median_wcs, blot_shape, blot_wcs,
                        pix_ratio, fillval=fillval)

    # since the blot data is larger and the wcs are equivalent the blot
    # will contain the median data + some fill values
    assert blotted.shape == blot_shape
    np.testing.assert_equal(blotted[:median_shape[0], :median_shape[1]], median_data)
    np.testing.assert_equal(blotted[median_shape[0]:, :], fillval)
    np.testing.assert_equal(blotted[:, median_shape[1]:], fillval)


def test_calc_gwcs_pixmap():
    # generate 2 wcses with different scales
    output_frame = gwcs.Frame2D(name="world")
    in_transform = models.Scale(1) & models.Scale(1)
    out_transform = models.Scale(2) & models.Scale(2)
    in_wcs = gwcs.WCS(in_transform, output_frame=output_frame)
    out_wcs = gwcs.WCS(out_transform, output_frame=output_frame)
    in_shape = (3, 4)
    pixmap = calc_gwcs_pixmap(in_wcs, out_wcs, in_shape)
    # we expect given the 2x scale difference to have a pixmap
    # with pixel coordinates / 2
    # use mgrid to generate these coordinates (and reshuffle to match the pixmap)
    expected = np.swapaxes(np.mgrid[:4, :3] / 2., 0, 2)
    np.testing.assert_equal(pixmap, expected)


def test_reproject():
    # generate 2 wcses with different scales
    output_frame = gwcs.Frame2D(name="world")
    wcs1 = gwcs.WCS(models.Scale(1) & models.Scale(1), output_frame=output_frame)
    wcs2 = gwcs.WCS(models.Scale(2) & models.Scale(2), output_frame=output_frame)
    project = reproject(wcs1, wcs2)
    pys, pxs = project(np.array([3]), np.array([1]))
    np.testing.assert_equal(pys, 1.5)
    np.testing.assert_equal(pxs, 0.5)


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
