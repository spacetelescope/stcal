"""
Utility functions for outlier detection routines
"""
import warnings

import numpy as np
from astropy.stats import sigma_clip
from drizzle.cdrizzle import tblot
from scipy import ndimage
from skimage.util import view_as_windows
import gwcs

from stcal.alignment.util import wcs_bbox_from_shape

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


__all__ = [
    "medfilt",
    "compute_weight_threshold",
    "flag_crs",
    "flag_resampled_crs",
    "gwcs_blot",
    "calc_gwcs_pixmap",
    "reproject",
]


def medfilt(arr, kern_size):
    """
    scipy.signal.medfilt (and many other median filters) have undefined behavior
    for nan inputs. See: https://github.com/scipy/scipy/issues/4800

    Parameters
    ----------
    arr : numpy.ndarray
        The input array

    kern_size : list of int
        List of kernel dimensions, length must be equal to arr.ndim.

    Returns
    -------
    filtered_arr : numpy.ndarray
        Input array median filtered with a kernel of size kern_size
    """
    padded = np.pad(arr, [[k // 2] for k in kern_size])
    windows = view_as_windows(padded, kern_size, np.ones(len(kern_size), dtype='int'))
    return np.nanmedian(windows, axis=np.arange(-len(kern_size), 0))


def compute_weight_threshold(weight, maskpt):
    '''
    Compute the weight threshold for a single image or cube.

    Parameters
    ----------
    weight : numpy.ndarray
        The weight array

    maskpt : float
        The percentage of the mean weight to use as a threshold for masking.

    Returns
    -------
    float
        The weight threshold for this integration.
    '''
    return np.mean(
        sigma_clip(
            weight[np.isfinite(weight) & (weight != 0)],
            sigma=3,
            maxiters=5,
            masked=False,
            copy=False,
        ),
    dtype='f8') * maskpt


def _abs_deriv(array):
    """Take the absolute derivate of a numpy array."""
    out = np.zeros_like(array)  # use same dtype as input

    # make output values nan where input is nan (for floating point input)
    if np.issubdtype(array.dtype, np.floating):
        out[np.isnan(array)] = np.nan

    # compute row-wise absolute diffference
    row_diff = np.abs(np.diff(array, axis=0))
    np.putmask(out[1:], np.isfinite(row_diff), row_diff)  # no need to do max yet

    # since these are absolute differences |r0-r1| = |r1-r0|
    # make a view of the target portion of the array
    row_offset_view = out[:-1]
    # compute an in-place maximum
    np.putmask(row_offset_view, row_diff > row_offset_view, row_diff)
    del row_diff

    # compute col-wise absolute difference
    col_diff = np.abs(np.diff(array, axis=1))
    col_offset_view = out[:, 1:]
    np.putmask(col_offset_view, col_diff > col_offset_view, col_diff)
    col_offset_view = out[:, :-1]
    np.putmask(col_offset_view, col_diff > col_offset_view, col_diff)
    return out


def flag_crs(
    sci_data,
    sci_err,
    blot_data,
    snr,
):
    """
    Straightforward detection of outliers for non-dithered data since
    sci_err includes all noise sources (photon, read, and flat for baseline).

    Parameters
    ----------
    sci_data : numpy.ndarray
        "Science" data possibly containing outliers.

    sci_err : numpy.ndarray
        Error estimates for sci_data.

    blot_data : numpy.ndarray
        Reference data used to detect outliers.

    snr : float
        Signal-to-noise ratio used during detection.

    Returns
    -------
    cr_mask : numpy.ndarray
        Boolean array where outliers (CRs) are true.
    """
    return np.greater(np.abs(sci_data - blot_data), snr * np.nan_to_num(sci_err))


def flag_resampled_crs(
    sci_data,
    sci_err,
    blot_data,
    snr1,
    snr2,
    scale1,
    scale2,
    backg,
):
    """
    Detect outliers (CRs) using resampled reference data.

    Parameters
    ----------

    sci_data : numpy.ndarray
        "Science" data possibly containing outliers

    sci_err : numpy.ndarray
        Error estimates for sci_data

    blot_data : numpy.ndarray
        Reference data used to detect outliers.

    snr1 : float
        Signal-to-noise ratio threshold used prior to smoothing.

    snr2 : float
        Signal-to-noise ratio threshold used after smoothing.

    scale1 : float
        Scale used prior to smoothing.

    scale2 : float
        Scale used after smoothing.

    backg : float
        Scalar background to subtract from the difference.

    Returns
    -------
    cr_mask : numpy.ndarray
        boolean array where outliers (CRs) are true
    """
    err_data = np.nan_to_num(sci_err)

    blot_deriv = _abs_deriv(blot_data)
    diff_noise = np.abs(sci_data - blot_data - backg)

    # Create a boolean mask based on a scaled version of
    # the derivative image (dealing with interpolating issues?)
    # and the standard n*sigma above the noise
    threshold1 = scale1 * blot_deriv + snr1 * err_data
    mask1 = np.greater(diff_noise, threshold1)

    # Smooth the boolean mask with a 3x3 boxcar kernel
    kernel = np.ones((3, 3), dtype=int)
    mask1_smoothed = ndimage.convolve(mask1, kernel, mode='nearest')

    # Create a 2nd boolean mask based on the 2nd set of
    # scale and threshold values
    threshold2 = scale2 * blot_deriv + snr2 * err_data
    mask2 = np.greater(diff_noise, threshold2)

    # Final boolean mask
    return mask1_smoothed & mask2


def gwcs_blot(median_data, median_wcs, blot_shape, blot_wcs, pix_ratio, fillval=0.0):
    """
    Resample the median data to recreate an input image based on
    the blot wcs.

    Parameters
    ----------
    median_data : numpy.ndarray
        The data to blot.

    median_wcs : gwcs.wcs.WCS
        The wcs for the median data.

    blot_shape : tuple of int
        The target blot data shape.

    blot_wcs : gwcs.wcs.WCS
        The target/blotted wcs.

    pix_ratio : float
        Pixel ratio.

    fillval : float, optional
        Fill value for missing data.

    Returns
    -------
    blotted : numpy.ndarray
        The blotted median data.

    blot_img : datamodel
        Datamodel containing header and WCS to define the 'blotted' image
    """
    # Compute the mapping between the input and output pixel coordinates
    pixmap = calc_gwcs_pixmap(blot_wcs, median_wcs, blot_shape)
    log.debug("Pixmap shape: {}".format(pixmap[:, :, 0].shape))
    log.debug("Sci shape: {}".format(blot_shape))
    log.info('Blotting {} <-- {}'.format(blot_shape, median_data.shape))

    outsci = np.full(blot_shape, fillval, dtype=np.float32)

    # Currently tblot cannot handle nans in the pixmap, so we need to give some
    # other value.  -1 is not optimal and may have side effects.  But this is
    # what we've been doing up until now, so more investigation is needed
    # before a change is made.  Preferably, fix tblot in drizzle.
    pixmap[np.isnan(pixmap)] = -1
    tblot(median_data, pixmap, outsci, scale=pix_ratio, kscale=1.0,
          interp='linear', exptime=1.0, misval=fillval, sinscl=1.0)

    return outsci


def calc_gwcs_pixmap(in_wcs, out_wcs, in_shape):
    """
    Return a pixel grid map from input frame to output frame.

    Parameters
    ----------
    in_wcs : gwcs.wcs.WCS
        Input/source wcs.

    out_wcs : gwcs.wcs.WCS
        Output/projected wcs.

    in_shape : list of int
        Input shape used to compute the input bounding box.

    Returns
    -------
    pixmap : numpy.ndarray
        Computed pixmap.
    """
    bb = wcs_bbox_from_shape(in_shape)
    log.debug("Bounding box from data shape: {}".format(bb))

    grid = gwcs.wcstools.grid_from_bounding_box(bb)
    return np.dstack(reproject(in_wcs, out_wcs)(grid[0], grid[1]))


def reproject(wcs1, wcs2):
    """
    Given two WCSs return a function which takes pixel
    coordinates in wcs1 and computes them in wcs2.

    It performs the forward transformation of ``wcs1`` followed by the
    inverse of ``wcs2``.

    Parameters
    ----------
    wcs1, wcs2 : gwcs.wcs.WCS
        WCS objects that have `pixel_to_world_values` and `world_to_pixel_values`
        methods.

    Returns
    -------
    _reproject :
        Function to compute the transformations.  It takes x, y
        positions in ``wcs1`` and returns x, y positions in ``wcs2``.
    """

    try:
        forward_transform = wcs1.pixel_to_world_values
        backward_transform = wcs2.world_to_pixel_values
    except AttributeError as err:
        raise TypeError("Input should be a WCS") from err

    def _reproject(x, y):
        sky = forward_transform(x, y)
        flat_sky = []
        for axis in sky:
            flat_sky.append(axis.flatten())
        det = backward_transform(*tuple(flat_sky))
        det_reshaped = []
        for axis in det:
            det_reshaped.append(axis.reshape(x.shape))
        return tuple(det_reshaped)
    return _reproject
