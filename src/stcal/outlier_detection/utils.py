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
    # necessary in order to assure that mask gets applied correctly
    if hasattr(weight, '_mask'):
        del weight._mask
    mask_zero_weight = np.equal(weight, 0.)
    mask_nans = np.isnan(weight)
    # Combine the masks
    weight_masked = np.ma.array(weight, mask=np.logical_or(
        mask_zero_weight, mask_nans))
    # Sigma-clip the unmasked data
    weight_masked = sigma_clip(weight_masked, sigma=3, maxiters=5)
    mean_weight = np.mean(weight_masked)
    # Mask pixels where weight falls below maskpt percent
    weight_threshold = mean_weight * maskpt
    return weight_threshold


def _abs_deriv(array):
    """Take the absolute derivate of a numpy array."""
    tmp = np.zeros(array.shape, dtype=np.float64)
    out = np.zeros(array.shape, dtype=np.float64)

    tmp[1:, :] = array[:-1, :]
    tmp, out = _absolute_subtract(array, tmp, out)
    tmp[:-1, :] = array[1:, :]
    tmp, out = _absolute_subtract(array, tmp, out)

    tmp[:, 1:] = array[:, :-1]
    tmp, out = _absolute_subtract(array, tmp, out)
    tmp[:, :-1] = array[:, 1:]
    tmp, out = _absolute_subtract(array, tmp, out)

    return out


def _absolute_subtract(array, tmp, out):
    tmp = np.abs(array - tmp)
    out = np.maximum(tmp, out)
    tmp = tmp * 0.
    return tmp, out


# TODO add tests
def flag_cr(
    sci_data,
    sci_err,
    blot_data,
    snr1,
    snr2,  # FIXME: unused for resample_data=False
    scale1,  # FIXME: unused for resample_data=False
    scale2,  # FIXME: unused for resample_data=False
    backg,  # FIXME: unused for resample_data=False
    resample_data,
):
    """
    Masks outliers in science image by updating DQ in-place

    Mask blemishes in dithered data by comparing a science image
    with a model image and the derivative of the model image.

    Parameters
    ----------

    FIXME: update these

    sci_image : ~jwst.datamodels.ImageModel
        the science data. Can also accept a CubeModel, but only if
        resample_data is False

    blot_array : np.ndarray
        the blotted median image of the dithered science frames.

    snr : str
        Signal-to-noise ratio

    scale : str
        scaling factor applied to the derivative

    backg : float
        Background value (scalar) to subtract

    resample_data : bool
        Boolean to indicate whether blot_image is created from resampled,
        dithered data or not

    Notes
    -----
    Accepting a CubeModel for sci_image and blot_image with resample_data=True
    appears to be a relatively simple extension, as the only thing that explicitly
    relies on the dimensionality is the kernel, which could be generalized.
    However, this is not currently needed, as CubeModels are only passed in for
    TSO data, where resampling is always False.
    """
    err_data = np.nan_to_num(sci_err)

    # create the outlier mask
    if resample_data:  # dithered outlier detection
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
        cr_mask = mask1_smoothed & mask2

    else:  # stack outlier detection
        diff_noise = np.abs(sci_data - blot_data)

        # straightforward detection of outliers for non-dithered data since
        # err_data includes all noise sources (photon, read, and flat for baseline)
        cr_mask = np.greater(diff_noise, snr1 * err_data)

    return cr_mask


# FIXME (or fixed) interp and sinscl were "options" only when provided
# as part of the step spec (which becomes outlierpars). As neither was
# in the spec (and providing unknown arguments causes an error), these
# were never configurable and always defaulted to linear and 1.0
def gwcs_blot(median_data, median_wcs, blot_data, blot_wcs, pix_ratio):
    """
    Resample the output/resampled image to recreate an input image based on
    the input image's world coordinate system

    Parameters
    ----------
    median_model : `~stdatamodels.jwst.datamodels.JwstDataModel`

    blot_img : datamodel
        Datamodel containing header and WCS to define the 'blotted' image
    """
    # Compute the mapping between the input and output pixel coordinates
    # TODO stcal.alignment.resample_utils.calc_pixmap does not work here
    pixmap = calc_gwcs_pixmap(blot_wcs, median_wcs, blot_data.shape)
    log.debug("Pixmap shape: {}".format(pixmap[:, :, 0].shape))
    log.debug("Sci shape: {}".format(blot_data.shape))
    log.info('Blotting {} <-- {}'.format(blot_data.shape, median_data.shape))

    outsci = np.zeros(blot_data.shape, dtype=np.float32)

    # Currently tblot cannot handle nans in the pixmap, so we need to give some
    # other value.  -1 is not optimal and may have side effects.  But this is
    # what we've been doing up until now, so more investigation is needed
    # before a change is made.  Preferably, fix tblot in drizzle.
    pixmap[np.isnan(pixmap)] = -1
    tblot(median_data, pixmap, outsci, scale=pix_ratio, kscale=1.0,
          interp='linear', exptime=1.0, misval=0.0, sinscl=1.0)

    return outsci


# TODO tests, duplicate in resample, resample_utils
def calc_gwcs_pixmap(in_wcs, out_wcs, shape):
    """ Return a pixel grid map from input frame to output frame.
    """
    bb = wcs_bbox_from_shape(shape)
    log.debug("Bounding box from data shape: {}".format(bb))

    grid = gwcs.wcstools.grid_from_bounding_box(bb)
    # TODO does stcal reproject work?
    pixmap = np.dstack(reproject(in_wcs, out_wcs)(grid[0], grid[1]))

    return pixmap


# TODO tests, duplicate in resample, assign_wcs, resample_utils
def reproject(wcs1, wcs2):
    """
    Given two WCSs or transforms return a function which takes pixel
    coordinates in the first WCS or transform and computes them in the second
    one. It performs the forward transformation of ``wcs1`` followed by the
    inverse of ``wcs2``.

    Parameters
    ----------
    wcs1, wcs2 : `~astropy.wcs.WCS` or `~gwcs.wcs.WCS`
        WCS objects that have `pixel_to_world_values` and `world_to_pixel_values`
        methods.

    Returns
    -------
    _reproject : func
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
        # Filter out RuntimeWarnings due to computed NaNs in the WCS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            det = backward_transform(*tuple(flat_sky))
        det_reshaped = []
        for axis in det:
            det_reshaped.append(axis.reshape(x.shape))
        return tuple(det_reshaped)
    return _reproject
