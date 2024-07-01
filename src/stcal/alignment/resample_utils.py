import logging
from copy import deepcopy

import numpy as np
from gwcs.wcstools import grid_from_bounding_box

from stcal.alignment import util

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def calc_pixmap(in_wcs, out_wcs, shape=None):
    """Return a pixel grid map from input frame to output frame.

    Parameters
    ----------
    in_wcs : `~astropy.wcs.WCS`
        Input WCS objects or transforms.
    out_wcs : `~astropy.wcs.WCS` or `~gwcs.wcs.WCS`
        output WCS objects or transforms.
    shape : tuple, optional
        Shape of grid in pixels. The default is None.

    Returns
    -------
    pixmap : ndarray of shape (xdim, ydim, 2)
        Reprojected pixel grid map. `pixmap[xin, yin]` returns `xout,
        yout` indices in the output image.
    """
    if shape:
        bb = util.wcs_bbox_from_shape(shape)
        log.debug("Bounding box from data shape: %s", bb)
    else:
        bb = util.wcs_bbox_from_shape(in_wcs.pixel_shape)
        log.debug("Bounding box from WCS: %s", bb)

    # creates 2 grids, one with rows of all x values * len(y) rows,
    # and the reverse for all y columns
    grid = grid_from_bounding_box(bb)
    transform_function = util.reproject(in_wcs, out_wcs)
    return np.dstack(transform_function(grid[0], grid[1]))


# is this allowed in stcal, since it operates on a datamodel?
# seems ok. jump step for example does use models
def make_output_wcs(input_models, ref_wcs=None,
                    pscale_ratio=None, pscale=None, rotation=None, shape=None,
                    crpix=None, crval=None):
    """Generate output WCS here based on footprints of all input WCS objects.

    Parameters
    ----------
    input_models : list of `DataModel objects`
        Each datamodel must have a ~gwcs.WCS object.

    pscale_ratio : float, optional
        Ratio of input to output pixel scale. Ignored when ``pscale``
        is provided.

    pscale : float, None, optional
        Absolute pixel scale in degrees. When provided, overrides
        ``pscale_ratio``.

    rotation : float, None, optional
        Position angle of output image Y-axis relative to North.
        A value of 0.0 would orient the final output image to be North up.
        The default of `None` specifies that the images will not be rotated,
        but will instead be resampled in the default orientation for the camera
        with the x and y axes of the resampled image corresponding
        approximately to the detector axes.

    shape : tuple of int, None, optional
        Shape of the image (data array) using ``numpy.ndarray`` convention
        (``ny`` first and ``nx`` second). This value will be assigned to
        ``pixel_shape`` and ``array_shape`` properties of the returned
        WCS object.

    crpix : tuple of float, None, optional
        Position of the reference pixel in the image array. If ``crpix`` is not
        specified, it will be set to the center of the bounding box of the
        returned WCS object.

    crval : tuple of float, None, optional
        Right ascension and declination of the reference pixel. Automatically
        computed if not provided.

    Returns
    -------
    output_wcs : object
        WCS object, with defined domain, covering entire set of input frames
    """
    if ref_wcs is None:
        wcslist = [i.meta.wcs for i in input_models]
        for w, i in zip(wcslist, input_models):
            if w.bounding_box is None:
                w.bounding_box = util.wcs_bbox_from_shape(i.data.shape)
        naxes = wcslist[0].output_frame.naxes

        if naxes != 2:
            msg = f"Output WCS needs 2 spatial axes \
                    but the supplied WCS has {naxes} axes."
            raise RuntimeError(msg)

        output_wcs = util.wcs_from_footprints(
            input_models,
            pscale_ratio=pscale_ratio,
            pscale=pscale,
            rotation=rotation,
            shape=shape,
            crpix=crpix,
            crval=crval
        )

    else:
        naxes = ref_wcs.output_frame.naxes
        if naxes != 2:
            msg = f"Output WCS needs 2 spatial axes \
                    but the supplied WCS has {naxes} axes."
            raise RuntimeError(msg)
        output_wcs = deepcopy(ref_wcs)
        if shape is not None:
            output_wcs.array_shape = shape

    # Check that the output data shape has no zero length dimensions
    if not np.prod(output_wcs.array_shape):
        msg = f"Invalid output frame shape: {tuple(output_wcs.array_shape)}"
        raise ValueError(msg)

    return output_wcs
