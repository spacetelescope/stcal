import logging

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
