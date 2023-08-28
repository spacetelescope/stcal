import gwcs
import logging
import numpy as np
from util import wcs_bbox_from_shape, reproject

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def calc_pixmap(in_wcs, out_wcs, shape=None):
    """Return a pixel grid map from input frame to output frame

    Parameters
    ----------
    in_wcs: `~astropy.wcs.WCS` 
        Input WCS objects or transforms.
    in_wcs: `~astropy.wcs.WCS`
        output WCS objects or transforms.
    shape : tuple, optional
        Shape of grid in pixels. The default is None.

    Returns
    -------
    pixmap
        reprojected pixel grid map
    """    
    if shape:
        bb = wcs_bbox_from_shape(shape)
        log.debug("Bounding box from data shape: {}".format(bb))
    else:
        bb = in_wcs.bounding_box
        log.debug("Bounding box from WCS: {}".format(in_wcs.bounding_box))

    grid = gwcs.wcstools.grid_from_bounding_box(bb)
    pixmap = np.dstack(reproject(in_wcs, out_wcs)(grid[0], grid[1]))
    return pixmap