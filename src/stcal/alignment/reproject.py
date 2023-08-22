import gwcs
import numpy as np
from astropy import wcs as fitswcs
from astropy.modeling import Model
from typing import Union


def reproject_coords(wcs1, wcs2):
    """
    Given two WCSs or transforms return a function which takes pixel
    coordinates in the first WCS or transform and computes them in pixel coordinates
    in the second one. It performs the forward transformation of ``wcs1`` followed by the
    inverse of ``wcs2``.

    Parameters
    ----------
    wcs1, wcs2 : `~astropy.wcs.WCS` or `~gwcs.wcs.WCS` or `~astropy.modeling.Model`
        WCS objects.

    Returns
    -------
    _reproject : func
        Function to compute the transformations.  It takes x, y
        positions in ``wcs1`` and returns x, y positions in ``wcs2``.
    """

    def _get_forward_transform_func(wcs1):
        """Get the forward transform function from the input WCS. If the wcs is a
        fitswcs.WCS object all_pix2world requres three inputs, the x (str, ndarrray),
        y (str, ndarray), and origin (int). The origin should be between 0, and 1
        https://docs.astropy.org/en/latest/wcs/index.html#loading-wcs-information-from-a-fits-file
        )
        """
        if isinstance(wcs1, fitswcs.WCS):
            forward_transform = wcs1.all_pix2world
        elif isinstance(wcs1, gwcs.WCS):
            forward_transform = wcs1.forward_transform
        elif issubclass(wcs1, Model):
            forward_transform = wcs1
        else:
            raise TypeError(
                "Expected input to be astropy.wcs.WCS or gwcs.WCS "
                "object or astropy.modeling.Model subclass"
            )
        return forward_transform

    def _get_backward_transform_func(wcs2):
        if isinstance(wcs2, fitswcs.WCS):
            backward_transform = wcs2.all_world2pix
        elif isinstance(wcs2, gwcs.WCS):
            backward_transform = wcs2.backward_transform
        elif issubclass(wcs2, Model):
            backward_transform = wcs2.inverse
        else:
            raise TypeError(
                "Expected input to be astropy.wcs.WCS or gwcs.WCS "
                "object or astropy.modeling.Model subclass"
            )
        return backward_transform

    def _reproject(x: Union[str, np.ndarray], y: Union[str, np.ndarray]) -> tuple:
        """
        Reprojects the input coordinates from one WCS to another.

        Parameters:
        -----------
        x : str or np.ndarray
            Array of x-coordinates to be reprojected.
        y : str or np.ndarray
            Array of y-coordinates to be reprojected.

        Returns:
        --------
        tuple
            Tuple of reprojected x and y coordinates.
        """
        # example inputs to resulting function (12, 13, 0) # third number is origin
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        if len(x) != len(y):
            raise ValueError("x and y must be the same length")
        sky = _get_forward_transform_func(wcs1)(x, y, 0)
        sky_back = np.array(_get_backward_transform_func(wcs2)(sky[0], sky[1], 0))
        new_sky = tuple(sky_back[:, :1].flatten())
        return tuple(new_sky)

    return _reproject
