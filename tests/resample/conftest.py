""" Test various utility functions """
import math
import pytest
from pathlib import Path

import asdf
from astropy.convolution import Gaussian2DKernel
import numpy as np

from stcal.alignment.util import wcs_bbox_from_shape

from . helpers import make_gwcs, make_nrcb5_model

DATADIR = "data"


@pytest.fixture(scope='function')
def wcs_gwcs():
    crval = (150.0, 2.0)
    crpix = (500.0, 500.0)
    shape = (1000, 1000)
    pscale = 0.06 / 3600
    return make_gwcs(crpix, crval, pscale, shape)


@pytest.fixture(scope="module")
def nrcb5_wcs_wcsinfo():
    """ Returns both wcs and wcsinfo """
    path = Path(__file__).parent.parent / DATADIR / "nrcb5-wcs.asdf"
    with asdf.open(path, lazy_load=False) as asdf_file:
        wcs = asdf_file.tree["wcs"]
        wcs.bounding_box = wcs_bbox_from_shape(wcs.array_shape)
        wcsinfo = asdf_file.tree["wcsinfo"]
        return wcs, wcsinfo


@pytest.fixture(scope="module")
def nrcb5_many_fluxes(nrcb5_wcs_wcsinfo):
    model = make_nrcb5_model(nrcb5_wcs_wcsinfo)
    model["dq"][:, :] = 1

    np.random.seed(0)

    patch_size = 21
    p2 = patch_size // 2
    # add border so that resampled partial pixels can be isolated
    # in the segmentation:
    border = 4
    pwb = patch_size + border

    fwhm2sigma = 2.0 * math.sqrt(2.0 * math.log(2.0))

    ny, nx = model["data"].shape

    stars = []

    for yc in range(border + p2, ny - pwb, pwb):
        for xc in range(border + p2, nx - pwb, pwb):
            sl = np.s_[yc - p2:yc + p2 + 1, xc - p2:xc + p2 + 1]
            flux = 1.0 + 99.0 * np.random.random()
            if np.random.random() > 0.7:
                # uniform image
                psf = np.full((patch_size, patch_size), flux)
            else:
                # "star":
                fwhm = 1.5 + 1.5 * np.random.random()
                sigma = fwhm / fwhm2sigma

                psf = flux * Gaussian2DKernel(
                    sigma,
                    x_size=patch_size,
                    y_size=patch_size
                ).array

            flux = psf.sum()

            model["data"][sl] = psf
            model["dq"][sl] = 0
            stars.append((xc, yc, flux, sl))

    model["stars"] = stars
    return model
