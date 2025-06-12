""" Test various utility functions """
import pytest
from pathlib import Path

import asdf
from astropy.convolution import Gaussian2DKernel
from astropy.stats.funcs import gaussian_sigma_to_fwhm
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
    np.random.seed(0)

    # just to have something "truly" irrational (that could not be
    # reproduced by accident as a result of normal processing):
    exptime = (np.pi + np.exp(1)) / np.euler_gamma

    model = make_nrcb5_model(nrcb5_wcs_wcsinfo, exptime=exptime)
    model["dq"][:, :] = 1

    patch_size = 21
    patch_area = patch_size * patch_size
    p2 = patch_size // 2
    # add border so that resampled partial pixels can be isolated
    # in the segmentation:
    border = 4
    pwb = patch_size + border

    ny, nx = model["data"].shape

    stars = []

    for yc in range(border + p2, ny - pwb, pwb):
        for xc in range(border + p2, nx - pwb, pwb):
            sl = np.s_[yc - p2:yc + p2 + 1, xc - p2:xc + p2 + 1]
            flux = 1.0 + 99.0 * np.random.random()
            if np.random.random() > 0.7:
                # uniform image
                psf = np.full((patch_size, patch_size), flux / patch_area)
            else:
                # "star":
                fwhm = 1.5 + 1.5 * np.random.random()
                sigma = fwhm / gaussian_sigma_to_fwhm

                psf = flux * Gaussian2DKernel(
                    sigma,
                    x_size=patch_size,
                    y_size=patch_size
                ).array

            mean_noise = (0.05 + 0.35 * np.random.random()) * flux / patch_area
            rdnoise = mean_noise * np.random.random((patch_size, patch_size))

            model["data"][sl] = psf
            model["dq"][sl] = 0

            model["var_rnoise"][sl] = rdnoise
            model["var_poisson"][sl] = psf
            var_patch = psf + rdnoise
            model["err"][sl] = np.sqrt(var_patch)

            stars.append((xc, yc, sl))

    model["stars"] = stars
    return model
