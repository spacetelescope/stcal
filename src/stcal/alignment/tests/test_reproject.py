import pytest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from stcal.alignment import reproject


def get_fake_wcs():
    fake_wcs1 = WCS(
        fits.Header(
            {
                "NAXIS": 2,
                "NAXIS1": 1,
                "NAXIS2": 1,
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRVAL1": 0,
                "CRVAL2": 0,
                "CRPIX1": 1,
                "CRPIX2": 1,
                "CDELT1": -0.1,
                "CDELT2": 0.1,
            }
        )
    )
    fake_wcs2 = WCS(
        fits.Header(
            {
                "NAXIS": 2,
                "NAXIS1": 1,
                "NAXIS2": 1,
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRVAL1": 0,
                "CRVAL2": 0,
                "CRPIX1": 1,
                "CRPIX2": 1,
                "CDELT1": -0.05,
                "CDELT2": 0.05,
            }
        )
    )
    return fake_wcs1, fake_wcs2


@pytest.mark.parametrize(
    "x_inp, y_inp, x_expected, y_expected",
    [
        (1000, 2000, 2000, 4000),  # string input test
        ([1000], [2000], 2000, 4000),  # array input test
        pytest.param(1, 2, 3, 4, marks=pytest.mark.xfail),  # expected failure test
    ],
)
def test__reproject(x_inp, y_inp, x_expected, y_expected):
    wcs1, wcs2 = get_fake_wcs()
    f = reproject.reproject_coords(wcs1, wcs2)
    x_out, y_out = f(x_inp, y_inp)
    assert np.allclose(x_out, x_expected, rtol=1e-05)
    assert np.allclose(y_out, y_expected, rtol=1e-05)
