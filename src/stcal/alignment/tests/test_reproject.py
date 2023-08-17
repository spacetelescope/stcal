import numpy as np
from astropy.io import fits
from stcal.alignment import reproject



def test__reproject():
    x_inp, y_inp = 1000, 2000
    # Create a test image with a single pixel
    fake_wcs1 = fits.Header({'NAXIS': 2, 'NAXIS1': 1, 'NAXIS2': 1, 'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN', 'CRVAL1': 0, 'CRVAL2': 0, 'CRPIX1': 1, 'CRPIX2': 1, 'CDELT1': -0.1, 'CDELT2': 0.1})
    fake_wcs2 = fits.Header({'NAXIS': 2, 'NAXIS1': 1, 'NAXIS2': 1, 'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN', 'CRVAL1': 0, 'CRVAL2': 0, 'CRPIX1': 1, 'CRPIX2': 1, 'CDELT1': -0.05, 'CDELT2': 0.05})

    # Call the function
    reproject.reproject_coords(fake_wcs1, fake_wcs2)