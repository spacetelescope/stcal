""" Test various utility functions """
import pytest

from astropy import wcs as fitswcs
import numpy as np

from . helpers import make_gwcs


@pytest.fixture(scope='function')
def wcs_gwcs():
    crval = (150.0, 2.0)
    crpix = (500.0, 500.0)
    shape = (1000, 1000)
    pscale = 0.06 / 3600
    return make_gwcs(crpix, crval, pscale, shape)


@pytest.fixture(scope='function')
def wcs_fitswcs(wcs_gwcs):
    fits_wcs = fitswcs.WCS(wcs_gwcs.to_fits_sip())
    fits_wcs.pixel_area = wcs_gwcs.pixel_area
    fits_wcs.pixel_scale = wcs_gwcs.pixel_scale
    return fits_wcs


@pytest.fixture(scope='module')
def wcs_slicedwcs(function):
    xmin, xmax = 100, 500
    slices = (slice(xmin, xmax), slice(xmin, xmax))
    sliced_wcs = fitswcs.wcsapi.SlicedLowLevelWCS(wcs_gwcs, slices)
    sliced_wcs.pixel_area = wcs_gwcs.pixel_area
    sliced_wcs.pixel_scale = wcs_gwcs.pixel_scale
    return sliced_wcs
