""" Test various utility functions """
import pytest

from . helpers import make_gwcs


@pytest.fixture(scope='function')
def wcs_gwcs():
    crval = (150.0, 2.0)
    crpix = (500.0, 500.0)
    shape = (1000, 1000)
    pscale = 0.06 / 3600
    return make_gwcs(crpix, crval, pscale, shape)
