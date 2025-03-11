""" Test various utility functions """
import pytest
from pathlib import Path

import asdf

from stcal.alignment.util import wcs_bbox_from_shape

from . helpers import make_gwcs

DATADIR = "data"


@pytest.fixture(scope='function')
def wcs_gwcs():
    crval = (150.0, 2.0)
    crpix = (500.0, 500.0)
    shape = (1000, 1000)
    pscale = 0.06 / 3600
    return make_gwcs(crpix, crval, pscale, shape)


@pytest.fixture()
def nrcb5_wcs_wcsinfo():
    """ Returns both wcs and wcsinfo """
    path = Path(__file__).parent.parent / DATADIR / "nrcb5-wcs.asdf"
    with asdf.open(path, lazy_load=False) as asdf_file:
        wcs = asdf_file.tree["wcs"]
        wcs.bounding_box = wcs_bbox_from_shape(wcs.array_shape)
        wcsinfo = asdf_file.tree["wcsinfo"]
        return wcs, wcsinfo
