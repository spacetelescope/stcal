"""Test astrometric utility functions for alignment"""
from copy import deepcopy
from pathlib import Path

import asdf
import numpy as np
import pytest
from astropy.modeling.models import Shift

from stcal.tweakreg import astrometric_utils as amutils
from stcal.tweakreg.tweakreg import (
    _is_wcs_correction_small,
    _wcs_to_skycoord,
)

# Define input GWCS specification to be used for these tests
WCS_NAME = "mosaic_long_i2d_gwcs.asdf"  # Derived using B7.5 Level 3 product
EXPECTED_NUM_SOURCES = 2469
EXPECTED_RADIUS = 0.02564497890604383
TEST_CATALOG = "GAIADR3"
DATADIR = "data"

# something
BKG_LEVEL = 0.001
N_EXAMPLE_SOURCES = 21


@pytest.fixture(scope="module")
def wcsobj():
    path = Path(__file__).parent / DATADIR / WCS_NAME
    with asdf.open(path) as asdf_file:
        return asdf_file["wcs"]


def test_radius(wcsobj):
    # compute radius
    radius, fiducial = amutils.compute_radius(wcsobj)

    # check results
    np.testing.assert_allclose(radius, EXPECTED_RADIUS, rtol=1e-6)


def test_get_catalog(wcsobj):
    # Get radius and fiducial
    radius, fiducial = amutils.compute_radius(wcsobj)

    # Get the catalog
    cat = amutils.get_catalog(fiducial[0], fiducial[1], sr=radius,
                              catalog=TEST_CATALOG)

    assert len(cat) == EXPECTED_NUM_SOURCES


def test_create_catalog(wcsobj):
    # Create catalog
    gcat = amutils.create_astrometric_catalog(
        wcsobj, "2016.0",
        catalog=TEST_CATALOG,
        output=None,
    )
    # check that we got expected number of sources
    assert len(gcat) == EXPECTED_NUM_SOURCES


def test_create_catalog_graceful_failure(wcsobj):
    """
    Ensure catalog retuns zero sources instead of failing outright
    when the bounding box is too small to find any sources
    """
    wcsobj.bounding_box = ((0, 0.5), (0, 0.5))

    # Create catalog
    gcat = amutils.create_astrometric_catalog(
        wcsobj, "2016.0",
        catalog=TEST_CATALOG,
        output=None,
    )
    # check that we got expected number of sources
    assert len(gcat) == 0


def fake_correctors(offset):
    path = Path(__file__).parent / DATADIR / WCS_NAME
    with asdf.open(path) as af:
        wcs = af.tree["wcs"]

    # Make a copy and add an offset at the end of the transform
    twcs = deepcopy(wcs)
    step = twcs.pipeline[0]
    step.transform = step.transform | Shift(offset) & Shift(offset)
    twcs.bounding_box = wcs.bounding_box

    class FakeCorrector:
        def __init__(self, wcs, original_skycoord):
            self.wcs = wcs
            self._original_skycoord = original_skycoord

        @property
        def meta(self):
            return {"original_skycoord": self._original_skycoord}

    return [FakeCorrector(twcs, _wcs_to_skycoord(wcs))]


@pytest.mark.parametrize(("offset", "is_good"),
                         [(1 / 3600, True), (11 / 3600, False)])
def test_is_wcs_correction_small(offset, is_good):
    """
    Test that the _is_wcs_correction_small method returns True for a small
    wcs correction and False for a "large" wcs correction. The values in this
    test are selected based on the current step default parameters:
        - use2dhist
        - searchrad
        - tolerance
    Changes to the defaults for these parameters will likely require updating
    the values uses for parametrizing this test.
    """
    correctors = fake_correctors(offset)
    assert _is_wcs_correction_small(correctors) == is_good
