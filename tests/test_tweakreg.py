"""Test astrometric utility functions for alignment"""

import copy
from copy import deepcopy
from pathlib import Path

import asdf
import contextlib
import numpy as np
import pytest
from astropy.modeling.models import Shift
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord

from stcal.tweakreg import astrometric_utils as amutils
from stcal.tweakreg.tweakreg import (
    TweakregError,
    _is_wcs_correction_small,
    _parse_refcat,
    _parse_sky_centroid,
    _wcs_to_skycoord,
    absolute_align,
    construct_wcs_corrector,
    relative_align,
)
from stcal.tweakreg.utils import _wcsinfo_from_wcs_transform
import requests

# Define input GWCS specification to be used for these tests
WCS_NAME = "mosaic_long_i2d_gwcs.asdf"  # Derived using B7.5 Level 3 product
EXPECTED_NUM_SOURCES = 2469

# more recent WCS with a defined input frame is necessary for some tests
WCS_NAME_2 = "nrcb1-wcs.asdf"

TEST_CATALOG = "GAIADR3"
CATALOG_FNAME = "ref_cat.ecsv"
DATADIR = "data"

# something
BKG_LEVEL = 0.001
N_EXAMPLE_SOURCES = 21


class MockConnectionError:
    def __init__(self, *args, **kwargs):
        raise requests.exceptions.ConnectionError


@pytest.fixture(scope="module")
def wcsobj():
    path = Path(__file__).parent / DATADIR / WCS_NAME
    with asdf.open(path, lazy_load=False) as asdf_file:
        return asdf_file.tree["wcs"]


@pytest.fixture(scope="module")
def wcsobj2():
    path = Path(__file__).parent / DATADIR / WCS_NAME_2
    with asdf.open(path, lazy_load=False) as asdf_file:
        return asdf_file.tree["wcs"]


def test_radius(wcsobj):
    # compute radius
    radius, _ = amutils.compute_radius(wcsobj)

    # check results
    EXPECTED_RADIUS = 0.02564497890604383
    np.testing.assert_allclose(radius, EXPECTED_RADIUS, rtol=1e-6)


def test_get_catalog(wcsobj):
    # Get radius and fiducial
    radius, fiducial = amutils.compute_radius(wcsobj)

    # Get the catalog
    cat = amutils.get_catalog(fiducial[0], fiducial[1], search_radius=radius,
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


def test_expected_fails_bad_separation():

    correctors = fake_correctors(0.0)
    separation = 1.0
    tolerance = 1.0
    with pytest.raises(TweakregError):
        relative_align(correctors,
                       separation=separation,
                       tolerance=tolerance)

    with pytest.raises(TweakregError):
        absolute_align(correctors, "GAIADR3",
                       None,
                       None,
                       None,
                       abs_separation=separation,
                       abs_tolerance=tolerance)


class AttrDict(dict):
    """Hack to be able to treat wcsinfo dict as an object so attributes
    can be accessed"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    @property
    def instance(self):
        return self


class Metadata:

    def __init__(self, wcs, epoch, group_id=None):
        self.wcs = wcs
        self.observation = AttrDict({"date": epoch})
        wcsinfo = _wcsinfo_from_wcs_transform(wcs)
        wcsinfo["v3yangle"] = 0.0
        wcsinfo["vparity"] = 1
        self.wcsinfo = AttrDict(wcsinfo)
        self.group_id = group_id


class MinimalDataWithWCS:

    def __init__(self, wcs, epoch="2016-01-01T00:00:00.0", group_id=None):
        self.meta = Metadata(wcs, epoch, group_id=group_id)
        self.data = np.zeros((512, 512))


@pytest.fixture(scope="module")
def datamodel(wcsobj2, group_id=None):
    return MinimalDataWithWCS(wcsobj2, group_id=group_id)


@pytest.fixture(scope="module")
def abs_refcat(datamodel):

    wcsobj = datamodel.meta.wcs
    radius, fiducial = amutils.compute_radius(wcsobj)
    return amutils.get_catalog(fiducial[0], fiducial[1], search_radius=radius,
                            catalog=TEST_CATALOG)


def test_parse_refcat(datamodel, abs_refcat, tmp_path):

    correctors = fake_correctors(0.0)
    cat = abs_refcat

    # save refcat to file
    cat.write(tmp_path / CATALOG_FNAME, format="ascii.ecsv", overwrite=True)

    # parse refcat from file
    epoch = Time(datamodel.meta.observation.date).decimalyear
    refcat = _parse_refcat(tmp_path / CATALOG_FNAME,
                           correctors,
                           datamodel.meta.wcs,
                           datamodel.meta.wcsinfo,
                           epoch)
    assert isinstance(refcat, Table)

    # find refcat from web
    refcat = _parse_refcat(TEST_CATALOG, correctors, datamodel.meta.wcs, datamodel.meta.wcsinfo, epoch)
    assert isinstance(refcat, Table)


def test_parse_sky_centroid(abs_refcat):

    # make a SkyCoord object out of the RA and DEC columns
    sky_centroid = SkyCoord(abs_refcat["ra"], abs_refcat["dec"], unit="deg")
    abs_refcat["sky_centroid"] = sky_centroid

    # test case where ra, dec, and sky_centroid are all present
    cat = abs_refcat.copy()
    with pytest.warns(UserWarning):
        cat_out = _parse_sky_centroid(cat)
    assert isinstance(cat_out, Table)
    assert np.all(cat["ra"] == cat_out["ra"])
    assert np.all(cat["dec"] == cat_out["dec"])
    assert "sky_centroid" not in cat_out.columns

    # test case where ra, dec are no longer present
    cat = abs_refcat.copy()
    cat.remove_columns(["ra", "dec"])
    cat_out = _parse_sky_centroid(cat)
    assert isinstance(cat_out, Table)
    assert np.all(cat["ra"] == cat_out["ra"])
    assert np.all(cat["dec"] == cat_out["dec"])

    # test case where neither present
    cat = abs_refcat.copy()
    cat.remove_columns(["ra", "dec", "sky_centroid"])
    with pytest.raises(KeyError):
        _parse_sky_centroid(cat)

    # test case where multiple RA or dec columns are present
    cat = abs_refcat.copy()
    cat["RA"] = cat["ra"]
    with pytest.raises(KeyError):
        _parse_sky_centroid(cat)

    # test case where no RA/DEC but multiple sky_centroid columns are present
    cat = abs_refcat.copy()
    cat.remove_columns(["ra", "dec"])
    cat["sky_cENTROID"] = cat["sky_centroid"]
    with pytest.raises(KeyError):
        _parse_sky_centroid(cat)


@pytest.fixture(scope="module")
def input_catalog(datamodel):
    """Get catalog from gaia, transform it to x,y in the image frame,
    use it as an input catalog"""
    # Get radius and fiducial
    w = datamodel.meta.wcs
    radius, fiducial = amutils.compute_radius(w)

    # Get the catalog
    cat = amutils.get_catalog(fiducial[0], fiducial[1], search_radius=radius,
                              catalog=TEST_CATALOG)

    x, y = w.world_to_pixel_values(cat["ra"].value, cat["dec"].value)
    return Table({"x": x, "y": y})


@pytest.fixture(scope="module")
def example_input(wcsobj2):

    m0 = MinimalDataWithWCS(wcsobj2)
    m0.data[:] = BKG_LEVEL
    n_sources = N_EXAMPLE_SOURCES
    rng = np.random.default_rng(26)
    xs = rng.choice(50, n_sources, replace=False) * 8 + 10
    ys = rng.choice(50, n_sources, replace=False) * 8 + 10
    for y, x in zip(ys, xs, strict=False):
        m0.data[y-1:y+2, x-1:x+2] = [
            [0.1, 0.6, 0.1],
            [0.6, 0.8, 0.6],
            [0.1, 0.6, 0.1],
        ]

    m1 = copy.deepcopy(m0)
    # give each a unique filename
    m0.meta.filename = "some_file_0.fits"
    m0.meta.group_id = "a"
    m1.meta.filename = "some_file_1.fits"
    m1.meta.group_id = "b"
    return [m0, m1]


@pytest.mark.parametrize("with_shift", [True, False])
def test_relative_align(example_input, input_catalog, with_shift):

    [m0, m1] = example_input
    cat1 = copy.deepcopy(input_catalog)
    if with_shift:
        m1.data[:-9] = m0.data[9:]
        m1.data[-9:] = BKG_LEVEL
        cat1["y"] -= 9

    correctors = [construct_wcs_corrector(dm.meta.wcs,
                                          dm.meta.wcsinfo.instance,
                                          cat,
                                          dm.meta.group_id) for (dm, cat) in \
                                          zip([m0, m1], [input_catalog, cat1], strict=True)]
    result = relative_align(correctors, minobj=5)

    # ensure wcses differ by a small amount due to the shift above
    # by projecting one point through each wcs and comparing the difference
    abs_delta = abs(result[1].wcs(0, 0)[0] - result[0].wcs(0, 0)[0])
    if with_shift:
        assert abs_delta > 1E-5
    else:
        assert abs_delta < 1E-12


def test_absolute_align(example_input, input_catalog):

    correctors = [construct_wcs_corrector(dm.meta.wcs,
                                          dm.meta.wcsinfo.instance,
                                          input_catalog,
                                          dm.meta.group_id) for dm in example_input]

    ref_model = example_input[0]
    result = absolute_align(correctors,
                            TEST_CATALOG,
                            ref_wcs=ref_model.meta.wcs,
                            ref_wcsinfo=ref_model.meta.wcsinfo,
                            epoch=Time(ref_model.meta.observation.date).decimalyear,
                            abs_minobj=5)
    for res in result:
        assert res.meta["group_id"] == 987654

    abs_delta = abs(result[1].wcs(0, 0)[0] - result[0].wcs(0, 0)[0])
    assert abs_delta < 1E-12

def test_get_catalog_timeout():
    """Test that get_catalog can raise an exception on timeout."""

    with pytest.raises(Exception) as exec_info:
        for dt in np.arange(1, 0, -0.01):
            with contextlib.suppress(requests.exceptions.ConnectionError):
                amutils.get_catalog(10, 10, search_radius=0.1, catalog="GAIADR3", timeout=dt)
    assert exec_info.type == requests.exceptions.Timeout


def test_get_catalog_raises_connection_error(monkeypatch):
    """Test that get_catalog can raise an exception on connection error."""

    monkeypatch.setattr("requests.get", MockConnectionError)

    with pytest.raises(Exception) as exec_info:
        amutils.get_catalog(10, 10, search_radius=0.1, catalog="GAIADR3")

    assert exec_info.type == requests.exceptions.ConnectionError
