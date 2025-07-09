"""Test astrometric utility functions for alignment"""

import copy
from copy import deepcopy
from pathlib import Path

import asdf
import contextlib
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling.models import Shift
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body_barycentric
from astropy.stats import mad_std

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

ARAD = np.pi / 180.0


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


@pytest.mark.parametrize(
    "ra, dec, sr",
    [
        (10, 10, 0.1),
        (10, -10, 0.1),
        (0, 0, 0.01),
    ],
)
@pytest.mark.parametrize("catalog_name", ["GAIADR1", "GAIADR2", "GAIADR3"])
def test_get_catalog_using_valid_parameters(ra, dec, sr, catalog_name):
    """Test that get_catalog works properly with valid input parameters."""

    assert len(amutils.get_catalog(ra, dec, search_radius=sr, catalog=catalog_name)) > 0


@pytest.mark.parametrize(
    "ra, dec, sr, catalog_name",
    [
        (10, 10, 0.1, "GAIDR3"),
        (-10, 10, 0.1, "GAIADR3"),
        (10, 100, 0.1, "GAIADR3"),
        (10, 100, 0.1, ""),
        (None, 100, 0.1, "GAIADR3"),
        (10, 10, 0.00014, "GAIADR3"),  # very small search radius -> no sources
    ],
)
def test_get_catalog_using_invalid_parameters(ra, dec, sr, catalog_name):
    """Test that get_catalog returns an empty table for invalid input parameters."""

    assert len(amutils.get_catalog(ra, dec, search_radius=sr, catalog=catalog_name)) == 0


@pytest.mark.parametrize(
    "ra, dec, epoch",
    [
        (10, 10, 2000),
        (10, 10, 2010.3),
        (10, 10, 2030),
        (10, -10, 2000),
        (10, -10, 2010.3),
        (10, -10, 2030),
        (0, 0, 2000),
        (0, 0, 2010.3),
        (0, 0, 2030),
    ],
)
def test_get_catalog_using_epoch(ra, dec, epoch):
    """Test that get_catalog returns coordinates corrected by proper motion
    and parallax. The idea is to fetch data for a specific epoch from the MAST VO API
    and compare them with the expected coordinates for that epoch.
    First, the data for a specific coordinates and epoch are fetched from the MAST VO
    API. Then, the data for the same coordinates are fetched for the Gaia's reference
    epoch of 2016.0, and corrected for proper motion and parallax using explicit
    calculations for the initially specified epoch. We then compare the results between
    the returned coordinates from the MAST VO API and the manually updated
    coordinates."""

    def get_parallax_correction_barycenter(epoch, gaia_ref_epoch_coords):
        """
        Calculates the parallax correction in the Earth barycenter frame for a given epoch
        and Gaia reference epoch coordinates (i.e. Gaia coordinates at the reference epoch).

        Parameters
        ----------
        epoch : float
            The epoch for which the parallax correction is calculated.
        gaia_ref_epoch_coords : dict
            The Gaia reference epoch coordinates, including 'ra', 'dec', and 'parallax'.

        Returns
        -------
        tuple
            A tuple containing the delta_ra and delta_dec values of the parallax correction
            in degrees.

        Examples
        --------
        .. code-block :: python
            epoch = 2022.5
            gaia_coords = {'ra': 180.0, 'dec': 45.0, 'parallax': 10.0}
            correction = get_parallax_correction_earth_barycenter(epoch, gaia_coords)
            print(correction)
            (0.001, -0.002)
        """

        obs_date = Time(epoch, format="decimalyear")
        earths_center_barycentric_coords = get_body_barycentric(
            "earth", obs_date, ephemeris="builtin"
        )
        earth_X = earths_center_barycentric_coords.x
        earth_Y = earths_center_barycentric_coords.y
        earth_Z = earths_center_barycentric_coords.z

        # angular displacement components
        # (see eq. 8.15 of "Spherical Astronomy" by Robert M. Green)
        delta_ra = (
            u.Quantity(gaia_ref_epoch_coords["parallax"], unit="mas").to(u.rad)
            * (1 / np.cos(gaia_ref_epoch_coords["dec"] * ARAD))
            * (
                earth_X.value * np.sin(gaia_ref_epoch_coords["ra"] * ARAD)
                - earth_Y.value * np.cos(gaia_ref_epoch_coords["ra"] * ARAD)
            )
        ).to("deg")
        delta_dec = (
            u.Quantity(gaia_ref_epoch_coords["parallax"], unit="mas").to(u.rad)
            * (
                earth_X.value
                * np.cos(gaia_ref_epoch_coords["ra"] * ARAD)
                * np.sin(gaia_ref_epoch_coords["dec"] * ARAD)
                + earth_Y.value
                * np.sin(gaia_ref_epoch_coords["ra"] * ARAD)
                * np.sin(gaia_ref_epoch_coords["dec"] * ARAD)
                - earth_Z.value * np.cos(gaia_ref_epoch_coords["dec"] * ARAD)
            )
        ).to("deg")

        return delta_ra, delta_dec

    def get_proper_motion_correction(epoch, gaia_ref_epoch_coords, gaia_ref_epoch):
        """
        Calculates the proper motion correction for a given epoch and Gaia reference epoch
        coordinates.

        Parameters
        ----------
        epoch : float
            The epoch for which the proper motion correction is calculated.
        gaia_ref_epoch_coords : dict
            A dictionary containing Gaia reference epoch coordinates.
        gaia_ref_epoch : float
            The Gaia reference epoch.

        Returns
        -------
        None

        Examples
        --------
        .. code-block:: python
        epoch = 2022.5
        gaia_coords = {
            "ra": 180.0,
            "dec": 45.0,
            "pmra": 2.0,
            "pmdec": 1.5
        }
        gaia_ref_epoch = 2020.0
        get_proper_motion_correction(epoch, gaia_coords, gaia_ref_epoch)
        """

        expected_new_dec = (
            np.array(
                gaia_ref_epoch_coords["dec"] * 3600
                + (epoch - gaia_ref_epoch) * gaia_ref_epoch_coords["pmdec"] / 1000
            )
            / 3600
        )
        average_dec = np.array(
            [
                np.mean([new, old])
                for new, old in zip(
                    expected_new_dec, gaia_ref_epoch_coords["dec"], strict=False
                )
            ]
        )
        pmra = gaia_ref_epoch_coords["pmra"] / np.cos(np.deg2rad(average_dec))

        # angular displacement components
        gaia_ref_epoch_coords["pm_delta_dec"] = u.Quantity(
            (epoch - gaia_ref_epoch) * gaia_ref_epoch_coords["pmdec"] / 1000,
            unit=u.arcsec,
        ).to(u.deg)
        gaia_ref_epoch_coords["pm_delta_ra"] = u.Quantity(
            (epoch - gaia_ref_epoch) * (pmra / 1000), unit=u.arcsec
        ).to(u.deg)

    def get_parallax_correction(epoch, gaia_ref_epoch_coords):
        """
        Calculates the parallax correction for a given epoch and Gaia reference epoch
        coordinates.

        Parameters
        ----------
        epoch : float
            The epoch for which to calculate the parallax correction.
        gaia_ref_epoch_coords : dict
            A dictionary containing the Gaia reference epoch coordinates:
            - "ra" : float
                The right ascension in degrees.
            - "dec" : float
                The declination in degrees.
            - "parallax" : float
                The parallax in milliarcseconds (mas).

        Returns
        -------
        None

        Notes
        -----
        This function calculates the parallax correction for a given epoch and Gaia
        reference epoch coordinates. It uses the `get_parallax_correction_barycenter`
        and `get_parallax_correction_mast` functions to obtain the parallax corrections
        based on different coordinate frames.

        Examples
        --------
        This function is typically used to add parallax correction columns to a main table
        of Gaia reference epoch coordinates.

        .. code-block:: python

            epoch = 2023.5
            gaia_coords = {"ra": 180.0, "dec": 30.0, "parallax": 2.5}
            get_parallax_correction(epoch, gaia_coords)
        """

        # get parallax correction using textbook calculations (i.e. Earth's barycenter)
        parallax_corr = get_parallax_correction_barycenter(
            epoch=epoch, gaia_ref_epoch_coords=gaia_ref_epoch_coords
        )

        # add parallax corrections columns to the main table
        gaia_ref_epoch_coords["parallax_delta_ra"] = parallax_corr[0]
        gaia_ref_epoch_coords["parallax_delta_dec"] = parallax_corr[1]


    result = amutils.get_catalog(ra, dec, epoch=epoch)

    # updated coordinates at the provided epoch
    returned_ra = np.array(result["ra"])
    returned_dec = np.array(result["dec"])

    # get GAIA data and update coords to requested epoch using pm measurements
    gaia_ref_epoch = 2016.0
    gaia_ref_epoch_coords_all = amutils.get_catalog(ra, dec, epoch=gaia_ref_epoch)

    gaia_ref_epoch_coords = gaia_ref_epoch_coords_all  # [mask]

    # calculate proper motion corrections
    get_proper_motion_correction(
        epoch=epoch,
        gaia_ref_epoch_coords=gaia_ref_epoch_coords,
        gaia_ref_epoch=gaia_ref_epoch,
    )
    # calculate parallax corrections
    get_parallax_correction(epoch=epoch, gaia_ref_epoch_coords=gaia_ref_epoch_coords)

    # calculate the expected coordinates value after corrections have been applied to
    # Gaia's reference epoch coordinates

    # textbook (barycentric frame)
    expected_ra = (
        gaia_ref_epoch_coords["ra"]
        + gaia_ref_epoch_coords["pm_delta_ra"]
        + gaia_ref_epoch_coords["parallax_delta_ra"]
    )
    expected_dec = (
        gaia_ref_epoch_coords["dec"]
        + gaia_ref_epoch_coords["pm_delta_dec"]
        + gaia_ref_epoch_coords["parallax_delta_dec"]
    )

    assert len(result) > 0

    # adopted tolerance: 2.8e-9 deg -> 10 uas (~0.0001 pix)
    assert np.median(returned_ra - expected_ra) < 2.8e-9
    assert np.median(returned_dec - expected_dec) < 2.8e-9

    assert mad_std(returned_ra - expected_ra) < 2.8e-9
    assert mad_std(returned_dec - expected_dec) < 2.8e-9

def test_create_catalog(wcsobj):
    # Create catalog
    gcat = amutils.create_astrometric_catalog(
        wcsobj, "2016.0",
        catalog=TEST_CATALOG,
        output=None,
    )
    # check that we got expected number of sources
    assert len(gcat) == EXPECTED_NUM_SOURCES


@pytest.mark.parametrize("num_sources", [5, 10, 15])
def test_create_catalog_variable_num_sources(wcsobj, num_sources):
    """
    Test fetching data from supported catalogs with variable number of sources.
    """
    # Create catalog
    gcat = amutils.create_astrometric_catalog(
        wcsobj, "2016.0",
        catalog=TEST_CATALOG,
        output=None,
        num_sources=num_sources,
    )
    # check that we got expected number of sources
    assert len(gcat) == num_sources


@pytest.mark.parametrize(
    "catalog, epoch, start_time",
    [
        ("GAIADR1", "2000.0", 2000),
        ("GAIADR2", "2010", 2010),
        ("GAIADR3", "2030.0", 2030),
        ("GAIADR3", "J2000", None),
        ("GAIADR3", 2030.0, 2030),
        ("GAIADR3", None , None),
    ],
)
def test_create_catalog_different_epochs(wcsobj, catalog, epoch, start_time):
    """
    Test fetching data from supported catalogs with different epochs.
    """
    # Create catalog
    gcat = amutils.create_astrometric_catalog(
        wcsobj, epoch,
        catalog=catalog,
        output=None,
    )
    if start_time is None:
        assert "epoch" not in gcat

    else:
        assert (gcat["epoch"] == start_time).all()


@pytest.mark.parametrize("write_format", ["ascii.ecsv", "fits"])
def test_write_astrometric_catalog(wcsobj, write_format, tmp_path):
    """
    Test writing the astrometric catalog to a file in different formats.
    """

    filename = tmp_path /f"astrometric_catalog.{write_format}"
    assert not filename.exists(), f"Catalog file {filename} already exists."

    amutils.create_astrometric_catalog(
        wcsobj, "2016.0",
        catalog=TEST_CATALOG,
        output=filename,
        table_format=write_format,
    )

    assert filename.exists(), f"Catalog file {filename} was not created."


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
    assert refcat.meta["name"] == CATALOG_FNAME

    # find refcat from web
    refcat = _parse_refcat(TEST_CATALOG, correctors, datamodel.meta.wcs, datamodel.meta.wcsinfo, epoch)
    assert isinstance(refcat, Table)
    assert refcat.meta["name"] == TEST_CATALOG


def test_parse_sky_centroid(abs_refcat):

    # make a SkyCoord object out of the RA and DEC columns
    sky_centroid = SkyCoord(abs_refcat["ra"], abs_refcat["dec"], unit="deg")
    abs_refcat["sky_centroid"] = sky_centroid

    # test case where ra, dec, and sky_centroid are all present
    cat = abs_refcat.copy()
    with pytest.warns(UserWarning):
        cat_out = _parse_sky_centroid(cat)
    assert isinstance(cat_out, Table)
    assert np.all(abs_refcat["ra"] == cat_out["RA"])
    assert np.all(abs_refcat["dec"] == cat_out["DEC"])
    assert "sky_centroid" not in cat_out.columns

    # test case where ra, dec are no longer present
    cat = abs_refcat.copy()
    cat.remove_columns(["ra", "dec"])
    cat_out = _parse_sky_centroid(cat)
    assert isinstance(cat_out, Table)
    assert np.all(abs_refcat["ra"] == cat_out["RA"])
    assert np.all(abs_refcat["dec"] == cat_out["DEC"])

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

    with pytest.raises(requests.exceptions.Timeout), contextlib.suppress(requests.exceptions.ConnectionError):
        for dt in np.arange(1, 0, -0.01):
            amutils.get_catalog(10, 10, search_radius=0.1, catalog="GAIADR3", timeout=dt, override=True)


def test_get_catalog_raises_connection_error(monkeypatch):
    """Test that get_catalog can raise an exception on connection error."""

    monkeypatch.setattr("requests.get", MockConnectionError)

    with pytest.raises(requests.exceptions.ConnectionError):
        amutils.get_catalog(10, 10, search_radius=0.1, catalog="GAIADR3", override=True)
