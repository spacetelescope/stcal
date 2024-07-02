"""Test astrometric utility functions for alignment"""
from copy import deepcopy
from pathlib import Path

import asdf
import numpy as np
import pytest
from astropy.modeling.models import Shift
from astropy.wcs import WCS
from gwcs.wcstools import grid_from_bounding_box
from photutils.segmentation import SourceCatalog, SourceFinder
from stdatamodels.jwst.datamodels import ImageModel

from stcal.tweakreg import astrometric_utils as amutils
from stcal.tweakreg.tweakreg import _is_wcs_correction_small, _parse_refcat, _wcs_to_skycoord, _construct_wcs_corrector, \
    relative_align, absolute_align, apply_tweakreg_solution
from stcal.tweakreg.utils import _wcsinfo_from_wcs_transform

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
    path = Path(__file__).parent / "data" / "mosaic_long_i2d_gwcs.asdf"
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


@pytest.fixture()
def example_wcs():
    path = Path(__file__).parent / "data" / "nrcb1-wcs.asdf"
    with asdf.open(path, lazy_load=False) as af:
        return af.tree["wcs"]


@pytest.fixture()
def example_input(example_wcs):
    m0 = ImageModel((512, 512))

    # add a wcs and wcsinfo
    m0.meta.wcs = example_wcs
    m0.meta.wcsinfo = _wcsinfo_from_wcs_transform(example_wcs)
    m0.meta.wcsinfo.v3yangle = 0.0
    m0.meta.wcsinfo.vparity = -1

    # and a few 'sources'
    m0.data[:] = BKG_LEVEL
    n_sources = N_EXAMPLE_SOURCES  # a few more than default minobj
    rng = np.random.default_rng(26)
    xs = rng.choice(50, n_sources, replace=False) * 8 + 10
    ys = rng.choice(50, n_sources, replace=False) * 8 + 10
    for y, x in zip(ys, xs):
        m0.data[y-1:y+2, x-1:x+2] = [
            [0.1, 0.6, 0.1],
            [0.6, 0.8, 0.6],
            [0.1, 0.6, 0.1],
        ]
    m0.meta.observation.date = "2019-01-01T00:00:00"
    m0.meta.filename = "some_file_0.fits"
    return m0


@pytest.mark.usefixtures("_jail")
def test_parse_refcat(example_input):
    """
    Ensure absolute catalog write creates a file and respects self.output_dir
    """

    OUTDIR = Path("outdir")
    Path.mkdir(OUTDIR)

    correctors = fake_correctors(0.0)
    _parse_refcat(TEST_CATALOG,
                  example_input,
                  correctors,
                  save_abs_catalog=True,
                  output_dir=OUTDIR)

    expected_outfile = OUTDIR / "fit_gaiadr3_ref.ecsv"

    assert Path.exists(expected_outfile)


def make_source_catalog(data):
    """
    Extremely lazy version of source detection step.
    """
    finder = SourceFinder(npixels=5)
    segment_map = finder(data, threshold=0.5)
    sources = SourceCatalog(data, segment_map).to_table()
    sources.rename_column("xcentroid", "x")
    sources.rename_column("ycentroid", "y")
    return sources


@pytest.mark.parametrize("with_shift", [True, False])
def test_relative_align(example_input, with_shift):
    """
    A simplified unit test for basic operation of the TweakRegStep
    when run with or without a small shift in the input image sources
    """
    shifted = example_input.copy()
    shifted.meta.filename = "some_file_1.fits"
    if with_shift:
        # shift 9 pixels so that the sources in one of the 2 images
        # appear at different locations (resulting in a correct wcs update)
        shifted.data[:-9] = example_input.data[9:]
        shifted.data[-9:] = BKG_LEVEL

    # assign images to different groups (so they are aligned to each other)
    example_input.meta.group_id = "a"
    shifted.meta.group_id = "b"

    # create source catalogs
    models = [example_input, shifted]
    source_catalogs = [make_source_catalog(m.data) for m in models]

    # construct correctors from the catalogs
    correctors = [_construct_wcs_corrector(m, cat) for m, cat in zip(models, source_catalogs)]

    # relative alignment of images to each other (if more than one group)
    correctors, local_align_failed = relative_align(correctors)

    # update the wcs in the models
    for (model, corrector) in zip(models, correctors):

        apply_tweakreg_solution(model, corrector, TEST_CATALOG,
                                sip_approx=True, sip_degree=3, sip_max_pix_error=0.1,
                                sip_max_inv_pix_error=0.1, sip_inv_degree=3,
                                sip_npoints=12)

    # and that the wcses differ by a small amount due to the shift above
    # by projecting one point through each wcs and comparing the difference
    abs_delta = abs(models[1].meta.wcs(0, 0)[0] - models[0].meta.wcs(0, 0)[0])
    if with_shift:
        assert abs_delta > 1E-5
    else:
        assert abs_delta < 1E-12

    # also test SIP approximation keywords
    # the first wcs is identical to the input and
    # does not have SIP approximation keywords --
    # they are normally set by assign_wcs
    assert np.allclose(models[0].meta.wcs(0, 0)[0], example_input.meta.wcs(0, 0)[0])
    for key in ["ap_order", "bp_order"]:
        assert key not in models[0].meta.wcsinfo.instance

    # for the second, SIP approximation should be present
    for key in ["ap_order", "bp_order"]:
        assert models[1].meta.wcsinfo.instance[key] == 3

    # evaluate fits wcs and gwcs for the approximation, make sure they agree
    wcs_info = models[1].meta.wcsinfo.instance
    grid = grid_from_bounding_box(models[1].meta.wcs.bounding_box)
    gwcs_ra, gwcs_dec = models[1].meta.wcs(*grid)
    fits_wcs = WCS(wcs_info)
    fitswcs_res = fits_wcs.pixel_to_world(*grid)

    assert np.allclose(fitswcs_res.ra.deg, gwcs_ra)
    assert np.allclose(fitswcs_res.dec.deg, gwcs_dec)
