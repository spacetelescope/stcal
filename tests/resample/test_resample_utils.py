"""Test various utility functions"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from stcal.resample.utils import (
    _get_inverse_variance,
    build_driz_weight,
    build_mask,
    calc_pixmap,
    compute_mean_pixel_area,
    get_tmeasure,
    is_flux_density,
    is_imaging_wcs,
    resample_range,
)

from .helpers import JWST_DQ_FLAG_DEF, make_gwcs, make_input_model

GOOD = 0
DQ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
BITVALUES = 2**0 + 2**2
BITVALUES_STR = f"{2**0}, {2**2}"
BITVALUES_INV_STR = f"~{2**0}, {2**2}"
JWST_NAMES = "DO_NOT_USE,JUMP_DET"
JWST_NAMES_INV = "~" + JWST_NAMES


@pytest.mark.parametrize(
    "dq, bitvalues, expected",
    [
        (DQ, 0, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])),
        (DQ, BITVALUES, np.array([1, 1, 0, 0, 1, 1, 0, 0, 0])),
        (DQ, BITVALUES_STR, np.array([1, 1, 0, 0, 1, 1, 0, 0, 0])),
        (DQ, BITVALUES_INV_STR, np.array([1, 0, 1, 0, 0, 0, 0, 0, 1])),
        (DQ, JWST_NAMES, np.array([1, 1, 0, 0, 1, 1, 0, 0, 0])),
        (DQ, JWST_NAMES_INV, np.array([1, 0, 1, 0, 0, 0, 0, 0, 1])),
        (DQ, None, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])),
    ],
)
def test_build_mask(dq, bitvalues, expected):
    """Test logic of mask building

    Parameters
    ----------
    dq: numpy.array
        The input data quality array

    bitvalues: int or str
        The bitvalues to match against

    expected: numpy.array
        Expected mask array
    """
    result = build_mask(dq, bitvalues, flag_name_map=JWST_DQ_FLAG_DEF)
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "data_shape, bbox, exception, truth",
    [
        ((1, 2, 3), ((1, 500), (0, 350)), True, None),
        ((1, 2, 3), None, True, None),
        ((1,), ((1, 500), (0, 350)), True, None),
        ((1,), None, True, None),
        ((1000, 800), ((1, 500),), True, None),
        ((1000, 800), ((1, 500), (0, 350), (0, 350)), True, None),
        ((1,), ((1, 500), (0, 350)), True, None),
        ((1200, 1400), ((700, 300), (600, 800)), False, (700, 700, 600, 800)),
        ((1200, 1400), ((600, 800), (700, 300)), False, (600, 800, 700, 700)),
        ((1200, 1400), ((300, 700), (600, 800)), False, (300, 700, 600, 800)),
        ((750, 470), ((300, 700), (600, 800)), False, (300, 469, 600, 749)),
        ((750, 470), ((-5, -1), (-800, -600)), False, (0, 0, 0, 0)),
        ((750, 470), None, False, (0, 469, 0, 749)),
        ((-750, -470), None, False, (0, 0, 0, 0)),
    ],
)
def test_resample_range(data_shape, bbox, exception, truth):
    if exception:
        with pytest.raises(ValueError):
            resample_range(data_shape, bbox)
        return

    xyminmax = resample_range(data_shape, bbox)
    assert np.allclose(xyminmax, truth, rtol=0, atol=1e-12)


def test_get_tmeasure():
    model = {
        "measurement_time": 12.34,
        "exposure_time": 23.45,
    }

    assert get_tmeasure(model) == (12.34, True)

    model["measurement_time"] = None
    assert get_tmeasure(model) == (23.45, False)

    del model["measurement_time"]
    assert get_tmeasure(model) == (23.45, False)

    del model["exposure_time"]
    with pytest.raises(KeyError):
        get_tmeasure(model)


def test_is_imaging_wcs(wcs_gwcs):
    assert is_imaging_wcs(wcs_gwcs)


def test_compute_mean_pixel_area(wcs_gwcs):
    area = np.deg2rad(wcs_gwcs.pixel_scale) ** 2
    assert abs(compute_mean_pixel_area(wcs_gwcs) / area - 1.0) < 1e-5


@pytest.mark.parametrize(
    "unit,result",
    [("Jy", True), ("MJy", True), ("MJy/sr", False), ("DN/s", False), ("bad_unit", False), (None, False)],
)
def test_is_flux_density(unit, result):
    assert is_flux_density(unit) is result


@pytest.mark.parametrize("weight_type", ["ivm", "exptime", "ivm-sky"])
def test_build_driz_weight(weight_type):
    """Check that correct weight map is returned of different weight types"""

    model = make_input_model((10, 10))

    # N.B.: all variance arrays are initialized to 1

    model["dq"][0] = JWST_DQ_FLAG_DEF.DO_NOT_USE
    model["measurement_time"] = 10.0
    model["var_rnoise"] /= 10.0
    model["var_sky"] /= 10.0

    weight_map = build_driz_weight(model, weight_type=weight_type, good_bits=GOOD)
    assert_array_equal(weight_map[0], 0)
    assert_array_equal(weight_map[1:], 10.0)
    assert weight_map.dtype == np.float32


@pytest.mark.parametrize("weight_type", ["ivm", "exptime", "ivm-sky"])
def test_build_driz_weight_zeros(weight_type):
    """Check that zero or not finite variance arrays return proper weight
    map."""
    model = make_input_model((10, 10))

    model["measurement_time"] = 0.0
    model["var_rnoise"] *= 0
    model["var_sky"] *= np.inf

    weight_map = build_driz_weight(model, weight_type=weight_type)

    assert_array_equal(weight_map, 0)


@pytest.mark.parametrize(
    "weight_type,var_array_name",
    [
        ("ivm", "var_rnoise"),
        ("exptime", "measurement_time"),
        ("ivm-sky", "var_sky"),
    ],
)
def test_build_driz_weight_none(weight_type, var_array_name):
    """Check that missing variance array returns equally weighted map set
    to 1."""
    model = make_input_model((10, 10))

    del model[var_array_name]

    weight_map = build_driz_weight(model, weight_type=weight_type)

    assert_array_equal(weight_map, 1)


@pytest.mark.parametrize("weight_type", ["ivm-smed", "ivm-med5"])
def test_unsupported_weight_type(weight_type):
    model = make_input_model((10, 10))
    with pytest.raises(ValueError, match=rf"^Invalid weight type: {repr(weight_type)}"):
        build_driz_weight(model, weight_type=weight_type)


@pytest.mark.parametrize("array_name", ["var_rnoise", "var_sky"])
def test_get_inverse_variance_valid_and_invalid(caplog, array_name):
    arr = np.array([[4.0, 0.0], [np.nan, 1.0]])
    inv = _get_inverse_variance(arr, arr.shape, array_name)
    assert np.isclose(inv[0, 0], 0.25)
    assert inv[0, 1] == 0
    assert inv[1, 0] == 0
    assert inv[1, 1] == 1.0

    # Wrong shape
    inv2 = _get_inverse_variance(None, (2, 2), array_name)
    np.testing.assert_array_equal(inv2, 1)

    inv3 = _get_inverse_variance(np.ones((1, 1)), (2, 2), array_name)
    np.testing.assert_array_equal(inv3, 1)

    # Warning message logged both times
    assert caplog.text.count(f"'{array_name}' array not available.") == 2


@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("stepsize", [1, 10, 100])
@pytest.mark.parametrize("in_shape", [(1000, 1000), (1234, 1435), (200, 300)])
@pytest.mark.parametrize("bbox_border", [None, 0, 2, 10, -5])
@pytest.mark.parametrize("shift", [(0, 0), (3, 5), (-5, -2)])
def test_pixmap(shift, bbox_border, in_shape, stepsize, order):
    out_shape = (987, 789)
    wcs1 = make_gwcs(crpix=(0, 0), crval=(0, 0), pscale=2.0e-5, shape=in_shape)
    wcs2 = make_gwcs(crpix=shift, crval=(0, 0), pscale=2.0e-5, shape=out_shape)
    if bbox_border is None:
        wcs1.bounding_box = None
    else:
        wcs1.bounding_box = (
            (bbox_border - 0.5, in_shape[1] - bbox_border + 0.5),
            (bbox_border - 0.5, in_shape[0] - bbox_border + 0.5),
        )

    # expected pixmap:
    y, x = np.indices(in_shape, dtype=np.float64)
    x += shift[0]
    y += shift[1]
    if bbox_border is not None and bbox_border > 0:
        x[:bbox_border, :] = np.nan
        x[-bbox_border + 1 :, :] = np.nan
        x[:, :bbox_border] = np.nan
        x[:, -bbox_border + 1 :] = np.nan
        y[:bbox_border, :] = np.nan
        y[-bbox_border + 1 :, :] = np.nan
        y[:, :bbox_border] = np.nan
        y[:, -bbox_border + 1 :] = np.nan

    ok_pixmap = np.dstack([x, y])

    # compute pixmap
    pixmap = calc_pixmap(wcs1, wcs2, stepsize=stepsize, order=order)

    (
        assert_allclose(pixmap, ok_pixmap, rtol=0, atol=1e-5, equal_nan=True),
        (
            f"Failed for shift={shift}, bbox_border={bbox_border}, in_shape={in_shape}, "
            f"stepsize={stepsize}, order={order}"
        ),
    )


def test_map_to_self():
    """
    Map a pixel array to itself. Should return the same array.
    This is a modified version of the test from the `drizzle` package.
    """
    input_wcs = make_gwcs(crpix=(0, 0), crval=(0, 0), pscale=2.0e-5, shape=(100, 100))
    shape = input_wcs.array_shape

    ok_pixmap = np.indices(shape, dtype="float64")
    ok_pixmap = ok_pixmap.transpose()

    pixmap = calc_pixmap(input_wcs, input_wcs)

    # Got x-y transpose right
    assert_equal(pixmap.shape, ok_pixmap.shape)

    # Mapping an array to itself
    assert_allclose(pixmap, ok_pixmap, rtol=1.0e-6, atol=1.0e-9)

    # user-provided shape
    pixmap = calc_pixmap(input_wcs, input_wcs, (12, 34))
    assert_equal(pixmap.shape, (12, 34, 2))

    # Check that an exception is raised for WCS without pixel_shape and
    # bounding_box:
    input_wcs.pixel_shape = None
    input_wcs.bounding_box = None
    with pytest.raises(ValueError):
        calc_pixmap(input_wcs, input_wcs)

    # user-provided shape when array_shape is not set:
    pixmap = calc_pixmap(input_wcs, input_wcs, (12, 34))
    assert_equal(pixmap.shape, (12, 34, 2))

    # from bounding box:
    input_wcs.bounding_box = ((5.3, 33.5), (2.8, 11.5))
    pixmap = calc_pixmap(input_wcs, input_wcs)
    assert_equal(pixmap.shape, (12, 34, 2))

    # from bounding box and pixel_shape (the later takes precedence):
    input_wcs.array_shape = shape
    pixmap = calc_pixmap(input_wcs, input_wcs)
    assert_equal(pixmap.shape, ok_pixmap.shape)


@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("stepsize", [1, 10])
def test_disable_gwcs_bbox(order, stepsize):
    """
    Map a pixel array to a translated version of itself.
    This is a modified version of the test from the `drizzle` package.
    """
    in_shape = (1024, 1048)

    first_wcs = make_gwcs(crpix=(0, 0), crval=(0, 0), pscale=2.0e-5, shape=in_shape)
    second_wcs = make_gwcs(crpix=(-2, -2), crval=(0, 0), pscale=2.0e-5, shape=in_shape)

    y, x = np.indices(first_wcs.array_shape, dtype="float64") - 2.0
    ok_pixmap = np.dstack([x, y])

    # disable both bounding boxes:
    pixmap = calc_pixmap(first_wcs, second_wcs, disable_bbox="both", order=order, stepsize=stepsize)
    assert_allclose(pixmap[2:, 2:], ok_pixmap[2:, 2:], rtol=1.0e-6, atol=1.0e-8)
    assert np.all(np.isfinite(pixmap[:2, :2]))
    assert np.all(np.isfinite(pixmap[-2:, -2:]))
    # check bbox was restored
    assert first_wcs.bounding_box is not None
    assert second_wcs.bounding_box is not None

    # disable "from" bounding box:
    pixmap = calc_pixmap(first_wcs, second_wcs, disable_bbox="from", order=order, stepsize=stepsize)

    assert_allclose(pixmap[2:, 2:], ok_pixmap[2:, 2:], rtol=1.0e-6, atol=1.0e-8)
    assert np.all(np.logical_not(np.isfinite(pixmap[:2, :])))
    assert np.all(np.logical_not(np.isfinite(pixmap[:, :2])))
    # check bbox was restored
    assert first_wcs.bounding_box is not None
    assert second_wcs.bounding_box is not None

    # disable "to" bounding boxes:
    pixmap = calc_pixmap(first_wcs, second_wcs, disable_bbox="to", order=order, stepsize=stepsize)
    assert_allclose(pixmap[2:, 2:], ok_pixmap[2:, 2:], rtol=1.0e-6, atol=1.0e-8)
    assert np.all(np.isfinite(pixmap[:2, :2]))
    assert np.all(pixmap[:2, :2] < 0.0)
    assert np.all(np.isfinite(pixmap[-2:, -2:]))
    # check bbox was restored
    assert first_wcs.bounding_box is not None
    assert second_wcs.bounding_box is not None

    # enable all bounding boxes:
    pixmap = calc_pixmap(first_wcs, second_wcs, disable_bbox="none", order=order, stepsize=stepsize)
    assert_allclose(pixmap[2:, 2:], ok_pixmap[2:, 2:], rtol=1.0e-6, atol=1.0e-8)
    assert np.all(np.logical_not(np.isfinite(pixmap[:2, :2])))
    # check bbox was restored
    assert first_wcs.bounding_box is not None
    assert second_wcs.bounding_box is not None
