""" Test various utility functions """
from numpy.testing import assert_array_equal
import numpy as np
import pytest

from stcal.resample.utils import (
    build_driz_weight,
    build_mask,
    compute_mean_pixel_area,
    get_tmeasure,
    is_flux_density,
    is_imaging_wcs,
    resample_range,
    _get_inverse_variance,
)

from . helpers import make_input_model, JWST_DQ_FLAG_DEF

GOOD = 0
DQ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
BITVALUES = 2**0 + 2**2
BITVALUES_STR = f'{2**0}, {2**2}'
BITVALUES_INV_STR = f'~{2**0}, {2**2}'
JWST_NAMES = 'DO_NOT_USE,JUMP_DET'
JWST_NAMES_INV = '~' + JWST_NAMES


@pytest.mark.parametrize(
    'dq, bitvalues, expected', [
        (DQ, 0, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])),
        (DQ, BITVALUES, np.array([1, 1, 0, 0, 1, 1, 0, 0, 0])),
        (DQ, BITVALUES_STR, np.array([1, 1, 0, 0, 1, 1, 0, 0, 0])),
        (DQ, BITVALUES_INV_STR, np.array([1, 0, 1, 0, 0, 0, 0, 0, 1])),
        (DQ, JWST_NAMES, np.array([1, 1, 0, 0, 1, 1, 0, 0, 0])),
        (DQ, JWST_NAMES_INV, np.array([1, 0, 1, 0, 0, 0, 0, 0, 1])),
        (DQ, None, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])),
    ]
)
def test_build_mask(dq, bitvalues, expected):
    """ Test logic of mask building

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
        ((1, ), ((1, 500), (0, 350)), True, None),
        ((1, ), None, True, None),
        ((1000, 800), ((1, 500), ), True, None),
        ((1000, 800), ((1, 500), (0, 350), (0, 350)), True, None),
        ((1, ), ((1, 500), (0, 350)), True, None),
        ((1200, 1400), ((700, 300), (600, 800)), False, (700, 700, 600, 800)),
        ((1200, 1400), ((600, 800), (700, 300)), False, (600, 800, 700, 700)),
        ((1200, 1400), ((300, 700), (600, 800)), False, (300, 700, 600, 800)),
        ((750, 470), ((300, 700), (600, 800)), False, (300, 469, 600, 749)),
        ((750, 470), ((-5, -1), (-800, -600)), False, (0, 0, 0, 0)),
        ((750, 470), None, False, (0, 469, 0, 749)),
        ((-750, -470), None, False, (0, 0, 0, 0)),
    ]
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
    area = np.deg2rad(wcs_gwcs.pixel_scale)**2
    assert abs(
        compute_mean_pixel_area(wcs_gwcs) / area - 1.0
    ) < 1e-5


@pytest.mark.parametrize('unit,result',
                         [('Jy', True), ('MJy', True),
                          ('MJy/sr', False), ('DN/s', False),
                          ('bad_unit', False), (None, False)])
def test_is_flux_density(unit, result):
    assert is_flux_density(unit) is result


@pytest.mark.parametrize("weight_type", ["ivm", "exptime"])
def test_build_driz_weight(weight_type):
    """Check that correct weight map is returned of different weight types"""

    model = make_input_model((10, 10))

    model["dq"][0] = JWST_DQ_FLAG_DEF.DO_NOT_USE
    model["measurement_time"] = 10.0
    model["var_rnoise"] /= 10.0

    weight_map = build_driz_weight(
        model,
        weight_type=weight_type,
        good_bits=GOOD
    )
    assert_array_equal(weight_map[0], 0)
    assert_array_equal(weight_map[1:], 10.0)
    assert weight_map.dtype == np.float32


@pytest.mark.parametrize("weight_type", ["ivm", None])
def test_build_driz_weight_zeros(weight_type):
    """Check that zero or not finite weight maps get set to 1"""
    model = make_input_model((10, 10))

    weight_map = build_driz_weight(model, weight_type=weight_type)

    assert_array_equal(weight_map, 1)


@pytest.mark.parametrize("weight_type", ["ivm-smed", "ivm-med5"])
def test_unsupported_weight_type(weight_type):
    model = make_input_model((10, 10))
    with pytest.raises(ValueError, match=fr"^Invalid weight type: {repr(weight_type)}"):
        build_driz_weight(model, weight_type=weight_type)


@pytest.mark.parametrize("array_name", ["var_rnoise", "var_sky"])
def test_get_inverse_variance_valid_and_invalid(array_name):
    arr = np.array([[4.0, 0.0], [np.nan, 1.0]])
    inv = _get_inverse_variance(arr, arr.shape, array_name)
    assert np.isclose(inv[0, 0], 0.25)
    assert inv[0, 1] == 0
    assert inv[1, 0] == 0
    assert inv[1, 1] == 1.0
    # Wrong shape
    with pytest.warns(RuntimeWarning, match=f"'{array_name}' array not available."):
        inv2 = _get_inverse_variance(None, (2, 2), array_name)
    assert np.all(inv2 == 1)
    with pytest.warns(RuntimeWarning, match=f"'{array_name}' array not available."):
        inv3 = _get_inverse_variance(np.ones((1, 1)), (2, 2), array_name)
    assert np.all(inv3 == 1)


