""" Test various utility functions """
import asdf
from asdf_astropy.testing.helpers import assert_model_equal

from gwcs import coordinate_frames as cf
from numpy.testing import assert_array_equal
import numpy as np
import pytest

from stcal.resample.utils import (
    build_mask,
    bytes2human,
    compute_mean_pixel_area,
    get_tmeasure,
    is_imaging_wcs,
    resample_range,
    load_custom_wcs,
)

from . helpers import JWST_DQ_FLAG_DEF


DQ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
BITVALUES = 2**0 + 2**2
BITVALUES_STR = f'{2**0}, {2**2}'
BITVALUES_INV_STR = f'~{2**0}, {2**2}'
JWST_NAMES = 'DO_NOT_USE,JUMP_DET'
JWST_NAMES_INV = '~' + JWST_NAMES


def _assert_frame_equal(a, b):
    """ Copied from `gwcs`'s test_wcs.py """
    __tracebackhide__ = True

    assert type(a) is type(b)

    if a is None:
        return

    if not isinstance(a, cf.CoordinateFrame):
        return a == b

    assert a.name == b.name  # nosec
    assert a.axes_order == b.axes_order  # nosec
    assert a.axes_names == b.axes_names  # nosec
    assert a.unit == b.unit  # nosec
    assert a.reference_frame == b.reference_frame  # nosec


def _assert_wcs_equal(a, b):
    """ Based on corresponding function from `gwcs`'s test_wcs.py """
    assert a.name == b.name  # nosec

    assert a.pixel_shape == b.pixel_shape
    assert a.array_shape == b.array_shape
    if a.array_shape is not None:
        assert a.array_shape == b.pixel_shape[::-1]

    assert len(a.available_frames) == len(b.available_frames)  # nosec
    for a_step, b_step in zip(a.pipeline, b.pipeline):
        _assert_frame_equal(a_step.frame, b_step.frame)
        assert_model_equal(a_step.transform, b_step.transform)


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


def test_load_custom_wcs_no_shape(tmpdir, wcs_gwcs):
    """
    Test loading a WCS from an asdf file.
    """
    wcs_file = str(tmpdir / "wcs.asdf")
    wcs_gwcs.pixel_shape = None
    wcs_gwcs.array_shape = None
    wcs_gwcs.bounding_box = None

    with asdf.AsdfFile({"wcs": wcs_gwcs}) as af:
        af.write_to(wcs_file)

    with pytest.raises(ValueError):
        load_custom_wcs(wcs_file, output_shape=None)


@pytest.mark.parametrize(
    "array_shape, pixel_shape, output_shape, expected",
    [
        # (None, None, None, (1000, 1000)),  # from the bounding box
        # # (None, (123, 456), None, (456, 123)),  # fails
        # ((456, 123), None, None, (456, 123)),
        # ((456, 123), None, (567, 890), (890, 567)),
        ((456, 123), (123, 456), (567, 890), (890, 567)),
        ((456, 123), (123, 456), None, (890, 567)),
    ]
)
def test_load_custom_wcs(tmpdir, wcs_gwcs, array_shape, pixel_shape,
                         output_shape, expected):
    """
    Test loading a WCS from an asdf file. `expected` is expected
    ``wcs.array_shape``.

    """
    wcs_file = str(tmpdir / "wcs.asdf")

    wcs_gwcs.pixel_shape = pixel_shape
    wcs_gwcs.array_shape = array_shape

    with asdf.AsdfFile({"wcs": wcs_gwcs}) as af:
        af.write_to(wcs_file)

    if output_shape is not None:
        wcs_gwcs.array_shape = output_shape[::-1]

    wcs_read = load_custom_wcs(wcs_file, output_shape=output_shape)

    assert wcs_read.array_shape == expected

    _assert_wcs_equal(wcs_gwcs, wcs_read)


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


@pytest.mark.parametrize(
        "n, readable",
        [
            (10000, "9.8K"),
            (100001221, "95.4M")
        ]
)
def test_bytes2human(n, readable):
    assert bytes2human(n) == readable


def test_is_imaging_wcs(wcs_gwcs):
    assert is_imaging_wcs(wcs_gwcs)


def test_compute_mean_pixel_area(wcs_gwcs):
    area = np.deg2rad(wcs_gwcs.pixel_scale)**2
    assert abs(
        compute_mean_pixel_area(wcs_gwcs) / area - 1.0
    ) < 1e-5
