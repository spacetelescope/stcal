import gwcs
import numpy as np
import pytest
from astropy import coordinates as coord
from astropy import units as u
from astropy import wcs as fitswcs
from astropy.io import fits
from astropy.modeling import models
from gwcs import coordinate_frames as cf

from stcal.alignment import resample_utils
from stcal.alignment.util import (
    _compute_fiducial_from_footprints,
    _validate_wcs_list,
    compute_fiducial,
    compute_s_region_imaging,
    compute_s_region_keyword,
    compute_scale,
    sregion_to_footprint,
    wcs_bbox_from_shape,
)


def _create_wcs_object_without_distortion(
    fiducial_world,
    pscale,
    shape,
):
    # subtract 0 shift to mimic a resampled WCS that does include shift transforms
    shift = models.Shift() & models.Shift()
    scale = models.Scale(pscale[0]) & models.Scale(pscale[1])

    tan = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(
        fiducial_world[0],
        fiducial_world[1],
        180,
    )

    det2sky = shift | scale | tan | celestial_rotation
    det2sky.name = "linear_transform"

    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(reference_frame=coord.FK5(), name="fk5", unit=(u.deg, u.deg))

    pipeline = [(detector_frame, det2sky), (sky_frame, None)]

    wcs_obj = gwcs.WCS(pipeline)

    wcs_obj.bounding_box = (
        (-0.5, shape[-1] - 0.5),
        (-0.5, shape[-2] - 0.5),
    )

    return wcs_obj


def _create_wcs_and_datamodel(fiducial_world, shape, pscale):
    wcs = _create_wcs_object_without_distortion(fiducial_world=fiducial_world, shape=shape, pscale=pscale)
    ra_ref, dec_ref = fiducial_world[0], fiducial_world[1]
    return DataModel(
        ra_ref=ra_ref,
        dec_ref=dec_ref,
        roll_ref=0,
        v2_ref=0,
        v3_ref=0,
        v3yangle=0,
        wcs=wcs,
    )


class WcsInfo:
    """
    JWST-like wcsinfo object
    """

    def __init__(self, ra_ref, dec_ref, roll_ref, v2_ref, v3_ref, v3yangle):
        self.ra_ref = ra_ref
        self.dec_ref = dec_ref
        self.ctype1 = "RA---TAN"
        self.ctype2 = "DEC--TAN"
        self.v2_ref = v2_ref
        self.v3_ref = v3_ref
        self.v3yangle = v3yangle
        self.roll_ref = roll_ref
        self.vparity = -1
        self.wcsaxes = 2
        self.s_region = ""
        self.instance = self.instance()

    def instance(self):
        return {
            "ra_ref": self.ra_ref,
            "dec_ref": self.dec_ref,
            "ctype1": self.ctype1,
            "ctype2": self.ctype2,
            "v2_ref": self.v2_ref,
            "v3_ref": self.v3_ref,
            "v3yangle": self.v3yangle,
            "roll_ref": self.roll_ref,
            "vparity": self.vparity,
            "wcsaxes": self.wcsaxes,
            "s_region": self.s_region,
        }


class Coordinates:
    def __init__(self):
        self.reference_frame = "ICRS"


class MetaData:
    def __init__(self, ra_ref, dec_ref, roll_ref, v2_ref, v3_ref, v3yangle, wcs=None):
        self.wcsinfo = WcsInfo(ra_ref, dec_ref, roll_ref, v2_ref, v3_ref, v3yangle)
        self.wcs = wcs
        self.coordinates = Coordinates()


class DataModel:
    """JWST-like datamodel object"""

    def __init__(self, ra_ref, dec_ref, roll_ref, v2_ref, v3_ref, v3yangle, wcs=None):
        self.meta = MetaData(ra_ref, dec_ref, roll_ref, v2_ref, v3_ref, v3yangle, wcs=wcs)


@pytest.mark.parametrize("footprint", [True, False])
def test_compute_fiducial(footprint):
    """Test that util.compute_fiducial can properly determine the center of the
    WCS's footprint.
    """

    shape = (3, 3)  # in pixels
    fiducial_world = (0, 0)  # in deg
    pscale = (0.000014, 0.000014)  # in deg/pixel

    wcs = _create_wcs_object_without_distortion(fiducial_world=fiducial_world, shape=shape, pscale=pscale)
    if footprint:
        footprint = wcs.footprint()
        computed_fiducial = _compute_fiducial_from_footprints([footprint])
    else:
        computed_fiducial = compute_fiducial([wcs])

    assert all(np.isclose(wcs(1, 1), computed_fiducial))


@pytest.mark.parametrize("pscales", [(0.000014, 0.000014), (0.000028, 0.000014)])
def test_compute_scale(pscales):
    """Test that util.compute_scale can properly determine the pixel scale of a
    WCS object.
    """
    shape = (3, 3)  # in pixels
    fiducial_world = (0, 0)  # in deg
    pscale = (pscales[0], pscales[1])  # in deg/pixel

    wcs = _create_wcs_object_without_distortion(fiducial_world=fiducial_world, shape=shape, pscale=pscale)
    expected_scale = np.sqrt(pscale[0] * pscale[1])

    computed_scale = compute_scale(wcs=wcs, fiducial=fiducial_world)

    assert np.isclose(expected_scale, computed_scale)


def test_sregion_to_footprint():
    """Test that util.sregion_to_footprint can properly convert an S_REGION
    string to a list of vertices.
    """
    s_region = (
        "POLYGON ICRS  1.000000000 2.000000000 3.000000000 4.000000000 "
        "5.000000000 6.000000000 7.000000000 8.000000000"
    )
    expected_footprint = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    footprint = sregion_to_footprint(s_region)

    assert footprint.shape == (4, 2)
    assert np.allclose(footprint, expected_footprint)


def test_validate_wcs_list():
    shape = (3, 3)  # in pixels
    fiducial_world = (10, 0)  # in deg
    pscale = (0.000028, 0.000028)  # in deg/pixel

    dm_1 = _create_wcs_and_datamodel(fiducial_world, shape, pscale)
    wcs_1 = dm_1.meta.wcs

    # shift fiducial by one pixel in both directions and create a new WCS
    fiducial_world = (
        fiducial_world[0] - 0.000028,
        fiducial_world[1] - 0.000028,
    )
    dm_2 = _create_wcs_and_datamodel(fiducial_world, shape, pscale)
    wcs_2 = dm_2.meta.wcs

    wcs_list = [wcs_1, wcs_2]

    assert _validate_wcs_list(wcs_list)


@pytest.mark.parametrize(
    ("wcs_list", "expected_error"),
    [
        ([], TypeError),
        ([1, 2, 3], TypeError),
        (["1", "2", "3"], TypeError),
        (["1", None, []], TypeError),
        ("1", TypeError),
        (1, ValueError),
        (None, ValueError),
    ],
)
def test_validate_wcs_list_invalid(wcs_list, expected_error):
    with pytest.raises(expected_error, match=r".*"):
        _validate_wcs_list(wcs_list)


def get_fake_wcs():
    fake_wcs1 = fitswcs.WCS(
        fits.Header(
            {
                "NAXIS": 2,
                "NAXIS1": 4,
                "NAXIS2": 4,
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRVAL1": 0,
                "CRVAL2": 0,
                "CRPIX1": 1,
                "CRPIX2": 1,
                "CDELT1": -0.1,
                "CDELT2": 0.1,
            }
        )
    )
    fake_wcs2 = fitswcs.WCS(
        fits.Header(
            {
                "NAXIS": 2,
                "NAXIS1": 5,
                "NAXIS2": 5,
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRVAL1": 0,
                "CRVAL2": 0,
                "CRPIX1": 1,
                "CRPIX2": 1,
                "CDELT1": -0.05,
                "CDELT2": 0.05,
            }
        )
    )
    return fake_wcs1, fake_wcs2


def test_wcs_bbox_from_shape_2d():
    bb = wcs_bbox_from_shape((512, 2048))
    assert bb == ((-0.5, 2047.5), (-0.5, 511.5))


def test_wcs_bbox_from_shape_3d():
    bb = wcs_bbox_from_shape((3, 32, 2048))
    assert bb == ((-0.5, 2047.5), (-0.5, 31.5))

    bb = wcs_bbox_from_shape((750, 45, 50))
    assert bb == ((-0.5, 49.5), (-0.5, 44.5))


@pytest.mark.parametrize(
    ("shape", "pixmap_expected_shape"),
    [
        (None, (4, 4, 2)),
        ((100, 200), (100, 200, 2)),
    ],
)
def test_calc_pixmap_shape(shape, pixmap_expected_shape):
    # TODO: add test for gwcs.WCS
    wcs1, wcs2 = get_fake_wcs()
    pixmap = resample_utils.calc_pixmap(wcs1, wcs2, shape=shape)
    assert pixmap.shape == pixmap_expected_shape


def test_calc_pixmap():
    # generate 2 wcses with different scales
    output_frame = gwcs.Frame2D(name="world")
    in_transform = models.Scale(1) & models.Scale(1)
    out_transform = models.Scale(2) & models.Scale(2)
    in_wcs = gwcs.WCS(in_transform, output_frame=output_frame)
    out_wcs = gwcs.WCS(out_transform, output_frame=output_frame)
    in_shape = (3, 4)
    pixmap = resample_utils.calc_pixmap(in_wcs, out_wcs, in_shape)
    # we expect given the 2x scale difference to have a pixmap
    # with pixel coordinates / 2
    # use mgrid to generate these coordinates (and reshuffle to match the pixmap)
    expected = np.swapaxes(np.mgrid[:4, :3] / 2.0, 0, 2)
    np.testing.assert_equal(pixmap, expected)


@pytest.mark.parametrize(
    ("model", "footprint", "expected_s_region", "expected_log_info"),
    [
        (
            _create_wcs_and_datamodel((10, 0), (3, 3), (0.000028, 0.000028)),
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            "POLYGON ICRS  1.000000000 2.000000000 3.000000000 4.000000000 5.000000000 6.000000000 7.000000000 8.000000000",  # noqa: E501
            "Update S_REGION to POLYGON ICRS  1.000000000 2.000000000 3.000000000 4.000000000 5.000000000 6.000000000 7.000000000 8.000000000",  # noqa: E501
        ),
        (
            _create_wcs_and_datamodel((10, 0), (3, 3), (0.000028, 0.000028)),
            np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0], [7.0, 8.0]]),
            None,
            "There are NaNs in s_region, S_REGION not updated.",
        ),
    ],
)
def test_compute_s_region_keyword(model, footprint, expected_s_region, expected_log_info, caplog):
    """
    Test that S_REGION keyword is being properly populated with the coordinate values.
    """
    model.meta.wcsinfo.s_region = compute_s_region_keyword(footprint)
    assert model.meta.wcsinfo.s_region == expected_s_region
    assert expected_log_info in caplog.text


@pytest.mark.parametrize(
    ("shape", "expected_bbox"),
    [
        ((100, 200), ((-0.5, 199.5), (-0.5, 99.5))),
        ((1, 1), ((-0.5, 0.5), (-0.5, 0.5))),
        ((0, 0), ((-0.5, -0.5), (-0.5, -0.5))),
    ],
)
def test_wcs_bbox_from_shape(shape, expected_bbox):
    """
    Test that the bounding box generated by wcs_bbox_from_shape is correct.
    """
    bbox = wcs_bbox_from_shape(shape)
    assert bbox == expected_bbox


@pytest.mark.parametrize(
    ("model", "bounding_box", "data"),
    [
        (
            _create_wcs_and_datamodel((10, 0), (3, 3), (0.000028, 0.000028)),
            ((-0.5, 2.5), (-0.5, 2.5)),
            None,
        ),
        (
            _create_wcs_and_datamodel((10, 0), (3, 3), (0.000028, 0.000028)),
            None,
            np.zeros((3, 3)),
        ),
    ],
)
def test_compute_s_region_imaging(model, bounding_box, data):
    """
    Test that S_REGION keyword is being properly updated with the coordinates
    corresponding to the footprint (same as WCS(bounding box)).
    """
    model.data = data
    model.meta.wcs.bounding_box = bounding_box
    expected_s_region_coords = [
        *model.meta.wcs(-0.5, -0.5),
        *model.meta.wcs(-0.5, 2.5),
        *model.meta.wcs(2.5, 2.5),
        *model.meta.wcs(2.5, -0.5),
    ]
    shape = data.shape if data is not None else None
    model.meta.wcsinfo.s_region = compute_s_region_imaging(model.meta.wcs, shape=shape, center=False)
    updated_s_region_coords = [float(x) for x in model.meta.wcsinfo.s_region.split(" ")[3:]]
    assert all(
        np.isclose(x, y) for x, y in zip(updated_s_region_coords, expected_s_region_coords, strict=False)
    )
