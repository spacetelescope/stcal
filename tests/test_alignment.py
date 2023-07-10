import numpy as np

from astropy.modeling import models
from astropy import coordinates as coord
from astropy import units as u

from gwcs import WCS
from gwcs import coordinate_frames as cf

import pytest

from stcal.alignment.util import (
    compute_fiducial,
    compute_scale,
    wcs_from_footprints,
)


def _create_wcs_object_without_distortion(
    fiducial_world,
    pscale,
    shape,
):
    fiducial_detector = tuple(shape.value)

    # subtract 1 to account for pixel indexing starting at 0
    shift = models.Shift(-(fiducial_detector[0] - 1)) & models.Shift(
        -(fiducial_detector[1] - 1)
    )

    scale = models.Scale(pscale[0].to("deg")) & models.Scale(
        pscale[1].to("deg")
    )

    tan = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(
        fiducial_world[0],
        fiducial_world[1],
        180 * u.deg,
    )

    det2sky = shift | scale | tan | celestial_rotation
    det2sky.name = "linear_transform"

    detector_frame = cf.Frame2D(
        name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix)
    )
    sky_frame = cf.CelestialFrame(
        reference_frame=coord.FK5(), name="fk5", unit=(u.deg, u.deg)
    )

    pipeline = [(detector_frame, det2sky), (sky_frame, None)]

    wcs_obj = WCS(pipeline)

    wcs_obj.bounding_box = (
        (-0.5, fiducial_detector[0] - 0.5),
        (-0.5, fiducial_detector[0] - 0.5),
    )

    return wcs_obj


def _create_wcs_and_datamodel(fiducial_world, shape, pscale):
    wcs = _create_wcs_object_without_distortion(
        fiducial_world=fiducial_world, shape=shape, pscale=pscale
    )
    ra_ref, dec_ref = fiducial_world[0].value, fiducial_world[1].value
    datamodel = DataModel(ra_ref=ra_ref, dec_ref=dec_ref, roll_ref=0,
                          v2_ref=0, v3_ref=0, v3yangle=0, wcs=wcs)
    return datamodel


class WcsInfo:
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


class Coordinates:
    def __init__(self):
        self.reference_frame = "ICRS"


class MetaData:
    def __init__(self, ra_ref, dec_ref, roll_ref, v2_ref, v3_ref, v3yangle, wcs=None):
        self.wcsinfo = WcsInfo(ra_ref, dec_ref, roll_ref, v2_ref, v3_ref, v3yangle)
        self.wcs = wcs
        self.coordinates=Coordinates()


class DataModel:
    def __init__(self, ra_ref, dec_ref, roll_ref, v2_ref, v3_ref, v3yangle, wcs=None):
        self.meta = MetaData(ra_ref, dec_ref, roll_ref, v2_ref, v3_ref, v3yangle, wcs=wcs)


def test_compute_fiducial():
    """Test that util.compute_fiducial can properly determine the center of the
    WCS's footprint.
    """

    shape = (3, 3) * u.pix
    fiducial_world = (0, 0) * u.deg
    pscale = (0.05, 0.05) * u.arcsec

    wcs = _create_wcs_object_without_distortion(
        fiducial_world=fiducial_world, shape=shape, pscale=pscale
    )

    computed_fiducial = compute_fiducial([wcs])

    assert all(np.isclose(wcs(1, 1), computed_fiducial))


@pytest.mark.parametrize("pscales", [(0.05, 0.05), (0.1, 0.05)])
def test_compute_scale(pscales):
    """Test that util.compute_scale can properly determine the pixel scale of a
    WCS object.
    """
    shape = (3, 3) * u.pix
    fiducial_world = (0, 0) * u.deg
    pscale = (pscales[0], pscales[1]) * u.arcsec

    wcs = _create_wcs_object_without_distortion(
        fiducial_world=fiducial_world, shape=shape, pscale=pscale
    )
    expected_scale = np.sqrt(pscale[0].to("deg") * pscale[1].to("deg")).value

    computed_scale = compute_scale(wcs=wcs, fiducial=fiducial_world.value)

    assert np.isclose(expected_scale, computed_scale)


def test_wcs_from_footprints():
    shape = (3, 3) * u.pix
    fiducial_world = (10, 0) * u.deg
    pscale = (0.1, 0.1) * u.arcsec

    dm_1 = _create_wcs_and_datamodel(fiducial_world, shape, pscale)
    wcs_1 = dm_1.meta.wcs

    # new fiducial will be shifted by one pixel in both directions
    fiducial_world -= pscale
    dm_2 = _create_wcs_and_datamodel(fiducial_world, shape, pscale)
    wcs_2 = dm_2.meta.wcs

    # check overlapping pixels have approximate the same world coordinate
    assert all(np.isclose(wcs_1(0, 1), wcs_2(1, 2)))
    assert all(np.isclose(wcs_1(1, 0), wcs_2(2, 1)))
    assert all(np.isclose(wcs_1(0, 0), wcs_2(1, 1)))
    assert all(np.isclose(wcs_1(1, 1), wcs_2(2, 2)))

    wcs = wcs_from_footprints([dm_1, dm_2])

    # check that center of calculated WCS matches the
    # expected position onto wcs_1 and wcs_2
    assert all(np.isclose(wcs(2, 2), wcs_1(0.5, 0.5)))
    assert all(np.isclose(wcs(2, 2), wcs_2(1.5, 1.5)))
