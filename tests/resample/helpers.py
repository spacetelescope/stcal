from astropy.nddata.bitmask import BitFlagNameMap
from astropy import coordinates as coord
from astropy.modeling import models as astmodels

from gwcs import coordinate_frames as cf
from gwcs.wcstools import wcs_from_fiducial
import numpy as np

from stcal.alignment import compute_s_region_imaging


class JWST_DQ_FLAG_DEF(BitFlagNameMap):
    DO_NOT_USE = 1
    SATURATED = 2
    JUMP_DET = 4


def make_gwcs(crpix, crval, pscale, shape):
    """ Simulate a gwcs from FITS WCS parameters.

    crpix - tuple of floats
    crval - tuple of floats (RA, DEC)
    pscale - pixel scale in degrees
    shape - array shape (numpy's convention)

    """
    prj = astmodels.Pix2Sky_TAN()
    fiducial = np.array(crval)

    pc = np.array([[-1., 0.], [0., 1.]])
    pc_matrix = astmodels.AffineTransformation2D(pc, name='pc_rotation_matrix')
    scale = (astmodels.Scale(pscale, name='cdelt1') &
             astmodels.Scale(pscale, name='cdelt2'))
    transform = pc_matrix | scale

    out_frame = cf.CelestialFrame(
        name='world',
        axes_names=('lon', 'lat'),
        reference_frame=coord.ICRS()
    )
    input_frame = cf.Frame2D(name="detector")
    wnew = wcs_from_fiducial(
        fiducial,
        coordinate_frame=out_frame,
        projection=prj,
        transform=transform,
        input_frame=input_frame
    )

    output_bounding_box = (
        (-0.5, float(shape[1]) - 0.5),
        (-0.5, float(shape[0]) - 0.5)
    )
    offset1, offset2 = crpix
    offsets = (astmodels.Shift(-offset1, name='crpix1') &
               astmodels.Shift(-offset2, name='crpix2'))

    wnew.insert_transform('detector', offsets, after=True)
    wnew.bounding_box = output_bounding_box

    tr = wnew.pipeline[0].transform
    pix_area = (
        np.deg2rad(tr['cdelt1'].factor.value) *
        np.deg2rad(tr['cdelt2'].factor.value)
    )

    wnew.pixel_area = pix_area
    wnew.pixel_scale = pscale
    wnew.pixel_shape = shape[::-1]
    wnew.array_shape = shape

    return wnew


def make_input_model(shape, crpix=(0, 0), crval=(0, 0), pscale=2.0e-5,
                     group_id=1, exptime=1):
    w = make_gwcs(
        crpix=crpix,
        crval=crval,
        pscale=pscale,
        shape=shape
    )

    model = {
        "data": np.zeros(shape, dtype=np.float32),
        "dq": np.zeros(shape, dtype=np.int32),

        # meta:
        "filename": "",
        "group_id": group_id,
        "s_region": compute_s_region_imaging(w),
        "wcs": w,
        "bunit_data": "MJy",

        "exposure_time": exptime,
        "start_time": 0.0,
        "end_time": exptime,
        "duration": exptime,
        "measurement_time": exptime,
        "effective_exposure_time": exptime,
        "elapsed_exposure_time": exptime,

        "pixelarea_steradians": w.pixel_area,
        "pixelarea_arcsecsq": w.pixel_area * (np.rad2deg(1) * 3600)**2,

        "level": 0.0,  # sky level
        "subtracted": False,
    }

    for arr in ["var_flat", "var_rnoise", "var_poisson"]:
        model[arr] = np.ones(shape, dtype=np.float32)

    model["err"] = np.sqrt(3.0) * np.ones(shape, dtype=np.float32)

    return model


def make_output_model(crpix, crval, pscale, shape):
    w = make_gwcs(
        crpix=crpix,
        crval=crval,
        pscale=pscale,
        shape=shape
    )

    model = {
        # WCS:
        "wcs": w,
        "pixelarea_steradians": w.pixel_area,
        "pixelarea_arcsecsq": w.pixel_area * (np.rad2deg(1) * 3600)**2,

        # main arrays:
        "data": None,
        "wht": None,
        "con": None,

        # error arrays:
        "var_rnoise": None,
        "var_flat": None,
        "var_poisson": None,
        "err": None,

        # drizzle info:
        "pointings": 0,

        # exposure time:
        "exposure_time": 0.0,
        "measurement_time": None,
        "start_time": None,
        "end_time": None,
        "duration": 0.0,

        # other meta:
        "filename": "",
        "s_region": compute_s_region_imaging(w),
        "bunit_data": "MJy",
    }

    return model
