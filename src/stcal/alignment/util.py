"""Common utility functions for datamodel alignment."""
from __future__ import annotations

import functools
import logging
import re
import typing
from typing import TYPE_CHECKING, Tuple
import warnings

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import astropy

import gwcs
import numpy as np
from astropy import wcs as fitswcs
from astropy.coordinates import SkyCoord
from astropy.modeling import models as astmodels
from astropy.utils.misc import isiterable
from gwcs.wcstools import wcs_from_fiducial

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

__all__ = [
    "compute_scale",
    "compute_fiducial",
    "calc_rotation_matrix",
    "compute_s_region_imaging",
    "compute_s_region_keyword",
    "sregion_to_footprint",
    "wcs_from_footprints",
    "wcs_from_sregions",
    "wcs_bbox_from_shape",
    "reproject",
]


def _calculate_fiducial_from_spatial_footprint(
    spatial_footprint: np.ndarray,
) -> tuple:
    """
    Calculates the fiducial coordinates from a given spatial footprint.

    Parameters
    ----------
    spatial_footprint : np.ndarray
        An Nx2 array containing the world coordinates of the WCS footprint's
        bounding box, where N is the number of bounding box positions.

    Returns
    -------
    lon_fiducial, lat_fiducial : np.ndarray, np.ndarray
        The world coordinates of the fiducial point in the output coordinate frame.
    """
    lon, lat = spatial_footprint.T
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    x_mid = (np.max(x) + np.min(x)) / 2.0
    y_mid = (np.max(y) + np.min(y)) / 2.0
    z_mid = (np.max(z) + np.min(z)) / 2.0
    lon_fiducial = np.rad2deg(np.arctan2(y_mid, x_mid)) % 360.0
    lat_fiducial = np.rad2deg(np.arctan2(z_mid, np.sqrt(x_mid**2 + y_mid**2)))
    return lon_fiducial, lat_fiducial


def _generate_tranform(
    wcs: gwcs.wcs.WCS,
    wcsinfo: dict,
    ref_fiducial: np.ndarray,
    pscale_ratio: float | None = None,
    pscale: float | None = None,
    rotation: float | None = None,
    transform: astmodels.Model | None = None,
) -> astmodels.Model:
    """
    Creates a transform from pixel to world coordinates based on a
    reference datamodel's WCS.

    Parameters
    ----------
    wcs : ~gwcs.wcs.WCS
        The WCS object.

    wcsinfo : dict
        A dictionary containing the WCS FITS keywords and corresponding values.

    ref_fiducial : np.array
        A two-elements array containing the world coordinates of the fiducial point.

    pscale_ratio : int, None, optional
        Ratio of input to output pixel scale. This parameter is only used when
        ``pscale=None`` and, in that case, it is passed on to ``compute_scale``.

    pscale : float, None, optional
        The plate scale. If `None`, the plate scale is calculated from the reference
        datamodel.

    rotation : float, None, optional
        Position angle of output image's Y-axis relative to North.
        A value of 0.0 would orient the final output image to be North up.
        The default of `None` specifies that the images will not be rotated,
        but will instead be resampled in the default orientation for the camera
        with the x and y axes of the resampled image corresponding
        approximately to the detector axes. Ignored when ``transform`` is
        provided. If `None`, the rotation angle is extracted from the
        reference model's ``meta.wcsinfo.roll_ref``.

    transform : ~astropy.modeling.Model, None, optional
        A transform between frames.

    Returns
    -------
    transform : ~astropy.modeling.Model
        An :py:mod:`~astropy` model containing the transform between frames.
    """
    if transform is None:
        sky_axes = wcs._get_axes_indices().tolist()  # noqa: SLF001
        v3yangle = np.deg2rad(wcsinfo["v3yangle"])
        vparity = wcsinfo["vparity"]
        if rotation is None:
            roll_ref = np.deg2rad(wcsinfo["roll_ref"])
        else:
            roll_ref = np.deg2rad(rotation) + (vparity * v3yangle)

        # reshape the rotation matrix returned from calc_rotation_matrix
        # into the correct shape for constructing the transformation
        pc = np.reshape(calc_rotation_matrix(roll_ref, v3yangle, vparity=vparity), (2, 2))

        rotation = astmodels.AffineTransformation2D(pc, name="pc_rotation_matrix")
        transform = [rotation]
        if sky_axes:
            if not pscale:
                pscale = compute_scale(wcs, ref_fiducial, pscale_ratio=pscale_ratio)
            transform.append(astmodels.Scale(pscale, name="cdelt1") & astmodels.Scale(pscale, name="cdelt2"))

        if transform:
            transform = functools.reduce(lambda x, y: x | y, transform)

    return transform


def _get_bounding_box_with_offsets(
        footprints: list[np.ndarray],
        ref_wcs: gwcs.wcs.WCS,
        fiducial: Sequence,
        crpix: Sequence | None,
        shape: Sequence | None
        ) -> Tuple[tuple, tuple, astmodels.Model]:
    """
    Calculates axis minimum values and bounding box.
    Calculates the offsets to the transform.

    Parameters
    ----------
    footprints : list
        A list of numpy arrays each of shape (N, 2) containing the
        (RA, Dec) vertices demarcating the footprint of the input WCSs.

    ref_wcs : ~gwcs.wcs.WCS
        The reference WCS object.

    fiducial : tuple
        A tuple containing the world coordinates of the fiducial point.

    crpix : list or tuple, None, optional
        0-indexed pixel coordinates of the reference pixel.

    shape : tuple, None, optional
        Shape (using `numpy.ndarray` convention) of the image array associated
        with the ``ref_wcs``.


    Returns
    -------

    tuple
        A tuple containing the bounding box region in the format
        ((x0_lower, x0_upper), (x1_lower, x1_upper), ...).

    tuple
        Shape of the image. When ``shape`` argument is `None`, shape is
        determined from the upper limit of the computed bounding box, otherwise
        input value of ``shape`` is returned.

    ~astropy.modeling.Model
        A model with the offsets to be added to the WCS's transform.

    """
    domain_bounds = np.hstack([ref_wcs.backward_transform(*f.T) for f in footprints])
    domain_min = np.min(domain_bounds, axis=1)
    domain_max = np.max(domain_bounds, axis=1)

    native_crpix = ref_wcs.backward_transform(*fiducial)

    if crpix is None:
        # shift the coordinates by domain_min so that all input footprints
        # will project to positive coordinates in the offsetted ref_wcs
        offsets = tuple(ncrp - dmin for ncrp, dmin in zip(native_crpix, domain_min))

    else:
        # assume 0-based CRPIX and require that fiducial would map to the user
        # defined crpix value:
        offsets = tuple(
            crp - ncrp for ncrp, crp in zip(native_crpix, crpix)
        )

    # Also offset domain limits:
    domain_min += offsets
    domain_max += offsets

    if shape is None:
        shape = tuple(int(dmax + 0.5) for dmax in domain_max[::-1])
        bounding_box = tuple(
            (-0.5, s - 0.5) for s in shape[::-1]
        )

    else:
        # trim upper bounding box limits
        shape = tuple(shape)
        bounding_box = tuple(
            (max(0, int(dmin + 0.5)) - 0.5, min(int(dmax + 0.5), sz) - 0.5)
            for dmin, dmax, sz in zip(domain_min, domain_max, shape[::-1])
        )

    model = astmodels.Shift(-offsets[0], name="crpix1")
    for k, shift in enumerate(offsets[1:]):
        model = model.__and__(astmodels.Shift(-shift, name=f"crpix{k + 2:d}"))

    return bounding_box, shape, model


def _calculate_fiducial(footprints: list[np.ndarray],
                        crval: Sequence | None = None) -> tuple:
    """
    Calculates the coordinates of the fiducial point and, if necessary, updates it with
    the values in CRVAL (the update is applied to spatial axes only).

    Parameters
    ----------
    footprints : list
        A list of numpy arrays each of shape (N, 2) containing the
        (RA, Dec) vertices demarcating the footprint of the input WCSs.

    crval : list, optional
        A reference world coordinate associated with the reference pixel. If not `None`,
        then the fiducial coordinates of the spatial axes will be updated with the
        values from ``crval``.

    Returns
    -------
    fiducial : tuple
        A tuple containing the world coordinate of the fiducial point.
    """
    if crval is not None:
        return tuple(crval)
    return _compute_fiducial_from_footprints(footprints)


def _calculate_new_wcs(wcs: gwcs.wcs.WCS,
                       shape: Sequence | None,
                       footprints: list[np.ndarray],
                       fiducial: tuple,
                       crpix: Sequence | None = None,
                       transform: astmodels.Model | None = None,
                       ) -> gwcs.wcs.WCS:
    """
    Calculates a new WCS object based on the combined footprints
    and reference WCS provided.

    Parameters
    ----------
    wcs : ~gwcs.wcs.WCS
        The reference WCS object.

    footprints : list
        A list of numpy arrays each of shape (N, 2) containing the
        (RA, Dec) vertices demarcating the footprint of the input WCSs.

    fiducial : tuple
        A tuple containing the location on the sky in some standard
        coordinate system.

    shape : list, None, optional
        The shape of the new WCS's pixel grid. If `None`, then the output bounding box
        will be used to determine it.

    crpix : tuple, None, optional
        0-indexed coordinates of the reference pixel.

    transform : ~astropy.modeling.Model, None, optional
        An optional transform to be prepended to the transform constructed by the
        fiducial point. The number of outputs of this transform must equal the number
        of axes in the coordinate frame.

    Returns
    -------
    wcs_new : ~gwcs.wcs.WCS
        The new WCS object that corresponds to the combined WCS objects in `wcs_list`.
    """
    wcs_new = wcs_from_fiducial(
        fiducial,
        coordinate_frame=wcs.output_frame,
        projection=astmodels.Pix2Sky_TAN(),
        transform=transform,
        input_frame=wcs.input_frame,
    )

    bounding_box, shape, offsets = _get_bounding_box_with_offsets(
        footprints,
        ref_wcs=wcs_new,
        fiducial=fiducial,
        crpix=crpix,
        shape=shape
    )

    if any(d < 2 for d in shape):
        raise ValueError(
            "Computed shape for the output image using provided "
            "WCS parameters is too small."
        )

    wcs_new.insert_transform("detector", offsets, after=True)
    wcs_new.bounding_box = bounding_box
    wcs_new.pixel_shape = shape[::-1]
    wcs_new.array_shape = shape
    return wcs_new


def _validate_wcs_list(wcs_list: list[gwcs.wcs.WCS]) -> bool:
    """
    Validates wcs_list.

    Parameters
    ----------
    wcs_list : list
        A list of WCS objects.

    Returns
    -------
    bool or Exception
        If wcs_list is valid, returns True. Otherwise, it will raise an error.

    Raises
    ------
    ValueError
        Raised whenever wcs_list is not an iterable.
    TypeError
        Raised whenever wcs_list is empty or any of its content is not an
        instance of WCS.
    """
    if not isiterable(wcs_list):
        msg = "Expected 'wcs_list' to be an iterable of WCS objects."
        raise ValueError(msg)

    if len(wcs_list):
        if not all(isinstance(w, gwcs.WCS) for w in wcs_list):
            msg = "All items in 'wcs_list' are to be instances of gwcs.wcs.WCS."
            raise TypeError(msg)
    else:
        msg = "'wcs_list' should not be empty."
        raise TypeError(msg)

    return True


def compute_scale(
    wcs: gwcs.wcs.WCS,
    fiducial: tuple | np.ndarray,
    disp_axis: int | None = None,
    pscale_ratio: float | None = None,
) -> float:
    """Compute the scale at the fiducial point on the detector..

    Parameters
    ----------
    wcs : ~gwcs.wcs.WCS
        Reference WCS object from which to compute a scaling factor.

    fiducial : tuple
        Input fiducial of (RA, DEC) or (RA, DEC, Wavelength) used in calculating
        reference points.

    disp_axis : int, None, optional
        Dispersion axis integer. Assumes the same convention as
        ``wcsinfo.dispersion_direction``

    pscale_ratio : int, None, optional
        Ratio of input to output pixel scale

    Returns
    -------
    scale : float
        Scaling factor for x and y or cross-dispersion direction.

    """
    spectral = "SPECTRAL" in wcs.output_frame.axes_type

    if spectral and disp_axis is None:
        msg = "If input WCS is spectral, a disp_axis must be given"
        raise ValueError(msg)

    crpix = np.array(wcs.invert(*fiducial, with_bounding_box=False))

    delta = np.zeros_like(crpix)
    spatial_idx = np.where(np.array(wcs.output_frame.axes_type) == "SPATIAL")[0]
    delta[spatial_idx[0]] = 1

    crpix_with_offsets = np.vstack((crpix, crpix + delta, crpix + np.roll(delta, 1))).T
    crval_with_offsets = wcs(*crpix_with_offsets, with_bounding_box=False)

    coords = SkyCoord(
        ra=crval_with_offsets[spatial_idx[0]],
        dec=crval_with_offsets[spatial_idx[1]],
        unit="deg",
    )
    xscale = np.abs(coords[0].separation(coords[1]).value)
    yscale = np.abs(coords[0].separation(coords[2]).value)

    if pscale_ratio is not None:
        xscale *= pscale_ratio
        yscale *= pscale_ratio

    if spectral:
        # Assuming scale doesn't change with wavelength
        # Assuming disp_axis is consistent with DataModel.meta.wcsinfo.dispersion.direction
        return float(yscale) if disp_axis == 1 else float(xscale)

    return float(np.sqrt(xscale * yscale))


def compute_fiducial(wcslist: list,
                     bounding_box: Sequence | None = None) -> np.ndarray:
    """
    Calculates the world coordinates of the fiducial point of a list of WCS objects.
    For a celestial footprint this is the center. For a spectral footprint, it is the
    beginning of its range.

    Parameters
    ----------
    wcslist : list
        A list containing all the WCS objects for which the fiducial is to be
        calculated.
    bounding_box : tuple, list, None, optional
        The bounding box over which the WCS is valid. It can be a either tuple of tuples
        or a list of lists of size 2 where each element represents a range of
        (low, high) values. The bounding_box is in the order of the axes, axes_order.
        For two inputs and axes_order(0, 1) the bounding box can be either
        ((xlow, xhigh), (ylow, yhigh)) or [[xlow, xhigh], [ylow, yhigh]].

    Returns
    -------
    fiducial : np.ndarray
        A two-elements array containing the world coordinates of the fiducial point
        in the combined output coordinate frame.

    Notes
    -----
    This function assumes all WCSs have the same output coordinate frame.
    """
    axes_types = wcslist[0].output_frame.axes_type
    spatial_axes = np.array(axes_types) == "SPATIAL"
    spectral_axes = np.array(axes_types) == "SPECTRAL"
    footprints = np.hstack([w.footprint(bounding_box=bounding_box).T for w in wcslist])
    spatial_footprint = footprints[spatial_axes]
    spectral_footprint = footprints[spectral_axes]

    fiducial = np.empty(len(axes_types))
    if spatial_footprint.any():
        fiducial[spatial_axes] = _calculate_fiducial_from_spatial_footprint(spatial_footprint.T)
    if spectral_footprint.any():
        fiducial[spectral_axes] = spectral_footprint.min()
    return fiducial


def _compute_fiducial_from_footprints(footprints: list[np.ndarray]) -> tuple:
    """
    Calculates the world coordinates of the fiducial point of a list of WCS objects.
    For a celestial footprint this is the center. For a spectral footprint, it is the
    beginning of its range.

    Parameters
    ----------
    footprints : list
        A list of numpy arrays each of shape (N, 2) containing the
        (RA, Dec) vertices demarcating the footprint of the input WCSs.

    bounding_box : tuple, list, None
        The bounding box over which the WCS is valid. It can be a either tuple of tuples
        or a list of lists of size 2 where each element represents a range of
        (low, high) values. The bounding_box is in the order of the axes, axes_order.
        For two inputs and axes_order(0, 1) the bounding box can be either
        ((xlow, xhigh), (ylow, yhigh)) or [[xlow, xhigh], [ylow, yhigh]].

    Returns
    -------
    fiducial : tuple
        A tuple containing the world coordinates of the fiducial point
        in the combined output coordinate frame.

    Notes
    -----
    This function assumes all WCSs have the same output coordinate frame.
    """
    spatial_footprint = np.vstack(footprints)
    return _calculate_fiducial_from_spatial_footprint(spatial_footprint)


def calc_rotation_matrix(roll_ref: float, v3i_yangle: float, vparity: int = 1) -> list[float]:
    r"""Calculate the rotation matrix.

    Parameters
    ----------
    roll_ref : float
        Telescope roll angle of V3 North over East at the ref. point in radians

    v3i_yangle : float
        The angle between ideal Y-axis and V3 in radians.

    vparity : int
        The x-axis parity, usually taken from the JWST SIAF parameter VIdlParity.
        Value should be "1" or "-1".

    Returns
    -------
    matrix: list
        A list containing the rotation matrix elements in column order.

    Notes
    -----
    The rotation matrix is

    .. math::
        PC = \\begin{bmatrix}
                pc_{1,1} & pc_{2,1} \\\\
                pc_{1,2} & pc_{2,2}
            \\end{bmatrix}
    """
    if vparity not in (1, -1):
        msg = f"vparity should be 1 or -1. Input was: {vparity}"
        raise ValueError(msg)

    rel_angle = roll_ref - (vparity * v3i_yangle)

    pc1_1 = vparity * np.cos(rel_angle)
    pc1_2 = np.sin(rel_angle)
    pc2_1 = vparity * -np.sin(rel_angle)
    pc2_2 = np.cos(rel_angle)

    return [pc1_1, pc1_2, pc2_1, pc2_2]


def wcs_from_footprints(
    wcs_list: list[gwcs.wcs.WCS],
    ref_wcs: gwcs.wcs.WCS,
    ref_wcsinfo: dict,
    transform: astropy.modeling.models.Model | None = None,
    bounding_box: Sequence | None = None,
    pscale_ratio: float | None = None,
    pscale: float | None = None,
    rotation: float | None = None,
    shape: Sequence | None = None,
    crpix: Sequence | None = None,
    crval: Sequence | None = None,
) -> gwcs.wcs.WCS:
    """
    Create a WCS from a list of input datamodels.

    A fiducial point in the output coordinate frame is created from  the
    footprints of all WCS objects. For a spatial frame this is the center
    of the union of the footprints. For a spectral frame the fiducial is in
    the beginning of the footprint range.
    If ``refmodel`` is None, the first WCS object in the list is considered
    a reference. The output coordinate frame and projection (for celestial frames)
    is taken from ``refmodel``.
    If ``transform`` is not supplied, a compound transform is created using
    CDELTs and PC.
    If ``bounding_box`` is not supplied, the `bounding_box` of the new WCS is computed
    from `bounding_box` of all input WCSs.

    Parameters
    ----------
    wcs_list : list
        A list of valid datamodels.

    ref_wcs :
        A valid datamodel whose WCS is used as reference for the creation of the output
        coordinate frame, projection, and scaling and rotation transforms.
        If not supplied the first model in the list is used as ``refmodel``.

    ref_wcsinfo : dict
        A dictionary containing the WCS FITS keywords and corresponding values.

    transform : ~astropy.modeling.Model, None, optional
        A transform, passed to :py:func:`gwcs.wcstools.wcs_from_fiducial`
        If not supplied `Scaling | Rotation` is computed from ``refmodel``.

    bounding_box : tuple, None, optional
        Bounding_box of the new WCS.
        If not supplied it is computed from the bounding_box of all inputs.

    pscale_ratio : float, None, optional
        Ratio of input to output pixel scale. Ignored when either
        ``transform`` or ``pscale`` are provided.

    pscale : float, None, optional
        Absolute pixel scale in degrees. When provided, overrides
        ``pscale_ratio``. Ignored when ``transform`` is provided.

    rotation : float, None, optional
        Position angle of output image's Y-axis relative to North.
        A value of 0.0 would orient the final output image to be North up.
        The default of `None` specifies that the images will not be rotated,
        but will instead be resampled in the default orientation for the camera
        with the x and y axes of the resampled image corresponding
        approximately to the detector axes. Ignored when ``transform`` is
        provided.

    shape : tuple of int, None, optional
        Shape of the image (data array) using ``np.ndarray`` convention
        (``ny`` first and ``nx`` second). This value will be assigned to
        ``pixel_shape`` and ``array_shape`` properties of the returned
        WCS object.

    crpix : tuple of float, None, optional
        Position of the reference pixel in the image array.  If ``crpix`` is not
        specified, it will be set to the center of the bounding box of the
        returned WCS object.

    crval : tuple of float, None, optional
        Right ascension and declination of the reference pixel. Automatically
        computed if not provided.

    wcs_list : list, None, optional
        A list of WCS objects. If not supplied, the WCS objects are extracted
        from the input datamodels.

    Returns
    -------
    wcs_new : ~gwcs.wcs.WCS
        The WCS object corresponding to the combined input footprints.

    """
    msg = ("wcs_from_footprints is deprecated and will be removed in a future release."
           "It is recommended to use wcs_from_sregions instead.")
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    _validate_wcs_list(wcs_list)
    footprints = [w.footprint() for w in wcs_list]
    if ref_wcs is None:
        ref_wcs = wcs_list[0]
    return wcs_from_sregions(
        footprints,
        ref_wcs,
        ref_wcsinfo,
        transform=transform,
        pscale_ratio=pscale_ratio,
        pscale=pscale,
        rotation=rotation,
        shape=shape,
        crpix=crpix,
        crval=crval,
    )


def sregion_to_footprint(s_region: str) -> np.ndarray:
    """
    Parse the s_region string and return the footprint as an Nx2 array.

    Parameters
    ----------
    s_region : str
        The S_REGION header keyword

    Returns
    -------
    footprint : np.array
        A 2D array of the footprint of the region, shape (N, 2)
    """
    no_prefix = re.sub(r"[a-zA-Z]", "", s_region)
    return np.array(no_prefix.split(), dtype=float).reshape(-1, 2)


def wcs_from_sregions(
    footprints: list[np.ndarray] | list[str],
    ref_wcs: gwcs.wcs.WCS,
    ref_wcsinfo: dict,
    transform: astropy.modeling.models.Model | None = None,
    pscale_ratio: float | None = None,
    pscale: float | None = None,
    rotation: float | None = None,
    shape: Sequence | None = None,
    crpix: Sequence | None = None,
    crval: Sequence | None = None,
) -> gwcs.wcs.WCS:
    """
    Create a WCS from a list of input s_regions or footprints.

    A fiducial point in the output coordinate frame is created from  the
    footprints of all WCS objects. For a spatial frame this is the center
    of the union of the footprints. For a spectral frame the fiducial is in
    the beginning of the footprint range.
    If ``refmodel`` is None, the first WCS object in the list is considered
    a reference. The output coordinate frame and projection (for celestial frames)
    is taken from ``refmodel``.
    If ``transform`` is not supplied, a compound transform is created using
    CDELTs and PC.
    If ``bounding_box`` is not supplied, the `bounding_box` of the new WCS is computed
    from `bounding_box` of all input WCSs.

    Parameters
    ----------
    footprints : list of np.ndarray or list of str
        If list elements are numpy arrays, each should have shape (N, 2) and contain
        (RA, Dec) vertices demarcating the footprint of the input WCSs.
        If list elements are strings, each should be the S_REGION header keyword
        containing (RA, Dec) vertices demarcating the footprint of the input WCSs.

    ref_wcs : ~gwcs.wcs.WCS
        A WCS used as reference for the creation of the output
        coordinate frame, projection, and scaling and rotation transforms.

    ref_wcsinfo : dict
        A dictionary containing the WCS FITS keywords and corresponding values.

    transform : ~astropy.modeling.Model, None, optional
        A transform, passed to :py:func:`gwcs.wcstools.wcs_from_fiducial`
        If not supplied `Scaling | Rotation` is computed from ``refmodel``.

    pscale_ratio : float, None, optional
        Ratio of input to output pixel scale. Ignored when either
        ``transform`` or ``pscale`` are provided.

    pscale : float, None, optional
        Absolute pixel scale in degrees. When provided, overrides
        ``pscale_ratio``. Ignored when ``transform`` is provided.

    rotation : float, None, optional
        Position angle of output image's Y-axis relative to North.
        A value of 0.0 would orient the final output image to be North up.
        The default of `None` specifies that the images will not be rotated,
        but will instead be resampled in the default orientation for the camera
        with the x and y axes of the resampled image corresponding
        approximately to the detector axes. Ignored when ``transform`` is
        provided.

    shape : tuple of int, None, optional
        Shape of the image (data array) using ``np.ndarray`` convention
        (``ny`` first and ``nx`` second). This value will be assigned to
        ``pixel_shape`` and ``array_shape`` properties of the returned
        WCS object.

    crpix : tuple of float, None, optional
        Position of the reference pixel in the image array.  If ``crpix`` is not
        specified, it will be set to the center of the bounding box of the
        returned WCS object.

    crval : tuple of float, None, optional
        Right ascension and declination of the reference pixel. Automatically
        computed if not provided.

    Returns
    -------
    wcs_new : ~gwcs.wcs.WCS
        The WCS object corresponding to the combined input footprints.

    """
    footprints = [sregion_to_footprint(s_region)
                  if isinstance(s_region, str) else s_region
                  for s_region in footprints]
    fiducial = _calculate_fiducial(footprints, crval=crval)

    transform = _generate_tranform(
        ref_wcs,
        wcsinfo=ref_wcsinfo,
        pscale_ratio=pscale_ratio,
        pscale=pscale,
        rotation=rotation,
        ref_fiducial=np.array([ref_wcsinfo["ra_ref"], ref_wcsinfo["dec_ref"]]),
        transform=transform,
    )

    return _calculate_new_wcs(
        wcs=ref_wcs,
        footprints=footprints,
        fiducial=fiducial,
        shape=shape,
        crpix=crpix,
        transform=transform,
    )


def compute_s_region_imaging(wcs: gwcs.wcs.WCS,
                             shape: Sequence | None = None,
                             center: bool = True) -> str | None:
    """
    Update the ``S_REGION`` keyword using the WCS footprint.

    Parameters
    ----------
    wcs : ~gwcs.wcs.WCS
        The WCS object.

    shape : tuple, None, optional
        Shape of input model data array. Used to compute the bounding box if not
        provided in the WCS object, and required in that case. The default is None.

    center : bool, None, optional
        Whether or not to use the center of the pixel as reference for the
        coordinates, by default True

    Returns
    -------
    s_region : str
        String containing the S_REGION object.
    """
    bbox = wcs.bounding_box
    if shape is None and bbox is None:
        msg = "If wcs.bounding_box is not specified, shape must be provided."
        raise ValueError(msg)

    if shape is not None and bbox is None:
        bbox = wcs_bbox_from_shape(shape)
        wcs.bounding_box = bbox

    # footprint is an array of shape (2, 4) as we
    # are interested only in the footprint on the sky
    ### TODO: we shouldn't use center=True in the call below because we want to
    ### calculate the coordinates of the footprint based on the *bounding box*,
    ### which means we are interested in each pixel's vertice, not its center.
    ### By using center=True, a difference of 0.5 pixel should be accounted for
    ### when comparing the world coordinates of the bounding box and the footprint.
    footprint = wcs.footprint(bbox, center=center, axis_type="spatial").T
    # take only imaging footprint
    footprint = footprint[:2, :]

    # Make sure RA values are all positive
    np.mod(footprint[0], 360, out=footprint[0])

    footprint = footprint.T
    return compute_s_region_keyword(footprint)


def wcs_bbox_from_shape(shape: Sequence) -> tuple:
    """Create a bounding box from the shape of the data.

    This is appropriate to attach to a wcs object

    Parameters
    ----------
    shape : tuple
        The shape attribute from a `np.ndarray` array

    Returns
    -------
    bbox : tuple
        Bounding box in x, y order.
    """
    return (-0.5, shape[-1] - 0.5), (-0.5, shape[-2] - 0.5)


def compute_s_region_keyword(footprint: np.ndarray) -> str | None:
    """Update the S_REGION keyword.

    Parameters
    ----------
    footprint :
        A 4x2 numpy array containing the coordinates of the vertices of the footprint.

    Returns
    -------
    s_region : str
        String containing the S_REGION object.
    """
    s_region = "POLYGON ICRS  {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}".format(
       *footprint.flatten()
    )
    if "nan" in s_region:
        # do not update s_region if there are NaNs.
        log.info("There are NaNs in s_region, S_REGION not updated.")
        return None
    log.info("Update S_REGION to %s", s_region)
    return s_region


def reproject(wcs1: gwcs.wcs.WCS, wcs2: gwcs.wcs.WCS) -> Callable:
    """
    Given two WCSs or transforms return a function which takes pixel
    coordinates in the first WCS or transform and computes them in pixel coordinates
    in the second one. It performs the forward transformation of ``wcs1`` followed by the
    inverse of ``wcs2``.

    Parameters
    ----------
    wcs1 : astropy.wcs.WCS or gwcs.wcs.WCS
        Input WCS objects or transforms.
    wcs2 : astropy.wcs.WCS or gwcs.wcs.WCS
        Output WCS objects or transforms.

    Returns
    -------
        Function to compute the transformations.  It takes x, y
        positions in ``wcs1`` and returns x, y positions in ``wcs2``.
    """

    def _get_forward_transform_func(wcs1):
        """Get the forward transform function from the input WCS. If the wcs is a
        fitswcs.WCS object all_pix2world requires three inputs, the x (str, ndarrray),
        y (str, ndarray), and origin (int). The origin should be between 0, and 1
        https://docs.astropy.org/en/latest/wcs/index.html#loading-wcs-information-from-a-fits-file
        ).
        """
        if isinstance(wcs1, fitswcs.WCS):
            forward_transform = wcs1.all_pix2world
        elif isinstance(wcs1, gwcs.WCS):
            forward_transform = wcs1.forward_transform
        else:
            msg = "Expected input to be astropy.wcs.WCS or gwcs.WCS object"
            raise TypeError(msg)
        return forward_transform

    def _get_backward_transform_func(wcs2):
        if isinstance(wcs2, fitswcs.WCS):
            backward_transform = wcs2.all_world2pix
        elif isinstance(wcs2, gwcs.WCS):
            backward_transform = wcs2.backward_transform
        else:
            msg = "Expected input to be astropy.wcs.WCS or gwcs.WCS object"
            raise TypeError(msg)
        return backward_transform

    def _reproject(x: float | np.ndarray, y: float | np.ndarray) -> tuple:
        """
        Reprojects the input coordinates from one WCS to another.

        Parameters
        ----------
        x : float or np.ndarray
            x-coordinate(s) to be reprojected.
        y : float or np.ndarray
            y-coordinate(s) to be reprojected.

        Returns
        -------
        tuple
            Tuple of np.ndarrays including reprojected x and y coordinates.
        """
        # example inputs to resulting function (12, 13, 0) # third number is origin
        # uses np.arrays for shape functionality
        if not isinstance(x, (np.ndarray)):
            x = np.array(x)
        if not isinstance(y, (np.ndarray)):
            y = np.array(y)
        if x.shape != y.shape:
            msg = "x and y must be the same length"
            raise ValueError(msg)
        sky = _get_forward_transform_func(wcs1)(x, y, 0)

        # rearrange into array including flattened x and y values
        flat_sky = [axis.flatten() for axis in sky]
        det = np.array(_get_backward_transform_func(wcs2)(flat_sky[0], flat_sky[1], 0))
        det_reshaped = [axis.reshape(x.shape) for axis in det]

        return tuple(det_reshaped)

    return _reproject
