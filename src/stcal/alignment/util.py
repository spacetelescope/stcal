"""Common utility functions for datamodel alignment."""
from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Protocol

import gwcs
import numpy as np
from astropy import units as u
from astropy import wcs as fitswcs
from astropy.coordinates import SkyCoord
from astropy.modeling import models as astmodels
from astropy.utils.misc import isiterable
from gwcs.wcstools import wcs_from_fiducial

if TYPE_CHECKING:
    from asdf import AsdfFile

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


__all__ = [
    "compute_scale",
    "compute_fiducial",
    "calc_rotation_matrix",
    "wcs_from_footprints",
    "reproject",
]


class SupportsDataWithWcs(Protocol):
    _asdf: AsdfFile

    def to_flat_dict():
        ...


def _calculate_fiducial_from_spatial_footprint(
    spatial_footprint: np.ndarray,
) -> np.ndarray:
    """
    Calculates the fiducial coordinates from a given spatial footprint.

    Parameters
    ----------
    spatial_footprint : numpy.ndarray
        A 2xN array containing the world coordinates of the WCS footprint's
        bounding box, where N is the number of bounding box positions.

    Returns
    -------
    lon_fiducial, lat_fiducial : numpy.ndarray, numpy.ndarray
        The world coordinates of the fiducial point in the output coordinate frame.
    """
    lon, lat = spatial_footprint
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
    refmodel: SupportsDataWithWcs,
    ref_fiducial: np.array,
    pscale_ratio: int | None = None,
    pscale: float | None = None,
    rotation: float | None = None,
    transform: astmodels.Model | None = None,
) -> astmodels.Model:
    """
    Creates a transform from pixel to world coordinates based on a
    reference datamodel's WCS.

    Parameters
    ----------
    refmodel :
        The datamodel that should be used as reference for calculating the
        transform parameters.

    pscale_ratio : int, None
        Ratio of input to output pixel scale. This parameter is only used when
        ``pscale=None`` and, in that case, it is passed on to ``compute_scale``.

    pscale : float, None
        The plate scale. If `None`, the plate scale is calculated from the reference
        datamodel.

    rotation : float, None
        Position angle of output image's Y-axis relative to North.
        A value of 0.0 would orient the final output image to be North up.
        The default of `None` specifies that the images will not be rotated,
        but will instead be resampled in the default orientation for the camera
        with the x and y axes of the resampled image corresponding
        approximately to the detector axes. Ignored when ``transform`` is
        provided. If `None`, the rotation angle is extracted from the
        reference model's ``meta.wcsinfo.roll_ref``.

    ref_fiducial : numpy.array
        A two-elements array containing the world coordinates of the fiducial point.

    transform : ~astropy.modeling.Model
        A transform between frames.

    Returns
    -------
    transform : ~astropy.modeling.Model
        An :py:mod:`~astropy` model containing the transform between frames.
    """
    if transform is None:
        sky_axes = refmodel.meta.wcs._get_axes_indices().tolist()  # noqa: SLF001
        v3yangle = np.deg2rad(refmodel.meta.wcsinfo.v3yangle)
        vparity = refmodel.meta.wcsinfo.vparity
        if rotation is None:
            roll_ref = np.deg2rad(refmodel.meta.wcsinfo.roll_ref)
        else:
            roll_ref = np.deg2rad(rotation) + (vparity * v3yangle)

        # reshape the rotation matrix returned from calc_rotation_matrix
        # into the correct shape for constructing the transformation
        pc = np.reshape(calc_rotation_matrix(roll_ref, v3yangle, vparity=vparity), (2, 2))

        rotation = astmodels.AffineTransformation2D(pc, name="pc_rotation_matrix")
        transform = [rotation]
        if sky_axes:
            if not pscale:
                pscale = compute_scale(refmodel.meta.wcs, ref_fiducial, pscale_ratio=pscale_ratio)
            transform.append(astmodels.Scale(pscale, name="cdelt1") & astmodels.Scale(pscale, name="cdelt2"))

        if transform:
            transform = functools.reduce(lambda x, y: x | y, transform)

    return transform


def _get_axis_min_and_bounding_box(ref_model, wcs_list, ref_wcs):
    """
    Calculates axis minimum values and bounding box.

    Parameters
    ----------
    ref_model :
        The reference datamodel for which to determine the minimum axis values and
        bounding box.

    wcs_list : list
        The list of WCS objects.

    ref_wcs : ~gwcs.wcs.WCS
        The reference WCS object.

    Returns
    -------
    tuple
        A tuple containing two elements:
            1 - a :py:class:`numpy.ndarray` with the minimum value in each axis;
            2 - a tuple containing the bounding box region in the format
            ((x0_lower, x0_upper), (x1_lower, x1_upper)).
    """
    footprints = [w.footprint().T for w in wcs_list]
    domain_bounds = np.hstack([ref_wcs.backward_transform(*f) for f in footprints])
    axis_min_values = np.min(domain_bounds, axis=1)
    domain_bounds = (domain_bounds.T - axis_min_values).T

    output_bounding_box = []
    for axis in ref_model.meta.wcs.output_frame.axes_order:
        axis_min, axis_max = (
            domain_bounds[axis].min(),
            domain_bounds[axis].max(),
        )
        # populate output_bounding_box
        output_bounding_box.append((axis_min, axis_max))

    output_bounding_box = tuple(output_bounding_box)
    return (axis_min_values, output_bounding_box)


def _calculate_fiducial(wcs_list, bounding_box, crval=None):
    """
    Calculates the coordinates of the fiducial point and, if necessary, updates it with
    the values in CRVAL (the update is applied to spatial axes only).

    Parameters
    ----------
    wcs_list : list
        A list of WCS objects.

    bounding_box : tuple, or list, optional
        The bounding box over which the WCS is valid. It can be a either tuple of tuples
        or a list of lists of size 2 where each element represents a range of
        (low, high) values. The bounding_box is in the order of the axes, axes_order.
        For two inputs and axes_order(0, 1) the bounding box can be either
        ((xlow, xhigh), (ylow, yhigh)) or [[xlow, xhigh], [ylow, yhigh]].

    crval : list, optional
        A reference world coordinate associated with the reference pixel. If not `None`,
        then the fiducial coordinates of the spatial axes will be updated with the
        values from ``crval``.

    Returns
    -------
    fiducial : numpy.ndarray
        A two-elements array containing the world coordinate of the fiducial point.
    """
    fiducial = compute_fiducial(wcs_list, bounding_box=bounding_box)
    if crval is not None:
        i = 0
        for k, axt in enumerate(wcs_list[0].output_frame.axes_type):
            if axt == "SPATIAL":
                # overwrite only spatial axes with user-provided CRVAL
                fiducial[k] = crval[i]
                i += 1
    return fiducial


def _calculate_offsets(fiducial, wcs, axis_min_values, crpix):
    """
    Calculates the offsets to the transform.

    Parameters
    ----------
    fiducial : numpy.ndarray
        A two-elements containing the world coordinates of the fiducial point.

    wcs : ~gwcs.wcs.WCS
        A WCS object. It will be used to determine the

    axis_min_values : numpy.ndarray
        A two-elements array containing the minimum pixel value for each axis.

    crpix : list or tuple
        Pixel coordinates of the reference pixel.

    Returns
    -------
    ~astropy.modeling.Model
        A model with the offsets to be added to the WCS's transform.

    Notes
    -----
    If ``crpix=None``, then ``fiducial``, ``wcs``, and ``axis_min_values`` must be
    provided, in which case, the offsets will be calculated using the WCS object to
    find the pixel coordinates of the fiducial point and then correct it by the minimum
    pixel value for each axis.
    """
    if crpix is None and fiducial is not None and wcs is not None and axis_min_values is not None:
        offset1, offset2 = wcs.backward_transform(*fiducial)
        offset1 -= axis_min_values[0]
        offset2 -= axis_min_values[1]
    else:
        offset1, offset2 = crpix

    return astmodels.Shift(-offset1, name="crpix1") & astmodels.Shift(-offset2, name="crpix2")


def _calculate_new_wcs(ref_model, shape, wcs_list, fiducial, crpix=None, transform=None):
    """
    Calculates a new WCS object based on the combined WCS objects provided.

    Parameters
    ----------
    ref_model :
        The reference model to be used when extracting metadata.

    shape : list
        The shape of the new WCS's pixel grid. If `None`, then the output bounding box
        will be used to determine it.

    wcs_list : list
        A list containing WCS objects.

    fiducial : numpy.ndarray
        A two-elements array containing the location on the sky in some standard
        coordinate system.

    crpix : tuple, optional
        The coordinates of the reference pixel.

    transform : ~astropy.modeling.Model
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
        coordinate_frame=ref_model.meta.wcs.output_frame,
        projection=astmodels.Pix2Sky_TAN(),
        transform=transform,
        input_frame=ref_model.meta.wcs.input_frame,
    )
    axis_min_values, output_bounding_box = _get_axis_min_and_bounding_box(ref_model, wcs_list, wcs_new)
    offsets = _calculate_offsets(
        fiducial=fiducial,
        wcs=wcs_new,
        axis_min_values=axis_min_values,
        crpix=crpix,
    )

    wcs_new.insert_transform("detector", offsets, after=True)
    wcs_new.bounding_box = output_bounding_box

    if shape is None:
        shape = [int(axs[1] - axs[0] + 0.5) for axs in output_bounding_box[::-1]]

    wcs_new.pixel_shape = shape[::-1]
    wcs_new.array_shape = shape
    return wcs_new


def _validate_wcs_list(wcs_list):
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


def wcsinfo_from_model(input_model: SupportsDataWithWcs) -> dict[str, np.ndarray | str | bool]:
    """
    Creates a dict {wcs_keyword: array_of_values} pairs from a datamodel.

    Parameters
    ----------
    input_model :
        The input datamodel.

    Returns
    -------
    wcsinfo : dict
        A dict containing the WCS FITS keywords and corresponding values.

    """
    defaults = {
        "CRPIX": 0,
        "CRVAL": 0,
        "CDELT": 1.0,
        "CTYPE": "",
        "CUNIT": u.Unit(""),
    }
    wcsaxes = input_model.meta.wcsinfo.wcsaxes
    wcsinfo = {"WCSAXES": wcsaxes}
    for key in ["CRPIX", "CRVAL", "CDELT", "CTYPE", "CUNIT"]:
        val = []
        for ax in range(1, wcsaxes + 1):
            k = (key + f"{ax}").lower()
            v = getattr(input_model.meta.wcsinfo, k, defaults[key])
            val.append(v)
        wcsinfo[key] = np.array(val)

    pc = np.zeros((wcsaxes, wcsaxes), dtype=np.float32)
    for i in range(1, wcsaxes + 1):
        for j in range(1, wcsaxes + 1):
            pc[i - 1, j - 1] = getattr(input_model.meta.wcsinfo, f"pc{i}_{j}", 1)
    wcsinfo["PC"] = pc
    wcsinfo["RADESYS"] = input_model.meta.coordinates.reference_frame
    wcsinfo["has_cd"] = False
    return wcsinfo


def compute_scale(
    wcs: gwcs.WCS,
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

    disp_axis : int
        Dispersion axis integer. Assumes the same convention as
        ``wcsinfo.dispersion_direction``

    pscale_ratio : int
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

    crpix = np.array(wcs.invert(*fiducial))

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
        return yscale if disp_axis == 1 else xscale

    return np.sqrt(xscale * yscale)


def compute_fiducial(wcslist: list, bounding_box: tuple | list | None = None) -> np.ndarray:
    """
    Calculates the world coordinates of the fiducial point of a list of WCS objects.
    For a celestial footprint this is the center. For a spectral footprint, it is the
    beginning of its range.

    Parameters
    ----------
    wcslist : list
        A list containing all the WCS objects for which the fiducial is to be
        calculated.

    bounding_box : tuple, list, None
        The bounding box over which the WCS is valid. It can be a either tuple of tuples
        or a list of lists of size 2 where each element represents a range of
        (low, high) values. The bounding_box is in the order of the axes, axes_order.
        For two inputs and axes_order(0, 1) the bounding box can be either
        ((xlow, xhigh), (ylow, yhigh)) or [[xlow, xhigh], [ylow, yhigh]].

    Returns
    -------
    fiducial : numpy.ndarray
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
        fiducial[spatial_axes] = _calculate_fiducial_from_spatial_footprint(spatial_footprint)
    if spectral_footprint.any():
        fiducial[spectral_axes] = spectral_footprint.min()
    return fiducial


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
    dmodels,
    refmodel=None,
    transform=None,
    bounding_box=None,
    pscale_ratio=None,
    pscale=None,
    rotation=None,
    shape=None,
    crpix=None,
    crval=None,
):
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
    dmodels : list
        A list of valid datamodels.

    refmodel :
        A valid datamodel whose WCS is used as reference for the creation of the output
        coordinate frame, projection, and scaling and rotation transforms.
        If not supplied the first model in the list is used as ``refmodel``.

    transform : ~astropy.modeling.Model
        A transform, passed to :py:func:`gwcs.wcstools.wcs_from_fiducial`
        If not supplied `Scaling | Rotation` is computed from ``refmodel``.

    bounding_box : tuple
        Bounding_box of the new WCS.
        If not supplied it is computed from the bounding_box of all inputs.

    pscale_ratio : float, None
        Ratio of input to output pixel scale. Ignored when either
        ``transform`` or ``pscale`` are provided.

    pscale : float, None
        Absolute pixel scale in degrees. When provided, overrides
        ``pscale_ratio``. Ignored when ``transform`` is provided.

    rotation : float, None
        Position angle of output image's Y-axis relative to North.
        A value of 0.0 would orient the final output image to be North up.
        The default of `None` specifies that the images will not be rotated,
        but will instead be resampled in the default orientation for the camera
        with the x and y axes of the resampled image corresponding
        approximately to the detector axes. Ignored when ``transform`` is
        provided.

    shape : tuple of int, None
        Shape of the image (data array) using ``numpy.ndarray`` convention
        (``ny`` first and ``nx`` second). This value will be assigned to
        ``pixel_shape`` and ``array_shape`` properties of the returned
        WCS object.

    crpix : tuple of float, None
        Position of the reference pixel in the image array.  If ``crpix`` is not
        specified, it will be set to the center of the bounding box of the
        returned WCS object.

    crval : tuple of float, None
        Right ascension and declination of the reference pixel. Automatically
        computed if not provided.

    Returns
    -------
    wcs_new : ~gwcs.wcs.WCS
        The WCS object corresponding to the combined input footprints.

    """
    wcs_list = [im.meta.wcs for im in dmodels]

    _validate_wcs_list(wcs_list)

    fiducial = _calculate_fiducial(wcs_list=wcs_list, bounding_box=bounding_box, crval=crval)

    refmodel = dmodels[0] if refmodel is None else refmodel

    transform = _generate_tranform(
        refmodel=refmodel,
        pscale_ratio=pscale_ratio,
        pscale=pscale,
        rotation=rotation,
        ref_fiducial=np.array([refmodel.meta.wcsinfo.ra_ref, refmodel.meta.wcsinfo.dec_ref]),
        transform=transform,
    )

    return _calculate_new_wcs(
        ref_model=refmodel,
        shape=shape,
        crpix=crpix,
        wcs_list=wcs_list,
        fiducial=fiducial,
        transform=transform,
    )


def update_s_region_imaging(model, center=True):
    """
    Update the ``S_REGION`` keyword using ``WCS.footprint``.

    Parameters
    ----------
    model :
        The input datamodel.
    center : bool, optional
        Whether or not to use the center of the pixel as reference for the
        coordinates, by default True
    """
    bbox = model.meta.wcs.bounding_box

    if bbox is None:
        bbox = wcs_bbox_from_shape(model.data.shape)

    # footprint is an array of shape (2, 4) as we
    # are interested only in the footprint on the sky
    ### TODO: we shouldn't use center=True in the call below because we want to
    ### calculate the coordinates of the footprint based on the *bounding box*,
    ### which means we are interested in each pixel's vertice, not its center.
    ### By using center=True, a difference of 0.5 pixel should be accounted for
    ### when comparing the world coordinates of the bounding box and the footprint.
    footprint = model.meta.wcs.footprint(bbox, center=center, axis_type="spatial").T
    # take only imaging footprint
    footprint = footprint[:2, :]

    # Make sure RA values are all positive
    negative_ind = footprint[0] < 0
    if negative_ind.any():
        footprint[0][negative_ind] = 360 + footprint[0][negative_ind]

    footprint = footprint.T
    update_s_region_keyword(model, footprint)


def wcs_bbox_from_shape(shape):
    """Create a bounding box from the shape of the data.

    This is appropriate to attach to a wcs object

    Parameters
    ----------
    shape : tuple
        The shape attribute from a `numpy.ndarray` array

    Returns
    -------
    bbox : tuple
        Bounding box in x, y order.
    """
    return (-0.5, shape[-1] - 0.5), (-0.5, shape[-2] - 0.5)


def update_s_region_keyword(model, footprint):
    """Update the S_REGION keyword.

    Parameters
    ----------
    model :
        The input model
    footprint : numpy.array
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
    else:
        model.meta.wcsinfo.s_region = s_region
        log.info("Update S_REGION to %s", model.meta.wcsinfo.s_region)


def reproject(wcs1, wcs2):
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
