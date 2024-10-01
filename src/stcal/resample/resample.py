import abc
from copy import deepcopy
import logging
import os
import sys
import warnings

import numpy as np
from scipy.ndimage import median_filter

from drizzle.resample import Drizzle
from drizzle.utils import calc_pixmap

import psutil
from spherical_geometry.polygon import SphericalPolygon
from astropy.nddata.bitmask import (
    bitfield_to_boolean_mask,
    interpret_bit_flags,
)

from .utils import bytes2human, get_tmeasure
from ..alignment.util import (
    compute_scale,
    wcs_bbox_from_shape,
    wcs_from_footprints,
)


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

__all__ = [
    "LibModelAccess",
    "OutputTooLargeError",
    "Resample",
    "resampled_wcs_from_models",
]

_SUPPORTED_CUSTOM_WCS_PARS = [
    'pixel_scale_ratio',
    'pixel_scale',
    'output_shape',
    'crpix',
    'crval',
    'rotation',
]


def _resample_range(data_shape, bbox=None):
    # Find range of input pixels to resample:
    if bbox is None:
        xmin = ymin = 0
        xmax = data_shape[1] - 1
        ymax = data_shape[0] - 1
    else:
        ((x1, x2), (y1, y2)) = bbox
        xmin = max(0, int(x1 + 0.5))
        ymin = max(0, int(y1 + 0.5))
        xmax = min(data_shape[1] - 1, int(x2 + 0.5))
        ymax = min(data_shape[0] - 1, int(y2 + 0.5))

    return xmin, xmax, ymin, ymax


class LibModelAccess(abc.ABC):
    # list of model attributes needed by this module. While this is not
    # required, it is helpful for subclasses to check they know how to
    # access these attributes.
    min_supported_attributes = [
        # arrays:
        "data",
        "dq",
        "var_rnoise",
        "var_poisson",
        "var_flat",

        # meta:
        "filename",
        "group_id",
        "s_region",
        "wcsinfo",
        "wcs",

        "exposure_time",
        "start_time",
        "end_time",
        "duration",
        "measurement_time",
        "effective_exposure_time",
        "elapsed_exposure_time",

        "pixelarea_steradians",
#        "pixelarea_arcsecsq",

        "level",  # sky level
        "subtracted",

        "weight_type",
        "pointings",
        "n_coadds",
    ]

    @abc.abstractmethod
    def iter_model(self, attributes=None):
        ...

    @property
    @abc.abstractmethod
    def n_models(self):
        ...

    @property
    @abc.abstractmethod
    def n_groups(self):
        ...


def resampled_wcs_from_models(
        input_models,
        pixel_scale_ratio=1.0,
        pixel_scale=None,
        output_shape=None,
        rotation=None,
        crpix=None,
        crval=None,
):
    """
    Computes the WCS of the resampled image from input models and
    specified WCS parameters.

    Parameters
    ----------

    input_models : LibModelAccess
        An object of `LibModelAccess`-derived type.

    pixel_scale_ratio : float, optional
        Desired pixel scale ratio defined as the ratio of the first model's
        pixel scale computed from this model's WCS at the fiducial point
        (taken as the ``ref_ra`` and ``ref_dec`` from the ``wcsinfo`` meta
        attribute of the first input image) to the desired output pixel
        scale. Ignored when ``pixel_scale`` is specified.

    pixel_scale : float, None, optional
        Desired pixel scale (in degrees) of the output WCS. When provided,
        overrides ``pixel_scale_ratio``.

    output_shape : tuple of two integers (int, int), None, optional
        Shape of the image (data array) using ``np.ndarray`` convention
        (``ny`` first and ``nx`` second). This value will be assigned to
        ``pixel_shape`` and ``array_shape`` properties of the returned
        WCS object.

    rotation : float, None, optional
        Position angle of output image's Y-axis relative to North.
        A value of 0.0 would orient the final output image to be North up.
        The default of `None` specifies that the images will not be rotated,
        but will instead be resampled in the default orientation for the
        camera with the x and y axes of the resampled image corresponding
        approximately to the detector axes. Ignored when ``transform`` is
        provided.

    crpix : tuple of float, None, optional
        Position of the reference pixel in the resampled image array.
        If ``crpix`` is not specified, it will be set to the center of the
        bounding box of the returned WCS object.

    crval : tuple of float, None, optional
        Right ascension and declination of the reference pixel.
        Automatically computed if not provided.

    Returns
    -------
    wcs : ~gwcs.wcs.WCS
        The WCS object corresponding to the combined input footprints.

    pscale_in : float
        Computed pixel scale (in degrees) of the first input image.

    pscale_out : float
        Computed pixel scale (in degrees) of the output image.

    """
    # build a list of WCS of all input models:
    wcs_list = []
    ref_wcsinfo = None
    for model_info, _ in input_models.iter_model(
                attributes=["data", "wcs", "wcsinfo"]
            ):
        # TODO: is deepcopy necessary? Is ModelLibrary read-only by default?
        w = deepcopy(model_info["wcs"])
        if ref_wcsinfo is None:
            ref_wcsinfo = model_info["wcsinfo"]
        # make sure all WCS objects have the bounding_box defined:
        if w.bounding_box is None:
            bbox = wcs_bbox_from_shape(model_info["data"].shape)
            w.bounding_box = bbox
        wcs_list.append(w)

    if output_shape is None:
        bounding_box = None
    else:
        bounding_box = wcs_bbox_from_shape(output_shape)

    pscale_in0 = compute_scale(
        wcs_list[0],
        fiducial=np.array([ref_wcsinfo["ra_ref"], ref_wcsinfo["dec_ref"]])
    )

    if pixel_scale is None:
        pixel_scale = pscale_in0 / pixel_scale_ratio
        log.info(
            f"Pixel scale ratio (pscale_in / pscale_out): {pixel_scale_ratio}"
        )
        log.info(f"Computed output pixel scale: {3600 * pixel_scale} arcsec.")
    else:
        pixel_scale_ratio = pscale_in0 / pixel_scale
        log.info(f"Output pixel scale: {3600 * pixel_scale} arcsec.")
        log.info(
            "Computed pixel scale ratio (pscale_in / pscale_out): "
            f"{pixel_scale_ratio}."
        )

    wcs = wcs_from_footprints(
        wcs_list=wcs_list,
        ref_wcs=wcs_list[0],
        ref_wcsinfo=ref_wcsinfo,
        pscale_ratio=pixel_scale_ratio,
        pscale=pixel_scale,
        rotation=rotation,
        bounding_box=bounding_box,
        shape=output_shape,
        crpix=crpix,
        crval=crval,
    )

    return wcs, pscale_in0, pixel_scale, pixel_scale_ratio


class OutputTooLargeError(RuntimeError):
    """Raised when the output is too large for in-memory instantiation"""


class Resample:
    """
    This is the controlling routine for the resampling process.

    Notes
    -----
    This routine performs the following operations::

      1. Extracts parameter settings from input model, such as pixfrac,
         weight type, exposure time (if relevant), and kernel, and merges
         them with any user-provided values.
      2. Creates output WCS based on input images and define mapping function
         between all input arrays and the output array.
      3. Updates output data model with output arrays from drizzle, including
         a record of metadata from all input models.
    """
    resample_suffix = 'i2d'
    resample_file_ext = '.fits'
    n_arrays_per_output = 2  # #flt-point arrays in the output (data, weight, var, err, etc.)

    # supported output arrays (subclasses can add more):
    output_array_types = {
        "data": np.float32,
        "wht": np.float32,
        "con": np.int32,
        "var_rnoise": np.float32,
        "var_flat": np.float32,
        "var_poisson": np.float32,
        "err": np.float32,
    }

    dq_flag_name_map = {}

    def __init__(self, input_models, pixfrac=1.0, kernel="square",
                 fillval=0.0, wht_type="ivm", good_bits=0,
                 output_wcs=None, wcs_pars=None, output_model=None,
                 accumulate=False, enable_ctx=True, enable_var=True,
                 allowed_memory=None):
        """
        Parameters
        ----------
        input_models : LibModelAccess
            A `LibModelAccess` object allowing iterating over all contained
            models of interest.

        kwargs : dict
            Other parameters.

            .. note::
                ``output_shape`` is in the ``x, y`` order.

        """
        # input models
        self._input_models = input_models

        self._output_model = None
        self._output_wcs = None
        self._enable_ctx = enable_ctx
        self._enable_var = enable_var
        self._accumulate = accumulate

        # these are attributes that are used only for information purpose
        # and are added to created the output_model only if they are not already
        # present there:
        self._pixel_scale_ratio = None
        self._output_pixel_scale = None

        # resample parameters
        self.pixfrac = pixfrac
        self.kernel = kernel
        self.fillval = fillval
        self.weight_type = wht_type
        self.good_bits = good_bits

        self._output_wcs = output_wcs

        self.input_file_names = []

        # check wcs_pars has supported keywords:
        if wcs_pars is None:
            wcs_pars = {}
        elif wcs_pars:
            unsup = []
            unsup = set(wcs_pars.keys()).difference(_SUPPORTED_CUSTOM_WCS_PARS)
            if unsup:
                raise KeyError(
                    "Unsupported custom WCS parameters: "
                    f"{','.join(map(repr, unsup))}."
                )

        # determine output WCS and set-up output model if needed:
        if output_model is None:
            if output_wcs is None:
                output_wcs, _, ps, ps_ratio = resampled_wcs_from_models(
                    input_models,
                    pixel_scale_ratio=wcs_pars.get("pixel_scale_ratio", 1.0),
                    pixel_scale=wcs_pars.get("pixel_scale"),
                    output_shape=wcs_pars.get("output_shape"),
                    rotation=wcs_pars.get("rotation"),
                    crpix=wcs_pars.get("crpix"),
                    crval=wcs_pars.get("crval"),
                )
                self._output_pixel_scale = ps  # degrees
                self._pixel_scale_ratio = ps_ratio
            else:
                self.check_output_wcs(output_wcs, wcs_pars)
                self._output_pixel_scale = np.rad2deg(
                    np.sqrt(_compute_image_pixel_area(output_wcs))
                )
                log.info(
                    "Computed output pixel scale: "
                    f"{3600 * self._output_pixel_scale} arcsec."
                )

            self._output_wcs = output_wcs

        else:
            self.validate_output_model(
                output_model=output_model,
                output_wcs=output_wcs,
                accumulate=accumulate,
                enable_ctx=enable_ctx,
                enable_var=enable_var,
            )
            self._output_model = output_model
            self._output_wcs = output_model["wcs"]
            if output_wcs:
                log.warning(
                    "'output_wcs' will be ignored. Using the 'wcs' supplied "
                    "by the 'output_model' instead."
                )
            self._output_pixel_scale = np.rad2deg(
                np.sqrt(_compute_image_pixel_area(output_wcs))
            )
            self._pixel_scale_ratio = output_model.get("wcs", None)
            log.info(
                "Computed output pixel scale: "
                f"{3600 * self._output_pixel_scale} arcsec."
            )

        self._output_array_shape = self._output_wcs.array_shape

        # Check that the output data shape has no zero length dimensions
        npix = np.prod(self._output_array_shape)
        if not npix:
            raise ValueError(
                f"Invalid output frame shape: {tuple(self._output_array_shape)}"
            )

        # set up output model (arrays, etc.)
        if self._output_model is None:
            self._output_model = self.create_output_model(
                allowed_memory=allowed_memory
            )

        self._group_ids = []

        log.info(f"Driz parameter kernel: {self.kernel}")
        log.info(f"Driz parameter pixfrac: {self.pixfrac}")
        log.info(f"Driz parameter fillval: {self.fillval}")
        log.info(f"Driz parameter weight_type: {self.weight_type}")

        log.debug(f"Output mosaic size: {self._output_wcs.pixel_shape}")

    def check_output_wcs(self, output_wcs, wcs_pars,
                         estimate_output_shape=True):
        """
        Check that provided WCS has expected properties and that its
        ``array_shape`` property is defined.

        """
        naxes = output_wcs.output_frame.naxes
        if naxes != 2:
            raise RuntimeError(
                "Output WCS needs 2 spatial axes but the "
                f"supplied WCS has {naxes} axes."
            )

        # make sure array_shape and pixel_shape are set:
        if output_wcs.array_shape is None and estimate_output_shape:
            if wcs_pars and "output_shape" in wcs_pars:
                output_wcs.array_shape = wcs_pars["output_shape"]
            else:
                if output_wcs.bounding_box:
                    halfpix = 0.5 + sys.float_info.epsilon
                    output_wcs.array_shape = (
                        int(output_wcs.bounding_box[1][1] + halfpix),
                        int(output_wcs.bounding_box[0][1] + halfpix),
                    )
                else:
                    # TODO: In principle, we could compute footprints of all
                    # input models, convert them to image coordinates using
                    # `output_wcs`, and then take max(x_i), max(y_i) as
                    # output image size.
                    raise ValueError(
                        "Unable to infer output image size from provided "
                        "inputs."
                    )

    @classmethod
    def output_model_attributes(cls, accumulate, enable_ctx, enable_var):
        """
        Returns a set of string keywords that must be present in an
        'output_model' that is provided as input at the class initialization.

        """
        # always required:
        attributes = {
            "data",
            "wcs",
            "wht",
        }

        if enable_ctx:
            attributes.add("con")
        if enable_var:
            attributes.update(
                ["var_rnoise", "var_poisson", "var_flat", "err"]
            )
        if accumulate:
            if enable_ctx:
                attributes.add("n_coadds")

            # additional attributes required for input parameter 'output_model'
            # when data and weight arrays are not None:
            attributes.update(
                {
                    "pixfrac",
                    "kernel",
                    "fillval",
                    "weight_type",
                    "pointings",
                    "exposure_time",
                    "measurement_time",
                    "start_time",
                    "end_time",
                    "duration",
                }
            )

        return attributes

    def validate_output_model(self, output_model, accumulate,
                              enable_ctx, enable_var):
        if output_model is None:
            if accumulate:
                raise ValueError(
                    "'output_model' must be defined when 'accumulate' is True."
                )
            return

        required_attributes = self.output_model_attributes(
            accumulate=accumulate,
            enable_ctx=enable_ctx,
            enable_var=enable_var,
        )

        for attr in required_attributes:
            if attr not in output_model:
                raise ValueError(
                    f"'output_model' dictionary must have '{attr}' set."
                )

        model_wcs = output_model["wcs"]
        self.check_output_wcs(model_wcs, estimate_output_shape=False)
        wcs_shape = model_wcs.array_shape
        ref_shape = output_model["data"].shape
        if accumulate and wcs_shape is None:
            raise ValueError(
                "Output model's 'wcs' must have 'array_shape' attribute "
                "set when 'accumulate' parameter is True."
            )

        if not np.array_equiv(wcs_shape, ref_shape):
            raise ValueError(
                "Output model's 'wcs.array_shape' value is not consistent "
                "with the shape of the data array."
            )

        for attr in required_attributes.difference(["data", "wcs"]):
            if (isinstance(output_model[attr], np.ndarray) and
                    not np.array_equiv(output_model[attr].shape, ref_shape)):
                raise ValueError(
                    "'output_wcs.array_shape' value is not consistent "
                    f"with the shape of the '{attr}' array."
                )

        # TODO: also check "pixfrac", "kernel", "fillval", "weight_type"
        # with initializer parameters. log a warning if different.

    def create_output_model(self, allowed_memory):
        """ Create a new "output model": a dictionary of data and meta fields.
        Check that there is enough memory to hold all arrays.
        """
        assert self._output_wcs is not None
        assert np.array_equiv(
            self._output_wcs.array_shape,
            self._output_array_shape
        )
        assert self._output_pixel_scale

        pix_area = self._output_pixel_scale**2

        output_model = {
            # WCS:
            "wcs": deepcopy(self._output_wcs),

            # main arrays:
            "data": None,
            "wht": None,
            "con": None,

            # resample parameters:
            "pixfrac": self.pixfrac,
            "kernel": self.kernel,
            "fillval": self.fillval,
            "weight_type": self.weight_type,

            # accumulate-specific:
            "n_coadds": 0,

            # pixel scale:
            "pixelarea_steradians": pix_area,
            "pixelarea_arcsecsq": pix_area * np.rad2deg(3600)**2,
            "pixel_scale_ratio": self._pixel_scale_ratio,

            # drizzle info:
            "pointings": 0,

            # exposure time:
            "exposure_time": 0.0,
            "measurement_time": None,
            "start_time": None,
            "end_time": None,
            "duration": 0.0,
        }

        if self._enable_var:
            output_model.update(
                {
                    "var_rnoise": None,
                    "var_flat": None,
                    "var_poisson": None,
                    "err": None,
                }
            )

        if allowed_memory:
            self.check_memory_requirements(list(output_model), allowed_memory)

        return output_model

    @property
    def output_model(self):
        return self._output_model

    @property
    def output_array_shape(self):
        return self._output_array_shape

    @property
    def group_ids(self):
        return self._group_ids

    def check_memory_requirements(self, arrays, allowed_memory):
        """ Called just before `create_output_model` returns to verify
        that there is enough memory to hold the output.

        """
        if allowed_memory is None and "DMODEL_ALLOWED_MEMORY" not in os.environ:
            return

        allowed_memory = float(allowed_memory)

        # get the available memory
        available_memory = (
            psutil.virtual_memory().available + psutil.swap_memory().total
        )

        # compute the output array size
        npix = npix = np.prod(self._output_array_shape)
        nmodels = len(self._input_models)
        nconpl = nmodels // 32 + (1 if nmodels % 32 else 0)  # #context planes
        required_memory = 0
        for arr in arrays:
            if arr in self.output_array_types:
                f = nconpl if arr == "con" else 1
                required_memory += f * self.output_array_types[arr].itemsize
        # add pixmap itemsize:
        required_memory += 2 * np.dtype(float).itemsize
        required_memory *= npix

        # compare used to available
        used_fraction = required_memory / available_memory
        if used_fraction > allowed_memory:
            raise OutputTooLargeError(
                f'Combined ImageModel size {self._output_wcs.array_shape} '
                f'requires {bytes2human(required_memory)}. '
                f'Model cannot be instantiated.'
            )

    def build_driz_weight(self, model_info, weight_type=None, good_bits=None):
        """Create a weight map for use by drizzle. """
        data = model_info["data"]
        dq = model_info["dq"]

        dqmask = bitfield_to_boolean_mask(
            dq,
            good_bits,
            good_mask_value=1,
            dtype=np.uint8,
            flag_name_map=self.dq_flag_name_map,
        )

        if weight_type and weight_type.startswith('ivm'):
            weight_type = weight_type.strip()
            selective_median = weight_type.startswith('ivm-smed')
            bitvalue = interpret_bit_flags(
                good_bits,
                flag_name_map=self.dq_flag_name_map
            )
            if bitvalue is None:
                bitvalue = 0

            # disable selective median if SATURATED flag is included
            # in good_bits:
            try:
                saturation = self.dq_flag_name_map["SATURATED"]
                if selective_median and not (bitvalue & saturation):
                    selective_median = False
                    weight_type = 'ivm'
            except AttributeError:
                pass

            var_rnoise = model_info["var_rnoise"]
            if (var_rnoise is not None and var_rnoise.shape == data.shape):
                with np.errstate(divide="ignore", invalid="ignore"):
                    inv_variance = var_rnoise**-1

                inv_variance[~np.isfinite(inv_variance)] = 1

                if weight_type != 'ivm':
                    ny, nx = data.shape

                    # apply a median filter to smooth the weight at saturated
                    # (or high read-out noise) single pixels. keep kernel size
                    # small to still give lower weight to extended CRs, etc.
                    ksz = weight_type[8 if selective_median else 7 :]
                    if ksz:
                        kernel_size = int(ksz)
                        if not (kernel_size % 2):
                            raise ValueError(
                                'Kernel size of the median filter in IVM '
                                'weighting must be an odd integer.'
                            )
                    else:
                        kernel_size = 3

                    ivm_copy = inv_variance.copy()

                    if selective_median:
                        # apply median filter selectively only at
                        # points of partially saturated sources:
                        jumps = np.where(
                            np.logical_and(dq & saturation, dqmask)
                        )
                        w2 = kernel_size // 2
                        for r, c in zip(*jumps):
                            x1 = max(0, c - w2)
                            x2 = min(nx, c + w2 + 1)
                            y1 = max(0, r - w2)
                            y2 = min(ny, r + w2 + 1)
                            data = ivm_copy[y1:y2, x1:x2][dqmask[y1:y2, x1:x2]]
                            if data.size:
                                inv_variance[r, c] = np.median(data)
                            # else: leave it as is

                    else:
                        # apply median to the entire inv-var array:
                        inv_variance = median_filter(
                            inv_variance,
                            size=kernel_size
                        )
                    bad_dqmask = np.logical_not(dqmask)
                    inv_variance[bad_dqmask] = ivm_copy[bad_dqmask]

            else:
                warnings.warn(
                    "var_rnoise array not available. "
                    "Setting drizzle weight map to 1",
                    RuntimeWarning
                )
                inv_variance = 1.0

            result = inv_variance * dqmask

        elif weight_type == "exptime":
            exptime = model_info["exposure_time"]
            result = exptime * dqmask

        else:
            result = np.ones(data.shape, dtype=data.dtype) * dqmask

        return result.astype(np.float32)

    def init_time_info(self):
        """ Initialize variables/arrays needed to process exposure time. """
        self._t_used_group_id = []

        self._total_exposure_time = self.output_model["exposure_time"]
        self._duration = self.output_model["duration"]
        self._total_measurement_time = self.output_model["measurement_time"]
        if self._total_measurement_time is None:
            self._total_measurement_time = 0.0

        if (start_time := self.output_model.get("start_time", None)) is None:
            self._exptime_start = []
        else:
            self._exptime_start[start_time]

        if (end_time := self.output_model.get("end_time", None)) is None:
            self._exptime_end = []
        else:
            self._exptime_end[end_time]

        self._measurement_time_success = []

    def update_total_time(self, model_info):
        """ A method called by the `~ResampleBase.run` method to process each
        image's time attributes.

        """
        if (group_id := model_info["group_id"]) in self._t_used_group_id:
            return
        self._t_used_group_id.append(group_id)

        self._exptime_start.append(model_info["start_time"])
        self._exptime_end.append(model_info["end_time"])

        t, success = get_tmeasure(model_info)
        self._total_exposure_time += model_info["exposure_time"]
        self._measurement_time_success.append(success)
        self._total_measurement_time += t

        self._duration += model_info["duration"]

    def finalize_time_info(self):
        """ Perform final computations for the total time and update relevant
        fileds of the output model.

        """
        attrs = {
            # basic exposure time attributes:
            "exposure_time": self._total_exposure_time,
            "start_time": min(self._exptime_start),
            "end_time": max(self._exptime_end),
            # Update other exposure time keywords:
            # XPOSURE (identical to the total effective exposure time, EFFEXPTM)
            "effective_exposure_time": self._total_exposure_time,
            # DURATION (identical to TELAPSE, elapsed time)
            "duration": self._duration,
            "elapsed_exposure_time": self._duration,
        }

        if all(self._measurement_time_success):
            attrs["measurement_time"] = self._total_measurement_time

        self._output_model.update(attrs)

    def init_resample_data(self):
        """ Create a `Drizzle` object to process image data. """
        om = self._output_model

        self.driz_data = Drizzle(
            kernel=self.kernel,
            fillval=self.fillval,
            out_shape=self._output_array_shape,
            out_img=om["data"],
            out_wht=om["wht"],
            out_ctx=om["con"],
            exptime=om["exposure_time"],
            begin_ctx_id=om["n_coadds"],
            max_ctx_id=om["n_coadds"] + self._input_models.n_models,
        )

    def init_resample_variance(self):
        """ Create a `Drizzle` objects to process image variance arrays. """
        self._var_rnoise_sum = np.full(self._output_array_shape, np.nan)
        self._var_poisson_sum = np.full(self._output_array_shape, np.nan)
        self._var_flat_sum = np.full(self._output_array_shape, np.nan)
        # self._total_weight_var_rnoise = np.zeros(self._output_array_shape)
        self._total_weight_var_poisson = np.zeros(self._output_array_shape)
        self._total_weight_var_flat = np.zeros(self._output_array_shape)

        # create resample objects for the three variance arrays:
        driz_init_kwargs = {
            'kernel': self.kernel,
            'fillval': np.nan,
            'out_shape': self._output_array_shape,
            # 'exptime': 1.0,
            'no_ctx': True,
        }
        self.driz_rnoise = Drizzle(**driz_init_kwargs)
        self.driz_poisson = Drizzle(**driz_init_kwargs)
        self.driz_flat = Drizzle(**driz_init_kwargs)

    def _check_var_array(self, model_info, array_name):
        """ Check that a variance array has the same shape as the model's
        data array.

        """
        array_data = model_info.get(array_name, None)
        sci_data = model_info["data"]
        model_name = _get_model_name(model_info)

        if array_data is None or array_data.size == 0:
            log.debug(
                f"No data for '{array_name}' for model "
                f"{repr(model_name)}. Skipping ..."
            )
            return False

        elif array_data.shape != sci_data.shape:
            log.warning(
                f"Data shape mismatch for '{array_name}' for model "
                f"{repr(model_name)}. Skipping ..."
            )
            return False

        return True

    def add_model(self, model_info, image_model):
        """ Resample and add data (variance, etc.) arrays to the output arrays.

        Parameters
        ----------

        model_info : dict
            A dictionary with data extracted from an image model needed for
            `Resample` to successfully process this model.

        image_model : object
            The original data model from which ``model`` data was extracted.
            It is not used by this method in this class but can be used
            by pipeline-specific subclasses to perform additional processing
            such as blend headers.

        """
        in_data = model_info["data"]

        if (group_id := model_info["group_id"]) not in self.group_ids:
            self.group_ids.append(group_id)
            self.output_model["pointings"] += 1

        self.input_file_names.append(model_info["filename"])

        # Check that input models are 2D images
        if in_data.ndim != 2:
            raise RuntimeError(
                f"Input model {_get_model_name(model_info)} "
                "is not a 2D image."
            )

        input_pixflux_area = model_info["pixelarea_steradians"]
        imwcs = model_info["wcs"]
        if (input_pixflux_area and
                'SPECTRAL' not in imwcs.output_frame.axes_type):
            if not np.array_equiv(imwcs.array_shape, in_data.shape):
                imwcs.array_shape = in_data.shape
            input_pixel_area = _compute_image_pixel_area(imwcs)
            if input_pixel_area is None:
                model_name = model_info["filename"]
                if not model_name:
                    model_name = "Unknown"
                raise ValueError(
                    "Unable to compute input pixel area from WCS of input "
                    f"image {repr(model_name)}."
                )
            iscale = np.sqrt(input_pixflux_area / input_pixel_area)
        else:
            iscale = 1.0

        # TODO: should weight_type=None here?
        in_wht = self.build_driz_weight(
            model_info,
            weight_type=self.weight_type,
            good_bits=self.good_bits
        )

        # apply sky subtraction
        blevel = model_info["level"]
        if not model_info["subtracted"] and blevel is not None:
            in_data = in_data - blevel

        xmin, xmax, ymin, ymax = _resample_range(
            in_data.shape,
            imwcs.bounding_box
        )

        pixmap = calc_pixmap(wcs_from=imwcs, wcs_to=self._output_wcs)

        add_image_kwargs = {
            'exptime': model_info["exposure_time"],
            'pixmap': pixmap,
            'scale': iscale,
            'weight_map': in_wht,
            'wht_scale': 1.0,
            'pixfrac': self.pixfrac,
            'in_units': 'cps',  # TODO: get units from data model
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
        }

        self.driz_data.add_image(in_data, **add_image_kwargs)

        if self._enable_var:
            self.resample_variance_data(model_info, add_image_kwargs)

    def run(self):
        """ Resample and coadd many inputs to a single output.

        1. Call methods that initialize data, variance, and time computations.
        2. Add input images (data, variances, etc) to output arrays.
        3. Perform final computations to compute variance and error
           arrays and total expose time information for the resampled image.

        """
        self.init_time_info()
        self.init_resample_data()
        if self._enable_var:
            self.init_resample_variance()

        for model_info, image_model in self._input_models.iter_model():
            self.add_model(model_info, image_model)
            self.update_total_time(model_info)

        # assign resampled arrays to the output model dictionary:
        self._output_model["data"] = self.driz_data.out_img.astype(
            dtype=self.output_array_types["data"]
        )
        self._output_model["wht"] = self.driz_data.out_wht.astype(
            dtype=self.output_array_types["wht"]
        )

        if self._enable_ctx:
            self._output_model["con"] = self.driz_data.out_ctx.astype(
                dtype=self.output_array_types["con"]
            )

        if self._enable_var:
            self.finalize_variance_processing()
            self.compute_errors()

        self.finalize_time_info()

    def resample_variance_data(self, data_model, add_image_kwargs):
        """ Resample and add input model's variance arrays to the output
        vararrays.

        """
        log.info("Resampling variance components")

        # Resample read-out noise and compute weight map for variance arrays
        if self._check_var_array(data_model, 'var_rnoise'):
            data = data_model["var_rnoise"]
            data = np.sqrt(data)

            # reset driz output arrays:
            self.driz_rnoise.out_img[:, :] = self.driz_rnoise.fillval
            self.driz_rnoise.out_wht[:, :] = 0.0

            self.driz_rnoise.add_image(data, **add_image_kwargs)
            var = self.driz_rnoise.out_img
            np.square(var, out=var)

            weight_mask = var > 0

            # Set the weight for the image from the weight type
            if self.weight_type == "ivm":
                weight_mask = var > 0
                weight = np.ones(self._output_array_shape)
                weight[weight_mask] = np.reciprocal(var[weight_mask])
                weight_mask &= (weight > 0.0)
                # Add the inverse of the resampled variance to a running sum.
                # Update only pixels (in the running sum) with valid new values:
                self._var_rnoise_sum[weight_mask] = np.nansum(
                    [
                        self._var_rnoise_sum[weight_mask],
                        weight[weight_mask]
                    ],
                    axis=0
                )
            elif self.weight_type == "exptime":
                weight = np.full(
                    self._output_array_shape,
                    get_tmeasure(data_model)[0],
                )
                weight_mask = np.ones(self._output_array_shape, dtype=bool)
                self._var_rnoise_sum = np.nansum(
                    [self._var_rnoise_sum, weight],
                    axis=0
                )
            else:
                weight = np.ones(self._output_array_shape)
                weight_mask = np.ones(self._output_array_shape, dtype=bool)
                self._var_rnoise_sum = np.nansum(
                    [self._var_rnoise_sum, weight],
                    axis=0
                )
        else:
            weight = np.ones(self._output_array_shape)
            weight_mask = np.ones(self._output_array_shape, dtype=bool)

        for var_name in ["var_flat", "var_poisson"]:
            if not self._check_var_array(data_model, var_name):
                continue
            data = data_model[var_name]
            data = np.sqrt(data)

            driz = getattr(self, var_name.replace("var", "driz"))
            var_sum = getattr(self, f"_{var_name}_sum")
            t_var_weight = getattr(self, f"_total_weight_{var_name}")

            # reset driz output arrays:
            driz.out_img[:, :] = driz.fillval
            driz.out_wht[:, :] = 0.0

            driz.add_image(data, **add_image_kwargs)
            var = driz.out_img
            np.square(var, out=var)

            mask = (var > 0) & weight_mask

            # Add the inverse of the resampled variance to a running sum.
            # Update only pixels (in the running sum) with valid new values:
            var_sum[mask] = np.nansum(
                [
                    var_sum[mask],
                    var[mask] * weight[mask] * weight[mask]
                ],
                axis=0
            )
            t_var_weight[mask] += weight[mask]

    def finalize_variance_processing(self):
        # We now have a sum of the weighted resampled variances.
        # Divide by the total weights, squared, and set in the output model.
        # Zero weight and missing values are NaN in the output.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value*", RuntimeWarning)
            warnings.filterwarnings("ignore", "divide by zero*", RuntimeWarning)

            # readout noise
            np.reciprocal(self._var_rnoise_sum, out=self._var_rnoise_sum)
            if self._accumulate and self._output_model["var_rnoise"]:
                self._output_model["var_rnoise"] += self._var_rnoise_sum.astype(
                    dtype=self.output_array_types["var_rnoise"]
                )
            else:
                self._output_model["var_rnoise"] = self._var_rnoise_sum.astype(
                    dtype=self.output_array_types["var_rnoise"]
                )

            # Poisson noise
            for _ in range(2):
                np.divide(
                    self._var_poisson_sum,
                    self._total_weight_var_poisson,
                    out=self._var_poisson_sum
                )

            if self._accumulate and self._output_model["var_poisson"]:
                self._output_model["var_poisson"] += self._var_rnoise_sum.astype(
                    dtype=self.output_array_types["var_poisson"]
                )
            else:
                self._output_model["var_poisson"] = self._var_rnoise_sum.astype(
                    dtype=self.output_array_types["var_poisson"]
                )

            # flat's noise
            for _ in range(2):
                np.divide(
                    self._var_flat_sum,
                    self._total_weight_var_flat,
                    out=self._var_flat_sum
                )

            if self._accumulate and self._output_model["var_flat"]:
                self._output_model["var_flat"] += self._var_rnoise_sum.astype(
                    dtype=self.output_array_types["var_flat"]
                )
            else:
                self._output_model["var_flat"] = self._var_rnoise_sum.astype(
                    dtype=self.output_array_types["var_flat"]
                )

            # free arrays:
            del self._var_rnoise_sum
            del self._var_poisson_sum
            del self._var_flat_sum
            # del self._total_weight_var_rnoise
            del self._total_weight_var_poisson
            del self._total_weight_var_flat

    def compute_errors(self):
        """ Computes total error of the resampled image. """
        vars = np.array(
            [
                self._output_model["var_rnoise"],
                self._output_model["var_poisson"],
                self._output_model["var_flat"],
            ]
        )
        all_nan_mask = np.any(np.isnan(vars), axis=0)

        err = np.sqrt(np.nansum(vars, axis=0)).astype(
            dtype=self.output_array_types["err"]
        )
        del vars
        err[all_nan_mask] = np.nan

        self._output_model["err"] = err


def _get_boundary_points(xmin, xmax, ymin, ymax, dx=None, dy=None, shrink=0):
    """
    xmin, xmax, ymin, ymax - integer coordinates of pixel boundaries
    step - distance between points along an edge
    shrink - number of pixels by which to reduce `shape`

    Returns a list of points and the area of the rectangle
    """
    nx = xmax - xmin + 1
    ny = ymax - ymin + 1

    if dx is None:
        dx = nx
    if dy is None:
        dy = ny

    if nx - 2 * shrink < 1 or ny - 2 * shrink < 1:
        raise ValueError("Image size is too small.")

    sx = max(1, int(np.ceil(nx / dx)))
    sy = max(1, int(np.ceil(ny / dy)))

    xmin += shrink
    xmax -= shrink
    ymin += shrink
    ymax -= shrink

    size = 2 * sx + 2 * sy
    x = np.empty(size)
    y = np.empty(size)

    b = np.s_[0:sx]  # bottom edge
    r = np.s_[sx:sx + sy]  # right edge
    t = np.s_[sx + sy:2 * sx + sy]  # top edge
    l = np.s_[2 * sx + sy:2 * sx + 2 * sy]  # left

    x[b] = np.linspace(xmin, xmax, sx, False)
    y[b] = ymin
    x[r] = xmax
    y[r] = np.linspace(ymin, ymax, sy, False)
    x[t] = np.linspace(xmax, xmin, sx, False)
    y[t] = ymax
    x[l] = xmin
    y[l] = np.linspace(ymax, ymin, sy, False)

    area = (xmax - xmin) * (ymax - ymin)
    center = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))

    return x, y, area, center, b, r, t, l


def _compute_image_pixel_area(wcs):
    """ Computes pixel area in steradians.
    """
    if wcs.array_shape is None:
        raise ValueError("WCS must have array_shape attribute set.")

    valid_polygon = False
    spatial_idx = np.where(np.array(wcs.output_frame.axes_type) == 'SPATIAL')[0]

    ny, nx = wcs.array_shape

    ((xmin, xmax), (ymin, ymax)) = wcs.bounding_box

    xmin = max(0, int(xmin + 0.5))
    xmax = min(nx - 1, int(xmax - 0.5))
    ymin = max(0, int(ymin + 0.5))
    ymax = min(ny - 1, int(ymax - 0.5))
    if xmin > xmax:
        (xmin, xmax) = (xmax, xmin)
    if ymin > ymax:
        (ymin, ymax) = (ymax, ymin)

    k = 0
    dxy = [1, -1, -1, 1]

    while xmin < xmax and ymin < ymax:
        try:
            x, y, image_area, center, b, r, t, l = _get_boundary_points(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                dx=min((xmax - xmin) // 4, 15),
                dy=min((ymax - ymin) // 4, 15)
            )
        except ValueError:
            return None

        world = wcs(x, y)
        ra = world[spatial_idx[0]]
        dec = world[spatial_idx[1]]

        limits = [ymin, xmax, ymax, xmin]

        for j in range(4):
            sl = [b, r, t, l][k]
            if not (np.all(np.isfinite(ra[sl])) and
                    np.all(np.isfinite(dec[sl]))):
                limits[k] += dxy[k]
                ymin, xmax, ymax, xmin = limits
                k = (k + 1) % 4
                break
            k = (k + 1) % 4
        else:
            valid_polygon = True
            break

        ymin, xmax, ymax, xmin = limits

    if not valid_polygon:
        return None

    world = wcs(*center)
    wcenter = (world[spatial_idx[0]], world[spatial_idx[1]])

    sky_area = SphericalPolygon.from_radec(ra, dec, center=wcenter).area()
    if sky_area > 2 * np.pi:
        log.warning(
            "Unexpectedly large computed sky area for an image. "
            "Setting area to: 4*Pi - area"
        )
        sky_area = 4 * np.pi - sky_area
    pix_area = sky_area / image_area

    return pix_area


def _get_model_name(model_info):
    model_name = model_info["filename"]
    if model_name is None or not model_name.strip():
        model_name = "Unknown"
    return model_name
