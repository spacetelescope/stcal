import logging
import math
import os
import warnings
import json
import abc
from copy import deepcopy
import sys

import numpy as np
from scipy.ndimage import median_filter
import psutil

from astropy import units as u
from astropy.nddata.bitmask import (
    bitfield_to_boolean_mask,
    interpret_bit_flags,
)
from drizzle.utils import calc_pixmap
from drizzle.resample import Drizzle
from stdatamodels.jwst.library.basic_utils import bytes2human


from stcal.resample.utils import (
    bytes2human,
    compute_wcs_pixel_area,
    get_tmeasure,
    resample_range,
)


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

__all__ = [
    "compute_wcs_pixel_area"
    "OutputTooLargeError",
    "Resample",
    "resampled_wcs_from_models",
    "UnsupportedWCSError",
]


class OutputTooLargeError(RuntimeError):
    """Raised when the output is too large for in-memory instantiation"""


class UnsupportedWCSError(RuntimeError):
    """ Raised when provided output WCS has an unexpected number of axes
    or has an unsupported structure.
    """


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

    def __init__(self, n_input_models=None, pixfrac=1.0, kernel="square",
                 fillval=0.0, wht_type="ivm", good_bits=0,
                 output_wcs=None, output_model=None,
                 accumulate=False, enable_ctx=True, enable_var=True,
                 compute_err=None,
                 allowed_memory=None):
        """
        Parameters
        ----------
        n_input_models : int, None, optional
            Number of input models expected to be resampled. When provided,
            this is used to estimate memory requirements and optimize memory
            allocation for the context array.

        pixfrac : float, optional
            The fraction of a pixel that the pixel flux is confined to. The
            default value of 1 has the pixel flux evenly spread across the
            image. A value of 0.5 confines it to half a pixel in the linear
            dimension, so the flux is confined to a quarter of the pixel area
            when the square kernel is used.

        kernel: {"square", "gaussian", "point", "turbo", "lanczos2", "lanczos3"}, optional
            The name of the kernel used to combine the input. The choice of
            kernel controls the distribution of flux over the kernel.
            The square kernel is the default.

            .. warning::
               The "gaussian" and "lanczos2/3" kernels **DO NOT**
               conserve flux.

        fillval: float, None, str, optional
            The value of output pixels that did not have contributions from
            input images' pixels. When ``fillval`` is either `None` or
            ``"INDEF"`` and ``out_img`` is provided, the values of ``out_img``
            will not be modified. When ``fillval`` is either `None` or
            ``"INDEF"`` and ``out_img`` is **not provided**, the values of
            ``out_img`` will be initialized to `numpy.nan`. If ``fillval``
            is a string that can be converted to a number, then the output
            pixels with no contributions from input images will be set to this
            ``fillval`` value.

        wht_type : {"exptime", "ivm"}, optional
            The weighting type for adding models' data. For ``wht_type="ivm"``
            (the default), the weighting will be determined per-pixel using
            the inverse of the read noise (VAR_RNOISE) array stored in each
            input image. If the ``VAR_RNOISE`` array does not exist,
            the variance is set to 1 for all pixels (i.e., equal weighting).
            If ``weight_type="exptime"``, the weight will be set equal
            to the measurement time (``TMEASURE``) when available and to
            the exposure time (``EFFEXPTM``) otherwise.

        good_bits : int, str, None, optional
            An integer bit mask, `None`, a Python list of bit flags, a comma-,
            or ``'|'``-separated, ``'+'``-separated string list of integer
            bit flags or mnemonic flag names that indicate what bits in models'
            DQ bitfield array should be *ignored* (i.e., zeroed).

            When co-adding models using :py:meth:`add_model`, any pixels with
            a non-zero DQ values are assigned a weight of zero and therefore
            they do not contribute to the output (resampled) data.
            ``good_bits`` provides a mean to ignore some of the DQ bitflags.

            When ``good_bits`` is an integer, it must be
            the sum of all the DQ bit values from the input model's
            DQ array that should be considered "good" (or ignored). For
            example, if pixels in the DQ array can be
            combinations of 1, 2, 4, and 8 flags and one wants to consider DQ
            "defects" having flags 2 and 4 as being acceptable, then
            ``good_bits`` should be set to 2+4=6. Then a pixel with DQ values
            2,4, or 6 will be considered a good pixel, while a pixel with
            DQ value, e.g., 1+2=3, 4+8=12, etc. will be flagged as
            a "bad" pixel.

            Alternatively, when ``good_bits`` is a string, it can be a
            comma-separated or '+' separated list of integer bit flags that
            should be summed to obtain the final "good" bits. For example,
            both "4,8" and "4+8" are equivalent to integer ``good_bits=12``.

            Finally, instead of integers, ``good_bits`` can be a string of
            comma-separated mnemonics. For example, for JWST, all the following
            specifications are equivalent:

            `"12" == "4+8" == "4, 8" == "JUMP_DET, DROPOUT"`

            In order to "translate" mnemonic code to integer bit flags,
            ``Resample.dq_flag_name_map`` attribute must be set to either
            a dictionary (with keys being mnemonc codes and the values being
            integer flags) or a `~astropy.nddata.BitFlagNameMap`.

            In order to reverse the meaning of the flags
            from indicating values of the "good" DQ flags
            to indicating the "bad" DQ flags, prepend '~' to the string
            value. For example, in order to exclude pixels with
            DQ flags 4 and 8 for computations and to consider
            as "good" all other pixels (regardless of their DQ flag),
            use a value of ``~4+8``, or ``~4,8``. A string value of
            ``~0`` would be equivalent to a setting of ``None``.

            | Default value (0) will make *all* pixels with non-zero DQ
            values be considered "bad" pixels, and the corresponding data
            pixels will be assigned zero weight and thus these pixels
            will not contribute to the output resampled data array.

            | Set `good_bits` to `None` to turn off the use of model's DQ
            array.

            For more details, see documentation for
            `astropy.nddata.bitmask.extend_bit_flag_map`.

        output_wcs : dict, WCS object, None
            Specifies output WCS either directly as a WCS or a dictionary
            with keys ``'wcs'`` (WCS object) and ``'pixel_scale'``
            (pixel scale in arcseconds). ``'pixel_scale'``, when provided,
            will be used for computation of drizzle scaling factor. When it is
            not provided, output pixel scale will be *estimated* from the
            provided WCS object. ``output_wcs`` object is required when
            ``output_model`` is `None`. ``output_wcs`` is ignored when
            ``output_model`` is provided.

        output_model : dict, None, optional
            A dictionary containing data arrays and other attributes that
            will be used to add new models to. use
            :py:meth:`Resample.output_model_attributes` to get the list of
            keywords that must be present. When ``accumulate`` is `False`,
            only the WCS object of the model will be used. When ``accumulate``
            is `True`, new models will be added to the existing data in the
            ``output_model``.

            When ``output_model`` is `None`, a new model will be created.

        accumulate : bool, optional
            Indicates whether resampled models should be added to the
            provided ``output_model`` data or if new arrays should be
            created.

        enable_ctx : bool, optional
            Indicates whether to create a context image. If ``disable_ctx``
            is set to `True`, parameters ``out_ctx``, ``begin_ctx_id``, and
            ``max_ctx_id`` will be ignored.

        enable_var : bool, optional
            Indicates whether to resample variance arrays.

        compute_err : {"from_var", "driz_err"}, None, optional
            - ``"from_var"``: compute output model's error array from
              all (Poisson, flat, readout) resampled variance arrays.
              Setting ``compute_err`` to ``"from_var"`` will assume
              ``enable_var`` was set to `True` regardless of actual
              value of the parameter ``enable_var``.
            - ``"driz_err"``: compute output model's error array by drizzling
              together all input models' error arrays.

            Error array will be assigned to ``'err'`` key of the output model.

            .. note::
                At this time, output error array is not equivalent to
                error propagation results.

        allowed_memory : float, None
            Fraction of memory allowed to be used for resampling. If
            ``allowed_memory`` is `None` then no check for available memory
            will be performed.

        """
        # to see if setting up arrays and drizzle is needed
        self._finalized = False
        self._n_res_models = 0

        self._n_predicted_input_models = n_input_models
        self.allowed_memory = allowed_memory
        self._output_model = output_model
        self._create_new_output_model = output_model is not None

        self._enable_ctx = enable_ctx
        self._enable_var = enable_var
        self._compute_err = compute_err
        self._accumulate = accumulate

        # these are attributes that are used only for information purpose
        # and are added to created the output_model only if they are
        # not already present there:
        self._pixel_scale_ratio = None
        self._output_pixel_scale = None  # in arcsec

        # resample parameters
        self.pixfrac = pixfrac
        self.kernel = kernel
        self.fillval = fillval
        self.good_bits = good_bits

        if wht_type in ["ivm", "exptime"]:
            self.weight_type = wht_type
        else:
            raise ValueError("Unexpected weight type: '{self.weight_type}'")

        self._output_wcs = output_wcs

        self.input_file_names = []
        self._group_ids = []

        # determine output WCS and set-up output model if needed:
        if output_model is None:
            if output_wcs is None:
                raise ValueError(
                    "Output WCS must be provided either through the "
                    "'output_wcs' parameter or the 'ouput_model' parameter. "
                )
            else:
                if isinstance(output_wcs, dict):
                    self._output_pixel_scale = output_wcs.get("pixel_scale")
                    self._pixel_scale_ratio = output_wcs.get(
                        "pixel_scale_ratio"
                    )
                    self._output_wcs = output_wcs.get("wcs")
                else:
                    self._output_wcs = output_wcs

                self.check_output_wcs(self._output_wcs)

        else:
            self.validate_output_model(
                output_model=output_model,
                accumulate=accumulate,
                enable_ctx=enable_ctx,
                enable_var=enable_var,
            )
            self._output_model = output_model
            self._output_wcs = output_model["wcs"]
            self._output_pixel_scale = output_model.get("pixel_scale")
            if output_wcs:
                log.warning(
                    "'output_wcs' will be ignored. Using the 'wcs' supplied "
                    "by the 'output_model' instead."
                )

        if self._output_pixel_scale is None:
            self._output_pixel_scale = 3600.0 * np.rad2deg(
                math.sqrt(compute_wcs_pixel_area(self._output_wcs))
            )
            log.info(
                "Computed output pixel scale: "
                f"{self._output_pixel_scale} arcsec."
            )
        else:
            log.info(
                f"Output pixel scale: {self._output_pixel_scale} arcsec."
            )

        self._output_array_shape = self._output_wcs.array_shape

        # Check that the output data shape has no zero length dimensions
        npix = np.prod(self._output_array_shape)
        if not npix:
            raise ValueError(
                "Invalid output frame shape: "
                f"{tuple(self._output_array_shape)}"
            )

        log.info(f"Driz parameter kernel: {self.kernel}")
        log.info(f"Driz parameter pixfrac: {self.pixfrac}")
        log.info(f"Driz parameter fillval: {self.fillval}")
        log.info(f"Driz parameter weight_type: {self.weight_type}")
        log.debug(
            f"Output mosaic size (nx, ny): {self._output_wcs.pixel_shape}"
        )

        # set up an empty (don't allocate arrays at this time) output model:
        if self._output_model is None:
            self._output_model = self.create_output_model()

        self.reset_arrays(reset_output=False, n_input_models=n_input_models)

    @classmethod
    def output_model_attributes(cls, accumulate, enable_ctx, enable_var,
                                compute_err):
        """
        Returns a set of string keywords that must be present in an
        'output_model' that is provided as input at the class initialization.

        Parameters
        ----------

        accumulate : bool, optional
            Indicates whether resampled models should be added to the
            provided ``output_model`` data or if new arrays should be
            created.

        enable_ctx : bool, optional
            Indicates whether to create a context image. If ``disable_ctx``
            is set to `True`, parameters ``out_ctx``, ``begin_ctx_id``, and
            ``max_ctx_id`` will be ignored.

        enable_var : bool, optional
            Indicates whether to resample variance arrays.

        compute_err : {"from_var", "driz_err"}, None, optional
            - ``"from_var"``: compute output model's error array from
              all (Poisson, flat, readout) resampled variance arrays.
              Setting ``compute_err`` to ``"from_var"`` will assume
              ``enable_var`` was set to `True` regardless of actual
              value of the parameter ``enable_var``.
            - ``"driz_err"``: compute output model's error array by drizzling
              together all input models' error arrays.

            Error array will be assigned to ``'err'`` key of the output model.

            .. note::
                At this time, output error array is not equivalent to
                error propagation results.

        """
        # always required:
        attributes = {
            "data",
            "wcs",
            "wht",
        }

        if enable_ctx:
            attributes.add("con")
        if compute_err:
            attributes.add("err")
        if enable_var:
            attributes.update(
                ["var_rnoise", "var_poisson", "var_flat"]
            )
            # TODO: if we want to support adding more data to
            # existing output models, we need to also store weights
            # for variance arrays:
            # var_rnoise_weight
            # var_flat_weight
            # var_poisson_weight
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

    def check_memory_requirements(self, output_model, allowed_memory,
                                  n_input_models=None):
        """ Called just before `create_output_model` returns to verify
        that there is enough memory to hold the output.

        Parameters
        ----------
        allowed_memory : float, None
            Fraction of memory allowed to be used for resampling. If

        output_model : dict, None, optional
            A dictionary containing data arrays and other attributes that
            will be used to add new models to. use
            :py:meth:`Resample.output_model_attributes` to get the list of
            keywords that must be present. When ``accumulate`` is `False`,
            only the WCS object of the model will be used. When ``accumulate``
            is `True`, new models will be added to the existing data in the
            ``output_model``.

            When ``output_model`` is `None`, a new model will be created.

        n_input_models : int, None, optional
            Number of input models expected to be resampled. When provided,
            this is used to estimate memory requirements and optimize memory
            allocation for the context array.


        """
        if ((allowed_memory is None and
                "DMODEL_ALLOWED_MEMORY" not in os.environ) or
                n_input_models is None):
            return

        allowed_memory = float(allowed_memory)

        # get the available memory
        available_memory = (
            psutil.virtual_memory().available + psutil.swap_memory().total
        )

        # compute the output array size
        npix = np.prod(self._output_array_shape)
        nconpl = n_input_models // 32 + (1 if n_input_models % 32 else 0)  # context planes
        required_memory = 0
        for arr in self.output_array_types:
            if arr in output_model:
                if arr == "con":
                    f = nconpl
                elif arr == "err":
                    if self._compute_err == "from_var":
                        f = 2  # data and weight arrays
                    elif self._compute_err == "driz_err":
                        f = 1
                elif arr.startswith("var"):
                    f = 3  # variance data, weight, and total arrays
                else:
                    f = 1

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

    def check_output_wcs(self, output_wcs, estimate_output_shape=True):
        """
        Check that provided WCS has expected properties and that its
        ``array_shape`` property is defined.

        Parameters
        ----------
        output_wcs : WCS object
            A WCS object corresponding to the output (resampled) image.

        estimate_output_shape : bool, optional
            Indicates whether to *estimate* pixel scale of the ``output_wcs``
            from

        """
        naxes = output_wcs.output_frame.naxes
        if naxes != 2:
            raise UnsupportedWCSError(
                "Output WCS needs 2 coordinate axes but the "
                f"supplied WCS has {naxes} axes."
            )

        # make sure array_shape and pixel_shape are set:
        if output_wcs.array_shape is None and estimate_output_shape:
            # if wcs_pars and "output_shape" in wcs_pars:
            #     output_wcs.array_shape = wcs_pars["output_shape"]
            # else:
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

    def validate_output_model(self, output_model, accumulate,
                              enable_ctx, enable_var):
        """ Checks that ``output_model`` dictionary has all the required
        keywords that the code would expect it to have based on the values
        of ``accumulate``, ``enable_ctx``, ``enable_var``. It will raise
        `ValueError` if `output_model` is missing required keywords/values.

        """
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

    def create_output_model(self):
        """ Create a new "output model": a dictionary of data and meta fields.
        Check that there is enough memory to hold all arrays by calling
        `check_memory_requirements`.

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
            "wcs": self._output_wcs,

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
            "pixelarea_steradians": pix_area / np.rad2deg(3600)**2,
            "pixelarea_arcsecsq": pix_area,
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
                    # TODO: if we want to support adding more data to
                    # existing output models, we need to also store weights
                    # for variance arrays:
                    # var_rnoise_weight
                    # var_flat_weight
                    # var_poisson_weight
                }
            )

        if self._compute_err is not None:
            output_model["err"] = None

        if self.allowed_memory:
            self.check_memory_requirements(
                output_model,
                self.allowed_memory,
                n_input_models=self._n_predicted_input_models,
            )

        return output_model

    @property
    def output_model(self):
        return self._output_model

    @property
    def output_array_shape(self):
        return self._output_array_shape

    @property
    def output_wcs(self):
        return self._output_wcs

    @property
    def group_ids(self):
        return self._group_ids

    def _get_intensity_scale(self, model):
        """
        Compute an intensity scale from the input and output pixel area.

        For imaging data, the scaling is used to account for differences
        between the nominal pixel area and the average pixel area for
        the input data.

        For spectral data, the scaling is used to account for flux
        conservation with non-unity pixel scale ratios, when the
        data units are flux density.

        Parameters
        ----------
        model : dict
            The input data model.

        Returns
        -------
        iscale : float
            The scale to apply to the input data before drizzling.

        """
        input_pixflux_area = model["pixelarea_steradians"]
        wcs = model["wcs"]

        if input_pixflux_area:
            if 'SPECTRAL' in wcs.output_frame.axes_type:
                # Use the nominal area as is
                input_pixel_area = input_pixflux_area

                # If input image is in flux density units, correct the
                # flux for the user-specified change to the spatial dimension
                if _is_flux_density(model["bunit_data"]):
                    input_pixel_area *= self.pscale_ratio
            else:
                input_pixel_area = compute_wcs_pixel_area(
                    wcs,
                    shape=model["data"].shape
                )
                if input_pixel_area is None:
                    model_name = model["filename"]
                    if not model_name:
                        model_name = "Unknown"
                    raise ValueError(
                        "Unable to compute input pixel area from WCS of input "
                        f"image {repr(model_name)}."
                    )

                if self._pixel_scale_ratio is None:
                    input_pscale = 3600.0 * np.rad2deg(
                        math.sqrt(input_pixel_area)
                    )

                    self._pixel_scale_ratio = (
                        self._output_pixel_scale / input_pscale
                    )

                    # update output model if "pixel_scale_ratio" was never
                    # set previously:
                    if (self._output_model is not None and
                            self._output_model["pixel_scale_ratio"] is None):
                        self._output_model["pixel_scale_ratio"] = self._pixel_scale_ratio

            iscale = math.sqrt(input_pixflux_area / input_pixel_area)

        else:
            iscale = 1.0

        return iscale

    @property
    def finalized(self):
        return self._finalized

    def reset_arrays(self, reset_output=True, n_input_models=None):
        """ Initialize/reset `Drizzle` objects, output model and arrays,
        and time counters. Output WCS and shape are not modified from
        `Resample` object initialization. This method needs to be called
        before calling :py:meth:`add_model` for the first time if
        :py:meth:`finalize` was previously called.

        Parameters
        ----------
        reset_output : bool, optional
            When `True` a new output model will be created. Otherwise new
            models will be resampled and added to existing output data arrays.

        n_input_models : int, None, optional
            Number of input models expected to be resampled. When provided,
            this is used to estimate memory requirements and optimize memory
            allocation for the context array.

        """
        self._n_predicted_input_models = n_input_models

        # set up an empty (don't allocate arrays at this time) output model:
        if reset_output or getattr(self, "_output_model", None) is None:
            self._output_model = self.create_output_model()

        om = self._output_model

        begin_ctx_id = om.get("n_coadds", 0)
        if n_input_models is None:
            max_ctx_id = None
        else:
            max_ctx_id = begin_ctx_id + n_input_models - 1

        self._driz = Drizzle(
            kernel=self.kernel,
            fillval=self.fillval,
            out_shape=self._output_array_shape,
            out_img=om["data"],
            out_wht=om["wht"],
            out_ctx=om["con"],
            exptime=om["exposure_time"],
            begin_ctx_id=begin_ctx_id,
            max_ctx_id=max_ctx_id,
        )

        # Also make a temporary model to hold error data
        if self._compute_err == "driz_err":
            self._driz_error = Drizzle(
                kernel=self.kernel,
                fillval=self.fillval,
                out_shape=self._output_array_shape,
                out_img=om["err"],
                exptime=om["exposure_time"],
                disable_ctx=True,
            )

        if self._enable_var:
            self.init_variance_arrays()

        self.init_time_counters()

        self._finalized = False

    def validate_input_model(self, model):
        """ Checks that ``model`` has all the required keywords needed for
        processing based on settings used during initialisation if the
        `Resample` object.

        Parameters
        ----------
        model : dict
            A dictionary containing data arrays and other meta attributes
            and values of actual models used by pipelines.

        Raises
        ------
        KeyError
            A `KeyError` is raised when ``model`` does not have a required
            keyword.

        """
        # TODO: do we need this to just raise a custom
        assert isinstance(model, dict)
        min_attributes = [
            # arrays:
            "data",
            "dq",

            # meta:
            "group_id",
            "s_region",
            "wcs",

            "exposure_time",
            "start_time",
            "end_time",
            "duration",
            "measurement_time",
            "effective_exposure_time",
            "elapsed_exposure_time",

            "pixelarea_steradians",
            # "pixelarea_arcsecsq",

            "level",  # sky level
            "subtracted",
        ]

        if self._enable_var:
            min_attributes += ["var_rnoise", "var_poisson", "var_flat"]

        if self._compute_err == "driz_err":
            min_attributes.append("err")

        if (not self._enable_var and self.weight_type is not None and
                self.weight_type.startswith('ivm')):
            min_attributes.append("var_rnoise")

        for attr in min_attributes:
            if attr not in model:
                raise KeyError(
                    f"Attempt to access non-existent key '{attr}' "
                    "in a data model."
                )

    def add_model(self, model):
        """ Resamples model image and either variance data (if ``enable_var``
        was `True`) or error data (if ``enable_err`` was `True`) and adds
        them using appropriate weighting to the corresponding
        arrays of the output model. It also updates resampled data weight,
        the context array (if ``enable_ctx`` is `True`), relevant output
        model's values such as "n_coadds".

        Whenever ``model`` has a unique group ID that was never processed
        before, the "pointings" value of the output model is incremented and
        the "group_id" attribute is updated. Also, time counters are updated
        with new values from the input ``model`` by calling
        :py:meth:`~Resample.update_time`.

        Parameters
        ----------
        model : dict
            A dictionary containing data arrays and other meta attributes
            and values of actual models used by pipelines.

        """
        if self._finalized:
            raise RuntimeError(
                "Resampling has been finalized and intermediate arrays have "
                "been freed. Unable to add new models. Call 'reset_arrays' "
                "to initialize a new output model and associated arrays."
            )
        self.validate_input_model(model)
        self._n_res_models += 1

        data = model["data"]
        wcs = model["wcs"]

        # Check that input models are 2D images
        if data.ndim != 2:
            raise RuntimeError(
                f"Input model '{model['filename']}' is not a 2D image."
            )

        self._output_model["n_coadds"] += 1

        if (group_id := model["group_id"]) not in self._group_ids:
            self.update_time(model)
            self._group_ids.append(group_id)
            self.output_model["pointings"] += 1

        iscale = self._get_intensity_scale(model)
        log.debug(f'Using intensity scale iscale={iscale}')

        pixmap = calc_pixmap(
            wcs,
            self.output_model["wcs"],
            data.shape,
        )

        log.info("Resampling science and variance data")

        weight = self.build_driz_weight(
            model,
            weight_type=self.weight_type,
            good_bits=self.good_bits,
        )

        # apply sky subtraction
        blevel = model["level"]
        if not model["subtracted"] and blevel is not None:
            data = data - blevel
            # self._output_model["subtracted"] = True

        xmin, xmax, ymin, ymax = resample_range(
            data.shape,
            wcs.bounding_box
        )

        add_image_kwargs = {
            'exptime': model["exposure_time"],
            'pixmap': pixmap,
            'scale': iscale,
            'weight_map': weight,
            'wht_scale': 1.0,  # hard-coded for JWST count-rate data
            'pixfrac': self.pixfrac,
            'in_units': 'cps',  # TODO: get units from data model
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
        }

        self._driz.add_image(data, **add_image_kwargs)

        if self._compute_err == "driz_err":
            self._driz_error.add_image(model["err"], **add_image_kwargs)

        if self._enable_var:
            self.resample_variance_arrays(model, pixmap, iscale, weight,
                                          xmin, xmax, ymin, ymax)

        # update output model (variance is too expensive so it's omitted)
        self._output_model["data"] = self._driz.out_img
        self._output_model["wht"] = self._driz.out_wht
        if self._driz.out_ctx is not None:
            self._output_model["con"] = self._driz.out_ctx

        if self._compute_err == "driz_err":
            # use resampled error
            self.output_model["err"] = self._driz_error.out_img

    def finalize(self, free_memory=True):
        """ Finalizes all computations and frees temporary objects.

        ``finalize`` calls :py:meth:`~Resample.finalize_resample_variance` and
        :py:meth:`~Resample.finalize_time_info`.

        .. warning::
          If ``enable_var=True`` and :py:meth:`~Resample.finalize` is called
          with ``free_memory=True`` then intermediate arrays holding variance
          weights will be lost and so continuing adding new models after
          a call to :py:meth:`~Resample.finalize` will result in incorrect
          variance.

        """
        if self._finalized:
            # can't finalize twice
            return
        self._finalized = free_memory

        self._output_model["pointings"] = len(self.group_ids)

        # assign resampled arrays to the output model dictionary:
        self._output_model["data"] = self._driz.out_img
        self._output_model["wht"] = self._driz.out_wht
        if self._driz.out_ctx is not None:
            # Since the context array is dynamic, it must be re-assigned
            # back to the product's `con` attribute.
            self._output_model["con"] = self._driz.out_ctx

        if free_memory:
            del self._driz

        # compute final variances:
        if self._enable_var:
            self.finalize_resample_variance(
                self._output_model,
                free_memory=free_memory
            )

        if self._compute_err == "driz_err":
            # use resampled error
            self.output_model["err"] = self._driz_error.out_img
            if free_memory:
                del self._driz_error

        elif self._enable_var:
            # compute error from variance arrays:
            var_components = [
                self._output_model["var_rnoise"],
                self._output_model["var_poisson"],
                self._output_model["var_flat"],
            ]
            if self._compute_err == "from_var":
                self.output_model["err"] = np.sqrt(
                    np.nansum(var_components, axis=0)
                )

                # nansum returns zero for input that is all NaN -
                # set those values to NaN instead
                all_nan = np.all(np.isnan(var_components), axis=0)
                self._output_model["err"][all_nan] = np.nan

            del var_components, all_nan

        self._finalized = True

        self.finalize_time_info()

        return

    def init_variance_arrays(self):
        """ Allocate arrays that hold co-added resampled variances and their
        weights. """
        shape = self.output_array_shape

        for noise_type in ["var_rnoise", "var_flat", "var_poisson"]:
            var_dtype = self.output_array_types[noise_type]
            kwd = f"{noise_type}_weight"
            if self._accumulate:
                wsum = self._output_model.get(noise_type)
                wt = self._output_model.get(kwd)
                if wsum is None or wt is None:
                    wsum = np.full(shape, np.nan, dtype=var_dtype)
                    wt = np.zeros(shape, dtype=var_dtype)
                else:
                    wsum = wsum * (wt * wt)
            else:
                wsum = np.full(shape, np.nan, dtype=var_dtype)
                wt = np.zeros(shape, dtype=var_dtype)

            setattr(self, f"_{noise_type}_wsum", wsum)
            setattr(self, f"_{noise_type}_weight", wt)

    def resample_variance_arrays(self, model, pixmap, iscale,
                                 weight_map, xmin, xmax, ymin, ymax):
        """ Resample and co-add variance arrays using appropriate weights
        and update total weights.

        Parameters
        ----------
        model : dict
            A dictionary containing data arrays and other meta attributes
            and values of actual models used by pipelines.

        pixmap : 3D array
            A mapping from input image (``data``) coordinates to resampled
            (``out_img``) coordinates. ``pixmap`` must be an array of shape
            ``(Ny, Nx, 2)`` where ``(Ny, Nx)`` is the shape of the input image.
            ``pixmap[..., 0]`` forms a 2D array of X-coordinates of input
            pixels in the ouput frame and ``pixmap[..., 1]`` forms a 2D array of
            Y-coordinates of input pixels in the ouput coordinate frame.

        iscale : float
            The scale to apply to the input variance data before drizzling.

        weight_map : 2D array, None, optional
            A 2D numpy array containing the pixel by pixel weighting.
            Must have the same dimensions as ``data``.

            When ``weight_map`` is `None`, the weight of input data pixels will
            be assumed to be 1.

        xmin : float, optional
            This and the following three parameters set a bounding rectangle
            on the input image. Only pixels on the input image inside this
            rectangle will have their flux added to the output image. Xmin
            sets the minimum value of the x dimension. The x dimension is the
            dimension that varies quickest on the image. If the value is zero,
            no minimum will be set in the x dimension. All four parameters are
            zero based, counting starts at zero.

        xmax : float, optional
            Sets the maximum value of the x dimension on the bounding box
            of the input image. If the value is zero, no maximum will
            be set in the x dimension, the full x dimension of the output
            image is the bounding box.

        ymin : float, optional
            Sets the minimum value in the y dimension on the bounding box. The
            y dimension varies less rapidly than the x and represents the line
            index on the input image. If the value is zero, no minimum  will be
            set in the y dimension.

        ymax : float, optional
            Sets the maximum value in the y dimension. If the value is zero, no
            maximum will be set in the y dimension, the full x dimension
            of the output image is the bounding box.

        """
        # Do the read noise variance first, so it can be
        # used for weights if needed
        pars = {
            'pixmap': pixmap,
            'iscale': iscale,
            'weight_map': weight_map,
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
        }

        if self._check_var_array(model, "var_rnoise"):
            rn_var = self._resample_one_variance_array(
                "var_rnoise",
                model=model,
                **pars,
            )

            # Find valid weighting values in the variance
            if rn_var is not None:
                mask = (rn_var > 0) & np.isfinite(rn_var)
            else:
                mask = np.full_like(rn_var, False)

            # Set the weight for the image from the weight type
            if self.weight_type == "ivm" and rn_var is not None:
                weight = np.ones(self.output_array_shape)
                weight[mask] = np.reciprocal(rn_var[mask])

            elif self.weight_type == "exptime":
                t, _ = get_tmeasure(model)
                weight = np.full(self.output_array_shape, t)

            # Weight and add the readnoise variance
            # Note: floating point overflow is an issue if variance weights
            # are used - it can't be squared before multiplication
            if rn_var is not None:
                # Add the inverse of the resampled variance to a running sum.
                # Update only pixels (in the running sum) with
                # valid new values:
                mask = (rn_var >= 0) & np.isfinite(rn_var) & (weight > 0)
                self._var_rnoise_wsum[mask] = np.nansum(
                    [
                        self._var_rnoise_wsum[mask],
                        rn_var[mask] * weight[mask] * weight[mask]
                    ],
                    axis=0
                )
                self._var_rnoise_weight[mask] += weight[mask]

        # Now do poisson and flat variance, updating only valid new values
        # (zero is a valid value; negative, inf, or NaN are not)
        if self._check_var_array(model, "var_poisson"):
            pn_var = self._resample_one_variance_array(
                "var_poisson",
                model=model,
                **pars,
            )
            if pn_var is not None:
                mask = (pn_var >= 0) & np.isfinite(pn_var) & (weight > 0)
                self._var_poisson_wsum[mask] = np.nansum(
                    [
                        self._var_poisson_wsum[mask],
                        pn_var[mask] * weight[mask] * weight[mask]
                    ],
                    axis=0
                )
                self._var_poisson_weight[mask] += weight[mask]

        if self._check_var_array(model, "var_flat"):
            flat_var = self._resample_one_variance_array(
                "var_flat",
                model=model,
                **pars,
            )
            if flat_var is not None:
                mask = (flat_var >= 0) & np.isfinite(flat_var) & (weight > 0)
                self._var_flat_wsum[mask] = np.nansum(
                    [
                        self._var_flat_wsum[mask],
                        flat_var[mask] * weight[mask] * weight[mask]
                    ],
                    axis=0
                )
                self._var_flat_weight[mask] += weight[mask]

    def finalize_resample_variance(self, output_model, free_memory=True):
        """ Compute variance for the resampled image from running sums and
        weights. Free memory (when ``free_memory=True``) that holds these
        running sums and weights arrays.
        """
        # Divide by the total weights, squared, and set in the output model.
        # Zero weight and missing values are NaN in the output.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value*", RuntimeWarning)
            warnings.filterwarnings("ignore", "divide by zero*", RuntimeWarning)

            output_variance = (
                self._var_rnoise_wsum / self._var_rnoise_weight /
                self._var_rnoise_weight
            ).astype(
                dtype=self.output_array_types["var_rnoise"]
            )
            output_model["var_rnoise"] = output_variance

            output_variance = (
                self._var_poisson_wsum / self._var_poisson_weight /
                self._var_poisson_weight
            ).astype(
                dtype=self.output_array_types["var_poisson"]
            )
            output_model["var_poisson"] = output_variance

            output_variance = (
                self._var_flat_wsum / self._var_flat_weight /
                self._var_flat_weight
            ).astype(
                dtype=self.output_array_types["var_flat"]
            )
            output_model["var_flat"] = output_variance

        if free_memory:
            self._finalized = True
            del (
                self._var_rnoise_wsum,
                self._var_poisson_wsum,
                self._var_flat_wsum,
                self._var_rnoise_weight,
                self._var_poisson_weight,
                self._var_flat_weight,
            )

    def _resample_one_variance_array(self, name, model, iscale,
                                     weight_map, pixmap,
                                     xmin=None, xmax=None, ymin=None,
                                     ymax=None):
        """Resample one variance image from an input model.

        The error image is passed to drizzle instead of the variance, to
        better match kernel overlap and user weights to the data, in the
        pixel averaging process. The drizzled error image is squared before
        returning.
        """
        variance = model.get(name)
        if variance is None or variance.size == 0:
            log.debug(
                f"No data for '{name}' for model "
                f"{repr(model['filename'])}. Skipping ..."
            )
            return

        elif variance.shape != model["data"].shape:
            log.warning(
                f"Data shape mismatch for '{name}' for model "
                f"{repr(model['filename'])}. Skipping ..."
            )
            return

        output_shape = self.output_array_shape

        # Resample the error array. Fill "unpopulated" pixels with NaNs.
        driz = Drizzle(
            out_shape=output_shape,
            kernel=self.kernel,
            fillval=np.nan,
            disable_ctx=True
        )

        # Call 'drizzle' to perform image combination
        log.info(f"Drizzling {variance.shape} --> {output_shape}")

        driz.add_image(
            data=np.sqrt(variance),
            exptime=model["exposure_time"],
            pixmap=pixmap,
            scale=iscale,
            weight_map=weight_map,
            wht_scale=1.0,  # hard-coded for JWST count-rate data
            pixfrac=self.pixfrac,
            in_units="cps",  # TODO: get units from data model
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )

        return driz.out_img ** 2

    def build_driz_weight(self, model, weight_type=None, good_bits=None):
        """ Create a weight map for use by drizzle.

        Parameters
        ----------
        wht_type : {"exptime", "ivm"}, optional
            The weighting type for adding models' data. For ``wht_type="ivm"``
            (the default), the weighting will be determined per-pixel using
            the inverse of the read noise (VAR_RNOISE) array stored in each
            input image. If the ``VAR_RNOISE`` array does not exist,
            the variance is set to 1 for all pixels (i.e., equal weighting).
            If ``weight_type="exptime"``, the weight will be set equal
            to the measurement time (``TMEASURE``) when available and to
            the exposure time (``EFFEXPTM``) otherwise.

        good_bits : int, str, None, optional
            An integer bit mask, `None`, a Python list of bit flags, a comma-,
            or ``'|'``-separated, ``'+'``-separated string list of integer
            bit flags or mnemonic flag names that indicate what bits in models'
            DQ bitfield array should be *ignored* (i.e., zeroed).

            See `Resample` for more information

        """
        data = model["data"]
        dq = model["dq"]

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

            var_rnoise = model["var_rnoise"]
            if (var_rnoise is not None and var_rnoise.shape == data.shape):
                with np.errstate(divide="ignore", invalid="ignore"):
                    inv_variance = var_rnoise**-1

                inv_variance[~np.isfinite(inv_variance)] = 1

                if weight_type != 'ivm':
                    ny, nx = data.shape

                    # apply a median filter to smooth the weight at saturated
                    # (or high read-out noise) single pixels. keep kernel size
                    # small to still give lower weight to extended CRs, etc.
                    ksz = weight_type[8 if selective_median else 7:]
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

            weight = inv_variance * dqmask

        elif weight_type == "exptime":
            t, _ = get_tmeasure(model)
            weight = np.full(data.shape, t)
            weight *= dqmask

        else:
            weight = np.ones(data.shape, dtype=data.dtype) * dqmask

        return weight.astype(np.float32)

    def init_time_counters(self):
        """ Initialize variables/arrays needed to process exposure time. """
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

    def update_time(self, model):
        """ A method called by the `~Resample.add_model` method to process each
        image's time attributes *only when ``model`` has a new group ID.

        """
        if model["group_id"] in self._group_ids:
            return

        self._exptime_start.append(model["start_time"])
        self._exptime_end.append(model["end_time"])

        t, success = get_tmeasure(model)
        self._total_exposure_time += model["exposure_time"]
        self._measurement_time_success.append(success)
        self._total_measurement_time += t

        self._duration += model["duration"]

    def finalize_time_info(self):
        """ Perform final computations for the total time and update relevant
        fileds of the output model.

        """
        assert self._n_res_models
        # basic exposure time attributes:
        self._output_model["exposure_time"] = self._total_exposure_time
        self._output_model["start_time"] = min(self._exptime_start)
        self._output_model["end_time"] = max(self._exptime_end)
        # Update other exposure time keywords:
        # XPOSURE (identical to the total effective exposure time,EFFEXPTM)
        self._output_model["effective_exposure_time"] = self._total_exposure_time
        # DURATION (identical to TELAPSE, elapsed time)
        self._output_model["duration"] = self._duration
        self._output_model["elapsed_exposure_time"] = self._duration

        if all(self._measurement_time_success):
            self._output_model["measurement_time"] = self._total_measurement_time

    def _check_var_array(self, model, array_name):
        """ Check that a variance array has the same shape as the model's
        data array.

        """
        array_data = model.get(array_name, None)
        sci_data = model["data"]
        model_name = _get_model_name(model)

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


def _get_model_name(model):
    model_name = model.get("filename")
    if model_name is None or not model_name.strip():
        model_name = "Unknown"
    return model_name


def _is_flux_density(bunit):
    """
    Differentiate between surface brightness and flux density data units.

    Parameters
    ----------
    bunit : str or `~astropy.units.Unit`
       Data units, e.g. 'MJy' (is flux density) or 'MJy/sr' (is not).

    Returns
    -------
    bool
        True if the units are equivalent to flux density units.
    """
    try:
        flux_density = u.Unit(bunit).is_equivalent(u.Jy)
    except (ValueError, TypeError):
        flux_density = False
    return flux_density
