import logging
import math
import warnings
import sys

import numpy as np

from drizzle.utils import calc_pixmap
from drizzle.resample import Drizzle

from stcal.resample.utils import (
    build_driz_weight,
    compute_mean_pixel_area,
    get_tmeasure,
    resample_range,
    is_flux_density,
)


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

__all__ = [
    "Resample",
    "UnsupportedWCSError",
]


class UnsupportedWCSError(RuntimeError):
    """ Raised when provided output WCS has an unexpected number of axes
    or has an unsupported structure.
    """


class Resample:
    """ Base class for resampling images.

    The main purpose of this class is to resample and add input images
    (data, variance array) to an output image defined by an output WCS.

    In particular, this class performs the following operations:

    1. Sets up output arrays based on arguments used at initialization.
    2. Based on information about the input images and user arguments, computes
       scale factors needed to convert resampled counts to fluxes.
    3. For each input image, computes coordinate transformations (``pixmap``)
       from the coordinate system of the input image to the coordinate system
       of the output image.
    4. Computes the weight image for each input image.
    5. Calls :py:class:`~drizzle.resample.Drizzle` methods to resample and
       combine input images and their variance/error arrays.
    6. Keeps track of total exposure time and other time-related quantities.

    """
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

    dq_flag_name_map = None

    def __init__(self, output_wcs, n_input_models=None, pixfrac=1.0,
                 kernel="square", fillval=0.0, weight_type="ivm", good_bits=0,
                 enable_ctx=True, enable_var=True, compute_err=None):
        """
        Parameters
        ----------
        output_wcs : dict
            Specifies output WCS as a dictionary
            with keys ``'wcs'`` (WCS object) and ``'pixel_scale'``
            (pixel scale in arcseconds). ``'pixel_scale'``, when provided,
            will be used for computation of drizzle scaling factor. When it is
            not provided, output pixel scale will be *estimated* from the
            provided WCS object.

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

        kernel : {"square", "gaussian", "point", "turbo", "lanczos2", "lanczos3"}, optional
            The name of the kernel used to combine the input. The choice of
            kernel controls the distribution of flux over the kernel.
            The square kernel is the default.

            .. warning::
               The "gaussian" and "lanczos2/3" kernels **DO NOT**
               conserve flux.

        fillval : float, None, str, optional
            The value of output pixels that did not have contributions from
            input images' pixels. When ``fillval`` is either `None` or
            ``"INDEF"`` and ``out_img`` is provided, the values of ``out_img``
            will not be modified. When ``fillval`` is either `None` or
            ``"INDEF"`` and ``out_img`` is **not provided**, the values of
            ``out_img`` will be initialized to `numpy.nan`. If ``fillval``
            is a string that can be converted to a number, then the output
            pixels with no contributions from input images will be set to this
            ``fillval`` value.

        weight_type : {"ivm", "exptime"}, optional
            The weighting type for adding models' data. For
            ``weight_type="ivm"`` (the default), the weighting will be
            determined per-pixel using the inverse of the read noise
            (VAR_RNOISE) array stored in each input image. If the
            ``VAR_RNOISE`` array does not exist, the variance is set to 1 for
            all pixels (i.e., equal weighting). If ``weight_type="exptime"``,
            the weight will be set equal to the measurement time
            when available and to the exposure time otherwise.

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

            Default value (0) will make *all* pixels with non-zero DQ
            values be considered "bad" pixels, and the corresponding data
            pixels will be assigned zero weight and thus these pixels
            will not contribute to the output resampled data array.

            Set `good_bits` to `None` to turn off the use of model's DQ
            array.

            For more details, see documentation for
            `astropy.nddata.bitmask.extend_bit_flag_map`.

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
        # to see if setting up arrays and drizzle is needed
        self._finalized = False
        self._n_res_models = 0

        self._enable_ctx = enable_ctx
        self._enable_var = enable_var
        self._compute_err = compute_err

        # these attributes are used only for informational purposes
        # and are added to created the output_model only if they are
        # not already present there:
        self._pixel_scale_ratio = None
        self._output_pixel_scale = None  # in arcsec

        # resample parameters
        self.pixfrac = pixfrac
        self.kernel = kernel
        self.fillval = fillval
        self.good_bits = good_bits

        if weight_type.startswith("ivm") or weight_type == "exptime":
            self.weight_type = weight_type
        else:
            raise ValueError("Unexpected weight type: '{self.weight_type}'")

        self._output_wcs = output_wcs

        self._group_ids = []

        # determine output WCS and set up output model if needed:
        if output_wcs is None:
            raise ValueError(
                "Output WCS must be provided either through the "
                "'output_wcs' parameter or the 'output_model' parameter. "
            )
        else:
            self._output_pixel_scale = output_wcs.get("pixel_scale")
            self._pixel_scale_ratio = output_wcs.get(
                "pixel_scale_ratio"
            )
            self._output_wcs = output_wcs.get("wcs")
            self.check_output_wcs(self._output_wcs)

        if self._output_pixel_scale is None:
            self._output_pixel_scale = 3600.0 * np.rad2deg(
                math.sqrt(
                    self.get_output_model_pixel_area({"wcs": self._output_wcs})
                )
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

        # Check that the output data shape has no zero-length dimensions
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

        # set up an empty output model (don't allocate arrays at this time):
        self.reset_arrays(n_input_models=n_input_models)

    def get_input_model_pixel_area(self, model):
        """
        Computes or retrieves pixel area of an input model. Currently,
        this is the average pixel area of the input model's pixels within
        either the bounding box (if available) or the entire data array.

        This value is used to compute a scale factor that will be applied
        to input image data. This scale factor takes into account the
        difference in the definition of the pixel area reported in model's
        ``meta`` and the pixel area at the location used to construct
        output WCS from the WCS of input models using ``pixel_scale_ratio``.

        The intensity scale factor is computed elsewhere as the ratio of the
        value of the pixel area in the meta to the area returned by this
        function.

        Subclasses can override this method to return the most appropriate
        pixel area value.

        Parameters
        ----------

        model : dict, None
            A dictionary containing data arrays and other meta attributes
            and values of actual models used by pipelines. In particular, it
            must have a keyword "wcs" and a WCS associated with it.

        Returns
        -------
        pix_area : float
            Pixel area in steradians.

        """
        pixel_area = compute_mean_pixel_area(
            model["wcs"],
            shape=model["data"].shape
        )
        return pixel_area

    def get_output_model_pixel_area(self, model):
        """
        Computes or retrieves pixel area of the output model. Currently,
        this is the average pixel area of the model's pixels within either
        the bounding box (if available) or the entire data array.

        Parameters
        ----------

        model : dict, None
            A dictionary containing data arrays and other meta attributes
            and values of actual models used by pipelines. In particular, it
            must have a keyword "wcs" and a WCS associated with it.

        Returns
        -------
        pix_area : float
            Pixel area in steradians.

        """
        pixel_area = compute_mean_pixel_area(model["wcs"])
        return pixel_area

    def check_output_wcs(self, output_wcs, estimate_output_shape=True):
        """
        Check that provided WCS has expected properties and that its
        ``array_shape`` property is defined. May modify ``output_wcs``.

        Parameters
        ----------
        output_wcs : WCS object
            A WCS object corresponding to the output (resampled) image.

        estimate_output_shape : bool, optional
            Indicates whether to *estimate* output image shape of the
            ``output_wcs`` from other available attributes such as
            ``bounding_box`` when ``output_wcs.array_shape`` is `None`.
            If ``estimate_output_shape`` is `True` and
            ``output_wcs.array_shape`` is `None`, upon return
            ``output_wcs.array_shape`` will be assigned an estimated value.

        """
        naxes = output_wcs.output_frame.naxes
        if 'SPECTRAL' in output_wcs.output_frame.axes_type:
            if naxes != 3:
                raise UnsupportedWCSError(
                    "Output spectral WCS needs 3 coordinate axes but the "
                    f"supplied WCS has {naxes} axes."
                )
            return

        if naxes != 2:
            raise UnsupportedWCSError(
                "Output WCS needs 2 coordinate axes but the "
                f"supplied WCS has {naxes} axes."
            )

        # make sure array_shape and pixel_shape are set:
        if output_wcs.array_shape is None and estimate_output_shape:
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

    def create_output_model(self):
        """ Create a new "output model": a dictionary of data and meta fields.

        Returns
        -------

        output_model : dict
            A dictionary of data model attributes and values.

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
                }
            )

        if self._compute_err is not None:
            output_model["err"] = None

        return output_model

    @property
    def output_model(self):
        """ Output (resampled) model. """
        return self._output_model

    @property
    def output_array_shape(self):
        """ Shape of the output model arrays. """
        return self._output_array_shape

    @property
    def output_wcs(self):
        """ WCS of the output (resampled) model. """
        return self._output_wcs

    @property
    def pixel_scale_ratio(self):
        """ Get the ratio of the output pixel scale to the input pixel scale.
        """
        return self._pixel_scale_ratio

    @property
    def output_pixel_scale(self):
        """ Get pixel scale of the output model in arcsec. """
        return self._output_pixel_scale  # in arcsec

    @property
    def group_ids(self):
        """ Get a list of all group IDs of models resampled and added to the
        output model.
        """
        return self._group_ids

    @property
    def enable_ctx(self):
        """ Indicates whether context array is enabled. """
        return self._enable_ctx

    @property
    def enable_var(self):
        """ Indicates whether variance arrays are resampled. """
        return self._enable_var

    @property
    def compute_err(self):
        """ Indicates whether error array is computed and how it is computed.
        """
        return self._compute_err

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
        photom_pixel_area = model["pixelarea_steradians"]
        wcs = model["wcs"]

        if photom_pixel_area:
            if 'SPECTRAL' in wcs.output_frame.axes_type:
                # Use the nominal area as is
                input_pixel_area = photom_pixel_area

                # If input image is in flux density units, correct the
                # flux for the user-specified change to the spatial dimension
                if is_flux_density(model["bunit_data"]):
                    iscale = 1.0 / math.sqrt(self.pixel_scale_ratio)
                else:
                    iscale = 1.0
            else:
                input_pixel_area = self.get_input_model_pixel_area(model)

                if input_pixel_area is None:
                    raise ValueError(
                        "Unable to compute input pixel area from WCS of input "
                        f"image {repr(_get_model_name(model))}."
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
                            self._output_model.get("pixel_scale_ratio") is None):
                        self._output_model["pixel_scale_ratio"] = self._pixel_scale_ratio

                iscale = math.sqrt(photom_pixel_area / input_pixel_area)

        else:
            iscale = 1.0

        return iscale

    def reset_arrays(self, n_input_models=None):
        """ Initialize/reset `Drizzle` objects, output model and arrays,
        and time counters and clears the "finalized" flag. Output WCS and shape
        are not modified from `Resample` object initialization. This method
        needs to be called before calling :py:meth:`add_model` for the first
        time after :py:meth:`finalize` was called.

        Parameters
        ----------
        n_input_models : int, None, optional
            Number of input models expected to be resampled. When provided,
            this is used to estimate memory requirements and optimize memory
            allocation for the context array.

        """
        # set up an empty output model (don't allocate arrays at this time):
        if not hasattr(self, "_output_model") or self.is_finalized():
            self._output_model = self.create_output_model()

        if n_input_models is None:
            max_ctx_id = None
        else:
            max_ctx_id = n_input_models - 1

        self._driz = Drizzle(
            kernel=self.kernel,
            fillval=self.fillval,
            out_shape=self._output_array_shape,
            out_img=self._output_model["data"],
            out_wht=self._output_model["wht"],
            out_ctx=self._output_model["con"],
            exptime=self._output_model["exposure_time"],
            begin_ctx_id=0,
            max_ctx_id=max_ctx_id,
            disable_ctx=not self._enable_ctx,
        )

        # Also make a temporary model to hold error data
        if self._compute_err == "driz_err":
            self._driz_error = Drizzle(
                kernel=self.kernel,
                fillval=self.fillval,
                out_shape=self._output_array_shape,
                out_img=self._output_model["err"],
                exptime=self._output_model["exposure_time"],
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
            "filename",
            "group_id",
            "wcs",

            "exposure_time",
            "start_time",
            "end_time",
            "duration",
            "measurement_time",

            "pixelarea_steradians",
            # "pixelarea_arcsecsq",

            "level",  # sky level
            "subtracted",
        ]

        if 'SPECTRAL' in model["wcs"].output_frame.axes_type:
            min_attributes.append("bunit_data")

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
        """ Resamples model image, variance data (if ``enable_var``
        is `True`) , and error data (if ``enable_err`` is `True`), and adds
        them to the corresponding
        arrays of the output model using appropriate weighting.
        It also updates the weight array and context array (if ``enable_ctx``
        is `True`) of the resampled data, as well as relevant metadata.

        Whenever ``model`` has a unique group ID that was never processed
        before, the "pointings" value of the output model is incremented and
        the "group_id" attribute is updated. Also, time counters are updated
        with new values from the input ``model`` by calling
        :py:meth:`~Resample.update_time` .

        Parameters
        ----------
        model : dict
            A dictionary containing data arrays and other meta attributes
            and values of actual models used by pipelines.

        """
        if self._finalized:
            raise RuntimeError(
                "Resampling has been finalized and no new models can be added "
                "to the resampled output model. Call 'reset_arrays' "
                "to initialize a new output model and associated arrays."
            )

        self.validate_input_model(model)
        self._n_res_models += 1

        data = model["data"]
        wcs = model["wcs"]

        # Check that input models are 2D images
        if data.ndim != 2:
            raise RuntimeError(
                f"Input model '{_get_model_name(model)}' is not a 2D image."
            )

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

        weight = build_driz_weight(
            model,
            weight_type=self.weight_type,
            good_bits=self.good_bits,
            flag_name_map=self.dq_flag_name_map
        )

        # apply sky subtraction
        blevel = model["level"]
        if not model["subtracted"] and blevel is not None:
            data = data - blevel

        xmin, xmax, ymin, ymax = resample_range(
            data.shape,
            wcs.bounding_box
        )

        add_image_kwargs = {
            'exptime': model["exposure_time"],
            'pixmap': pixmap,
            'scale': iscale,
            'weight_map': weight,
            'wht_scale': 1.0,
            'pixfrac': self.pixfrac,
            'in_units': 'cps',
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

    def is_finalized(self):
        """ Indicates whether all attributes of the ``output_model`` have been
        computed from intermediate (running) values.
        """
        return self._finalized

    def finalize(self):
        """ Performs final computations from any intermediate values,
        sets ouput model values, and optionally frees temporary/intermediate
        objects.

        ``finalize`` calls :py:meth:`~Resample.finalize_resample_variance` and
        :py:meth:`~Resample.finalize_time_info`.

        .. warning::
          Once the resample process has been finalized, adding new models to
          the output resampled model is not allowed.

        """
        if self._finalized:
            # can't finalize twice
            return

        self._finalized = True

        self._output_model["pointings"] = len(self.group_ids)

        # assign resampled arrays to the output model dictionary:
        self._output_model["data"] = self._driz.out_img
        self._output_model["wht"] = self._driz.out_wht
        if self._driz.out_ctx is not None:
            # Since the context array is dynamic, it must be re-assigned
            # back to the product's `con` attribute.
            self._output_model["con"] = self._driz.out_ctx

        del self._driz

        # compute final variances:
        if self._enable_var:
            self.finalize_resample_variance(self._output_model)

        if self._compute_err == "driz_err":
            # use resampled error
            self.output_model["err"] = self._driz_error.out_img
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
                del all_nan

            del var_components

        self.finalize_time_info()
        return

    def init_variance_arrays(self):
        """ Allocate arrays that hold co-added resampled variances and their
        weights. """
        shape = self.output_array_shape

        for noise_type in ["var_rnoise", "var_flat", "var_poisson"]:
            var_dtype = self.output_array_types[noise_type]
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
            pixels in the ouput frame and ``pixmap[..., 1]`` forms a 2D array
            of Y-coordinates of input pixels in the ouput coordinate frame.

        iscale : float
            The scale to apply to the input variance data before drizzling.

        weight_map : numpy.ndarray, None, optional
            A 2D ``numpy`` array containing the pixel by pixel weighting.
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
            if self.weight_type.startswith("ivm") and rn_var is not None:
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

    def finalize_resample_variance(self, output_model):
        """ Compute variance for the resampled image from running sums and
        weights. Free memory that holds these running sums and weights arrays.

        output_model : dict, None
            A dictionary containing data arrays and other attributes that
            will be used to add new models to. use
            :py:meth:`Resample.output_model_attributes` to get the list of
            keywords that must be present. When ``accumulate`` is `False`,
            only the WCS object of the model will be used. When ``accumulate``
            is `True`, new models will be added to the existing data in the
            ``output_model``.

        """
        # Divide by the total weights, squared, and set in the output model.
        # Zero weight and missing values are NaN in the output.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value*", RuntimeWarning)
            warnings.filterwarnings("ignore", "divide by zero*", RuntimeWarning)

            output_variance = (
                self._var_rnoise_wsum / (self._var_rnoise_weight *
                                         self._var_rnoise_weight)
            ).astype(
                dtype=self.output_array_types["var_rnoise"]
            )
            output_model["var_rnoise"] = output_variance

            output_variance = (
                self._var_poisson_wsum / (self._var_poisson_weight *
                                          self._var_poisson_weight)
            ).astype(
                dtype=self.output_array_types["var_poisson"]
            )
            output_model["var_poisson"] = output_variance

            output_variance = (
                self._var_flat_wsum / (self._var_flat_weight *
                                       self._var_flat_weight)
            ).astype(
                dtype=self.output_array_types["var_flat"]
            )
            output_model["var_flat"] = output_variance

            del (
                self._var_rnoise_wsum,
                self._var_poisson_wsum,
                self._var_flat_wsum,
                self._var_rnoise_weight,
                self._var_poisson_weight,
                self._var_flat_weight,
            )
            self._finalized = True

    def _resample_one_variance_array(self, name, model, iscale,
                                     weight_map, pixmap,
                                     xmin=None, xmax=None, ymin=None,
                                     ymax=None):
        """Resample one variance image from an input model.

        The error image is passed to drizzle instead of the variance in order
        to better match kernel overlap and user weights to the data during the
        pixel averaging process. The drizzled error image is squared before
        returning.

        """
        variance = model.get(name)
        if variance is None or variance.size == 0:
            log.debug(
                f"No data for '{name}' for model "
                f"{repr(_get_model_name(model))}. Skipping ..."
            )
            return

        elif variance.shape != model["data"].shape:
            log.warning(
                f"Data shape mismatch for '{name}' for model "
                f"{repr(_get_model_name(model))}. Skipping ..."
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
            wht_scale=1.0,
            pixfrac=self.pixfrac,
            in_units="cps",
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )

        return driz.out_img ** 2

    def init_time_counters(self):
        """ Initialize variables/arrays needed to process exposure time. """
        self._total_exposure_time = 0
        self._duration = 0
        self._total_measurement_time = 0
        if self._total_measurement_time is None:
            self._total_measurement_time = 0.0

        if (start_time := self.output_model.get("start_time", None)) is None:
            self._exptime_start = []
        else:
            self._exptime_start = [start_time]

        if (end_time := self.output_model.get("end_time", None)) is None:
            self._exptime_end = []
        else:
            self._exptime_end = [end_time]

        self._measurement_time_success = []

    def update_time(self, model):
        """
        A method called by the :py:meth:`~Resample.add_model` method to
        process each image's time attributes *only* when ``model`` has a new
        group ID.

        """
        if model["group_id"] in self._group_ids:
            return

        self._exptime_start.append(model["start_time"])
        self._exptime_end.append(model["end_time"])

        if (exposure_time := model["exposure_time"]) is not None:
            self._total_exposure_time += exposure_time

        t, success = get_tmeasure(model)
        self._measurement_time_success.append(success)
        if t is not None:
            self._total_measurement_time += t

        if (duration := model["duration"]) is not None:
            self._duration += duration

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
        # DURATION (identical to elapsed time)
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
    """ Return the value of ``"filename"`` from the model dictionary or
    ``"Unknown"`` when ``"filename"`` is either not present or it is `None`.

    """
    model_name = model.get("filename")
    if model_name is None or not model_name.strip():
        model_name = "Unknown"
    return model_name
