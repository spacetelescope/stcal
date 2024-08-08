import logging
import os
import warnings
from copy import deepcopy
import sys
import abc
from pathlib import Path, PurePath

import numpy as np
from scipy.ndimage import median_filter

from drizzle.resample import Drizzle
from drizzle.utils import calc_pixmap

import psutil
from spherical_geometry.polygon import SphericalPolygon

from stdatamodels.dqflags import interpret_bit_flags
from stdatamodels.jwst.datamodels.dqflags import pixel

from stdatamodels.jwst import datamodels
from stdatamodels.jwst.library.basic_utils import bytes2human

from .utils import get_tmeasure, build_mask


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

__all__ = [
    "OutputTooLargeError",
    "ResampleModelIO",
    "ResampleBase",
    "ResampleCoAdd",
    "ResampleSingle"
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


class ResampleModelIO(abc.ABC):
    @abc.abstractmethod
    def open_model(self, file_name):
        ...

    @abc.abstractmethod
    def get_model_meta(self, file_name, fields):
        ...

    @abc.abstractmethod
    def close_model(self, model):
        ...

    @abc.abstractmethod
    def save_model(self, model):
        ...

    @abc.abstractmethod
    def write_model(self, model, file_name):
        ...

    @abc.abstractmethod
    def new_model(self, image_shape=None, file_name=None):
        """ Return a new model for the resampled output """
        ...


class OutputTooLargeError(RuntimeError):
    """Raised when the output is too large for in-memory instantiation"""


def output_wcs_from_input_wcs(input_wcs_list, pixel_scale_ratio=1.0,
                              pixel_scale=None, output_shape=None,
                              crpix=None, crval=None, rotation=None):
    # TODO: should be replaced with a version that lives in stcal and
    # uses s_region
    w = deepcopy(input_wcs_list[0])  # this is bad
    return {
        'output_wcs': w,
        'pscale': np.rad2deg(np.sqrt(_compute_image_pixel_area(w))),
        'pscale_ratio': 1.0,
        'crpix': None
    }


class ResampleBase(abc.ABC):
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

    def __init__(self, input_models,
                 pixfrac=1.0, kernel="square", fillval=0.0, wht_type="ivm",
                 good_bits=0, output_wcs=None, wcs_pars=None,
                 in_memory=True, allowed_memory=None, **kwargs):
        """
        Parameters
        ----------
        input_models : list of objects
            list of data models, one for each input image

        output : str
            filename for output

        kwargs : dict
            Other parameters.

            .. note::
                ``output_shape`` is in the ``x, y`` order.

            .. note::
                ``in_memory`` controls whether or not the resampled
                array from ``resample_many_to_many()``
                should be kept in memory or written out to disk and
                deleted from memory. Default value is `True` to keep
                all products in memory.
        """
        self._output_model = None
        self._output_filename = None
        self._output_wcs = None
        self._output_array_shape = None
        self._close_output = False
        self._output_pixel_scale = None
        self._template_output_model = None

        # input models
        self._input_models = input_models
        # a lightweight data model with meta from first input model but no data.
        # it will be updated by 'load_input_meta()' below
        self._first_model_meta = None

        # resample parameters
        self.pixfrac = pixfrac
        self.kernel = kernel
        self.fillval = fillval
        self.weight_type = wht_type
        self.good_bits = good_bits
        self.in_memory = in_memory

        self._user_output_wcs = output_wcs

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
        # WCS parameters (should be deleted once not needed;
        # once an output WCS was created)
        self._wcs_pars = wcs_pars

        # process additional kwags specific to subclasses and store
        # unprocessed/unrecognized kwargs in ukwargs and warn about these
        # unrecognized kwargs
        ukwargs = self.process_kwargs(kwargs)
        self._warn_extra_kwargs(ukwargs)

        # load meta necessary for output WCS (and other) computations:
        self.load_input_meta(
            all=self._output_model is None and output_wcs is None
        )

        # computed average pixel scale of the first input image:
        input_pscale0 = np.rad2deg(
            np.sqrt(_compute_image_pixel_area(self._input_wcs_list[0]))
        )

        # compute output pixel scale, WCS, set-up output model
        if self._output_model:
            self._output_wcs = deepcopy(self._output_model.meta.wcs)
            self._output_array_shape = self._output_model.data.shape
            # TODO: extract any useful info from the output image before we close it:
            # if meta has pixel scale, populate it from there, if not:
            self._output_pixel_scale = np.rad2deg(
                np.sqrt(_compute_image_pixel_area(self._output_wcs))
            )
            self._pixel_scale_ratio = self._output_pixel_scale / input_pscale0
            log.info(f'Computed output pixel scale: {self._output_pixel_scale} arcsec.')

            self._create_output_template_model()  # create template before possibly closing output
            if self._close_output and not self.in_memory:
                self.close_model(self._output_model)
                self._output_model = None

        elif output_wcs:
            naxes = output_wcs.output_frame.naxes
            if naxes != 2:
                raise RuntimeError(
                    "Output WCS needs 2 spatial axes but the "
                    f"supplied WCS has {naxes} axes."
                )
            self._output_wcs = deepcopy(output_wcs)
            if wcs_pars and "output_shape" in wcs_pars:
                self._output_array_shape = wcs_pars["output_shape"]
            else:
                self._output_array_shape = self._output_wcs.array_shape
                if not self._output_array_shape and output_wcs.bounding_box:
                    halfpix = 0.5 + sys.float_info.epsilon
                    self._output_array_shape = (
                        int(output_wcs.bounding_box[1][1] + halfpix),
                        int(output_wcs.bounding_box[0][1] + halfpix),
                    )
                else:
                    raise ValueError(
                        "Unable to infer output image size from provided inputs."
                    )
                self._output_wcs.array_shape = self._output_array_shape

            self._output_pixel_scale = np.rad2deg(
                np.sqrt(_compute_image_pixel_area(self._output_wcs))
            )
            self._pixel_scale_ratio = self._output_pixel_scale / input_pscale0
            log.info(f'Computed output pixel scale: {self._output_pixel_scale} arcsec.')
            self._create_output_template_model()

        else:
            # build output WCS and calculate ouput image shape
            if "pixel_scale" in wcs_pars and wcs_pars['pixel_scale'] is not None:
                self._pixel_scale_ratio = wcs_pars["pixel_scale"] / input_pscale0
                log.info(f'Output pixel scale: {wcs_pars["pixel_scale"]} arcsec.')
                log.info(f'Computed output pixel scale ratio: {self._pixel_scale_ratio}.')
            else:
                self._pixel_scale_ratio = wcs_pars.get("pixel_scale_ratio", 1.0)
                log.info(f'Output pixel scale ratio: {self._pixel_scale_ratio}')
                self._output_pixel_scale = input_pscale0 * self._pixel_scale_ratio
                wcs_pars = wcs_pars.copy()
                wcs_pars["pixel_scale"] = self._output_pixel_scale
                log.info(f'Computed output pixel scale: {self._output_pixel_scale} arcsec.')

            w, ps = self._compute_output_wcs(**wcs_pars)
            self._output_wcs = w
            self._output_pixel_scale = ps
            self._output_array_shape = self._output_wcs.array_shape
            self._create_output_template_model()

        # Check that the output data shape has no zero length dimensions
        npix = np.prod(self._output_array_shape)
        if not npix:
            raise ValueError(
                f"Invalid output frame shape: {tuple(self._output_array_shape)}"
            )

        assert self._pixel_scale_ratio
        log.info(f"Driz parameter kernel: {self.kernel}")
        log.info(f"Driz parameter pixfrac: {self.pixfrac}")
        log.info(f"Driz parameter fillval: {self.fillval}")
        log.info(f"Driz parameter weight_type: {self.weight_type}")

        self.check_memory_requirements(allowed_memory)

        log.debug('Output mosaic size: {}'.format(self._output_wcs.pixel_shape))

    @property
    def output_model(self):
        return self._output_model

    def process_kwargs(self, kwargs):
        """ A method called by ``__init__`` to process input kwargs before
        output WCS is created and before output model template is created.

        Returns
        -------
        kwargs : dict
            Unrecognized/not processed ``kwargs``.

        """
        return {k : v for k, v in kwargs.items()}

    def _warn_extra_kwargs(self, kwargs):
        for k in kwargs:
            log.warning(f"Unrecognized argument '{k}' will be ignored.")

    def check_memory_requirements(self, allowed_memory):
        """ Called just before '_pre_run_callback()' is called to verify
        that there is enough memory to hold the output. """
        if allowed_memory is None and "DMODEL_ALLOWED_MEMORY" not in os.environ:
            return

        allowed_memory = float(allowed_memory)
        # make a small image model to get the dtype
        dtype = datamodels.ImageModel((1, 1)).data.dtype

        # get the available memory
        available_memory = psutil.virtual_memory().available + psutil.swap_memory().total

        # compute the output array size
        npix = npix = np.prod(self._output_array_shape)
        nmodels = len(self._input_models)
        nconpl = nmodels // 32 + (1 if nmodels % 32 else 0)
        required_memory = npix * (3 * dtype.itemsize + nconpl * 4)

        # compare used to available
        used_fraction = required_memory / available_memory
        if used_fraction > allowed_memory:
            raise OutputTooLargeError(
                f'Combined ImageModel size {self._output_wcs.array_shape} '
                f'requires {bytes2human(required_memory)}. '
                f'Model cannot be instantiated.'
            )

    def _compute_output_wcs(self, **wcs_pars):
        """ returns a diustortion-free WCS object and its pixel scale """
        owcs = output_wcs_from_input_wcs(self._input_wcs_list, **wcs_pars)
        return owcs['output_wcs'], owcs['pscale']

    def load_input_meta(self, all=True):
        # if 'all=False', load meta from the first image only

        # set-up list for WCS
        self._input_wcs_list = []
        self._input_s_region = []
        self._input_file_names = []
        self._close_input_models = []

        for k, model in enumerate(self._input_models):
            close_model = isinstance(model, str)
            self._close_input_models.append(close_model)
            if close_model:
                self._input_file_names.append(model)
                model = self.open_model(model)
                if self.in_memory:
                    self._input_models[k] = model
            else:
                self._input_file_names.append(model.meta.filename)
            # extract all info needed from *this* model:
            w = deepcopy(model.meta.wcs)
            w.array_shape = model.data.shape
            self._input_wcs_list.append(w)

            # extract other useful data
            #    - S_REGION:
            self._input_s_region.append(model.meta.wcsinfo.s_region)

            # store first model's entire meta (except for WCS and data):
            if self._first_model_meta is None:
                self._first_model_meta = self.new_model()
                self._first_model_meta.update(model)

            if close_model and not self.in_memory:
                self.close_model(model)

            if not all:
                break

    def blend_output_metadata(self, output_model):
        # TODO: Not sure about funct. signature and also I don't like it needs
        # to open input files again
        pass

    def build_driz_weight(self, model, weight_type=None, good_bits=None):
        """Create a weight map for use by drizzle
        """
        dqmask = build_mask(model.dq, good_bits)

        if weight_type and weight_type.startswith('ivm'):
            weight_type = weight_type.strip()
            selective_median = weight_type.startswith('ivm-smed')

            bitvalue = interpret_bit_flags(good_bits, mnemonic_map=pixel)
            if bitvalue is None:
                bitvalue = 0
            saturation = pixel['SATURATED']

            if selective_median and not (bitvalue & saturation):
                selective_median = False
                weight_type = 'ivm'

            if (model.hasattr("var_rnoise") and model.var_rnoise is not None and
                    model.var_rnoise.shape == model.data.shape):
                with np.errstate(divide="ignore", invalid="ignore"):
                    inv_variance = model.var_rnoise**-1

                inv_variance[~np.isfinite(inv_variance)] = 1

                if weight_type != 'ivm':
                    ny, nx = inv_variance.shape

                    # apply a median filter to smooth the weight at saturated
                    # (or high read-out noise) single pixels. keep kernel size
                    # small to still give lower weight to extended CRs, etc.
                    ksz = weight_type[8 if selective_median else 7 :]
                    if ksz:
                        kernel_size = int(ksz)
                        if not (kernel_size % 2):
                            raise ValueError(
                                'Kernel size of the median filter in IVM weighting'
                                ' must be an odd integer.'
                            )
                    else:
                        kernel_size = 3

                    ivm_copy = inv_variance.copy()

                    if selective_median:
                        # apply median filter selectively only at
                        # points of partially saturated sources:
                        jumps = np.where(
                            np.logical_and(model.dq & saturation, dqmask)
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

        elif weight_type == 'exptime':
            exptime = model.meta.exposure.exposure_time
            result = exptime * dqmask

        else:
            result = np.ones(model.data.shape, dtype=model.data.dtype) * dqmask

        return result.astype(np.float32)

    @abc.abstractmethod
    def run(self):
        ...

    def _create_output_template_model(self):
        pass

    def update_exposure_times(self):
        """Modify exposure time metadata in-place"""
        total_exposure_time = 0.
        exptime_start = []
        exptime_end = []
        duration = 0.0
        total_exptime = 0.0
        measurement_time_success = []
        for exposure in self._input_models.models_grouped:
            total_exposure_time += exposure[0].meta.exposure.exposure_time
            t, success = get_tmeasure(exposure[0])
            measurement_time_success.append(success)
            total_exptime += t
            exptime_start.append(exposure[0].meta.exposure.start_time)
            exptime_end.append(exposure[0].meta.exposure.end_time)
            duration += exposure[0].meta.exposure.duration

        # Update some basic exposure time values based on output_model
        self._output_model.meta.exposure.exposure_time = total_exposure_time
        if not all(measurement_time_success):
            self._output_model.meta.exposure.measurement_time = total_exptime
        self._output_model.meta.exposure.start_time = min(exptime_start)
        self._output_model.meta.exposure.end_time = max(exptime_end)

        # Update other exposure time keywords:
        # XPOSURE (identical to the total effective exposure time, EFFEXPTM)
        self._output_model.meta.exposure.effective_exposure_time = total_exptime
        # DURATION (identical to TELAPSE, elapsed time)
        self._output_model.meta.exposure.duration = duration
        self._output_model.meta.exposure.elapsed_exposure_time = duration


class ResampleCoAdd(ResampleBase):
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

    def __init__(self, input_models, output, accum=False,
                 pixfrac=1.0, kernel="square", fillval=0.0, wht_type="ivm",
                 good_bits=0, output_wcs=None, wcs_pars=None,
                 in_memory=True, allowed_memory=None):
        """
        Parameters
        ----------
        input_models : list of objects
            list of data models, one for each input image

        output : DataModel, str
            filename for output

        kwargs : dict
            Other parameters.

            .. note::
                ``output_shape`` is in the ``x, y`` order.

            .. note::
                ``in_memory`` controls whether or not the resampled
                array from ``resample_many_to_many()``
                should be kept in memory or written out to disk and
                deleted from memory. Default value is `True` to keep
                all products in memory.
        """
        self._accum = accum

        super().__init__(
            input_models,
            pixfrac, kernel, fillval, wht_type,
            good_bits, output_wcs, wcs_pars,
            in_memory, allowed_memory, output=output
        )

    def process_kwargs(self, kwargs):
        """ A method called by ``__init__`` to process input kwargs before
        output WCS is created and before output model template is created.
        """
        kwargs = super().process_kwargs(kwargs)
        output = kwargs.pop("output", None)
        accum = kwargs.pop("accum", False)

        # Load the model if accum is True
        if isinstance(output, str):
            self._output_filename = output
            if accum:
                try:
                    self._output_model = self.open_model(output)
                    self._close_output = True
                    log.info(
                        "Output model has been loaded and it will be used to "
                        "accumulate new data."
                    )
                    if self._user_output_wcs:
                        log.info(
                            "'output_wcs' will be ignored when 'output' is "
                            "provided and accum=True"
                        )
                    if self._wcs_pars:
                        log.info(
                            "'wcs_pars' will be ignored when 'output' is "
                            "provided and accum=True"
                        )
                except FileNotFoundError:
                    pass

        elif output is not None:
            self._output_filename = output.meta.filename
            self._output_model = output
            self._close_output = False

        return kwargs

    def blend_output_metadata(self, output_model):
        pass

    def extra_pre_resample_setup(self):
        pass

    def post_process_resample_model(self, data_model, driz_init_kwargs, add_image_kwargs):
        pass

    def finalize_resample(self):
        pass

    def _create_new_output_model(self):
        # this probably needs to be an abstract class.
        # also this is mostly needed for "single" drizzle.
        output_model = self.new_model(None)

        # update meta data and wcs

        # TODO: don't like this as it means reloading first image (again)
        output_model.update(self._first_model_meta)
        output_model.meta.wcs = deepcopy(self._output_wcs)

        pix_area = self._output_pixel_scale**2
        output_model.meta.photometry.pixelarea_steradians = pix_area
        output_model.meta.photometry.pixelarea_arcsecsq = (
            pix_area * np.rad2deg(3600)**2
        )

        return output_model

    def build_output_model_name(self):
        fnames = {f for f in self._input_file_names if f is not None}

        if not fnames:
            return "resampled_data_{resample_suffix}{resample_file_ext}"

        # TODO: maybe remove ending suffix for single file names?
        prefix = os.path.commonprefix(
            [PurePath(f).stem.strip('_- ') for f in fnames]
        )

        return prefix + "{resample_suffix}{resample_file_ext}"

    def create_output_model(self, resample_results):
        # this probably needs to be an abstract class (different telescopes
        # may want to save different arrays and ignore others).

        if not self._output_model and self._output_filename:
            if self._accum and Path(self._output_filename).is_file():
                self._output_model = self.open_model(self._output_filename)
            else:
                self._output_model = self._create_new_output_model()
                self._close_output = not self.in_memory

        if self._output_filename is None:
            self._output_filename = self.build_output_model_name()

        self._output_model.data = resample_results.out_img

        self.update_exposure_times()
        self.finalize_resample()

        self._output_model.meta.resample.weight_type = self.weight_type
        self._output_model.meta.resample.pointings = len(self._input_models.group_names)
        # TODO: also store the number of images added in total: ncoadds?

        self.blend_output_metadata(self._output_model)

        self._output_model.write(self._output_filename, overwrite=True)

        if self._close_output and not self.in_memory:
            self.close_model(self._output_model)
            self._output_model = None

    def run(self):
        """Resample and coadd many inputs to a single output.

        Used for stage 3 resampling
        """

        # TODO: repetiveness of code below should be compactified via using
        # getattr as in orig code and maybe making an alternative method to
        # the original resample_variance_array
        ninputs = len(self._input_models)

        do_accum = (
            self._accum and
            (
                self._output_model or
                (self._output_filename and Path(self._output_filename).is_file())
            )
        )

        if do_accum and self._output_model is None:
            self._output_model = self.open_model(self._output_filename)

            # get old data:
            data = self._output_model.data  # use .copy()?
            wht = self._output_model.wht  # use .copy()?
            ctx = self._output_model.con  # use .copy()?
            t_exptime = self._output_model.meta.exptime
            # TODO: we need something to store total number of images that
            #       have been used to create the resampled output, something
            #       similar to output_model.meta.resample.pointings
            ncoadds = self._output_model.meta.resample.ncoadds  # ???? (not sure about name)
            self.accum_output_arrays = True

        else:
            ncoadds = 0
            data = None
            wht = None
            ctx = None
            t_exptime = 0.0
            self.accum_output_arrays = False

        driz_data = Drizzle(
            kernel=self.kernel,
            fillval=self.fillval,
            out_shape=self._output_array_shape,
            out_img=data,
            out_wht=wht,
            out_ctx=ctx,
            exptime=t_exptime,
            begin_ctx_id=ncoadds,
            max_ctx_id=ncoadds + ninputs,
        )

        self.extra_pre_resample_setup()

        log.info("Resampling science data")
        for img in self._input_models:
            input_pixflux_area = img.meta.photometry.pixelarea_steradians
            if (input_pixflux_area and
                    'SPECTRAL' not in img.meta.wcs.output_frame.axes_type):
                img.meta.wcs.array_shape = img.data.shape
                input_pixel_area = _compute_image_pixel_area(img.meta.wcs)
                if input_pixel_area is None:
                    raise ValueError(
                        "Unable to compute input pixel area from WCS of input "
                        f"image {repr(img.meta.filename)}."
                    )
                iscale = np.sqrt(input_pixflux_area / input_pixel_area)
            else:
                iscale = 1.0

            img.meta.iscale = iscale

            # TODO: should weight_type=None here?
            in_wht = self.build_driz_weight(
                img,
                weight_type=self.weight_type,
                good_bits=self.good_bits
            )

            # apply sky subtraction
            blevel = img.meta.background.level
            if not img.meta.background.subtracted and blevel is not None:
                in_data = img.data - blevel
            else:
                in_data = img.data

            xmin, xmax, ymin, ymax = _resample_range(
                in_data.shape,
                img.meta.wcs.bounding_box
            )

            pixmap = calc_pixmap(wcs_from=img.meta.wcs, wcs_to=self._output_wcs)

            add_image_kwargs = {
                'exptime': img.meta.exposure.exposure_time,
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

            driz_data.add_image(in_data, **add_image_kwargs)

            self.post_process_resample_model(img, None, add_image_kwargs)

        # TODO: see what to do about original update_exposure_times()

        return self.create_output_model(driz_data)


class ResampleSingle(ResampleBase):
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

    def __init__(self, input_models,
                 pixfrac=1.0, kernel="square", fillval=0.0, wht_type="ivm",
                 good_bits=0, output_wcs=None, wcs_pars=None,
                 in_memory=True, allowed_memory=None):
        """
        Parameters
        ----------
        input_models : list of objects
            list of data models, one for each input image

        output : DataModel, str
            filename for output

        kwargs : dict
            Other parameters.

            .. note::
                ``output_shape`` is in the ``x, y`` order.

            .. note::
                ``in_memory`` controls whether or not the resampled
                array from ``resample_many_to_many()``
                should be kept in memory or written out to disk and
                deleted from memory. Default value is `True` to keep
                all products in memory.

        """
        super().__init__(
            input_models,
            pixfrac=pixfrac,
            kernel=kernel,
            fillval=fillval,
            wht_type=wht_type,
            good_bits=good_bits,
            output_wcs=output_wcs,
            wcs_pars=wcs_pars,
            in_memory=in_memory,
            allowed_memory=allowed_memory,
        )

    def build_output_name_from_input_name(self, input_file_name):
        """ Form output file name from input image name """
        indx = input_file_name.rfind('.')
        output_type = input_file_name[indx:]
        output_root = '_'.join(
            input_file_name.replace(output_type, '').split('_')[:-1]
        )
        output_file_name = f'{output_root}_outlier_i2d{output_type}'
        return output_file_name

    def _create_output_template_model(self):
        # this probably needs to be an abstract class.
        # also this is mostly needed for "single" drizzle.
        self._template_output_model = self.new_model()
        self._template_output_model.update(self._first_model_meta)
        self._template_output_model.meta.wcs = deepcopy(self._output_wcs)

        pix_area = self._output_pixel_scale**2
        self._template_output_model.meta.photometry.pixelarea_steradians = pix_area
        self._template_output_model.meta.photometry.pixelarea_arcsecsq = (
            pix_area * np.rad2deg(3600)**2
        )

    def create_output_model_single(self, file_name, resample_results):
        # this probably needs to be an abstract class
        output_model = self._template_output_model.copy()
        output_model.data = resample_results.out_img
        if self.in_memory:
            return output_model
        else:
            output_model.write(file_name, overwrite=True)
            self.close_model(output_model.close)
            log.info(f"Saved resampled model to {file_name}")
            return file_name

    def resample(self):
        """Resample many inputs to many outputs where outputs have a common frame.

        Coadd only different detectors of the same exposure, i.e. map NRCA5 and
        NRCB5 onto the same output image, as they image different areas of the
        sky.

        Used for outlier detection
        """
        output_models = []  # ModelContainer()

        for exposure in self._input_models.models_grouped:
            driz = Drizzle(
                kernel=self.kernel,
                fillval=self.fillval,
                out_shape=self._output_array_shape,
                max_ctx_id=0
            )

            # Determine output file type from input exposure filenames
            # Use this for defining the output filename
            output_filename = self.resampled_output_name_from_input_name(
                exposure[0].meta.filename
            )

            log.info(f"{len(exposure)} exposures to drizzle together")

            exptime = None

            for img in exposure:
                img = self.open_model(img)
                if exptime is None:
                    exptime = exposure[0].meta.exposure.exposure_time

                # compute image intensity correction due to the difference
                # between where in the input image
                # img.meta.photometry.pixelarea_steradians was computed and
                # the average input pixel area.

                input_pixflux_area = img.meta.photometry.pixelarea_steradians
                if (input_pixflux_area and
                        'SPECTRAL' not in img.meta.wcs.output_frame.axes_type):
                    img.meta.wcs.array_shape = img.data.shape
                    input_pixel_area = _compute_image_pixel_area(img.meta.wcs)
                    if input_pixel_area is None:
                        raise ValueError(
                            "Unable to compute input pixel area from WCS of input "
                            f"image {repr(img.meta.filename)}."
                        )
                    iscale = np.sqrt(input_pixflux_area / input_pixel_area)
                else:
                    iscale = 1.0

                # TODO: should weight_type=None here?
                in_wht = self.build_driz_weight(
                    img,
                    weight_type=self.weight_type,
                    good_bits=self.good_bits
                )

                # apply sky subtraction
                blevel = img.meta.background.level
                if not img.meta.background.subtracted and blevel is not None:
                    data = img.data - blevel
                else:
                    data = img.data

                xmin, xmax, ymin, ymax = _resample_range(
                    data.shape,
                    img.meta.wcs.bounding_box
                )

                pixmap = calc_pixmap(wcs_from=img.meta.wcs, wcs_to=self._output_wcs)

                driz.add_image(
                    data,
                    exptime=exposure[0].meta.exposure.exposure_time,
                    pixmap=pixmap,
                    scale=iscale,
                    weight_map=in_wht,
                    wht_scale=1.0,
                    pixfrac=self.pixfrac,
                    in_units='cps',  # TODO: get units from data model
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                )

                del data
                self.close_model(img)

            output_models.append(
                self.create_output_model_single(
                    output_filename,
                    driz
                )
            )

        return output_models  # or maybe just a plain list - not ModelContainer?


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
