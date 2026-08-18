from types import SimpleNamespace

import pytest

from stcal.resample.resample import Resample


def bare_resampler():
    return object.__new__(Resample)


def test_create_output_model_requires_wcs():
    resampler = bare_resampler()
    resampler._output_wcs = None
    with pytest.raises(RuntimeError, match="Output WCS is not configured"):
        resampler.create_output_model()


def test_create_output_model_requires_consistent_shape():
    resampler = bare_resampler()
    resampler._output_wcs = SimpleNamespace(array_shape=(2, 2))
    resampler._output_array_shape = (3, 3)
    with pytest.raises(RuntimeError, match="array shape does not match"):
        resampler.create_output_model()


def test_create_output_model_requires_pixel_scale():
    resampler = bare_resampler()
    resampler._output_wcs = SimpleNamespace(array_shape=(2, 2))
    resampler._output_array_shape = (2, 2)
    resampler._output_pixel_scale = None
    with pytest.raises(RuntimeError, match="Output pixel scale is not configured"):
        resampler.create_output_model()


def test_validate_input_model_rejects_non_mapping():
    resampler = bare_resampler()
    with pytest.raises(TypeError, match="Input model must be a dictionary"):
        resampler.validate_input_model([])


def test_finalize_time_info_requires_resampled_models():
    resampler = bare_resampler()
    resampler._n_res_models = 0
    with pytest.raises(RuntimeError, match="before any models have been resampled"):
        resampler.finalize_time_info()
