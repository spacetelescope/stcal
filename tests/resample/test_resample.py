import pytest

from stcal.resample import Resample

import numpy as np

from . helpers import (
    make_gwcs,
    make_input_model,
    make_output_model,
    JWST_DQ_FLAG_DEF,
)


@pytest.mark.parametrize(
    "weight_type", ["ivm", "exptime"]
)
def test_resample_mostly_defaults(weight_type):
    crval = (150.0, 2.0)
    crpix = (500.0, 500.0)
    shape = (1000, 1000)
    pscale = 0.06 / 3600

    w = make_gwcs(
        crpix=(600, 600),
        crval=crval,
        pscale=pscale,
        shape=(1200, 1200)
    )
    output_wcs = {
        "wcs": w,
        "pixel_scale": pscale * 3600,
    }

    nmodels = 4

    resample = Resample(
        n_input_models=nmodels,
        output_wcs=output_wcs,
        weight_type=weight_type
    )
    resample.dq_flag_name_map = JWST_DQ_FLAG_DEF

    influx = 0.0
    ttime = 0.0

    for k in range(nmodels):
        exptime = k + 1
        im = make_input_model(
            shape=shape,
            crpix=tuple(i - 6 * k for i in crpix),
            crval=crval,
            pscale=pscale,
            group_id=k + 1,
            exptime=exptime
        )
        data_val = k + 0.5
        im["data"][:, :] = data_val

        influx += data_val * (exptime if weight_type == "exptime" else 1)
        ttime += exptime

        resample.add_model(im)

    resample.finalize()

    # test cannot add another model after finalize():
    with pytest.raises(RuntimeError):
        resample.add_model(im)

    odata = resample.output_model["data"]
    oweight = resample.output_model["wht"]

    assert resample.output_model["pointings"] == nmodels
    assert resample.output_model["exposure_time"] == ttime

    # next assert assumes constant IVM
    assert np.allclose(
        np.sum(odata * oweight, dtype=float),
        influx * np.prod(shape),
        atol=0,
        rtol=1e-6,
    )

    assert np.nansum(resample.output_model["var_flat"]) > 0.0
    assert np.nansum(resample.output_model["var_poisson"]) > 0.0
    assert np.nansum(resample.output_model["var_rnoise"]) > 0.0


@pytest.mark.parametrize(
    "compute_err,weight_type",
    [
        ("from_var", "ivm-smed"),
        ("driz_err", "ivm-med5")
    ]
)
def test_resample_compute_error_mode(compute_err, weight_type):
    crval = (150.0, 2.0)
    crpix = (500.0, 500.0)
    shape = (1000, 1000)
    out_shape = (1200, 1200)
    pscale = 0.06 / 3600

    nmodels = 4

    output_model = make_output_model(
        crpix=(600, 600),
        crval=crval,
        pscale=pscale,
        shape=out_shape,
    )
    output_model["wht"] = np.zeros(out_shape, dtype=np.float32)

    resample = Resample(
        n_input_models=nmodels,
        output_wcs=output_model,
        weight_type=weight_type,
        compute_err=compute_err,
    )
    resample.dq_flag_name_map = JWST_DQ_FLAG_DEF

    influx = 0.0
    ttime = 0.0

    for k in range(nmodels):
        exptime = k + 1
        im = make_input_model(
            shape=shape,
            crpix=tuple(i - 6 * k for i in crpix),
            crval=crval,
            pscale=pscale,
            group_id=k + 1,
            exptime=exptime
        )
        data_val = k + 0.5
        im["data"][:, :] = data_val

        influx += data_val
        ttime += exptime

        resample.add_model(im)

    resample.finalize()

    odata = resample.output_model["data"]
    oweight = resample.output_model["wht"]

    assert resample.output_model["pointings"] == nmodels
    assert resample.output_model["exposure_time"] == ttime

    # next assert assumes constant IVM
    assert np.allclose(
        np.sum(odata * oweight, dtype=float),
        influx * np.prod(shape),
        atol=0,
        rtol=1e-6,
    )

    assert np.nansum(resample.output_model["var_flat"]) > 0.0
    assert np.nansum(resample.output_model["var_poisson"]) > 0.0
    assert np.nansum(resample.output_model["var_rnoise"]) > 0.0
