from stcal.resample import Resample

import numpy as np

from . helpers import make_gwcs, make_input_model, JWST_DQ_FLAG_DEF


def test_resample_defaults():
    crval = (150.0, 2.0)
    crpix = (500.0, 500.0)
    shape = (1000, 1000)
    pscale = 0.06 / 3600

    output_wcs = make_gwcs(
        crpix=(600, 600),
        crval=crval,
        pscale=pscale,
        shape=(1200, 1200)
    )

    nmodels = 4

    resample = Resample(n_input_models=nmodels, output_wcs=output_wcs)
    resample.dq_flag_name_map = JWST_DQ_FLAG_DEF

    influx = 0.0
    ttime = 0.0

    for k in range(nmodels):
        im = make_input_model(
            crpix=tuple(i - 6 * k for i in crpix),
            crval=crval,
            pscale=pscale,
            shape=shape,
            group_id=k + 1,
            exptime=k + 1
        )
        im["data"][:, :] = k + 0.5
        influx += k + 0.5
        ttime += k + 1

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
