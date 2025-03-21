import pytest

from drizzle.utils import calc_pixmap
from stcal.resample import Resample
from stcal.resample.utils import (
    build_driz_weight,
    resample_range,
)

import numpy as np

from . helpers import (
    make_gwcs,
    make_input_model,
    JWST_DQ_FLAG_DEF,
)


class _CustomResample(Resample):
    def add_model_hook(self, model, pixmap, iscale, weight_map, xmin, xmax, ymin, ymax):

        data = model["data"]
        wcs = model["wcs"]
        iscale_ref = self._get_intensity_scale(model)

        pixmap_ref = calc_pixmap(
            wcs,
            self.output_model["wcs"],
            data.shape,
        )

        weight_ref = build_driz_weight(
            model,
            weight_type=self.weight_type,
            good_bits=self.good_bits,
            flag_name_map=self.dq_flag_name_map
        )

        xyrange = resample_range(
            data.shape,
            wcs.bounding_box
        )

        assert np.allclose(pixmap_ref, pixmap)
        assert np.allclose(weight_ref, weight_map)
        assert np.allclose(xyrange, (xmin, xmax, ymin, ymax))
        assert abs(iscale - iscale_ref) < 1.0e-6

        raise RuntimeError("raised by subclass' add_model_hook")


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


def test_resample_add_model_hook():
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

    nmodels = 1

    resample = _CustomResample(
        n_input_models=nmodels,
        output_wcs=output_wcs,
        weight_type="exptime"
    )
    resample.dq_flag_name_map = JWST_DQ_FLAG_DEF

    im = make_input_model(
        shape=shape,
        crpix=crpix,
        crval=crval,
        pscale=pscale,
        group_id=1,
        exptime=10.0
    )

    with pytest.raises(RuntimeError, match="raised by subclass' add_model_hook") as err_info:
        resample.add_model(im)

    assert str(err_info.value) == "raised by subclass' add_model_hook"
