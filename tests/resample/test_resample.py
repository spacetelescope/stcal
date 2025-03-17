import pytest

from drizzle.utils import calc_pixmap
from stcal.resample import Resample
from stcal.resample.utils import (
    build_driz_weight,
    resample_range,
)

import numpy as np
from drizzle.utils import calc_pixmap

from stcal.resample import Resample
from stcal.alignment.util import wcs_from_footprints

from . helpers import (
    make_gwcs,
    make_input_model,
    make_output_model,
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


@pytest.mark.parametrize(
    "compute_err", ["from_var", "driz_err"]
)
def test_resample_compute_error_mode(compute_err):
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
        weight_type="ivm",
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

        ttime += exptime
        influx += data_val

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

@pytest.mark.parametrize(
    "compute_err",
    [
        ("from_var"),
        ("driz_err")
    ]
)
def test_resample_error_scaling(compute_err):
    crval = (150.0, 2.0)
    crpix = (500.0, 500.0)
    shape = (1000, 1000)
    out_shape = (500, 500)
    weight_type = 'exptime'
    pscale_in = 0.1 / 3600 # Input pixel scale 0.1 arcsec
    pscale_out = 0.2 / 3600 # Output pixel scale 0.2 arcsec
    sb_in = 1.0 # Input surface brightness
    sb_err_in = 0.1 # Input surface brightness error

    output_model = make_output_model(
        crpix=(250, 250),
        crval=crval,
        pscale=pscale_out,
        shape=out_shape,
    )

    resample = Resample(
        n_input_models=1,
        output_wcs=output_model,
        weight_type=weight_type,
        compute_err=compute_err,
    )
    resample.dq_flag_name_map = JWST_DQ_FLAG_DEF

    im = make_input_model(
        shape=shape,
        crpix=tuple(i - 6 for i in crpix),
        crval=crval,
        pscale=pscale_in,
        group_id=1
    )
    im["data"][:, :] = sb_in
    im["err"][:, :] = sb_err_in
    im["var_poisson"][:, :] = sb_err_in ** 2
    im["var_rnoise"][:, :] = 0
    im["var_flat"][:, :] = 0
    resample.add_model(im)

    resample.finalize()

    odata = resample.output_model["data"]
    oerr = resample.output_model["err"]

    # Surface brightness should be unchanged with new pixel scale
    assert np.allclose(np.nanmedian(odata),
                       sb_in,
                       atol=0,
                       rtol=1e-6)
    # Surface brightness error should have scaled with pixel area
    assert np.allclose(np.nanmedian(oerr),
                       sb_err_in * (pscale_in / pscale_out),
                       atol=0,
                       rtol=1e-6)


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

@pytest.mark.parametrize("kernel", ["square", "turbo", "point"])
@pytest.mark.parametrize("pscale_ratio", [0.55, 1.0, 1.2])
def test_resample_photometry(nrcb5_many_fluxes, pscale_ratio, kernel):
    """ test surface-brightness photometry """
    model = nrcb5_many_fluxes

    wcs = model["wcs"]
    wcsinfo = model["wcsinfo"]
    stars = model["stars"]

    output_wcs = wcs_from_footprints(
        [wcs],
        wcs,
        wcsinfo,
        pscale_ratio=pscale_ratio
    )

    resample = Resample(
        n_input_models=1,
        output_wcs={"wcs": output_wcs},
        weight_type="exptime",
        enable_var=False,
        compute_err=False,
        fillval="NAN",
        kernel=kernel
    )
    resample.dq_flag_name_map = JWST_DQ_FLAG_DEF
    resample.add_model(model)
    resample.finalize()

    # for efficiency, instead of doing this patch-by-patch,
    # multiply resampled data by resampled image weight
    out_wht = resample.output_model['wht']
    out_data = resample.output_model["data"] * out_wht

    pixmap = calc_pixmap(
        wcs,
        output_wcs,
        model["data"].shape,
    )

    dim3 = (slice(None, None, None), )
    for _, _, fin, sl in stars:
        xyout = pixmap[sl + dim3]
        xmin = int(np.floor(xyout[:, :, 0].min() - 0.5))
        xmax = int(np.ceil(xyout[:, :, 0].max() + 1.5))
        ymin = int(np.floor(xyout[:, :, 1].min() - 0.5))
        ymax = int(np.ceil(xyout[:, :, 1].max() + 1.5))
        fout = np.nansum(out_data[ymin:ymax, xmin:xmax])

        assert np.allclose(fin, fout, rtol=1.0e-6, atol=0.0)
