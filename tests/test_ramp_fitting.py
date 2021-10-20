import numpy as np

from stcal.ramp_fitting.ramp_fit import ramp_fit_data
from stcal.ramp_fitting.ramp_fit_class import RampData

DELIM = "-" * 70

# single group intergrations fail in the GLS fitting
# so, keep the two method test separate and mark GLS test as
# expected to fail.  Needs fixing, but the fix is not clear
# to me. [KDG - 19 Dec 2018]

dqflags = {
    'DO_NOT_USE':       2**0,   # Bad pixel. Do not use.
    'SATURATED':        2**1,   # Pixel saturated during exposure
    'JUMP_DET':         2**2,   # Jump detected during exposure
    'NO_GAIN_VALUE':    2**19,  # Gain cannot be measured
    'UNRELIABLE_SLOPE': 2**24,  # Slope variance large (i.e., noisy pixel)
}


# -----------------------------------------------------------------------------
#                           Test Suite


def base_neg_med_rates_single_integration():
    """
    Ensures negative Poisson
    """
    nints, ngroups, nrows, ncols = 1, 10, 1,1
    rnoise_val, gain_val = 10., 1.
    nframes, gtime, dtime = 1, 1., 1
    dims = (nints, ngroups, nrows, ncols)
    var = (rnoise_val, gain_val)
    tm = (nframes, gtime, dtime)
    ramp_data, rnoise, gain = setup_inputs(dims, var, tm)

    # Set up negative ramp
    neg_ramp = np.array([k + 1 for k in range(ngroups)])
    nslope = -0.5
    neg_ramp = neg_ramp * nslope
    ramp_data.data[0, :, 0, 0] = neg_ramp

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    slopes, cube, optional, gls_dummy = ramp_fit_data(
        ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags)

    return slopes, cube, optional, gls_dummy


def test_neg_med_rates_single_integration_slope():
    """
    Make sure the negative ramp has negative slope, the Poisson variance
    is zero, readnoise is non-zero  and the ERR array is a function of
    only RNOISE.
    """
    slopes, cube, optional, gls_dummy = \
        base_neg_med_rates_single_integration()

    sdata, sdq, svp, svr, serr = slopes
    assert(sdata[0, 0] < 0.)
    assert(svp[0, 0] == 0.)
    assert(svr[0, 0] != 0.)
    assert(np.sqrt(svr[0, 0]) == serr[0, 0])


def test_neg_med_rates_single_integration_integ():
    """
    Make sure that for the single integration data the single integration
    is the same as the slope data.
    """
    slopes, cube, optional, gls_dummy = \
        base_neg_med_rates_single_integration()

    sdata, sdq, svp, svr, serr = slopes
    idata, idq, ivp, ivr, int_times, ierr = cube
    tol = 1e-6

    np.testing.assert_allclose(idata[0, :, :], sdata, tol)
    np.testing.assert_allclose(ivp[0, :, :], svp, tol)
    np.testing.assert_allclose(ivr[0, :, :], svr, tol)
    np.testing.assert_allclose(ierr[0, :, :], serr, tol)


def test_neg_med_rates_single_integration_optional():
    """
    Make sure that for the single integration data the optional results
    is the same as the slope data.
    """
    slopes, cube, optional, gls_dummy = \
        base_neg_med_rates_single_integration()

    sdata, sdq, svp, svr, serr = slopes
    oslope, osigslope, ovp, ovr, \
        oyint, osigyint, opedestal, oweights, ocrmag = optional

    tol = 1e-6
    assert(oslope.shape[1] == 1)  # Max segments is 1 because clean ramp
    np.testing.assert_allclose(oslope[0, 0, :, :], sdata, tol)
    np.testing.assert_allclose(ovp[0, 0, :, :], svp, tol)
    np.testing.assert_allclose(ovr[0, 0, :, :], svr, tol)


def base_neg_med_rates_multi_integrations():
    """
    Ensures negative Poisson
    """
    nints, ngroups, nrows, ncols = 3, 10, 1, 1
    rnoise_val, gain_val = 10., 1.
    nframes, gtime, dtime = 1, 1., 1
    dims = (nints, ngroups, nrows, ncols)
    var = (rnoise_val, gain_val)
    tm = (nframes, gtime, dtime)
    ramp_data, rnoise, gain = setup_inputs(dims, var, tm)

    # Set up negative ramp
    neg_ramp = np.array([k + 1 for k in range(ngroups)])
    nslope = -0.5
    neg_ramp = neg_ramp * nslope
    ramp_data.data[0, :, 0, 0] = neg_ramp
    for k in range(1, nints):
        n = k + 1
        ramp_data.data[k, :, 0, 0] = neg_ramp * n

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    slopes, cube, optional, gls_dummy = ramp_fit_data(
        ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags)

    return slopes, cube, optional, gls_dummy, dims


def test_neg_med_rates_multi_integrations_slopes():
    slopes, cube, optional, gls_dummy, dims = \
        base_neg_med_rates_multi_integrations()

    nints, ngroups, nrows, ncols = dims

    sdata, sdq, svp, svr, serr = slopes
    assert(sdata[0, 0] < 0.)
    assert(svp[0, 0] == 0.)
    assert(svr[0, 0] != 0.)
    assert(np.sqrt(svr[0, 0]) == serr[0, 0])


def test_neg_med_rates_multi_integration_integ():
    """
    Make sure that for the multi-integration data with a negative slope
    results in zero Poisson info and the ERR array a function of only
    RNOISE.
    """
    slopes, cube, optional, gls_dummy, dims = \
        base_neg_med_rates_multi_integrations()

    sdata, sdq, svp, svr, serr = slopes
    idata, idq, ivp, ivr, int_times, ierr = cube
    tol = 1e-6

    sdata, sdq, svp, svr, serr = slopes
    idata, idq, ivp, ivr, int_times, ierr = cube

    np.testing.assert_allclose(ivp[:, 0, 0], np.array([0., 0., 0.]), tol)
    np.testing.assert_allclose(ierr, np.sqrt(ivr), tol)


def test_neg_med_rates_multi_integration_optional():
    """
    """
    slopes, cube, optional, gls_dummy, dims = \
        base_neg_med_rates_multi_integrations()

    sdata, sdq, svp, svr, serr = slopes
    oslope, osigslope, ovp, ovr, \
        oyint, osigyint, opedestal, oweights, ocrmag = optional

    tol = 1e-6
    assert(oslope.shape[1] == 1)  # Max segments is 1 because clean ramp
    np.testing.assert_allclose(ovp[:, 0, 0, 0], np.zeros(3), tol)


def base_neg_med_rates_single_integration_multi_segment():
    """
    Ensures negative Poisson
    """
    nints, ngroups, nrows, ncols = 1, 15, 2, 1
    rnoise_val, gain_val = 10., 1.
    nframes, gtime, dtime = 1, 1., 1
    dims = (nints, ngroups, nrows, ncols)
    var = (rnoise_val, gain_val)
    tm = (nframes, gtime, dtime)
    ramp_data, rnoise, gain = setup_inputs(dims, var, tm)

    # Set up negative ramp
    neg_ramp = np.array([k + 1 for k in range(ngroups)])
    nslope = -0.5
    neg_ramp = neg_ramp * nslope
    ramp_data.data[0, :, 0, 0] = neg_ramp

    ramp_data.data[0, 5:, 1, 0] = ramp_data.data[0, 5:, 1, 0] + 50
    ramp_data.groupdq[0, 5, 1, 0] = dqflags["JUMP_DET"]
    ramp_data.data[0, 10:, 1, 0] = ramp_data.data[0, 10:, 1, 0] + 50
    ramp_data.groupdq[0, 10, 1, 0] = dqflags["JUMP_DET"]

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    slopes, cube, optional, gls_dummy = ramp_fit_data(
        ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags)

    return slopes, cube, optional, gls_dummy, dims


def test_neg_med_rates_single_integration_multi_segment_optional():
    """
    Test a ramp with multiple segments to make sure the right number of
    segments are created and to make sure all Poisson segements are set to
    zero.
    """
    slopes, cube, optional, gls_dummy, dims = \
        base_neg_med_rates_single_integration_multi_segment()

    oslope, osigslope, ovp, ovr, \
        oyint, osigyint, opedestal, oweights, ocrmag = optional

    neg_ramp_poisson = ovp[0, :, 0, 0]
    tol = 1e-6

    assert(ovp.shape[1] == 3)
    np.testing.assert_allclose(neg_ramp_poisson, np.zeros(3), tol)


# -----------------------------------------------------------------------------
#                           Set up functions

# Need test for multi-ints near zero with positive and negative slopes
def setup_inputs(dims, var, tm):
    nints, ngroups, nrows, ncols = dims
    rnoise, gain = var
    nframes, gtime, dtime = tm

    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    err = np.ones(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    pixdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint8)
    int_times = np.zeros((nints,))

    base_array = np.array([k + 1 for k in range(ngroups)])
    base, inc = 1.5, 1.5
    for row in range(nrows):
        for col in range(ncols):
            data[0, :, row, col] = base_array * base
            base = base + inc

    for c_int in range(1, nints):
        data[c_int, :, :, :] = data[0, :, :, :].copy()

    ramp_data = RampData()
    ramp_data.set_arrays(
        data=data, err=err, groupdq=gdq, pixeldq=pixdq, int_times=int_times)
    ramp_data.set_meta(
        name="MIRI", frame_time=dtime, group_time=gtime, groupgap=0,
        nframes=nframes, drop_frames1=None)
    ramp_data.set_dqflags(dqflags)

    gain = np.ones(shape=(nrows, ncols), dtype=np.float64) * gain
    rnoise = np.full((nrows, ncols), rnoise, dtype=np.float32)

    return ramp_data, rnoise, gain

# -----------------------------------------------------------------------------


# Main product
def print_slope_data(slopes):
    sdata, sdq, svp, svr, serr = slopes
    print("Slope Data:")
    print(sdata)


def print_slope_poisson(slopes):
    sdata, sdq, svp, svr, serr = slopes
    print("Poisson:")
    print(svp)


def print_slope_readnoise(slopes):
    sdata, sdq, svp, svr, serr = slopes
    print("Readnoise:")
    print(svr)


def print_slope_err(slopes):
    sdata, sdq, svp, svr, serr = slopes
    print("Err:")
    print(serr)


def print_slopes(slopes):
    print(DELIM)
    print("**** SLOPES")
    print(DELIM)
    print_slope_data(slopes)

    print(DELIM)
    print_slope_poisson(slopes)

    print(DELIM)
    print_slope_readnoise(slopes)

    print(DELIM)
    print_slope_err(slopes)

    print(DELIM)


def print_integ_data(integ_info):
    idata, idq, ivp, ivr, int_times, ierr = integ_info
    print("Integration data:")
    print(idata)


def print_integ_poisson(integ_info):
    idata, idq, ivp, ivr, int_times, ierr = integ_info
    print("Integration Poisson:")
    print(ivp)


def print_integ_rnoise(integ_info):
    idata, idq, ivp, ivr, int_times, ierr = integ_info
    print("Integration read noise:")
    print(ivr)


def print_integ_err(integ_info):
    idata, idq, ivp, ivr, int_times, ierr = integ_info
    print("Integration err:")
    print(ierr)


def print_integ(integ_info):
    print(DELIM)
    print("**** INTEGRATIONS")
    print(DELIM)
    print_integ_data(integ_info)

    print(DELIM)
    print_integ_poisson(integ_info)

    print(DELIM)
    print_integ_rnoise(integ_info)

    print(DELIM)
    print_integ_err(integ_info)

    print(DELIM)


def print_optional_data(optional):
    oslope, osigslope, ovar_poisson, ovar_rnoise, \
        oyint, osigyint, opedestal, oweights, ocrmag = optional
    print("Optional results slopes:")
    print(f"Dimensions: {oslope.shape}")
    print(oslope)


def print_optional_poisson(optional):
    oslope, osigslope, ovar_poisson, ovar_rnoise, \
        oyint, osigyint, opedestal, oweights, ocrmag = optional
    print("Optional results Poisson:")
    print(f"Dimensions: {ovar_poisson.shape}")
    print(ovar_poisson)


def print_optional_rnoise(optional):
    oslope, osigslope, ovar_poisson, ovar_rnoise, \
        oyint, osigyint, opedestal, oweights, ocrmag = optional
    print("Optional results read noise:")
    print(f"Dimensions: {ovar_rnoise.shape}")
    print(ovar_rnoise)


def print_optional(optional):
    print(DELIM)
    print("**** OPTIONAL RESULTS")
    print(DELIM)
    print_optional_data(optional)

    print(DELIM)
    print_optional_poisson(optional)

    print(DELIM)
    print_optional_rnoise(optional)

    print(DELIM)


def print_all_info(slopes, cube, optional):
    """
    sdata, sdq, svp, svr, serr = slopes
    idata, idq, ivp, ivr, int_times, ierr = cube
    oslope, osigslope, ovp, ovr, \
        oyint, osigyint, opedestal, oweights, ocrmag = optional
    """

    print(" ")
    print_slopes(slopes)
    print_integ(cube)
    print_optional(optional)
