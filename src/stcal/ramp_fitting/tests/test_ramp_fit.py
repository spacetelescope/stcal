import numpy as np

from stcal.ramp_fitting.ramp_fit import ramp_fit_data
from .helper_functions import setup_inputs
from .helper_functions import dqflags


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
