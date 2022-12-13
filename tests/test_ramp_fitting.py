import numpy as np

from stcal.ramp_fitting.ramp_fit import ramp_fit_data
from stcal.ramp_fitting.ramp_fit_class import RampData

DELIM = "-" * 70

# single group intergrations fail in the GLS fitting
# so, keep the two method test separate and mark GLS test as
# expected to fail.  Needs fixing, but the fix is not clear
# to me. [KDG - 19 Dec 2018]

dqflags = {
    'GOOD':             0,      # Good pixel.
    'DO_NOT_USE':       2**0,   # Bad pixel. Do not use.
    'SATURATED':        2**1,   # Pixel saturated during exposure.
    'JUMP_DET':         2**2,   # Jump detected during exposure.
    'NO_GAIN_VALUE':    2**19,  # Gain cannot be measured.
    'UNRELIABLE_SLOPE': 2**24,  # Slope variance large (i.e., noisy pixel).
}

GOOD = dqflags["GOOD"]
DNU = dqflags["DO_NOT_USE"]
SAT = dqflags["SATURATED"]
JUMP = dqflags["JUMP_DET"]


# -----------------------------------------------------------------------------
#                           Test Suite

def base_neg_med_rates_single_integration():
    """
    Creates single integration data for testing ensuring negative median rates.
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
    idata, idq, ivp, ivr, ierr = cube
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
    Creates multi-integration data for testing ensuring negative median rates.
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
    """
    Test computing median rates of a ramp with multiple integrations.
    """
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
    idata, idq, ivp, ivr, ierr = cube
    tol = 1e-6

    sdata, sdq, svp, svr, serr = slopes
    idata, idq, ivp, ivr, ierr = cube

    np.testing.assert_allclose(ivp[:, 0, 0], np.array([0., 0., 0.]), tol)
    np.testing.assert_allclose(ierr, np.sqrt(ivr), tol)


def test_neg_med_rates_multi_integration_optional():
    """
    Make sure that for the multi-integration data with a negative slope with
    one segment has only one segment in the optional results product as well
    as zero Poisson variance.
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
    Creates single integration, multi-segment data for testing ensuring
    negative median rates.
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


def test_utils_dq_compress_final():
    """
    If there is any integration that has usable data, the DO_NOT_USE flag
    should not be set in the final DQ flag, even if it is set for one or more
    integrations.

    Set up a multi-integration 3 pixel data array each ramp as the following:
    1. Both integrations having all groups saturated.
        - Since all groups are saturated in all integrations the final DQ value
          for this pixel should have the DO_NOT_USE flag set.  Ramp fitting
          will flag a pixel as DO_NOT_USE in an integration if all groups in
          that integration are saturated.
    2. Only one integration with all groups saturated.
        - Since all groups are saturated in only one integration the final DQ
          value for this pixel should not have the DO_NOT_USE flag set, even
          though it is set in one of the integrations.
    3. No group saturated in any integration.
        - This is a "normal" pixel where there is usable information in both
          integrations.  Neither integration should have the DO_NOT_SET flag
          set, nor should it be set in the final DQ.
    """
    nints, ngroups, nrows, ncols = 2, 5, 1, 3
    rnoise_val, gain_val = 10., 1.
    nframes, gtime, dtime = 1, 1., 1
    dims = (nints, ngroups, nrows, ncols)
    var = (rnoise_val, gain_val)
    tm = (nframes, gtime, dtime)
    ramp_data, rnoise, gain = setup_inputs(dims, var, tm)

    ramp_data.groupdq[0, :, 0, 0] = np.array([dqflags["SATURATED"]] * ngroups)
    ramp_data.groupdq[1, :, 0, 0] = np.array([dqflags["SATURATED"]] * ngroups)

    ramp_data.groupdq[0, :, 0, 1] = np.array([dqflags["SATURATED"]] * ngroups)

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    slopes, cube, optional, gls_dummy = ramp_fit_data(
        ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags)

    dq = slopes[1]
    idq = cube[1]

    # Make sure DO_NOT_USE is set in the expected integrations.
    assert(idq[0, 0, 0] & dqflags["DO_NOT_USE"])
    assert(idq[1, 0, 0] & dqflags["DO_NOT_USE"])

    assert(idq[0, 0, 1] & dqflags["DO_NOT_USE"])
    assert(not (idq[1, 0, 1] & dqflags["DO_NOT_USE"]))

    assert(not (idq[0, 0, 2] & dqflags["DO_NOT_USE"]))
    assert(not (idq[1, 0, 2] & dqflags["DO_NOT_USE"]))

    # Make sure DO_NOT_USE is set in the expected final DQ.
    assert(dq[0, 0] & dqflags["DO_NOT_USE"])
    assert(not(dq[0, 1] & dqflags["DO_NOT_USE"]))
    assert(not(dq[0, 2] & dqflags["DO_NOT_USE"]))


def jp_2326_test_setup():
    """
    Sets up data for MIRI testing DO_NOT_USE flags at the beginning of ramps.
    """
    # Set up ramp data
    ramp = np.array([120.133545, 117.85222, 87.38832, 66.90588,  51.392555,
                     41.65941,   32.15081,  24.25277, 15.955284, 9.500946])
    dnu = dqflags["DO_NOT_USE"]
    dq = np.array([dnu, 0, 0, 0, 0, 0, 0, 0, 0, dnu])

    nints, ngroups, nrows, ncols = 1, len(ramp), 1, 1
    data = np.zeros((nints, ngroups, nrows, ncols))
    gdq = np.zeros((nints, ngroups, nrows, ncols), dtype=np.uint8)
    err = np.zeros((nints, ngroups, nrows, ncols))
    pdq = np.zeros((nrows, ncols), dtype=np.uint32)

    data[0, :, 0, 0] = ramp.copy()
    gdq[0, :, 0, 0] = dq.copy()

    ramp_data = RampData()
    ramp_data.set_arrays(
        data=data, err=err, groupdq=gdq, pixeldq=pdq)
    ramp_data.set_meta(
        name="MIRI", frame_time=2.77504, group_time=2.77504, groupgap=0,
        nframes=1, drop_frames1=None)
    ramp_data.set_dqflags(dqflags)

    # Set up gain and read noise
    gain = np.ones(shape=(nrows, ncols), dtype=np.float32) * 5.5
    rnoise = np.ones(shape=(nrows, ncols), dtype=np.float32) * 1000.

    return ramp_data, gain, rnoise


def test_miri_ramp_dnu_at_ramp_beginning():
    """
    Tests a MIRI ramp with DO_NOT_USE in the first two groups and last group.
    This test ensures these groups are properly excluded.
    """
    ramp_data, gain, rnoise = jp_2326_test_setup()
    ramp_data.groupdq[0, 1, 0, 0] = dqflags["DO_NOT_USE"]

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    slopes1, cube, optional, gls_dummy = ramp_fit_data(
        ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags)

    s1 = slopes1[0]
    tol = 1e-6
    ans = -4.1035075

    assert abs(s1[0, 0] - ans) < tol


def test_miri_ramp_dnu_and_jump_at_ramp_beginning():
    """
    Tests a MIRI ramp with DO_NOT_USE in the first and last group, with a
    JUMP_DET in the second group. This test ensures the DO_NOT_USE groups are
    properly excluded, while the JUMP_DET group is included.
    """
    ramp_data, gain, rnoise = jp_2326_test_setup()
    ramp_data.groupdq[0, 1, 0, 0] = dqflags["JUMP_DET"]

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    slopes2, cube, optional, gls_dummy = ramp_fit_data(
        ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags)

    s2 = slopes2[0]
    tol = 1e-6
    ans = -4.9032097

    assert abs(s2[0, 0] - ans) < tol


def test_2_group_cases():
    """
    Tests the special cases of 2 group ramps.  Create multiple pixel ramps
    with two groups to test the various DQ cases.
    """
    base_group = [-12328.601, -4289.051]
    base_err = [0., 0.]
    gain_val = 0.9699
    rnoise_val = 9.4552

    possibilities = [
        # Both groups are good
        [GOOD, GOOD],

        # Both groups are bad.  Note saturated 0th group kills group 1.
        [SAT, GOOD],
        [DNU | SAT, GOOD],
        [DNU, SAT],

        # One group is bad, while the other group is good.
        [DNU, GOOD],
        [GOOD, DNU],
        [GOOD, DNU | SAT],
    ]
    nints, ngroups, nrows, ncols = 1, 2, 1, len(possibilities)
    dims = nints, ngroups, nrows, ncols
    npix = nrows * ncols

    # Create the ramp data with a pixel for each of the possibilities above.
    # Set the data to the base data of 2 groups and set up the group DQ flags
    # are taken from the 'possibilities' list above.

    # Resize gain and read noise arrays.
    rnoise = np.ones((1, npix)) * rnoise_val
    gain = np.ones((1, npix)) * gain_val
    pixeldq = np.zeros((1, npix), dtype=np.uint32)

    data = np.zeros(dims, dtype=np.float32)  # Science data
    for k in range(npix):
        data[0, :, 0, k] = np.array(base_group)

    err = np.zeros(dims, dtype=np.float32)  # Error data
    for k in range(npix):
        err[0, :, 0, k] = np.array(base_err)

    groupdq = np.zeros(dims, dtype=np.uint8)  # Group DQ
    for k in range(npix):
        groupdq[0, :, 0, k] = np.array(possibilities[k])

    # Setup the RampData class to run ramp fitting on.
    ramp_data = RampData()

    ramp_data.set_arrays(data, err, groupdq, pixeldq)

    ramp_data.set_meta(
        name="NIRSPEC",
        frame_time=14.58889,
        group_time=14.58889,
        groupgap=0,
        nframes=1,
        drop_frames1=None)

    ramp_data.set_dqflags(dqflags)

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    slopes, cube, optional, gls_dummy = ramp_fit_data(
        ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags)

    # Check the outputs
    data, dq, var_poisson, var_rnoise, err = slopes
    chk_dt = np.array([[551.0735, np.nan, np.nan, np.nan, -293.9943, -845.0678, -845.0677]])
    chk_dq = np.array([[GOOD, DNU | SAT, DNU | SAT, DNU, GOOD, GOOD, GOOD]])
    chk_vp = np.array([[38.945766, 0., 0., 0., 38.945766, 38.945766, 0.]])
    chk_vr = np.array([[0.420046, 0.420046, 0.420046, 0., 0.420046, 0.420046, 0.420046]])
    chk_er = np.array([[6.274218, 0.64811, 0.64811, 0., 6.274218, 6.274218, 0.64811]])

    tol = 1.e-6
    np.testing.assert_allclose(data, chk_dt, tol)
    np.testing.assert_allclose(dq, chk_dq, tol)
    np.testing.assert_allclose(var_poisson, chk_vp, tol)
    np.testing.assert_allclose(var_rnoise, chk_vr, tol)
    np.testing.assert_allclose(err, chk_er, tol)


def run_one_group_ramp_suppression(nints, suppress):
    """
    Forms the base of the one group suppression tests.  Create three ramps
    using three pixels with two integrations.  In the first integration:
        The first ramp has no good groups.
        The second ramp has one good groups.
        The third ramp has all good groups.

    In the second integration all pixels have all good groups.
    """
    # Define the data.
    ngroups, nrows, ncols = 5, 1, 3
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    nframes, group_time, frame_time = 1, 5.0, 1
    var = rnoise, gain
    tm = nframes, group_time, frame_time

    # Using the above create the classes and arrays.
    ramp_data, rnoise2d, gain2d = setup_inputs(dims, var, tm)

    arr = np.array([k + 1 for k in range(ngroups)], dtype=float)

    sat = ramp_data.flags_saturated
    sat_dq = np.array([sat] * ngroups, dtype=ramp_data.groupdq.dtype)
    zdq = np.array([0] * ngroups, dtype=ramp_data.groupdq.dtype)

    ramp_data.data[0, :, 0, 0] = arr
    ramp_data.data[0, :, 0, 1] = arr
    ramp_data.data[0, :, 0, 2] = arr

    ramp_data.groupdq[0, :, 0, 0] = sat_dq  # All groups sat
    ramp_data.groupdq[0, :, 0, 1] = sat_dq  # 0th good, all others sat
    ramp_data.groupdq[0, 0, 0, 1] = 0
    ramp_data.groupdq[0, :, 0, 2] = zdq     # All groups good

    if nints > 1:
        ramp_data.data[1, :, 0, 0] = arr
        ramp_data.data[1, :, 0, 1] = arr
        ramp_data.data[1, :, 0, 2] = arr

        # All good ramps
        ramp_data.groupdq[1, :, 0, 0] = zdq
        ramp_data.groupdq[1, :, 0, 1] = zdq
        ramp_data.groupdq[1, :, 0, 2] = zdq

    ramp_data.suppress_one_group_ramps = suppress

    algo = "OLS"
    save_opt, ncores, bufsize = False, "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, dqflags)

    return slopes, cube, dims


def test_one_group_ramp_suppressed_one_integration():
    """
    Tests one group ramp fitting where suppression turned on.
    """
    slopes, cube, dims = run_one_group_ramp_suppression(1, True)
    nints, ngroups, nrows, ncols = dims
    tol = 1e-5

    # Check slopes information
    sdata, sdq, svp, svr, serr = slopes

    check = np.array([[np.nan, np.nan, 1.0000002]])
    np.testing.assert_allclose(sdata, check, tol)

    check = np.array([[DNU | SAT, DNU, GOOD]])
    np.testing.assert_allclose(sdq, check, tol)

    check = np.array([[0., 0., 0.01]])
    np.testing.assert_allclose(svp, check, tol)

    check = np.array([[0., 0., 0.19999999]])
    np.testing.assert_allclose(svr, check, tol)

    check = np.array([[0., 0., 0.45825756]])
    np.testing.assert_allclose(serr, check, tol)

    # Check slopes information
    cdata, cdq, cvp, cvr, cerr = cube

    check = np.array([[[np.nan, np.nan, 1.0000001]]])
    np.testing.assert_allclose(cdata, check, tol)

    check = np.array([[[DNU | SAT, DNU, GOOD]]])
    np.testing.assert_allclose(cdq, check, tol)

    check = np.array([[[0., 0., 0.01]]])
    np.testing.assert_allclose(cvp, check, tol)

    check = np.array([[[0., 0., 0.19999999]]])
    np.testing.assert_allclose(cvr, check, tol)

    check = np.array([[[0., 0., 0.4582576]]])
    np.testing.assert_allclose(cerr, check, tol)


def test_one_group_ramp_not_suppressed_one_integration():
    """
    Tests one group ramp fitting where suppression turned off.
    """
    slopes, cube, dims = run_one_group_ramp_suppression(1, False)
    nints, ngroups, nrows, ncols = dims
    tol = 1e-5

    # Check slopes information
    sdata, sdq, svp, svr, serr = slopes

    check = np.array([[np.nan, 1., 1.0000002]])
    np.testing.assert_allclose(sdata, check, tol)

    check = np.array([[DNU | SAT, GOOD, GOOD]])
    np.testing.assert_allclose(sdq, check, tol)

    check = np.array([[0., 0.04, 0.01]])
    np.testing.assert_allclose(svp, check, tol)

    check = np.array([[0., 3.9999995, 0.19999999]])
    np.testing.assert_allclose(svr, check, tol)

    check = np.array([[0., 2.009975, 0.45825756]])
    np.testing.assert_allclose(serr, check, tol)

    # Check slopes information
    cdata, cdq, cvp, cvr, cerr = cube

    check = np.array([[[np.nan, 1., 1.0000001]]])
    np.testing.assert_allclose(cdata, check, tol)

    check = np.array([[[DNU | SAT, GOOD, GOOD]]])
    np.testing.assert_allclose(cdq, check, tol)

    check = np.array([[[0., 0.04, 0.01]]])
    np.testing.assert_allclose(cvp, check, tol)

    check = np.array([[[0., 3.9999995, 0.19999999]]])
    np.testing.assert_allclose(cvr, check, tol)

    check = np.array([[[0., 2.0099752, 0.4582576]]])
    np.testing.assert_allclose(cerr, check, tol)


def test_one_group_ramp_suppressed_two_integrations():
    """
    Test one good group ramp and two integrations with
    suppression suppression turned on.
    """
    slopes, cube, dims = run_one_group_ramp_suppression(2, True)
    nints, ngroups, nrows, ncols = dims
    tol = 1e-5

    # Check slopes information
    sdata, sdq, svp, svr, serr = slopes

    check = np.array([[1.0000001, 1.0000001, 1.0000002]])
    np.testing.assert_allclose(sdata, check, tol)

    check = np.array([[GOOD, GOOD, GOOD]])
    np.testing.assert_allclose(sdq, check, tol)

    check = np.array([[0.005, 0.01, 0.005]])
    np.testing.assert_allclose(svp, check, tol)

    check = np.array([[0.19999999, 0.19999999, 0.09999999]])
    np.testing.assert_allclose(svr, check, tol)

    check = np.array([[0.45276925, 0.45825756, 0.32403702]])
    np.testing.assert_allclose(serr, check, tol)

    # Check slopes information
    cdata, cdq, cvp, cvr, cerr = cube

    check = np.array([[[np.nan,    np.nan,    1.0000001]],
                      [[1.0000001, 1.0000001, 1.0000001]]])
    np.testing.assert_allclose(cdata, check, tol)

    check = np.array([[[DNU | SAT, DNU, GOOD]],
                      [[GOOD, GOOD, GOOD]]])
    np.testing.assert_allclose(cdq, check, tol)

    check = np.array([[[0.,    0.,    0.01]],
                      [[0.005, 0.01, 0.01]]])
    np.testing.assert_allclose(cvp, check, tol)

    check = np.array([[[0.,         0.,         0.19999999]],
                      [[0.19999999, 0.19999999, 0.19999999]]])
    np.testing.assert_allclose(cvr, check, tol)

    check = np.array([[[0.,         0.,         0.4582576]],
                      [[0.45276922, 0.4582576, 0.4582576]]])
    np.testing.assert_allclose(cerr, check, tol)


def test_one_group_ramp_not_suppressed_two_integrations():
    """
    Test one good group ramp and two integrations with
    suppression suppression turned off.
    """
    slopes, cube, dims = run_one_group_ramp_suppression(2, False)
    nints, ngroups, nrows, ncols = dims
    tol = 1e-5

    # Check slopes information
    sdata, sdq, svp, svr, serr = slopes

    check = np.array([[1.0000001, 1.0000002, 1.0000002]])
    np.testing.assert_allclose(sdata, check, tol)

    check = np.array([[GOOD, GOOD, GOOD]])
    np.testing.assert_allclose(sdq, check, tol)

    check = np.array([[0.005, 0.008, 0.005]])
    np.testing.assert_allclose(svp, check, tol)

    check = np.array([[0.19999999, 0.19047618, 0.09999999]])
    np.testing.assert_allclose(svr, check, tol)

    check = np.array([[0.45276925, 0.44550666, 0.32403702]])
    np.testing.assert_allclose(serr, check, tol)

    # Check slopes information
    cdata, cdq, cvp, cvr, cerr = cube

    check = np.array([[[np.nan,    1.,        1.0000001]],
                      [[1.0000001, 1.0000001, 1.0000001]]])
    np.testing.assert_allclose(cdata, check, tol)

    check = np.array([[[DNU | SAT, GOOD, GOOD]],
                      [[GOOD, GOOD, GOOD]]])
    np.testing.assert_allclose(cdq, check, tol)

    check = np.array([[[0.,    0.04, 0.01]],
                      [[0.005, 0.01, 0.01]]])
    np.testing.assert_allclose(cvp, check, tol)

    check = np.array([[[0.,         3.9999995,  0.19999999]],
                      [[0.19999999, 0.19999999, 0.19999999]]])
    np.testing.assert_allclose(cvr, check, tol)

    check = np.array([[[0.,         2.0099752, 0.4582576]],
                      [[0.45276922, 0.4582576, 0.4582576]]])
    np.testing.assert_allclose(cerr, check, tol)


def create_zero_frame_data():
    """
    A two integration three pixel image.

    The first integration:
    1. Good first group with the remainder of the ramp being saturated.
    2. Saturated ramp.
    3. Saturated ramp with ZEROFRAME data used for the first group.

    The second integration has all good groups with half the data values.
    """
    # Create meta data.
    frame_time, nframes, groupgap = 10.736, 4, 1
    group_time = (nframes + groupgap) * frame_time
    nints, ngroups, nrows, ncols = 2, 5, 1, 3
    rnval, gval = 10., 5.

    # Create arrays for RampData.
    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    err = np.ones(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    pixdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint8)
    zframe = np.ones(shape=(nints, nrows, ncols), dtype=np.float32)

    # Create base ramps for each pixel in each integration.
    base_slope = 2000.0
    base_arr = [8000. + k * base_slope for k in range(ngroups)]
    base_ramp = np.array(base_arr, dtype=np.float32)

    data[0, :, 0, 0] = base_ramp
    data[0, :, 0, 1] = base_ramp
    data[0, :, 0, 2] = base_ramp
    data[1, :, :, :] = data[0, :, :, :] / 2.

    # ZEROFRAME data.
    fdn = (data[0, 1, 0, 0] - data[0, 0, 0, 0]) / (nframes + groupgap)
    dummy = data[0, 0, 0, 2] - (fdn * 2.5)
    zframe[0, 0, :] *= dummy
    zframe[0, 0, 1] = 0.  # ZEROFRAME is saturated too.
    fdn = (data[1, 1, 0, 0] - data[1, 0, 0, 0]) / (nframes + groupgap)
    dummy = data[1, 0, 0, 2] - (fdn * 2.5)
    zframe[1, 0, :] *= dummy

    # Set up group DQ array.
    gdq[0, :, :, :] = dqflags["SATURATED"]
    gdq[0, 0, 0, 0] = dqflags["GOOD"]

    # Create RampData for testing.
    ramp_data = RampData()
    ramp_data.set_arrays(
        data=data, err=err, groupdq=gdq, pixeldq=pixdq)
    ramp_data.set_meta(
        name="NIRCam", frame_time=frame_time, group_time=group_time,
        groupgap=groupgap, nframes=nframes, drop_frames1=None)
    ramp_data.set_dqflags(dqflags)

    ramp_data.suppress_one_group_ramps = False
    ramp_data.zeroframe = zframe

    # Create variance arrays
    gain = np.ones((nrows, ncols), np.float32) * gval
    rnoise = np.ones((nrows, ncols), np.float32) * rnval

    return ramp_data, gain, rnoise


def test_zeroframe():
    """
    A two integration three pixel image.

    The first integration:
    1. Good first group with the remainder of the ramp being saturated.
    2. Saturated ramp.
    3. Saturated ramp with ZEROFRAME data used for the first group.

    The second integration has all good groups with half the data values.
    """
    ramp_data, gain, rnoise = create_zero_frame_data()

    algo, save_opt, ncores, bufsize = "OLS", False, "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo,
        "optimal", ncores, dqflags)

    tol = 1.e-5

    # Check slopes information
    sdata, sdq, svp, svr, serr = slopes

    check = np.array([[44.256306, 18.62891, 23.787909]])
    np.testing.assert_allclose(sdata, check, tol, tol)

    check = np.array([[GOOD, GOOD, GOOD]])
    np.testing.assert_allclose(sdq, check, tol, tol)

    check = np.array([[0.06246654, 0.00867591, 0.29745975]])
    np.testing.assert_allclose(svp, check, tol, tol)

    check = np.array([[0.00041314, 0.0004338, 0.00043293]])
    np.testing.assert_allclose(svr, check, tol, tol)

    check = np.array([[0.2507582, 0.09544477, 0.54579544]])
    np.testing.assert_allclose(serr, check, tol, tol)

    # Check slopes information
    cdata, cdq, cvp, cvr, cerr = cube

    # The third pixel in integration zero has good data
    # because the zeroframe has good data, so the ramp
    # is not fully saturated.
    check = np.array([[[149.0313, np.nan, 130.40239]],
                      [[18.62891, 18.62891, 18.62891]]])
    np.testing.assert_allclose(cdata, check, tol, tol)

    check = np.array([[[GOOD, DNU | SAT, GOOD]],
                      [[GOOD, GOOD, GOOD]]])
    np.testing.assert_allclose(cdq, check, tol, tol)

    check = np.array([[[0.31233272, 0., 6.246655]],
                      [[0.07808318, 0.00867591, 0.31233275]]])
    np.testing.assert_allclose(cvp, check, tol, tol)

    check = np.array([[[0.00867591, 0., 0.21689774]],
                      [[0.0004338, 0.0004338, 0.0004338]]])
    np.testing.assert_allclose(cvr, check, tol, tol)

    check = np.array([[[0.56657624, 0., 2.542352]],
                      [[0.2802088, 0.09544477, 0.55925536]]])
    np.testing.assert_allclose(cerr, check, tol, tol)


def test_all_sat():
    """
    Test all ramps in all integrations saturated.
    """
    nints, ngroups, nrows, ncols = 2, 5, 1, 3
    rnval, gval = 10., 5.
    frame_time, nframes, groupgap = 10.736, 4, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp, gain, rnoise = create_blank_ramp_data(dims, var, tm)
    ramp.groupdq[:, 0, :, :] = ramp.flags_saturated

    algo, save_opt, ncores, bufsize = "OLS", False, "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp, bufsize, save_opt, rnoise, gain, algo,
        "optimal", ncores, dqflags)

    assert slopes is None
    assert cube is None


def test_dq_multi_int_dnu():
    """
    Tests to make sure that integration DQ flags get set when all groups
    in an integration are set to DO_NOT_USE.
    """
    nints, ngroups, nrows, ncols = 2, 5, 1, 1
    rnval, gval = 10., 5.
    frame_time, nframes, groupgap = 10.736, 4, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp, gain, rnoise = create_blank_ramp_data(dims, var, tm)
    base_arr = [(k + 1) * 100 for k in range(ngroups)]
    dq_arr = [ramp.flags_do_not_use] * ngroups

    ramp.data[0, :, 0, 0] = np.array(base_arr)
    ramp.data[1, :, 0, 0] = np.array(base_arr)
    ramp.groupdq[0, :, 0, 0] = np.array(dq_arr)

    algo, save_opt, ncores, bufsize = "OLS", False, "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp, bufsize, save_opt, rnoise, gain, algo,
        "optimal", ncores, dqflags)

    tol = 1.e-5

    # Check slopes information
    sdata, sdq, svp, svr, serr = slopes

    check = np.array([[1.8628913]])
    np.testing.assert_allclose(sdata, check, tol, tol)

    check = np.array([[0]])
    np.testing.assert_allclose(sdq, check, tol, tol)

    check = np.array([[0.00173518]])
    np.testing.assert_allclose(svp, check, tol, tol)

    check = np.array([[0.0004338]])
    np.testing.assert_allclose(svr, check, tol, tol)

    check = np.array([[0.04657228]])
    np.testing.assert_allclose(serr, check, tol, tol)

    # Check slopes information
    cdata, cdq, cvp, cvr, cerr = cube

    check = np.array([[[np.nan]],
                      [[1.8628913]]])
    np.testing.assert_allclose(cdata, check, tol, tol)

    check = np.array([[[dqflags["DO_NOT_USE"]]],
                      [[0]]])
    np.testing.assert_allclose(cdq, check, tol, tol)

    check = np.array([[[0.]],
                      [[0.00173518]]])
    np.testing.assert_allclose(cvp, check, tol, tol)

    check = np.array([[[2.5000000e+07]],
                      [[4.3379547e-04]]])
    np.testing.assert_allclose(cvr, check, tol, tol)

    check = np.array([[[0.]],
                      [[0.04657228]]])
    np.testing.assert_allclose(cerr, check, tol, tol)


def get_new_saturation():
    """
    Three columns (pixels) with two integrations each.
    1. One integ good, one partially saturated.
    2. One integ partially saturated, one fully saturated.
    2. Both integrations fully saturated.
    """
    nints, ngroups, nrows, ncols = 2, 20, 1, 3
    rnval, gval = 10., 5.
    frame_time, nframes, groupgap = 10.736, 4, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp, gain, rnoise = create_blank_ramp_data(dims, var, tm)

    bramp = [ 149.3061,  299.0544,  449.9949,  599.7617,  749.7327,
              900.117 , 1049.314 , 1200.6003, 1350.0906, 1500.7772,
             1649.3098, 1799.8952, 1949.1304, 2100.1875, 2249.85  ,
             2399.1154, 2550.537 , 2699.915 , 2850.0734, 2999.7891]

    # Set up ramp data.
    for integ in range(nints):
        for col in range(ncols):
            ramp.data[integ, :, 0, col] = np.array(bramp)

    #                    Set up DQ's.
    # Set up col 0
    # One integ no sat, one with jump and saturated
    dq = [GOOD, GOOD, GOOD, GOOD, GOOD, GOOD, GOOD, GOOD, GOOD, GOOD,
          GOOD, GOOD, GOOD, GOOD, GOOD, GOOD, GOOD, GOOD, GOOD, GOOD]
    ramp.groupdq[0, :, 0, 0] = np.array(dq)
    dq = [GOOD, GOOD, GOOD, GOOD, GOOD, JUMP, JUMP, GOOD, GOOD, GOOD,
          GOOD, GOOD, GOOD, GOOD, GOOD, SAT,  SAT,  SAT,  SAT,  SAT]
    ramp.groupdq[1, :, 0, 0] = np.array(dq)

    # Set up col 1
    # One integ with jump and saturated, one fully saturated
    dq = [GOOD, GOOD, GOOD, GOOD, GOOD, JUMP, JUMP, GOOD, GOOD, GOOD,
          GOOD, GOOD, GOOD, GOOD, GOOD, SAT,  SAT,  SAT,  SAT,  SAT]
    ramp.groupdq[0, :, 0, 1] = np.array(dq)
    dq = [SAT, SAT, SAT, SAT, SAT, SAT, SAT, SAT, SAT, SAT,
          SAT, SAT, SAT, SAT, SAT, SAT, SAT, SAT, SAT, SAT]
    ramp.groupdq[1, :, 0, 1] = np.array(dq)

    # Set up col 2
    # One integ fully saturated
    ramp.groupdq[0, :, 0, 2] = np.array(dq)
    ramp.groupdq[1, :, 0, 2] = np.array(dq)

    return ramp, gain, rnoise


def test_new_saturation():
    """
    Test the updated saturation flag setting implemented
    in JP-2988.  Integration level saturation is now only
    set if all groups in the integration are saturated.
    The saturation flag is set for the pixel only if all
    integrations are saturated.  If the pixel is flagged
    as saturated, then it must also be marked as do not
    use.
    """
    ramp, gain, rnoise = get_new_saturation()

    save_opt, ncores, bufsize, algo = False, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp, bufsize, save_opt, rnoise, gain, algo,"optimal", ncores, dqflags)

    tol = 1.e-5

    # Check slopes information
    sdata, sdq, svp, svr, serr = slopes

    check = np.array([[2.797567 , 2.8022935, np.nan]])
    np.testing.assert_allclose(sdata, check, tol, tol)

    check = np.array([[JUMP, JUMP, DNU | SAT]])
    np.testing.assert_allclose(sdq, check, tol, tol)

    check = np.array([[0.00033543, 0.00043342, 0.]])
    np.testing.assert_allclose(svp, check, tol, tol)

    check = np.array([[5.9019785e-06, 6.1970772e-05, 0.0000000e+00]])
    np.testing.assert_allclose(svr, check, tol, tol)

    check = np.array([[0.01847528, 0.02225729, 0.]])
    np.testing.assert_allclose(serr, check, tol, tol)

    # Check slopes information
    cdata, cdq, cvp, cvr, cerr = cube

    check = np.array([[[2.7949152, 2.8022935, np.nan]],
                      [[2.8020892, np.nan, np.nan]]])
    np.testing.assert_allclose(cdata, check, tol, tol)

    check = np.array([[[GOOD, JUMP, DNU | SAT]],
                      [[JUMP, DNU | SAT, DNU | SAT]]])
    np.testing.assert_allclose(cdq, check, tol, tol)

    check = np.array([[[0.00054729, 0.00043342, 0.]],
                      [[0.00086654, 0.        , 0.]]])
    np.testing.assert_allclose(cvp, check, tol, tol)

    check = np.array([[[6.5232398e-06, 6.1970772e-05, 3.3333334e+07]],
                      [[6.1970772e-05, 3.3333334e+07, 3.3333334e+07]]])
    np.testing.assert_allclose(cvr, check, tol, tol)

    check = np.array([[[0.02353317, 0.02258242, 0.]],
                      [[0.03073696, 0.        , 0.]]])
    np.testing.assert_allclose(cerr, check, tol, tol)


def create_blank_ramp_data(dims, var, tm):
    """
    Create empty RampData classes, as well as gain and read noise arrays,
    based on dimensional, variance, and timing input.
    """
    nints, ngroups, nrows, ncols = dims
    rnval, gval = var
    frame_time, nframes, groupgap = tm
    group_time = (nframes + groupgap) * frame_time

    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    err = np.ones(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    pixdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint8)

    ramp_data = RampData()
    ramp_data.set_arrays(
        data=data, err=err, groupdq=gdq, pixeldq=pixdq)
    ramp_data.set_meta(
        name="NIRSpec", frame_time=frame_time, group_time=group_time,
        groupgap=groupgap, nframes=nframes, drop_frames1=None)
    ramp_data.set_dqflags(dqflags)

    gain = np.ones(shape=(nrows, ncols), dtype=np.float64) * gval
    rnoise = np.ones(shape=(nrows, ncols), dtype=np.float64) * rnval

    return ramp_data, gain, rnoise


# -----------------------------------------------------------------------------
#                           Set up functions

def setup_inputs(dims, var, tm):
    """
    Given dimensions, variances, and timing data, this creates test data to
    be used for unit tests.
    """
    nints, ngroups, nrows, ncols = dims
    rnoise, gain = var
    nframes, gtime, dtime = tm

    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    err = np.ones(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    pixdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint8)

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
        data=data, err=err, groupdq=gdq, pixeldq=pixdq)
    ramp_data.set_meta(
        name="MIRI", frame_time=dtime, group_time=gtime, groupgap=0,
        nframes=nframes, drop_frames1=None)
    ramp_data.set_dqflags(dqflags)

    gain = np.ones(shape=(nrows, ncols), dtype=np.float64) * gain
    rnoise = np.full((nrows, ncols), rnoise, dtype=np.float32)

    return ramp_data, rnoise, gain

# -----------------------------------------------------------------------------

###############################################################################
# The functions below are only used for DEBUGGING tests and developing tests. #
###############################################################################


def base_print(label, arr):
    arr_str = np.array2string(arr, max_line_width=np.nan, separator=", ")
    print(label)
    print(arr_str)


def print_slope_data(slopes):
    sdata, sdq, svp, svr, serr = slopes
    base_print("Slope Data:", sdata)


def print_slope_dq(slopes):
    sdata, sdq, svp, svr, serr = slopes
    base_print("Data Quality:", sdq)


def print_slope_poisson(slopes):
    sdata, sdq, svp, svr, serr = slopes
    base_print("Poisson:", svp)


def print_slope_readnoise(slopes):
    sdata, sdq, svp, svr, serr = slopes
    base_print("Readnoise:", svr)


def print_slope_err(slopes):
    sdata, sdq, svp, svr, serr = slopes
    base_print("Err:", serr)


def print_slopes(slopes):
    print(DELIM)
    print("**** SLOPES")
    print(DELIM)
    print_slope_data(slopes)

    print(DELIM)
    print_slope_dq(slopes)

    print(DELIM)
    print_slope_poisson(slopes)

    print(DELIM)
    print_slope_readnoise(slopes)

    print(DELIM)
    print_slope_err(slopes)

    print(DELIM)


def print_integ_data(integ_info):
    idata, idq, ivp, ivr, ierr = integ_info
    base_print("Integration data:", idata)


def print_integ_dq(integ_info):
    idata, idq, ivp, ivr, ierr = integ_info
    base_print("Integration DQ:", idq)


def print_integ_poisson(integ_info):
    idata, idq, ivp, ivr, ierr = integ_info
    base_print("Integration Poisson:", ivp)


def print_integ_rnoise(integ_info):
    idata, idq, ivp, ivr, ierr = integ_info
    base_print("Integration read noise:", ivr)


def print_integ_err(integ_info):
    idata, idq, ivp, ivr, ierr = integ_info
    base_print("Integration err:", ierr)


def print_integ(integ_info):
    print(DELIM)
    print("**** INTEGRATIONS")
    print(DELIM)
    print_integ_data(integ_info)

    print(DELIM)
    print_integ_dq(integ_info)

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
    idata, idq, ivp, ivr, ierr = cube
    oslope, osigslope, ovp, ovr, \
        oyint, osigyint, opedestal, oweights, ocrmag = optional
    """

    print(" ")
    print_slopes(slopes)
    print_integ(cube)
    print_optional(optional)


def print_ramp_data(ramp_data):
    print(DELIM)
    print_ramp_data_data(ramp_data)
    print(DELIM)
    print_ramp_data_dq(ramp_data)
    print(DELIM)


def print_ramp_data_data(ramp_data):
    base_print("RampData Data:", ramp_data.data)


def print_ramp_data_dq(ramp_data):
    base_print("RampData Data Quality:", ramp_data.groupdq)
