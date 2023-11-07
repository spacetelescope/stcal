import pytest
import inspect
import numpy as np
import numpy.testing as npt

from stcal.ramp_fitting.ramp_fit import ramp_fit_data
from stcal.ramp_fitting.ramp_fit_class import RampData


#
# The first 12 tests are for a single ramp in a single integration. The ramps
#  have a variety of GROUPDQ vectors, with 1 or more segments in each ramp.  The
#  comparison of the PRIMARY output results are partly to verify the slopes and
#  variances of the combination of the segments in a ramp within the single
#  integration.  The comparison of the OPTIONAL output results are to verify the
#  results for each of the individual segments in a ramp.  Within each test is a
#  description of classification ('CASE') within the code of all of the segments
#  for the pixel based on the ramp's GROUPDQ, and the resulting segments and
#  their SCI values (these are mostly for my reference).
#

DELIM = "-" * 80

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


def test_pix_0():
    """
    CASE A: segment has >2 groups, at end of ramp.
    SCI seg is [15., 25., 35., 54., 55., 65., 75., 94., 95., 105.](A)
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [15., 25., 35., 54., 55., 65., 75., 94., 95., 105.], dtype=np.float32)
    dq = [GOOD] * ngroups
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo,"optimal", ncores, dqflags)

    # Set truth values for PRIMARY results:
    # [data, dq, err, var_p, var_r]
    p_true = [1.0117551, GOOD, 0.0921951, 0.0020202, 0.00647973]

    # Set truth values for OPTIONAL results:
    # [slope, sigslope, var_poisson, var_rnoise, yint, sigyint, ped, weights]
    o_true = [1.0117551, 4.874572, 0.0020202, 0.00647973,
              15.911023, 27.789335, 13.988245, 13841.038]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


def test_pix_1():
    """
    CASE H: the segment has a good 1st group and a bad 2nd group, so is a
      single group. If there are no later and longer segments in the ramp,
      this group's data will be used in the 'fit'. If there are later and
      longer segments, this group's data will not be used.
    CASE F: segment has 2 good groups not at array end.
    SCI segs are: seg0[15] (H, ignored), seg1[35, 54] (F)
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [15., 25., 35., 54., 55., 65., 75., 94., 95., 105.], dtype=np.float32)
    dq = [GOOD] * ngroups
    dq[1] = JUMP
    dq[2] = JUMP
    dq[4:] = [SAT] * (ngroups - 4)
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo,"optimal", ncores, dqflags)

    # Set truth values for PRIMARY results:
    p_true = [1.8999999, JUMP, 1.05057204, 0.03454545, 1.0691562]

    # Set truth values for OPTIONAL results:
    o_true = [1.9, 56.870003, 0.03454545, 1.0691562, -3., 56.870003,
              13.1, 0.82091206]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)



###############################################################################
# utilities


def set_scalars():
    """
    Set needed scalars for the size of the dataset, and other values needed for
    the fit.
    """

    nints = 1
    ngroups = 10
    nrows = 1
    ncols = 1

    timing = 10.0
    gain = 5.5
    readnoise = 10.34

    return nints, ngroups, nrows, ncols, timing, gain, readnoise


def create_blank_ramp_data(dims, var, timing, ts_name="NIRSpec"):
    """
    Create empty RampData classes, as well as gain and read noise arrays,
    based on dimensional, variance, and timing input.
    """
    nints, ngroups, nrows, ncols = dims
    rnval, gval = var
    frame_time = timing
    group_time = timing
    nframes = 1
    groupgap = 0

    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    err = np.ones(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    pixdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint8)

    ramp_data = RampData()
    ramp_data.set_arrays(
        data=data, err=err, groupdq=gdq, pixeldq=pixdq)
    ramp_data.set_meta(
        name=ts_name, frame_time=frame_time, group_time=group_time,
        groupgap=groupgap, nframes=nframes, drop_frames1=None)
    ramp_data.set_dqflags(dqflags)

    gain = np.ones(shape=(nrows, ncols), dtype=np.float32) * gval
    rnoise = np.ones(shape=(nrows, ncols), dtype=np.float32) * rnval

    return ramp_data, gain, rnoise


def debug_pri(p_true, new_info, pix):
    data, dq, vp, vr, err = new_info

    print(DELIM)
    dbg_print(f"data   = {data[0, pix]}")
    dbg_print(f"p_true = {p_true[0]}")
    print(DELIM)
    dbg_print(f"dq     = {dq[0, pix]}")
    dbg_print(f"p_true = {p_true[1]}")
    print(DELIM)
    dbg_print(f"vp     = {vp[0, pix]}")
    dbg_print(f"p_true = {p_true[3]}")
    print(DELIM)
    dbg_print(f"vr     = {vr[0, pix]}")
    dbg_print(f"p_true = {p_true[4]}")
    print(DELIM)
    dbg_print(f"err    = {err[0, pix]}")
    dbg_print(f"p_true = {p_true[2]}")
    print(DELIM)


def assert_pri(p_true, new_info, pix):
    """
    Compare true and fit values of primary output for extensions
    SCI, DQ, ERR, VAR_POISSON, VAR_RNOISE.
    """

    data, dq, var_poisson, var_rnoise, err = new_info

    npt.assert_allclose(data[0, pix], p_true[0], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(dq[0, pix], p_true[1], atol=1E-1)
    npt.assert_allclose(err[0, pix], p_true[2], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(var_poisson[0, pix], p_true[3], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(var_rnoise[0, pix], p_true[4], atol=2E-5, rtol=2e-5)

    return None


def debug_opt(o_true, opt_info, pix):
    (slope, sigslope, var_poisson, var_rnoise,
        yint, sigyint, pedestal, weights, crmag) = opt_info

    opt_slope = slope[0, :, 0, pix]
    opt_sigslope = sigslope[0, :, 0, pix]
    opt_var_poisson = var_poisson[0, :, 0, pix]
    opt_var_rnoise = var_rnoise[0, :, 0, pix]
    opt_yint = yint[0, :, 0, pix]
    opt_sigyint = sigyint[0, :, 0, pix]
    opt_pedestal = pedestal[:, 0, pix]
    opt_weights = weights[0, :, 0, pix]

    print(DELIM)
    dbg_print(f"slope  = {opt_slope}")
    dbg_print(f"o_true = {o_true[0]}")
    print(DELIM)
    dbg_print(f"sigslope = {opt_sigslope}")
    dbg_print(f"o_true   = {o_true[1]}")
    print(DELIM)
    dbg_print(f"var_p  = {opt_var_poisson}")
    dbg_print(f"o_true = {o_true[2]}")
    print(DELIM)
    dbg_print(f"var_r  = {opt_var_rnoise}")
    dbg_print(f"o_true = {o_true[3]}")
    print(DELIM)
    dbg_print(f"yint   = {opt_yint}")
    dbg_print(f"o_true = {o_true[4]}")
    print(DELIM)
    dbg_print(f"sigyint = {opt_sigyint}")
    dbg_print(f"o_true  = {o_true[5]}")
    print(DELIM)
    dbg_print(f"pedestal = {opt_pedestal}")
    dbg_print(f"o_true   = {o_true[6]}")
    print(DELIM)
    dbg_print(f"weights = {opt_weights}")
    dbg_print(f"o_true  = {o_true[7]}")
    print(DELIM)


def assert_opt(o_true, opt_info, pix):
    """
    Compare true and fit values of optional output for extensions SLOPE,
    SIGSLOPE, VAR_POISSON, VAR_RNOISE, YINT, SIGYINT, PEDESTAL, and WEIGHTS.
    Selecting the particular (and only) ramp in the optional output, which is
    [0,:,0,0]
    """
    (slope, sigslope, var_poisson, var_rnoise,
        yint, sigyint, pedestal, weights, crmag) = opt_info

    opt_slope = slope[0, :, 0, pix]
    opt_sigslope = sigslope[0, :, 0, pix]
    opt_var_poisson = var_poisson[0, :, 0, pix]
    opt_var_rnoise = var_rnoise[0, :, 0, pix]
    opt_yint = yint[0, :, 0, pix]
    opt_sigyint = sigyint[0, :, 0, pix]
    opt_pedestal = pedestal[:, 0, pix]
    opt_weights = weights[0, :, 0, pix]

    npt.assert_allclose(opt_slope, o_true[0], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(opt_sigslope, o_true[1], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(opt_var_poisson, o_true[2], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(opt_var_rnoise, o_true[3], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(opt_yint, o_true[4], atol=2E-2)
    npt.assert_allclose(opt_sigyint, o_true[5], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(opt_pedestal, o_true[6], atol=2E-5, rtol=3e-5)
    npt.assert_allclose(opt_weights, o_true[7], atol=2E-5, rtol=2e-5)

    return None


def dbg_print(string):
    """
    Print string with line number and filename.
    """
    cf = inspect.currentframe()
    line_number = cf.f_back.f_lineno
    finfo = inspect.getframeinfo(cf.f_back)
    fname = os.path.basename(finfo.filename)
    print(f"[{fname}:{line_number}] {string}")
