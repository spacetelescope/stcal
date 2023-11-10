import inspect
from pathlib import Path

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

# single group integrations fail in the GLS fitting
# so, keep the two method test separate and mark GLS test as
# expected to fail.  Needs fixing, but the fix is not clear
# to me. [KDG - 19 Dec 2018]

dqflags = {
    "GOOD": 0,  # Good pixel.
    "DO_NOT_USE": 2**0,  # Bad pixel. Do not use.
    "SATURATED": 2**1,  # Pixel saturated during exposure.
    "JUMP_DET": 2**2,  # Jump detected during exposure.
    "NO_GAIN_VALUE": 2**19,  # Gain cannot be measured.
    "UNRELIABLE_SLOPE": 2**24,  # Slope variance large (i.e., noisy pixel).
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
        [15.0, 25.0, 35.0, 54.0, 55.0, 65.0, 75.0, 94.0, 95.0, 105.0], dtype=np.float32
    )
    dq = [GOOD] * ngroups
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    # [data, dq, err, var_p, var_r]
    p_true = [1.0117551, GOOD, 0.0921951, 0.0020202, 0.00647973]

    # Set truth values for OPTIONAL results:
    # [slope, sigslope, var_poisson, var_rnoise, yint, sigyint, ped, weights]
    o_true = [1.0117551, 4.874572, 0.0020202, 0.00647973, 15.911023, 27.789335, 13.988245, 13841.038]

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
        [15.0, 25.0, 35.0, 54.0, 55.0, 65.0, 75.0, 94.0, 95.0, 105.0], dtype=np.float32
    )
    dq = [GOOD] * ngroups
    dq[1] = JUMP
    dq[2] = JUMP
    dq[4:] = [SAT] * (ngroups - 4)
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    p_true = [1.8999999, JUMP, 1.05057204, 0.03454545, 1.0691562]

    # Set truth values for OPTIONAL results:
    o_true = [1.9, 56.870003, 0.03454545, 1.0691562, -3.0, 56.870003, 13.1, 0.82091206]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


def test_pix_2():
    """
    CASE B: segment has >2 groups, not at end of ramp.
    CASE F: (twice) segment has 2 good groups not at array end.
    SCI segs are: seg0[15,25,35](B), seg1[54,55](F), seg2[65,75](F)
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [15.0, 25.0, 35.0, 54.0, 55.0, 65.0, 75.0, 94.0, 95.0, 105.0], dtype=np.float32
    )
    dq = [GOOD, GOOD, GOOD, JUMP, GOOD, JUMP, GOOD, JUMP, SAT, SAT]
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    p_true = [0.84833729, JUMP, 0.42747884, 0.00454545, 0.1781927]

    # Set truth values for OPTIONAL results for all segments
    o_true = [
        [1.0000001, 0.1, 1.0],  # slopes for 3 segments
        [28.435, 56.870003, 56.870003],  # sigslope
        [0.00909091, 0.01818182, 0.01818182],  # var_poisson
        [0.26728904, 1.0691562, 1.0691562],  # var_rnoise
        [14.999998, 51.0, 15.0],  # yint
        [36.709427, 56.870003, 56.870003],  # sigyint
        [14.151663],  # pedestal
        [13.091425, 0.84580624, 0.84580624],  # weights
    ]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


def test_pix_3():
    """
    CASE B: segment has >2 groups, not at end of ramp.
    CASE E: segment has 2 good groups, at end of ramp.
    SCI segs are: seg0[15,25,35,54,55,65,75,94](B), seg1[95,105](E)
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [15.0, 25.0, 35.0, 54.0, 55.0, 65.0, 75.0, 94.0, 95.0, 105.0], dtype=np.float32
    )
    dq = [GOOD] * ngroups
    dq[-2] = JUMP
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    p_true = [1.0746869, JUMP, 0.12186482, 0.00227273, 0.01257831]

    # Set truth values for OPTIONAL results:
    o_true = [
        [1.0757396, 1.0],
        [6.450687, 56.870003],
        [0.0025974, 0.01818182],
        [0.01272805, 1.0691562],
        [14.504965, 15.0],
        [27.842508, 56.870003],
        [13.925313],
        [4.2576841e03, 8.458062e-01],
    ]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


def test_pix_4():
    """
    CASE G: segment is the good 1st group of the entire ramp, and no later
      groups are good.
    SCI seg is seg0[15](G)
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [15.0, 25.0, 35.0, 54.0, 1055.0, 1065.0, 1075.0, 2594.0, 2595.0, 2605.0], dtype=np.float32
    )
    dq = [GOOD] + [SAT] * (ngroups - 1)
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    p_true = [1.5, GOOD, 1.047105, 0.02727273, 1.0691562]

    # Set truth values for OPTIONAL results:
    o_true = [1.5, 0.0, 0.02727273, 1.0691562, 0.0, 0.0, 13.5, 0.8318386]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


"""
    NOTE:
        There are small differences in the slope computation due to architectural
        differences of C and python.

--------------------------------------------------------------------------------
*** [2627] Segment 2, Integration 0 ***
Debug - [slope_fitter.c:3030]     sumx  = 0.018815478310
[ols_fit.py:3818]           sumx = array([0.01881548])

Debug - [slope_fitter.c:3031]     sumxx = 0.132461413741
[ols_fit.py:3819]          sumxx = array([0.13246141])

Debug - [slope_fitter.c:3032]     sumy  = 6.023876190186
[ols_fit.py:3820]           sumy = array([6.0238767], dtype=float32)

Debug - [slope_fitter.c:3033]     sumxy = 39.258270263672
[ols_fit.py:3821]          sumxy = array([39.25826825])

Debug - [slope_fitter.c:3034]     sumw  = 0.002894689096
[ols_fit.py:3822]     nreads_wtd = array([0.00289469], dtype=float32)

================================================================================
================================================================================

    num = (nreads_wtd * sumxy - sumx * sumy)
    denominator = nreads_wtd * sumxx - sumx**2
    invden = 1. / denominator

--------------------------------------------------------------------------------
Debug - [slope_fitter.c:2628] num = 0.000298373401
[ols_fit.py:3281]          num    = 0.000298359918

Debug - [slope_fitter.c:2629] den = 0.000029412389
[ols_fit.py:3282]          den    = 0.000029412383

Debug - [slope_fitter.c:2630] invden = 33999.277343750000
[ols_fit.py:3283]             invden = 33999.284936596494

Debug - [slope_fitter.c:2631] slope = 10.144479751587
[ols_fit.py:3284]            slope  = 10.144023881026

Debug - [slope_fitter.c:2632] gtime = 10.000000000000
Debug - [slope_fitter.c:2633] seg->slope = 1.014447927475
"""


# @pytest.mark.skip(reason="C architecture gives small differences for slope.")
def test_pix_5():
    """
    CASE B: segment has >2 groups, not at end of ramp.
    CASE A: segment has >2 groups, at end of ramp.
    SCI segs are: seg0[15, 25, 35, 54](B), seg1[ 2055, 2065, 2075, 2094, 2095,
       2105](A)
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [15.0, 25.0, 35.0, 54.0, 2055.0, 2065.0, 2075.0, 2094.0, 2095.0, 2105.0], dtype=np.float32
    )
    dq = [GOOD] * ngroups
    dq[4] = JUMP
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # XXX see the note above for the differences in C and python testing values.
    # Set truth values for PRIMARY results:
    p_true_p = [1.076075, JUMP, 0.16134359, 0.00227273, 0.02375903]
    # p_true_c = [1.076122522354126, JUMP, 0.16134359, 0.00227273, 0.02375903]  # To be used with C
    p_true = p_true_p

    # Set truth values for OPTIONAL results:
    oslope_p = [1.2799551, 1.0144024]
    # oslope_c = [1.2799551, 1.0144479]  # To be used with C
    o_true = [
        oslope_p,
        [18.312422, 9.920552],
        [0.00606061, 0.00363636],
        [0.10691562, 0.03054732],
        [13.537246, 2015.0737],
        [35.301933, 67.10882],
        [13.923912],
        [78.34764, 855.78046],
    ]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


def test_pix_6():
    """
    CASE F: segment has 2 good groups not at array end
    CASE A: segment has >2 groups, at end of ramp.
    SCI segs are: seg0[15,25](F), seg1[54, 55, 65, 375, 394, 395, 405](A)
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [15.0, 25.0, 35.0, 54.0, 55.0, 65.0, 375.0, 394.0, 395.0, 405.0], dtype=np.float32
    )
    dq = [GOOD] * ngroups
    dq[2] = JUMP
    dq[3] = JUMP
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    p_true = [6.092052, JUMP, 0.14613187, 0.0025974, 0.01875712]

    # Set truth values for OPTIONAL results:
    o_true = [
        [1.0, 6.195652],
        [56.870003, 8.8936615],
        [0.01818182, 0.0030303],
        [1.0691562, 0.01909207],
        [15.0, -143.2391],
        [56.870003, 58.76999],
        [8.907948],
        [8.4580624e-01, 2.0433204e03],
    ]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


def test_pix_7():
    """
    CASE B: segment has >2 groups, not at end of ramp.
    SCI seg is seg0[15,25,35,54,55,65,75,94](B)
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [15.0, 25.0, 35.0, 54.0, 55.0, 65.0, 75.0, 94.0, 195.0, 205.0], dtype=np.float32
    )
    dq = [GOOD] * (ngroups - 2) + [JUMP, JUMP]
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    p_true = [1.0757396, JUMP, 0.12379601, 0.0025974, 0.01272805]

    # Set truth values for OPTIONAL results:
    o_true = [1.0757396, 6.450687, 0.0025974, 0.01272805, 14.504951, 27.842508, 13.92426, 4257.684]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


def test_pix_8():
    """
    CASE H: the segment has a good 1st group and a bad 2nd group.
    CASE B: segment has >2 groups, not at end of ramp.
    SCI segs are: seg0[15](H, ignored), seg1[25, 35, 54, 55, 65, 75](B)
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [15.0, 25.0, 35.0, 54.0, 55.0, 65.0, 75.0, 94.0, 95.0, 105.0], dtype=np.float32
    )
    dq = [GOOD, JUMP, GOOD, GOOD, GOOD, GOOD, GOOD, SAT, SAT, SAT]
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    p_true = [0.98561335, JUMP, 0.1848883, 0.00363636, 0.03054732]

    # Set truth values for OPTIONAL results:
    o_true = [0.98561335, 9.920554, 0.00363636, 0.03054732, 16.508228, 39.383667, 14.014386, 855.78046]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


def test_pix_9():
    """
    CASE F: segment has 2 good groups not at array end.
    CASE B: segment has >2 groups, not at end of ramp.
    CASE E: segment has 2 good groups, at end of ramp.
    SCI seg are: seg0[15,25](F), seg1[54, 55, 65, 75, 94](B), seg2[95, 105](E)
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [15.0, 25.0, 35.0, 54.0, 55.0, 65.0, 75.0, 94.0, 95.0, 105.0], dtype=np.float32
    )
    dq = [GOOD, GOOD, JUMP, JUMP, GOOD, GOOD, GOOD, GOOD, JUMP, GOOD]
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    p_true = [0.9999994, JUMP, 0.22721863, 0.0030303, 0.048598]

    # Set truth values for OPTIONAL results:
    o_true = [
        [1.0, 0.9999994, 1.0],
        [56.870003, 13.036095, 56.870003],
        [0.01818182, 0.00454545, 0.01818182],
        [1.0691562, 0.05345781, 1.0691562],
        [15.0, 20.119896, 15.0],
        [56.870003, 68.618195, 56.870003],
        [14.0],
        [0.84580624, 297.23172, 0.84580624],
    ]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


def test_pix_10():
    """
    CASE F: segment has 2 good groups not at array end.
    CASE B: segment has >2 groups, not at end of ramp.
    CASE A: segment has >2 groups, at end of ramp.
    SCI segs are: seg0[15,25](F), seg1[35,54,55](B), seg2[65,75,94,95,105](A)
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [15.0, 25.0, 35.0, 54.0, 55.0, 65.0, 75.0, 94.0, 95.0, 105.0], dtype=np.float32
    )
    dq = [GOOD, GOOD, JUMP, GOOD, GOOD, JUMP, GOOD, GOOD, GOOD, GOOD]
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    p_true = [1.0, JUMP, 0.21298744, 0.0025974, 0.04276625]

    # Set truth values for OPTIONAL results:
    o_true = [
        [1.0, 1.0000014, 0.99999964],
        [56.870003, 28.434996, 13.036095],
        [0.01818182, 0.00909091, 0.00454545],
        [1.0691562, 0.26728904, 0.05345781],
        [15.0, 17.999956, 15.000029],
        [56.870003, 88.40799, 93.73906],
        [14.0],
        [0.84580624, 13.091425, 297.23172],
    ]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


def test_pix_11():
    """
    CASE F: segment has 2 good groups not at array end.
    SCI seg is: seg0[15,25](F)
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [15.0, 25.0, 35.0, 54.0, 55.0, 65.0, 75.0, 94.0, 95.0, 105.0], dtype=np.float32
    )
    dq = [GOOD, GOOD] + [SAT] * (ngroups - 2)
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    p_true = [1.0, GOOD, 1.042755, 0.01818182, 1.0691562]

    # Set truth values for OPTIONAL results:
    o_true = [1.0, 56.870003, 0.01818182, 1.0691562, 15.0, 56.870003, 14.0, 0.84580624]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


def test_pix_12():
    """
    CASE NGROUPS=2: the segment has a good 1st group and a saturated 2nd group,
      so is a single group. Group 1's data will be used in the 'fit'.
    """

    # XXX problem with C

    nints, ngroups, nrows, ncols = 1, 2, 1, 2
    gain, rnoise = 5.5, 10.34
    timing = 10.0
    dims = (nints, ngroups, nrows, ncols)
    var = (rnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array([15.0, 59025.0], dtype=np.float32)
    ramp_data.groupdq[0, :, 0, 0] = np.array([0, SAT])
    ramp_data.data[0, :, 0, 1] = np.array([61000.0, 61000.0], dtype=np.float32)
    ramp_data.groupdq[0, :, 0, 1] = np.array([SAT, SAT])

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results for pixel 1:
    # slope, dq, err, var_p, var_r
    # slope = group1 / deltatime = 15 / 10 = 1.5
    # dq = 2 (saturation) because group2 is saturated, but DNU is *not* set
    p_true = [1.5, GOOD, 1.047105, 0.027273, 1.069156]

    # Set truth values for OPTIONAL results:
    # slope, sig_slope, var_p, var_r, yint, sig_yint, pedestal, weights
    # slope = group1 / deltatime = 15 / 10 = 1.5
    # sig_slope, yint, sig_yint, and pedestal are all zero, because only 1 good group
    o_true = [1.5, 0.0, 0.027273, 1.069156, 0.0, 0.0, 13.5, 0.831839]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)

    # Set truth values for PRIMARY results for pixel 2:
    # slope, dq, err, var_p, var_r
    # slope = zero, because no good data
    # dq = 3 (saturation + do_not_use) because both groups are saturated
    p_true = [np.nan, 3, 0.0, 0.0, 0.0]

    # Set truth values for OPTIONAL results:
    # slope, sig_slope, var_p, var_r, yint, sig_yint, pedestal, weights
    # all values zero, because no good data
    o_true = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    assert_pri(p_true, slopes, 1)
    assert_opt(o_true, ols_opt, 1)


# -------------- start of MIRI tests: all have only a single segment-----
def test_miri_0():
    """
    MIRI data with ramp's 0th and final groups are flagged as DNU
    SCI seg is: [8888., 25., 35., 54., 55., 65., 75., 94., 95., 888.]
    GROUPDQ is: [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing, ts_name="MIRI")

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [8888.0, 25.0, 35.0, 54.0, 55.0, 65.0, 75.0, 94.0, 95.0, 888.0], dtype=np.float32
    )
    dq = [DNU] + [GOOD] * (ngroups - 2) + [DNU]
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    p_true = [1.025854, GOOD, 0.12379601, 0.0025974, 0.01272805]

    # Set truth values for OPTIONAL results:
    o_true = [1.025854, 6.450687, 0.0025974, 0.01272805, 26.439266, 27.842508, 23.974146, 4257.684]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


def test_miri_1():
    """
    MIRI data with ramp's 0th and final groups flagged as DNU; 0th group
    is also as a cosmic ray
    SCI seg is: [7777., 125., 135., 154., 165., 175., 185., 204., 205., 777.]
    GROUPDQ is: [5, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing, ts_name="MIRI")

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [7777.0, 125.0, 135.0, 154.0, 165.0, 175.0, 185.0, 204.0, 205.0, 777.0], dtype=np.float32
    )
    dq = [DNU | JUMP] + [GOOD] * (ngroups - 2) + [DNU]
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    p_true = [1.1996487, GOOD, 0.12379601, 0.0025974, 0.01272805]

    # Set truth values for OPTIONAL results:
    o_true = [1.1996487, 6.450687, 0.0025974, 0.01272805, 126.110214, 27.842508, 123.800354, 4257.684]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


def test_miri_2():
    """
    MIRI data with ramp's 0th and final groups flagged as both DNU
    and as CR.
    SCI seg is: [4444., 25., 35., 54., 55., 65., 75., 94., 95., 444.]
    GROUPDQ is: [5, 0, 0, 0, 0, 0, 0, 0, 0, 5]
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing, ts_name="MIRI")

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [4444.0, 25.0, 35.0, 54.0, 55.0, 65.0, 75.0, 94.0, 95.0, 444.0], dtype=np.float32
    )
    dq = [DNU | JUMP] + [GOOD] * (ngroups - 2) + [DNU | JUMP]
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    p_true = [1.025854, GOOD, 0.12379601, 0.0025974, 0.01272805]

    # Set truth values for OPTIONAL results:
    o_true = [1.025854, 6.450687, 0.0025974, 0.01272805, 26.439266, 27.842508, 23.974146, 4257.684]

    assert_pri(p_true, slopes, 0)
    assert_opt(o_true, ols_opt, 0)


def test_miri_3():
    """
    MIRI data with ramp's 0th and final groups flagged as DNU, and final
    group also flagged as CR.
    SCI seg is: [6666., 25., 35., 54., 55., 65., 75., 94., 95., 666.]
    GROUPDQ is: [1, 0, 0, 0, 0, 0, 0, 0, 0, 5]
    """
    nints, ngroups, nrows, ncols, timing, gain, readnoise = set_scalars()
    dims = (nints, ngroups, nrows, ncols)
    var = (readnoise, gain)
    ramp_data, gain, rnoise = create_blank_ramp_data(dims, var, timing, ts_name="MIRI")

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_data.data[0, :, 0, 0] = np.array(
        [6666.0, 25.0, 35.0, 54.0, 55.0, 65.0, 75.0, 94.0, 95.0, 666.0], dtype=np.float32
    )
    dq = [DNU] + [GOOD] * (ngroups - 2) + [DNU | JUMP]
    ramp_data.groupdq[0, :, 0, 0] = np.array(dq)

    save_opt, ncores, bufsize, algo = True, "none", 1024 * 30000, "OLS"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )

    # Set truth values for PRIMARY results:
    p_true = [1.025854, GOOD, 0.12379601, 0.0025974, 0.01272805]

    # Set truth values for OPTIONAL results:
    o_true = [1.025854, 6.450687, 0.0025974, 0.01272805, 26.439266, 27.842508, 23.974146, 4257.684]

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
    ramp_data.set_arrays(data=data, err=err, groupdq=gdq, pixeldq=pixdq)
    ramp_data.set_meta(
        name=ts_name,
        frame_time=frame_time,
        group_time=group_time,
        groupgap=groupgap,
        nframes=nframes,
        drop_frames1=None,
    )
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

    npt.assert_allclose(data[0, pix], p_true[0], atol=2e-5, rtol=2e-5)
    npt.assert_allclose(dq[0, pix], p_true[1], atol=1e-1)
    npt.assert_allclose(err[0, pix], p_true[2], atol=2e-5, rtol=2e-5)
    npt.assert_allclose(var_poisson[0, pix], p_true[3], atol=2e-5, rtol=2e-5)
    npt.assert_allclose(var_rnoise[0, pix], p_true[4], atol=2e-5, rtol=2e-5)


def debug_opt(o_true, opt_info, pix):
    (slope, sigslope, var_poisson, var_rnoise, yint, sigyint, pedestal, weights, crmag) = opt_info

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
    (slope, sigslope, var_poisson, var_rnoise, yint, sigyint, pedestal, weights, crmag) = opt_info

    opt_slope = slope[0, :, 0, pix]
    opt_sigslope = sigslope[0, :, 0, pix]
    opt_var_poisson = var_poisson[0, :, 0, pix]
    opt_var_rnoise = var_rnoise[0, :, 0, pix]
    opt_yint = yint[0, :, 0, pix]
    opt_sigyint = sigyint[0, :, 0, pix]
    opt_pedestal = pedestal[:, 0, pix]
    opt_weights = weights[0, :, 0, pix]

    npt.assert_allclose(opt_slope, o_true[0], atol=2e-5, rtol=2e-5)
    npt.assert_allclose(opt_sigslope, o_true[1], atol=2e-5, rtol=2e-5)
    npt.assert_allclose(opt_var_poisson, o_true[2], atol=2e-5, rtol=2e-5)
    npt.assert_allclose(opt_var_rnoise, o_true[3], atol=2e-5, rtol=2e-5)
    npt.assert_allclose(opt_yint, o_true[4], atol=2e-2)
    npt.assert_allclose(opt_sigyint, o_true[5], atol=2e-5, rtol=2e-5)
    npt.assert_allclose(opt_pedestal, o_true[6], atol=2e-5, rtol=3e-5)
    npt.assert_allclose(opt_weights, o_true[7], atol=2e-5, rtol=2e-5)


def dbg_print(string):
    """
    Print string with line number and filename.
    """
    cf = inspect.currentframe()
    line_number = cf.f_back.f_lineno
    finfo = inspect.getframeinfo(cf.f_back)
    fname = Path(finfo.filename).name
    print(f"[{fname}:{line_number}] {string}")
