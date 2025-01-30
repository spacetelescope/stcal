import numpy as np
import pytest

from stcal.ramp_fitting.ramp_fit import ramp_fit_class, ramp_fit_data
from stcal.ramp_fitting.ramp_fit_class import RampData
from stcal.ramp_fitting.likely_fit import likely_ramp_fit


test_dq_flags = {
    "GOOD": 0,
    "DO_NOT_USE": 1,
    "SATURATED": 2,
    "JUMP_DET": 4,
    "NO_GAIN_VALUE": 8,
    "UNRELIABLE_SLOPE": 16,
}

GOOD = test_dq_flags["GOOD"]
DNU = test_dq_flags["DO_NOT_USE"]
JMP = test_dq_flags["JUMP_DET"]
SAT = test_dq_flags["SATURATED"]
NGV = test_dq_flags["NO_GAIN_VALUE"]
USLOPE = test_dq_flags["UNRELIABLE_SLOPE"]

DELIM = "-" * 70


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
    pixdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint8)
    dark_current = np.zeros(shape=(nrows, ncols), dtype = np.float32)

    ramp_data = RampData()
    ramp_data.set_arrays(data=data, groupdq=gdq, pixeldq=pixdq, average_dark_current=dark_current)
    ramp_data.set_meta(
        name="NIRSpec",
        frame_time=frame_time,
        group_time=group_time,
        groupgap=groupgap,
        nframes=nframes,
        drop_frames1=None,
    )
    ramp_data.set_dqflags(test_dq_flags)

    gain = np.ones(shape=(nrows, ncols), dtype=np.float32) * gval
    rnoise = np.ones(shape=(nrows, ncols), dtype=np.float32) * rnval

    return ramp_data, gain, rnoise


# -----------------------------------------------------------------------------


def test_basic_ramp():
    """
    Test a basic ramp with a linear progression up the ramp.  Compare the
    integration results from the LIKELY algorithm to the OLS algorithm.
    """
    nints, ngroups, nrows, ncols = 1, 10, 1, 1
    rnval, gval = 10.0, 5.0
    frame_time, nframes, groupgap = 10.736, 4, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp_data, gain2d, rnoise2d = create_blank_ramp_data(dims, var, tm)

    # Create a simple linear ramp.
    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data.data[0, :, 0, 0] = ramp

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, "optimal", ncores, test_dq_flags
    )

    data = cube[0][0, 0, 0]
    ddiff = (ramp_data.data[0, ngroups-1, 0, 0] - ramp_data.data[0, 0, 0, 0])
    check = ddiff / float(ngroups-1)
    check = check / ramp_data.group_time
    tol = 1.e-5
    diff = abs(data - check)
    assert diff < tol
    slope = slopes[0][0, 0]
    assert abs(slope - data) < tol


    # Check against OLS.
    ramp_data1, gain2d1, rnoise2d1 = create_blank_ramp_data(dims, var, tm)

    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data1.data[0, :, 0, 0] = ramp

    save_opt, algo, ncores = False, "OLS", "none"
    slopes1, cube1, ols_opt1, gls_opt1 = ramp_fit_data(
        ramp_data1, 512, save_opt, rnoise2d1, gain2d1, algo, "optimal", ncores, test_dq_flags
    )

    data1 = cube1[0][0, 0, 0]
    diff = abs(data - data1)
    assert diff < tol


def test_basic_ramp_multi_pixel():
    """
    Test a basic ramp with a linear progression up the ramp.  Compare the
    integration results from the LIKELY algorithm to the OLS algorithm.
    """
    nints, ngroups, nrows, ncols = 1, 10, 2, 2
    rnval, gval = 10.0, 5.0
    frame_time, nframes, groupgap = 10.736, 4, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp_data, gain2d, rnoise2d = create_blank_ramp_data(dims, var, tm)

    # Create a simple linear ramp.
    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data.data[0, :, 0, 0] = ramp
    ramp_data.data[0, :, 0, 1] = ramp
    ramp_data.data[0, :, 1, 0] = ramp
    ramp_data.data[0, :, 1, 1] = ramp

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, "optimal", ncores, test_dq_flags
    )

    ramp_data1, gain2d1, rnoise2d1 = create_blank_ramp_data(dims, var, tm)

    # Create a simple linear ramp.
    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data1.data[0, :, 0, 0] = ramp
    ramp_data1.data[0, :, 0, 1] = ramp
    ramp_data1.data[0, :, 1, 0] = ramp
    ramp_data1.data[0, :, 1, 1] = ramp

    save_opt, algo, ncores = False, "OLS", "none"
    slopes1, cube1, ols_opt1, gls_opt1 = ramp_fit_data(
        ramp_data1, 512, save_opt, rnoise2d1, gain2d1, algo, "optimal", ncores, test_dq_flags
    )

    data, dq, vp, vr, err = slopes
    data1, dq1, vp1, vr1, err1 = slopes1

    tol = 1.e-4
    np.testing.assert_allclose(data, data1, tol)


def test_basic_ramp_2integ():
    """
    Test a basic ramp with a linear progression up the ramp.  Compare the
    integration results from the LIKELY algorithm to the OLS algorithm.
    """
    nints, ngroups, nrows, ncols = 2, 10, 1, 1
    rnval, gval = 10.0, 5.0
    frame_time, nframes, groupgap = 10.736, 4, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp_data, gain2d, rnoise2d = create_blank_ramp_data(dims, var, tm)

    # Create a simple linear ramp.
    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data.data[0, :, 0, 0] = ramp
    ramp_data.data[1, :, 0, 0] = ramp

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, "optimal", ncores, test_dq_flags
    )

    # Check against OLS.
    ramp_data1, gain2d1, rnoise2d1 = create_blank_ramp_data(dims, var, tm)

    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data1.data[0, :, 0, 0] = ramp
    ramp_data1.data[1, :, 0, 0] = ramp

    save_opt, algo, ncores = False, "OLS", "none"
    slopes1, cube1, ols_opt1, gls_opt1 = ramp_fit_data(
        ramp_data1, 512, save_opt, rnoise2d1, gain2d1, algo, "optimal", ncores, test_dq_flags
    )

    tol = 1.e-5
    data = cube[0][0, 0, 0]
    data1 = cube1[0][0, 0, 0]
    diff = abs(data - data1)
    assert diff < tol

    dbg_print_slope_slope1(slopes, slopes1, (0, 0))


def flagged_ramp_data():
    nints, ngroups, nrows, ncols = 1, 20, 1, 1
    rnval, gval = 10.0, 5.0
    frame_time, nframes, groupgap = 10.736, 4, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp_data, gain2d, rnoise2d = create_blank_ramp_data(dims, var, tm)

    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data.data[0, :, 0, 0] = ramp
    ramp_data.data[0, 10:, 0, 0] += 150.  # Add a jump.

    # Create segments in the ramp, including a jump and saturation at the end.
    dq = np.array([GOOD] * ngroups)
    dq[2] = DNU
    dq[17:] = SAT
    dq[10] = JMP
    ramp_data.groupdq[0, :, 0, 0] = dq

    return ramp_data, gain2d, rnoise2d


def test_flagged_ramp():
    """
    Test flagged ramp.  The flags will cause segments, as well as ramp
    truncation.  Compare the integration results from the LIKELY algorithm
    to the OLS algorithm.
    """
    ramp_data, gain2d, rnoise2d = flagged_ramp_data()

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, "optimal", ncores, test_dq_flags
    )

    data = cube[0][0, 0, 0]
    dq = cube[1][0, 0, 0]

    # Check against OLS.
    ramp_data, gain2d, rnoise2d = flagged_ramp_data()

    save_opt, algo, ncores = False, "OLS", "none"
    slopes1, cube1, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, "optimal", ncores, test_dq_flags
    )

    data_ols = cube1[0][0, 0, 0]
    dq_ols = cube1[1][0, 0, 0]

    tol = 1.e-5
    diff = abs(data - data_ols)
    assert diff < tol
    assert dq == dq_ols

    dbg_print_slope_slope1(slopes, slopes1, (0, 0))


def test_random_ramp():
    """
    Created a slope with a base slope of 150., with random Poisson
    noise with lambda 5.0.  At group 4 is a jump of 1100.0.
    """
    nints, ngroups, nrows, ncols = 1, 10, 1, 1
    rnval, gval = 10.0, 5.0
    frame_time, nframes, groupgap = 10.736, 5, 2

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp_data, gain2d, rnoise2d = create_blank_ramp_data(dims, var, tm)

    # A randomly generated ramp by setting up a ramp that has a slope of 150.
    # with some randomly added Poisson values, with lambda=5., and a jump
    # at group 4.
    ramp = np.array([153., 307., 457., 604., 1853., 2002., 2159., 2308., 2459., 2601.])
    ramp_data.data[0, :, 0, 0] = ramp

    # Create a jump, but don't mark it to make sure it gets detected.
    dq = np.array([GOOD] * ngroups)
    ramp_data.groupdq[0, :, 0, 0] = dq

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, "optimal", ncores, test_dq_flags
    )

    data, dq, vp, vr, err = slopes
    tol = 1.e-4

    assert abs(data[0, 0] - 1.9960526) < tol
    assert dq[0, 0] == JMP
    assert abs(vp[0, 0] - 0.00064461) < tol
    assert abs(vr[0, 0] - 0.00018037) < tol


def test_long_ramp():
    """
    Test a long ramp with hundreds of groups.
    """
    nints, ngroups, nrows, ncols = 1, 200, 1, 1
    rnval, gval = 10.0, 5.0
    frame_time, nframes, groupgap = 10.736, 4, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp_data, gain2d, rnoise2d = create_blank_ramp_data(dims, var, tm)

    # Create a simple linear ramp.
    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data.data[0, :, 0, 0] = ramp

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, "optimal", ncores, test_dq_flags
    )

    data = cube[0][0, 0, 0]
    ddiff = (ramp_data.data[0, ngroups-1, 0, 0] - ramp_data.data[0, 0, 0, 0])
    check = ddiff / float(ngroups-1)
    check = check / ramp_data.group_time
    tol = 1.e-5
    diff = abs(data - check)
    assert diff < tol

    # Check against OLS.
    ramp_data1, gain2d1, rnoise2d1 = create_blank_ramp_data(dims, var, tm)

    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data1.data[0, :, 0, 0] = ramp

    save_opt, algo, ncores = False, "OLS", "none"
    slopes1, cube1, ols_opt1, gls_opt1 = ramp_fit_data(
        ramp_data1, 512, save_opt, rnoise2d1, gain2d1, algo, "optimal", ncores, test_dq_flags
    )

    data1 = cube1[0][0, 0, 0]
    diff = abs(data - data1)
    assert diff < tol
    dbg_print_slope_slope1(slopes, slopes1, (0, 0))


@pytest.mark.parametrize("ngroups", [1, 2])
def test_too_few_group_ramp(ngroups):
    """
    It's supposed to fail.  The likelihood algorithm needs at least two
    groups to work.
    """
    nints, nrows, ncols = 1, 1, 1
    rnval, gval = 10.0, 5.0
    frame_time, nframes, groupgap = 10.736, 1, 0

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp_data, gain2d, rnoise2d = create_blank_ramp_data(dims, var, tm)

    # Create a simple linear ramp.
    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data.data[0, :, 0, 0] = ramp

    with pytest.raises(ValueError):
        likely_ramp_fit(ramp_data, rnoise2d, gain2d)


@pytest.mark.parametrize("nframes", [1, 2, 4, 8])
def test_short_group_ramp(nframes):
    """
    Test short ramps with various nframes.
    """
    nints, ngroups, nrows, ncols = 1, 4, 1, 1
    rnval, gval = 10.0, 5.0
    # frame_time, nframes, groupgap = 10.736, 4, 1
    frame_time, groupgap = 10.736, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp_data, gain2d, rnoise2d = create_blank_ramp_data(dims, var, tm)

    # Create a simple linear ramp.
    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data.data[0, :, 0, 0] = ramp

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, "optimal", ncores, test_dq_flags
    )

    data = cube[0][0, 0, 0]
    ddiff = (ramp_data.data[0, ngroups-1, 0, 0] - ramp_data.data[0, 0, 0, 0])
    check = ddiff / float(ngroups-1)
    check = check / ramp_data.group_time
    tol = 1.e-5
    diff = abs(data - check)
    assert diff < tol

    # Check against OLS.
    ramp_data1, gain2d1, rnoise2d1 = create_blank_ramp_data(dims, var, tm)

    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data1.data[0, :, 0, 0] = ramp

    save_opt, algo, ncores = False, "OLS", "none"
    slopes1, cube1, ols_opt1, gls_opt1 = ramp_fit_data(
        ramp_data1, 512, save_opt, rnoise2d1, gain2d1, algo, "optimal", ncores, test_dq_flags
    )

    data1 = cube1[0][0, 0, 0]
    diff = abs(data - data1)
    assert diff < tol
    # dbg_print_slope_slope1(slopes, slopes1, (0, 0))


def data_small_good_groups():
    nints, ngroups, nrows, ncols = 1, 10, 1, 1
    rnval, gval = 10.0, 5.0
    frame_time, nframes, groupgap = 10.736, 4, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp_data, gain2d, rnoise2d = create_blank_ramp_data(dims, var, tm)

    # Create a simple linear ramp.
    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data.data[0, :, 0, 0] = ramp

    dq = np.array([SAT] * ngroups, dtype=np.uint8)
    ramp_data.groupdq[0, :, 0, 0] = dq

    return ramp_data, gain2d, rnoise2d


# XXX One good group in a ramp may not be any good
# @pytest.mark.parametrize("ngood", [1, 2])
@pytest.mark.parametrize("ngood", [2])
def test_small_good_groups(ngood):
    """
    Test ramps with only one or two good groups.
    """
    ramp_data, gain2d, rnoise2d = data_small_good_groups()
    ramp_data.groupdq[0, :ngood, 0, 0] = GOOD

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, "optimal", ncores, test_dq_flags
    )

    lik_slope = slopes[0][0, 0]


    # Check against OLS.
    ramp_data1, gain2d1, rnoise2d1 = data_small_good_groups()
    ramp_data1.groupdq[0, :ngood, 0, 0] = GOOD

    save_opt, algo, ncores = False, "OLS", "none"
    slopes1, cube1, ols_opt1, gls_opt1 = ramp_fit_data(
        ramp_data1, 512, save_opt, rnoise2d1, gain2d1, algo, "optimal", ncores, test_dq_flags
    )
    ols_slope = slopes1[0][0, 0]

    tol = 1.e-4
    diff = abs(ols_slope - lik_slope)
    if ngood==2:
        assert diff < tol
    else:
        print(f"ols_slope = {ols_slope}")
        print(f"lik_slope = {lik_slope}")
        print(f"diff = {diff}")


def test_jump_detect():
    """
    Create a simple ramp with a (2, 2) image that has a jump in two
    different ramps and the computed slopes are still close.
    """
    nints, ngroups, nrows, ncols = 1, 10, 2, 2
    rnval, gval = 10.0, 5.0
    frame_time, nframes, groupgap = 10.736, 5, 2
    # frame_time, nframes, groupgap = 1., 1, 0

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp_data, gain2d, rnoise2d = create_blank_ramp_data(dims, var, tm)

    # Create a ramp with a jump to see if it gets detected.
    base, cr, jump_loc = 15., 1000., 6
    ramp = np.array([(k+1) * base for k in range(ngroups)])
    ramp_data.data[0, :, 0, 1] = ramp
    if nrows > 1:
        ramp_data.data[0, :, 1, 0] = ramp
    ramp[jump_loc:] += cr
    ramp_data.data[0, :, 0, 0] = ramp
    ramp[jump_loc-1] += cr
    if nrows > 1:
        ramp_data.data[0, :, 1, 1] = ramp

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, "optimal", ncores, test_dq_flags
    )

    data, dq, vp, vr, err = slopes
    slope_est = base / ramp_data.group_time

    tol = 1.e-4
    assert abs(data[0, 0] - slope_est) < tol
    assert abs(data[0, 1] - slope_est) < tol
    assert abs(data[1, 0] - slope_est) < tol
    assert abs(data[1, 1] - slope_est) < tol
    assert dq[0, 0] == JMP
    assert dq[0, 1] == GOOD
    assert dq[1, 0] == GOOD
    assert dq[1, 1] == JMP


def test_too_few_groups(caplog):
    """
    Ensure 
    """
    nints, ngroups, nrows, ncols = 1, 3, 1, 1
    rnval, gval = 10.0, 5.0
    frame_time, nframes, groupgap = 10.736, 4, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp_data, gain2d, rnoise2d = create_blank_ramp_data(dims, var, tm)

    # Create a simple linear ramp.
    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data.data[0, :, 0, 0] = ramp

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, "optimal", ncores, test_dq_flags
    )

    expected_log = "ramp fitting algorithm is being changed to OLS_C"
    assert expected_log in caplog.text


# -----------------------------------------------------------------
#                              DEBUG
# -----------------------------------------------------------------
def dbg_print_basic_ramp(ramp_data, pix=(0, 0)):
    row, col = pix
    nints = ramp_data.data.shape[0]
    data = ramp_data.data[:, :, row, col]
    dq = ramp_data.groupdq[:, :, row, col]

    print(" ")
    print(DELIM)
    print(f"Data Shape: {ramp_data.data.shape}")
    print(DELIM)
    print("Data:")
    for integ in range(nints):
        arr_str = np.array2string(data[integ, :], max_line_width=np.nan, separator=", ")
        print(f"[{integ}] {arr_str}")
    print(DELIM)

    print("DQ:")
    for integ in range(nints):
        arr_str = np.array2string(dq[integ, :], max_line_width=np.nan, separator=", ")
        print(f"[{integ}] {arr_str}")
    print(DELIM)


def dbg_print_slopes(slope, pix=(0, 0), label=None):
    data, dq, vp, vr, err = slope
    row, col = pix

    print(" ")
    print(DELIM)
    if label is not None:
        print("Slope Information: ({label})")
    else:
        print("Slope Information:")
    print(f"    Pixel = ({row}, {col})")

    print(f"data = {data[row, col]}")
    print(f"dq = {dq[row, col]}")
    print(f"vp = {vp[row, col]}")
    print(f"vr = {vr[row, col]}\n")

    print(DELIM)


def dbg_print_cube(cube, pix=(0, 0), label=None):
    data, dq, vp, vr, err = cube
    row, col = pix
    nints = data.shape[0]

    print(" ")
    print(DELIM)
    if label is not None:
        print("Cube Information: ({label})")
    else:
        print("Cube Information:")
    print(f"    Pixel = ({row}, {col})")
    print(f"    Number of Integrations = {nints}")

    print(f"data = {data[:, row, col]}")
    print(f"dq = {dq[:, row, col]}")
    print(f"vp = {vp[:, row, col]}")
    print(f"vr = {vr[:, row, col]}")

    print(DELIM)


def dbg_print_slope_slope1(slopes, slopes1, pix):
    data, dq, vp, vr, err = slopes
    data1, dq1, vp1, vr1, err1 = slopes1
    row, col = pix

    print(" ")
    print(DELIM)
    print("Slope Information:")
    print(f"    Pixel = ({row}, {col})")

    print(f"data LIK = {data[row, col]:.12f}")
    print(f"data OLS = {data1[row, col]:.12f}\n")

    # print(f"dq LIK = {dq[row, col]}")
    # print(f"dq OLS = {dq1[row, col]}\n")

    print(f"vp LIK = {vp[row, col]:.12f}")
    print(f"vp OLS = {vp1[row, col]:.12f}\n")

    print(f"vr LIK = {vr[row, col]:.12f}")
    print(f"vr OLS = {vr1[row, col]:.12f}\n")

    print(DELIM)


def dbg_print_cube_cube1(cube, cube1, pix):
    data, dq, vp, vr, err = cube
    data1, dq1, vp1, vr1, err1 = cube1
    row, col = pix
    nints = data1.shape[0]

    print(" ")
    print(DELIM)
    print("Cube Information:")
    print(f"    Pixel = ({row}, {col})")
    print(f"    Number of Integrations = {nints}")

    print(f"data LIK = {data[:, row, col]}")
    print(f"data OLS = {data1[:, row, col]}\n")

    # print(f"dq LIK = {dq[:, row, col]}")
    # print(f"dq OLS = {dq1[:, row, col]}\n")

    print(f"vp LIK = {vp[:, row, col]}")
    print(f"vp OLS = {vp1[:, row, col]}\n")

    print(f"vr LIK = {vr[:, row, col]}")
    print(f"vr OLS = {vr1[:, row, col]}\n")

    print(DELIM)


def array_string(arr, prec=4):
    return np.array2string(arr, precision=prec, max_line_width=np.nan, separator=", ")
