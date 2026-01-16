import numpy as np
import pytest

from stcal.ramp_fitting.likely_fit import likely_ramp_fit
from stcal.ramp_fitting.ramp_fit import ramp_fit_data
from stcal.ramp_fitting.ramp_fit_class import RampData

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
    dark_current = np.zeros(shape=(nrows, ncols), dtype=np.float32)

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


def create_linear_ramp(nints=1, ngroups=10, nrows=1, ncols=1, nframes=4):
    rnval, gval = 10.0, 5.0
    frame_time, groupgap = 10.736, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp_data, gain2d, rnoise2d = create_blank_ramp_data(dims, var, tm)

    # Create a simple linear ramp.
    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data.data[:, :, :, :] = ramp[None, :, None, None]

    return ramp_data, gain2d, rnoise2d


def create_flagged_ramp_data():
    nints, ngroups, nrows, ncols = 1, 20, 1, 1
    ramp_data, gain2d, rnoise2d = create_linear_ramp(nints, ngroups, nrows, ncols)

    # Add a jump
    ramp_data.data[0, 10:, 0, 0] += 150.0

    # Create segments in the ramp, including a jump and saturation at the end.
    dq = np.array([GOOD] * ngroups)
    dq[2] = DNU
    dq[17:] = SAT
    dq[10] = JMP
    ramp_data.groupdq[0, :, 0, 0] = dq

    return ramp_data, gain2d, rnoise2d


def test_basic_ramp():
    """
    Test a basic ramp with a linear progression up the ramp.

    Compare the integration results from the LIKELY algorithm to
    the OLS algorithm.
    """
    nints, ngroups, nrows, ncols = 1, 10, 1, 1
    ramp_data, gain2d, rnoise2d = create_linear_ramp(nints, ngroups, nrows, ncols)

    # Fit with likelihood
    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    data = cube['slope'][0, 0, 0]
    ddiff = ramp_data.data[0, ngroups - 1, 0, 0] - ramp_data.data[0, 0, 0, 0]
    check = ddiff / float(ngroups - 1)
    check = check / ramp_data.group_time
    tol = 1.0e-5
    np.testing.assert_allclose(data, check, tol)
    np.testing.assert_allclose(slopes['slope'][0, 0], data, tol)

    # Check against OLS fit
    algo = "OLS"
    slopes1, cube1, ols_opt1 = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)
    np.testing.assert_allclose(data, cube1['slope'][0, 0, 0], tol)


def test_basic_ramp_multi_pixel():
    """
    Test a basic ramp with a linear progression up the ramp.

    Use a 2x2 image instead of a single pixel. Compare the
    integration results from the LIKELY algorithm to the OLS algorithm.
    """
    nints, ngroups, nrows, ncols = 1, 10, 2, 2
    ramp_data, gain2d, rnoise2d = create_linear_ramp(nints, ngroups, nrows, ncols)

    # Fit with likelihood
    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    # Fit with OLS
    algo = "OLS"
    slopes1, cube1, ols_opt1 = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    data_tol = 1e-4
    err_tol = 0.1
    np.testing.assert_allclose(slopes['slope'], slopes1['slope'], data_tol)
    np.testing.assert_array_equal(slopes['dq'], slopes1['dq'])
    for key in ['var_poisson', 'var_rnoise', 'err']:
        np.testing.assert_allclose(slopes[key], slopes1[key], err_tol)


def test_basic_ramp_2integ():
    """
    Test a basic ramp with a linear progression up the ramp, 2 integrations.

    Compare the integration results from the LIKELY algorithm to the OLS algorithm.
    """
    nints, ngroups, nrows, ncols = 2, 10, 1, 1
    ramp_data, gain2d, rnoise2d = create_linear_ramp(nints, ngroups, nrows, ncols)

    # Fit with likelihood
    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    # Check against OLS fit
    algo = "OLS"
    slopes1, cube1, ols_opt1 = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    tol = 1.0e-5
    np.testing.assert_allclose(cube['slope'], cube1['slope'], tol)


def test_flagged_ramp():
    """
    Test flagged ramp.  The flags will cause segments, as well as ramp
    truncation.  Compare the integration results from the LIKELY algorithm
    to the OLS algorithm.
    """
    ramp_data, gain2d, rnoise2d = create_flagged_ramp_data()

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    data = cube['slope'][0, 0, 0]
    dq = cube['dq'][0, 0, 0]

    # Check against OLS.
    algo = "OLS"
    slopes1, cube1, ols_opt = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    data_ols = cube1['slope'][0, 0, 0]
    dq_ols = cube1['dq'][0, 0, 0]

    tol = 1.0e-5
    np.testing.assert_allclose(data, data_ols, tol)
    np.testing.assert_equal(dq, dq_ols)


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
    ramp = np.array([153.0, 307.0, 457.0, 604.0, 1853.0, 2002.0, 2159.0, 2308.0, 2459.0, 2601.0])
    ramp_data.data[0, :, 0, 0] = ramp

    # Create a jump, but don't mark it to make sure it gets detected.
    dq = np.array([GOOD] * ngroups)
    ramp_data.groupdq[0, :, 0, 0] = dq

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    tol = 1.0e-4

    assert abs(slopes['slope'][0, 0] - 1.9960526) < tol
    assert slopes['dq'][0, 0] == JMP
    assert abs(slopes['var_poisson'][0, 0] - 0.00064461) < tol
    assert abs(slopes['var_rnoise'][0, 0] - 0.00018037) < tol


def test_long_ramp():
    """
    Test a long ramp with hundreds of groups.
    """
    nints, ngroups, nrows, ncols = 1, 200, 1, 1
    ramp_data, gain2d, rnoise2d = create_linear_ramp(nints, ngroups, nrows, ncols)

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    data = cube['slope'][0, 0, 0]
    ddiff = ramp_data.data[0, ngroups - 1, 0, 0] - ramp_data.data[0, 0, 0, 0]
    check = ddiff / ((ngroups - 1) * ramp_data.group_time)
    tol = 1.0e-5
    np.testing.assert_allclose(data, check, tol)

    # Check against OLS.
    algo = "OLS"
    slopes1, cube1, ols_opt1 = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)
    np.testing.assert_allclose(data, cube1['slope'][0, 0, 0], tol)


@pytest.mark.parametrize("ngroups", [1, 2])
def test_too_few_group_ramp(ngroups):
    """
    Test a ramp with too few groups.

    It's supposed to fail.  The likelihood algorithm needs at least two
    groups to work.
    """
    nints, nrows, ncols = 1, 1, 1
    ramp_data, gain2d, rnoise2d = create_linear_ramp(nints, ngroups, nrows, ncols)

    with pytest.raises(ValueError):
        likely_ramp_fit(ramp_data, rnoise2d, gain2d)


@pytest.mark.parametrize("nframes", [1, 2, 4, 8])
def test_short_group_ramp(nframes):
    """
    Test short ramps with various nframes.
    """
    nints, ngroups, nrows, ncols = 1, 4, 1, 1
    ramp_data, gain2d, rnoise2d = create_linear_ramp(nints, ngroups, nrows, ncols, nframes)

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    data = cube['slope'][0, 0, 0]
    ddiff = ramp_data.data[0, ngroups - 1, 0, 0] - ramp_data.data[0, 0, 0, 0]
    check = ddiff / ((ngroups - 1) * ramp_data.group_time)
    tol = 1.0e-5
    np.testing.assert_allclose(data, check, tol)

    # Check against OLS.
    algo = "OLS"
    slopes1, cube1, ols_opt1 = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)
    np.testing.assert_allclose(data, cube1['slope'][0, 0, 0], tol)


def data_small_good_groups():
    nints, ngroups, nrows, ncols = 1, 10, 1, 1
    ramp_data, gain2d, rnoise2d = create_linear_ramp(nints, ngroups, nrows, ncols)

    dq = np.array([SAT] * ngroups, dtype=np.uint8)
    ramp_data.groupdq[0, :, 0, 0] = dq

    return ramp_data, gain2d, rnoise2d


def test_small_good_groups():
    """Test ramps with only two good groups."""
    ngood = 2
    ramp_data, gain2d, rnoise2d = data_small_good_groups()
    ramp_data.groupdq[0, :ngood, 0, 0] = GOOD

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    lik_slope = slopes['slope'][0, 0]

    # Check against OLS.
    ramp_data1, gain2d1, rnoise2d1 = data_small_good_groups()
    ramp_data1.groupdq[0, :ngood, 0, 0] = GOOD

    save_opt, algo, ncores = False, "OLS", "none"
    slopes1, cube1, ols_opt1 = ramp_fit_data(
        ramp_data1, save_opt, rnoise2d1, gain2d1, algo, "optimal", ncores
    )
    ols_slope = slopes1['slope'][0, 0]

    tol = 1.0e-4
    np.testing.assert_allclose(ols_slope, lik_slope, tol)


def test_jump_detect():
    """
    Test a ramp with jumps.

    Create a simple ramp with a (2, 2) image that has a jump in two
    different ramps and the computed slopes are still close.
    """
    nints, ngroups, nrows, ncols, nframes = 1, 10, 2, 2, 5
    ramp_data, gain2d, rnoise2d = create_linear_ramp(nints, ngroups, nrows, ncols, nframes)

    # Create a ramp with a jump to see if it gets detected.
    base, cr, jump_loc = 15.0, 1000.0, 6
    ramp = np.array([(k + 1) * base for k in range(ngroups)])
    ramp_data.data[0, :, 0, 1] = ramp
    if nrows > 1:
        ramp_data.data[0, :, 1, 0] = ramp
    ramp[jump_loc:] += cr
    ramp_data.data[0, :, 0, 0] = ramp
    ramp[jump_loc - 1] += cr
    if nrows > 1:
        ramp_data.data[0, :, 1, 1] = ramp

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    slope_est = base / ramp_data.group_time

    tol = 1.0e-4
    np.testing.assert_allclose(slopes['slope'], slope_est, tol)
    assert slopes['dq'][0, 0] == JMP
    assert slopes['dq'][0, 1] == GOOD
    assert slopes['dq'][1, 0] == GOOD
    assert slopes['dq'][1, 1] == JMP


def test_too_few_groups(caplog):
    """Check for a warning message."""
    nints, ngroups, nrows, ncols = 1, 3, 1, 1
    ramp_data, gain2d, rnoise2d = create_linear_ramp(nints, ngroups, nrows, ncols)

    save_opt, algo, ncores = False, "LIKELY", "none"
    ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    expected_log = "ramp fitting algorithm is being changed to OLS_C"
    assert expected_log in caplog.text


def test_zeroframe():
    """
    Test a multi-int ramp with a separate zeroframe.

    Compare the integration results from the LIKELY algorithm to the OLS algorithm.
    """
    nints, ngroups, nrows, ncols, nframes = 2, 10, 2, 2, 4
    ramp_data, gain2d, rnoise2d = create_linear_ramp(nints, ngroups, nrows, ncols, nframes)

    ramp_data.zeroframe = np.full((nints, nrows, ncols), 5.0, dtype=np.float32)

    # Fit with likelihood
    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    # Expected value
    np.testing.assert_allclose(slopes['slope'], 0.372503, 1e-5)

    # Check against OLS fit
    algo = "OLS"
    slopes1, cube1, ols_opt1 = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    # Fit will not be identical to OLS for unrealistic zeroframe
    tol = 1e-3
    np.testing.assert_allclose(cube['slope'], cube1['slope'], tol)


def test_zeroframe_bad_group(caplog):
    """Test a multi-int ramp with a separate zeroframe, but first group has only 1 read."""
    nints, ngroups, nrows, ncols, nframes = 2, 10, 2, 2, 1
    ramp_data, gain2d, rnoise2d = create_linear_ramp(nints, ngroups, nrows, ncols, nframes)

    ramp_data.zeroframe = np.full((nints, nrows, ncols), 5.0, dtype=np.float32)

    # Fit with likelihood
    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)

    # Warning is issued
    assert "Zero frame is present, but the first group has < 2 reads" in caplog.text

    # Check against OLS fit - it should be the same, since the zeroframe is ignored in this case
    algo = "OLS"
    slopes1, cube1, ols_opt1 = ramp_fit_data(ramp_data, save_opt, rnoise2d, gain2d, algo, "optimal", ncores)
    tol = 1e-7
    np.testing.assert_allclose(slopes['slope'], slopes1['slope'], tol)


# -----------------------------------------------------------------
#                              DEBUG
# -----------------------------------------------------------------


def dbg_print_basic_ramp(ramp_data, pix=(0, 0)):
    row, col = pix
    nints = ramp_data.data.shape[0]
    data = ramp_data.data[:, :, row, col]
    dq = ramp_data.groupdq[:, :, row, col]

    print(" ")  # noqa: T201
    print(DELIM)  # noqa: T201
    print(f"Data Shape: {ramp_data.data.shape}")  # noqa: T201
    print(DELIM)  # noqa: T201
    print("Data:")  # noqa: T201
    for integ in range(nints):
        arr_str = np.array2string(data[integ, :], max_line_width=np.nan, separator=", ")
        print(f"[{integ}] {arr_str}")  # noqa: T201
    print(DELIM)  # noqa: T201

    print("DQ:")  # noqa: T201
    for integ in range(nints):
        arr_str = np.array2string(dq[integ, :], max_line_width=np.nan, separator=", ")
        print(f"[{integ}] {arr_str}")  # noqa: T201
    print(DELIM)  # noqa: T201


def dbg_print_slopes(slope, pix=(0, 0), label=None):
    data, dq, vp, vr, err = slope
    row, col = pix

    print(" ")  # noqa: T201
    print(DELIM)  # noqa: T201
    if label is not None:
        print("Slope Information: ({label})")  # noqa: T201
    else:
        print("Slope Information:")  # noqa: T201
    print(f"    Pixel = ({row}, {col})")  # noqa: T201

    print(f"data = {data[row, col]}")  # noqa: T201
    print(f"dq = {dq[row, col]}")  # noqa: T201
    print(f"vp = {vp[row, col]}")  # noqa: T201
    print(f"vr = {vr[row, col]}\n")  # noqa: T201

    print(DELIM)  # noqa: T201


def dbg_print_cube(cube, pix=(0, 0), label=None):
    data, dq, vp, vr, err = cube
    row, col = pix
    nints = data.shape[0]

    print(" ")  # noqa: T201
    print(DELIM)  # noqa: T201
    if label is not None:
        print("Cube Information: ({label})")  # noqa: T201
    else:
        print("Cube Information:")  # noqa: T201
    print(f"    Pixel = ({row}, {col})")  # noqa: T201
    print(f"    Number of Integrations = {nints}")  # noqa: T201

    print(f"data = {data[:, row, col]}")  # noqa: T201
    print(f"dq = {dq[:, row, col]}")  # noqa: T201
    print(f"vp = {vp[:, row, col]}")  # noqa: T201
    print(f"vr = {vr[:, row, col]}")  # noqa: T201

    print(DELIM)  # noqa: T201


def dbg_print_slope_slope1(slopes, slopes1, pix):
    data, dq, vp, vr, err = slopes
    data1, dq1, vp1, vr1, err1 = slopes1
    row, col = pix

    print(" ")  # noqa: T201
    print(DELIM)  # noqa: T201
    print("Slope Information:")  # noqa: T201
    print(f"    Pixel = ({row}, {col})")  # noqa: T201

    print(f"data LIK = {data[row, col]:.12f}")  # noqa: T201
    print(f"data OLS = {data1[row, col]:.12f}\n")  # noqa: T201

    # print(f"dq LIK = {dq[row, col]}")
    # print(f"dq OLS = {dq1[row, col]}\n")

    print(f"vp LIK = {vp[row, col]:.12f}")  # noqa: T201
    print(f"vp OLS = {vp1[row, col]:.12f}\n")  # noqa: T201

    print(f"vr LIK = {vr[row, col]:.12f}")  # noqa: T201
    print(f"vr OLS = {vr1[row, col]:.12f}\n")  # noqa: T201

    print(DELIM)  # noqa: T201


def dbg_print_cube_cube1(cube, cube1, pix):
    data, dq, vp, vr, err = cube
    data1, dq1, vp1, vr1, err1 = cube1
    row, col = pix
    nints = data1.shape[0]

    print(" ")  # noqa: T201
    print(DELIM)  # noqa: T201
    print("Cube Information:")  # noqa: T201
    print(f"    Pixel = ({row}, {col})")  # noqa: T201
    print(f"    Number of Integrations = {nints}")  # noqa: T201

    print(f"data LIK = {data[:, row, col]}")  # noqa: T201
    print(f"data OLS = {data1[:, row, col]}\n")  # noqa: T201

    # print(f"dq LIK = {dq[:, row, col]}")
    # print(f"dq OLS = {dq1[:, row, col]}\n")

    print(f"vp LIK = {vp[:, row, col]}")  # noqa: T201
    print(f"vp OLS = {vp1[:, row, col]}\n")  # noqa: T201

    print(f"vr LIK = {vr[:, row, col]}")  # noqa: T201
    print(f"vr OLS = {vr1[:, row, col]}\n")  # noqa: T201

    print(DELIM)  # noqa: T201


from stcal.jump.jump_class import JumpData

DQFLAGS = {
    "GOOD": 0,
    "DO_NOT_USE": 1,
    "SATURATED": 2,
    "JUMP_DET": 4,
    "NO_GAIN_VALUE": 8,
    "REFERENCE_PIXEL": 2147483648,
}

GOOD = DQFLAGS["GOOD"]
DNU = DQFLAGS["DO_NOT_USE"]
SAT = DQFLAGS["SATURATED"]
JUMP = DQFLAGS["JUMP_DET"]
NGV = DQFLAGS["NO_GAIN_VALUE"]
REF = DQFLAGS["REFERENCE_PIXEL"]


def test_flag_large_events_withsnowball():
    nints, ngroups, nrows, ncols = 1, 20, 100, 100
    ramp_data, gain2d, rnoise2d = create_linear_ramp(nints, ngroups, nrows, ncols)

    # square of saturation surrounded by jump -> snowball
    # 112 pixels (121 minus 9) initially have a jump.
    ramp_data.data[0, 10:, 46:57, 46:57] += 300
    ramp_data.data[0, 10:, 50:53, 50:53] = 1e5
    ramp_data.groupdq[0, 10:, 50:53, 50:53] = SAT

    jump_data = JumpData(dqflags=DQFLAGS)
    jump_data.expand_large_events = 1
    jump_data.min_sat_area = 1
    jump_data.min_jump_area = 6
    jump_data.expand_factor = 1.9
    jump_data.edge_size = 0
    jump_data.sat_required_snowball = True
    jump_data.min_sat_radius_extend = 0.5
    jump_data.sat_expand = 1.1

    # Fit several ways: with and without snowball detection on.
    # In each case, make sure that all jumps are found, so that the variance
    # on the slope image is close to zero, and then check the number of
    # flagged pixels.

    image_info = likely_ramp_fit(ramp_data, rnoise2d, gain2d, jump_data=jump_data)[0]
    assert np.std(image_info['slope']) < 1e-5
    n_jump_expanded = np.sum(image_info['dq'] == JUMP)

    # Check that the uncertainties are the same for all pixels with jumps
    # and save this average uncertainty

    meanerr_jumppixels_new = np.mean(image_info['err'][image_info['dq'] == JUMP])
    assert np.std(image_info['err'][image_info['dq'] == JUMP]) < 1e-6

    # Now without snowball flagging

    image_info = likely_ramp_fit(ramp_data, rnoise2d, gain2d)[0]
    assert np.std(image_info['slope']) < 1e-5
    n_jump_original = np.sum(image_info['dq'] == JUMP)

    # Check that the uncertainties are the same for all pixels with jumps
    # originally flagged.  Check that this uncertainty matches the new value.

    meanerr_jumppixels_orig = np.mean(image_info['err'][image_info['dq'] == JUMP])
    assert np.std(image_info['err'][image_info['dq'] == JUMP]) < 1e-6
    assert np.abs(meanerr_jumppixels_orig - meanerr_jumppixels_new) < 1e-5

    assert n_jump_original == 112 and n_jump_expanded > 300 and n_jump_expanded < 600
