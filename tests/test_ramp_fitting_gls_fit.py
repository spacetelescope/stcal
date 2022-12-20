import pytest
import numpy as np

from stcal.ramp_fitting.ramp_fit import ramp_fit_data
from stcal.ramp_fitting.ramp_fit import ramp_fit_class

test_dq_flags = {
    "GOOD": 0,
    "DO_NOT_USE": 1,
    "SATURATED": 2,
    "JUMP_DET": 4,
    "NO_GAIN_VALUE": 8,
    "UNRELIABLE_SLOPE": 16,
}

GOOD = test_dq_flags["GOOD"]
DO_NOT_USE = test_dq_flags["DO_NOT_USE"]
JUMP_DET = test_dq_flags["JUMP_DET"]
SATURATED = test_dq_flags["SATURATED"]
NO_GAIN_VALUE = test_dq_flags["NO_GAIN_VALUE"]
UNRELIABLE_SLOPE = test_dq_flags["UNRELIABLE_SLOPE"]

DELIM = "-" * 70


def setup_inputs(dims, gain, rnoise, group_time, frame_time):
    """
    Creates test data for testing.  All ramp data is zero.

    Parameters
    ----------
    dims: tuple
        Four dimensions (nints, ngroups, nrows, ncols)

    gain: float
        Gain noise

    rnoise: float
        Read noise

    group_time: float
        Group time

    frame_time: float
        Frame time

    Return
    ------
    ramp_class: RampClass
        A RampClass with all zero data.

    gain: ndarray
        A 2-D array for gain noise for each pixel.

    rnoise: ndarray
        A 2-D array for read noise for each pixel.
    """
    nints, ngroups, nrows, ncols = dims

    ramp_class = ramp_fit_class.RampData()  # Create class

    # Create zero arrays according to dimensions
    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    err = np.ones(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    groupdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint8)
    pixeldq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)

    # Set clas arrays
    ramp_class.set_arrays(data, err, groupdq, pixeldq)

    # Set class meta
    ramp_class.set_meta(
        name="MIRI",
        frame_time=frame_time,
        group_time=group_time,
        groupgap=0,
        nframes=1,
        drop_frames1=0,
    )

    # Set class data quality flags
    ramp_class.set_dqflags(test_dq_flags)

    # Set noise arrays
    gain = np.ones(shape=(nrows, ncols), dtype=np.float64) * gain
    rnoise = np.full((nrows, ncols), rnoise, dtype=np.float32)

    return ramp_class, gain, rnoise


# -----------------------------------------------------------------------------


def test_one_group_small_buffer():
    """
    Checks to make sure if a single group is used, it works.
    TODO: It does not work.  GLS needs to be modified to work edge cases.
    """
    nints, ngroups, nrows, ncols = 1, 1, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    ramp_data.data[0, 0, 50, 50] = 10.0

    save_opt, algo, ncores = False, "GLS", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    data = slopes[0]
    tol = 1.0e-6
    np.testing.assert_allclose(data[50, 50], 10.0, tol)


def test_two_integrations():
    """
    A test to see if GLS is correctly combining integrations.
    """
    nints, ngroups, nrows, ncols = 2, 11, 1, 1
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 1, 5
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    row, col = 0, 0

    ramp = np.asarray([x * 100 for x in range(11)])
    ramp_data.data[0, :, row, col] = ramp
    ramp_data.data[1, :, row, col] = ramp * 2

    save_opt, algo, ncores = False, "GLS", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    np.testing.assert_allclose(slopes[0][row, col], 133.3377685, 1e-6)


def test_one_group_two_integrations():
    """
    Test for multiple integrations with only one group.
    """
    nints, ngroups, nrows, ncols = 2, 1, 1, 1
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    ramp_data.data[0, 0, 0, 0] = 10.
    ramp_data.data[1, 0, 0, 0] = 11.

    save_opt, algo, ncores = False, "GLS", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    data = slopes[0]

    check = 10.5
    np.testing.assert_allclose(data[0, 0], check, 1e-6)


def test_nocrs_noflux():
    """
    Make sure no data returns all zeros.
    """
    nints, ngroups, nrows, ncols = 1, 5, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    save_opt, algo, ncores = False, "GLS", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    assert 0 == np.max(slopes[0])
    assert 0 == np.min(slopes[0])


@pytest.mark.skip(reason="Getting all NaN's, but expecting all zeros.")
def test_nocrs_noflux_firstrows_are_nan():
    """
    The 12 rows are set to NaN.  Not sure why this is expected to return
    all zeros.
    """
    nints, ngroups, nrows, ncols = 1, 5, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )
    ramp_data.data[0:, 0:12, :] = np.nan

    save_opt, algo, ncores = False, "GLS", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    assert 0 == np.max(slopes[0])
    assert 0 == np.min(slopes[0])


def test_error_when_frame_time_not_set():
    """
    Frame time is needed, so make sure an exception gets raised.
    """
    nints, ngroups, nrows, ncols = 1, 5, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )
    ramp_data.data[0:, 0:12, :] = np.nan

    ramp_data.frame_time = None  # Must be set

    save_opt, algo, ncores = False, "GLS", "none"
    with pytest.raises(UnboundLocalError):
        slopes, cube, ols_opt, gls_opt = ramp_fit_data(
            ramp_data, 512, save_opt, rnoise2d, gain2d, algo,
            "optimal", ncores, test_dq_flags)


def test_five_groups_two_integrations_Poisson_noise_only():
    """
    Multi-group ramp, with multi-integrations, with large poisson noise.
    """
    nints, ngroups, nrows, ncols = 2, 5, 1, 1
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 7, 2000
    group_time, frame_time = 3.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    row, col = 0, 0
    ramp_data.data[0, 0, row, col] = 10.0
    ramp_data.data[0, 1, row, col] = 15.0
    ramp_data.data[0, 2, row, col] = 25.0
    ramp_data.data[0, 3, row, col] = 33.0
    ramp_data.data[0, 4, row, col] = 60.0
    ramp_data.data[1, 0, row, col] = 10.0
    ramp_data.data[1, 1, row, col] = 15.0
    ramp_data.data[1, 2, row, col] = 25.0
    ramp_data.data[1, 3, row, col] = 33.0
    ramp_data.data[1, 4, row, col] = 160.0

    save_opt, algo, ncores = False, "GLS", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    out_slope = slopes[0][row, col]
    deltaDN1 = 50
    deltaDN2 = 150
    check = (deltaDN1 + deltaDN2) / 2.0

    np.testing.assert_allclose(out_slope, check, 75.0, 1e-6)


def test_bad_gain_values():
    """
    Test for bad gain values where gain values are negative
    and NaN.
    """
    nints, ngroups, nrows, ncols = 1, 5, 10, 11
    r1, c1, r2, c2 = 3, 3, 7, 7
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 7, 2000
    group_time, frame_time = 3.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )
    gain2d[r1, c1] = -10
    gain2d[r2, c2] = np.nan

    # save_opt, algo, ncores = False, "OLS", "none"
    save_opt, algo, ncores = False, "GLS", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    # data, dq, var_poisson, var_rnoise, err = slopes
    data, dq, err = slopes
    flag_check = NO_GAIN_VALUE | DO_NOT_USE

    assert dq[r1, c1] == flag_check
    assert dq[r2, c2] == flag_check

    # These asserts are wrong for some reason
    assert(0 == np.max(data))
    assert(0 == np.min(data))


def test_simple_ramp():
    """
    Here given a 10 group ramp with an exact slope of 20/group.
    The output slope should be 20.
    """
    nints, ngroups, nrows, ncols = 1, 10, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 7, 2000
    group_time, frame_time = 3.0, 3

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data.data[0, :, 50, 50] = ramp

    save_opt, algo, ncores = False, "GLS", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    ans = slopes[0][50, 50]
    check = 20. / 3
    tol = 1.e-5
    np.testing.assert_allclose(ans, check, tol)


def test_read_noise_only_fit():
    """
    Checks ramp fit GLS against polyfit, but it is slightly off.
    """
    nints, ngroups, nrows, ncols = 1, 5, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 50, 1
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    ramp_arr = [10., 15., 25., 33., 60.]
    ramp_data.data[0, :, 50, 50] = np.array(ramp_arr)

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    xvalues = np.arange(5) * 1.0
    yvalues = np.array(ramp_arr)
    coeff = np.polyfit(xvalues, yvalues, 1)
    ans = slopes[0][50, 50]
    check = coeff[0]
    tol = 1.e-2
    # print(f"ans = {ans}")         # 11.78866004
    # print(f"check = {check}")     # 11.79999999
    np.testing.assert_allclose(ans, check, tol)


@pytest.mark.skip(reason="GLS not sure what expected value is.")
def test_photon_noise_only_fit():
    """

    """
    nints, ngroups, nrows, ncols = 1, 5, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 1, 1000
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    ramp_arr = [10., 15., 25., 33., 60.]
    ramp_data.data[0, :, 50, 50] = np.array(ramp_arr)

    check = (ramp_data.data[0,4,50,50] - ramp_data.data[0,0,50,50]) / 4.0

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    ans = slopes[0][50, 50]
    tol = 1.e-2
    # print(f"ans = {ans}")         #  8.6579208
    # print(f"check = {check}")     # 12.5
    np.testing.assert_allclose(ans, check, tol)


@pytest.mark.skip(reason="GLS not sure what expected value is.")
def test_photon_noise_only_bad_last_group():
    """

    """
    nints, ngroups, nrows, ncols = 1, 5, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 1, 1000
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )
    ramp_data.data[0, 0, 50, 50] = 10.0
    ramp_data.data[0, 1, 50, 50] = 15.0
    ramp_data.data[0, 2, 50, 50] = 25.0
    ramp_data.data[0, 3, 50, 50] = 33.0
    ramp_data.data[0, 4, 50, 50] = 60.0

    check = (ramp_data.data[0,3,50,50] - ramp_data.data[0,0,50,50]) / 3.0

    ramp_data.groupdq[0,4,:,:] = DO_NOT_USE

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    ans = slopes[0][50, 50]
    tol = 1.e-2
    # print(f"ans = {ans}")         # 8.6579208
    # print(f"check = {check}")     # 7.6666666
    np.testing.assert_allclose(ans, check, tol)


@pytest.mark.skip(reason="GLS not sure what expected value is.")
def test_photon_noise_with_unweighted_fit():
    """

    """

    nints, ngroups, nrows, ncols = 1, 5, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 1, 1000
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )
    ramp_data.data[0, 0, 50, 50] = 10.0
    ramp_data.data[0, 1, 50, 50] = 15.0
    ramp_data.data[0, 2, 50, 50] = 25.0
    ramp_data.data[0, 3, 50, 50] = 33.0
    ramp_data.data[0, 4, 50, 50] = 60.0

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "unweighted", ncores, test_dq_flags)

    xvalues = np.arange(5) * 1.0
    yvalues = np.array([10,15,25,33,60])
    coeff = np.polyfit(xvalues, yvalues, 1)
    check = coeff[0]
    ans = slopes[0][50, 50]
    tol = 1.e-5
    # print(f"ans = {ans}")         #  8.6579208
    # print(f"check = {check}")     # 11.7999999
    np.testing.assert_allclose(ans, check, tol)


def test_two_groups_fit():
    """
    Ensure pixels with two group ramps and saturated groups get their
    final DQ flags set properly.
    """
    nints, ngroups, nrows, ncols = 1, 2, 1, 3
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )
    ramp_data.data[0, 0, 0, 0] = 10.0
    ramp_data.data[0, 1, 0, 0] = 15.0
    ramp_data.data[0, 0, 0, 1] = 20.0
    ramp_data.data[0, 0, 0, 2] = 200.0
    ramp_data.data[0, 1, 0, 2] = 600.0
    check = (ramp_data.data[0, 1, 0, 0] - ramp_data.data[0, 0, 0, 0])

    ramp_data.drop_frames1 = 0
    # 2nd group is saturated
    ramp_data.groupdq[0, 1, 0, 1] = SATURATED

    # 1st group is saturated
    ramp_data.groupdq[0, 0, 0, 2] = SATURATED
    ramp_data.groupdq[0, 1, 0, 2] = SATURATED

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    ans_data = slopes[0][0, 0]
    ans_dq = slopes[1]
    tol = 1.e-5
    np.testing.assert_allclose(ans_data, check, tol)

    assert ans_dq[0, 0] == GOOD
    assert ans_dq[0, 1] == UNRELIABLE_SLOPE
    assert ans_dq[0, 2] == SATURATED | DO_NOT_USE


def test_four_groups_oneCR_orphangroupatend_fit():
    """

    """
    nints, ngroups, nrows, ncols = 1, 4, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )
    ramp_data.data[0, 0, 50, 50] = 10.0
    ramp_data.data[0, 1, 50, 50] = 15.0
    ramp_data.data[0, 2, 50, 50] = 20.0
    ramp_data.data[0, 3, 50, 50] = 145.0

    ramp_data.groupdq[0,3,50,50] = JUMP_DET

    check = (ramp_data.data[0,1,50,50] - ramp_data.data[0,0,50,50])

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    ans = slopes[0][50, 50]
    tol = 1.e-6
    np.testing.assert_allclose(ans, check, tol)


def test_four_groups_two_CRs_at_end():
    """

    """

    nints, ngroups, nrows, ncols = 1, 4, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    ramp_data.data[0, 0, 50, 50] = 10.0
    ramp_data.data[0, 1, 50, 50] = 15.0
    ramp_data.data[0, 2, 50, 50] = 25.0
    ramp_data.data[0, 3, 50, 50] = 145.0
    check = (ramp_data.data[0,1,50,50] - ramp_data.data[0,0,50,50])

    ramp_data.groupdq[0,2,50,50] = JUMP_DET
    ramp_data.groupdq[0,3,50,50] = JUMP_DET

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    ans = slopes[0][50, 50]
    tol = 1.e-6
    np.testing.assert_allclose(ans, check, tol)


@pytest.mark.skip(reason="GLS code does not [yet] handle all groups as jump.")
def test_four_groups_four_CRs():
    """

    """
    nints, ngroups, nrows, ncols = 1, 10, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )
    ramp_data.data[0, 0, 50, 50] = 10.0
    ramp_data.data[0, 1, 50, 50] = 15.0
    ramp_data.data[0, 2, 50, 50] = 25.0
    ramp_data.data[0, 3, 50, 50] = 145.0

    ramp_data.groupdq[0,0,50,50] = JUMP_DET
    ramp_data.groupdq[0,1,50,50] = JUMP_DET
    ramp_data.groupdq[0,2,50,50] = JUMP_DET
    ramp_data.groupdq[0,3,50,50] = JUMP_DET

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    ans = slopes[0][50, 50]
    check = 0
    tol = 1.e-6
    # print(f"ans = {ans}")
    # print(f"check = {check}")
    np.testing.assert_allclose(ans, check, tol)


@pytest.mark.skip(reason="GLS code does not [yet] handle only one good group.")
def test_four_groups_three_CRs_at_end():
    """

    """
    nints, ngroups, nrows, ncols = 1, 4, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    ramp_data.data[0, 0, 50, 50] = 10.0
    ramp_data.data[0, 1, 50, 50] = 15.0
    ramp_data.data[0, 2, 50, 50] = 25.0
    ramp_data.data[0, 3, 50, 50] = 145.0

    ramp_data.groupdq[0,1,50,50] = JUMP_DET
    ramp_data.groupdq[0,2,50,50] = JUMP_DET
    ramp_data.groupdq[0,3,50,50] = JUMP_DET

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    ans = slopes[0][50, 50]
    check = 10.0
    tol = 1.e-6
    # print(f"ans = {ans}")
    # print(f"check = {check}")
    np.testing.assert_allclose(ans, check, tol)


def test_four_groups_CR_causes_orphan_1st_group():
    """

    """
    nints, ngroups, nrows, ncols = 1, 4, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10000, 0.01
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    ramp_data.data[0, 0, 50, 50] = 10.0
    ramp_data.data[0, 1, 50, 50] = 125.0
    ramp_data.data[0, 2, 50, 50] = 145.0
    ramp_data.data[0, 3, 50, 50] = 165.0

    ramp_data.groupdq[0,1,50,50] = JUMP_DET

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    ans = slopes[0][50, 50]
    check = 20.0
    tol = 1.e-6
    np.testing.assert_allclose(ans, check, tol)


def test_one_group_fit():
    """

    """
    nints, ngroups, nrows, ncols = 1, 1, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1.0, 1

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )
    ramp_data.data[0, 0, 50, 50] = 10.0

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    ans = slopes[0][50, 50]
    check = 10.0
    tol = 1.e-6
    np.testing.assert_allclose(ans, check, tol)


@pytest.mark.skip(reason="GLS not sure what expected value is.")
def test_two_groups_unc():
    """

    """
    deltaDN = 5  # TODO: Not sure wha this is supposed to be.
    nints, ngroups, nrows, ncols = 1, 2, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 2
    group_time, frame_time = 3.0, 3.0

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    ramp_data.data[0, 0, 50, 50] = 10.0
    ramp_data.data[0, 1, 50, 50] = 10.0 + deltaDN

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    ans = slopes[2][50, 50]
    check = np.sqrt(
        (deltaDN / gain) / group_time**2 + (rnoise**2 / group_time**2))
    tol = 1.e-6
    # print(f"ans = {ans}")
    # print(f"check = {check}")
    np.testing.assert_allclose(ans, check, tol)


@pytest.mark.skip(reason="GLS does not comopute VAR_XXX arrays.")
def test_five_groups_unc():
    """

    """
    '''
        grouptime=3.0
        # deltaDN = 5
        ingain = 2
        inreadnoise =7
        ngroups=5
    ramp_data, gdq, rnModel, pixdq, err, gain = setup_inputs(ngroups=ngroups,
            gain=ingain, readnoise=inreadnoise, deltatime=grouptime)
    '''
    nints, ngroups, nrows, ncols = 1, 5, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 7, 2
    group_time, frame_time = 3.0, 3

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    ramp_data.data[0, 0, 50, 50] = 10.0
    ramp_data.data[0, 1, 50, 50] = 15.0
    ramp_data.data[0, 2, 50, 50] = 25.0
    ramp_data.data[0, 3, 50, 50] = 33.0
    ramp_data.data[0, 4, 50, 50] = 60.0

    # check = np.median(np.diff(ramp_data.data[0,:,50,50])) / grouptime

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    '''
    # Not sure what to do with this test.  The VAR_XXX arrays don't get
    # computed in GLS.

        # out_slope=slopes[0].data[500, 500]
        # deltaDN = 50
        delta_time = (ngroups - 1) * grouptime
        # delta_electrons = median_slope * ingain *delta_time
        single_sample_readnoise = np.float64(inreadnoise / np.sqrt(2))
        np.testing.assert_allclose(slopes[0].var_poisson[50,50],
            ((median_slope)/(ingain*delta_time)), 1e-6)
        np.testing.assert_allclose(slopes[0].var_rnoise[50,50],
            (12 * single_sample_readnoise**2/(ngroups * (ngroups**2 - 1) * grouptime**2)),  1e-6)
        np.testing.assert_allclose(slopes[0].err[50,50],
            np.sqrt(slopes[0].var_poisson[50,50]  + slopes[0].var_rnoise[50,50] ),  1e-6)
    '''


@pytest.mark.skip(reason="GLS doesn't produce the optional results product, yet.")
def test_oneCR_10_groups_combination():
    """

    """
    nints, ngroups, nrows, ncols = 1, 10, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 7, 200
    group_time, frame_time = 3.0, 3

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    # two segments perfect fit, second segment has twice the slope
    ramp_data.data[0, 0, 50, 50] = 15.0
    ramp_data.data[0, 1, 50, 50] = 20.0
    ramp_data.data[0, 2, 50, 50] = 25.0
    ramp_data.data[0, 3, 50, 50] = 30.0
    ramp_data.data[0, 4, 50, 50] = 35.0
    ramp_data.data[0, 5, 50, 50] = 140.0
    ramp_data.data[0, 6, 50, 50] = 150.0
    ramp_data.data[0, 7, 50, 50] = 160.0
    ramp_data.data[0, 8, 50, 50] = 170.0
    ramp_data.data[0, 9, 50, 50] = 180.0

    ramp_data.groupdq[0,5,50,50] = JUMP_DET

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    ans = slopes[0][50, 50]
    print(f"ans = {ans}")

    # TODO Need to add the optional results product to GLS

    '''
    segment_groups  = 5
    single_sample_readnoise = np.float64(inreadnoise / np.sqrt(2))

    #check that the segment variance is as expected
    np.testing.assert_allclose(opt_model.var_rnoise[0,0,50,50],
        (12.0 * single_sample_readnoise**2 \
         / (segment_groups * (segment_groups**2 - 1) * grouptime**2)),
        rtol=1e-6)
    # check the combined slope is the average of the two segments since
    # they have the same number of groups
    np.testing.assert_allclose(slopes.data[50, 50], 2.5,rtol=1e-5)

    #check that the slopes of the two segments are correct
    np.testing.assert_allclose(opt_model.slope[0,0,50, 50], 5/3.0,rtol=1e-5)
    np.testing.assert_allclose(opt_model.slope[0,1,50, 50], 10/3.0,rtol=1e-5)
    '''


@pytest.mark.skip(reason="GLS doesn't produce the optional results product, yet.")
def test_oneCR_10_groups_combination_noisy2ndSegment():
    """

    """
    nints, ngroups, nrows, ncols = 1, 10, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    # use large gain to show that Poisson noise doesn't affect the recombination
    rnoise, gain = 7, 200
    group_time, frame_time = 3.0, 3

    ramp_data, gain2d, rnoise2d = setup_inputs(
        dims, gain, rnoise, group_time, frame_time
    )

    # two segments perfect fit, second segment has twice the slope
    ramp_data.data[0, 0, 50, 50] = 15.0
    ramp_data.data[0, 1, 50, 50] = 20.0
    ramp_data.data[0, 2, 50, 50] = 25.0
    ramp_data.data[0, 3, 50, 50] = 30.0
    ramp_data.data[0, 4, 50, 50] = 35.0
    ramp_data.data[0, 5, 50, 50] = 135.0
    ramp_data.data[0, 6, 50, 50] = 155.0
    ramp_data.data[0, 7, 50, 50] = 160.0
    ramp_data.data[0, 8, 50, 50] = 168.0
    ramp_data.data[0, 9, 50, 50] = 180.0

    ramp_data.groupdq[0,5,50,50] = JUMP_DET

    save_opt, algo, ncores, bufsize = False, "GLS", "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo,
        "optimal", ncores, test_dq_flags)

    '''
    avg_slope = (opt_model.slope[0,0,50,50] + opt_model.slope[0,1,50,50])/2.0
    # even with noiser second segment, final slope should be just the average
    # since they have the same number of groups
    np.testing.assert_allclose(slopes.data[50, 50], avg_slope,rtol=1e-5)
    '''
