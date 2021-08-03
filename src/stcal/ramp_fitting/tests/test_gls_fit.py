import copy
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


@pytest.mark.skip(reason="GLS code does not [yet] handle single group integrations.")
def test_one_group_small_buffer_fit_gls():
    nints, ngroups, nrows, ncols = 1, 1, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1., 1

    ramp_data, gain2d, rnoise2d = setup_inputs(dims, gain, rnoise, group_time, frame_time)

    ramp_data.data[0, 0, 50, 50] = 10.0

    save_opt, algo, ncores = False, "GLS", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, 'optimal', ncores, test_dq_flags)

    data = slopes[0]
    tol = 1.e-6
    np.testing.assert_allclose(data[50, 50], 10.0, tol)


@pytest.mark.skip(reason="GLS does not correctly combine the slopes for integrations into the exposure slope.")
def test_gls_vs_ols_two_ints_ols():
    """
    A test to see if GLS is correctly combining integrations. The combination should only use the read noise variance.
    The current version of GLS does not work correctly.
    """
    # nints, ngroups, nrows, ncols = 1, 11, 103, 102
    nints, ngroups, nrows, ncols = 2, 11, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 1, 5
    group_time, frame_time = 1., 1

    ramp_data, gain2d, rnoise2d = setup_inputs(dims, gain, rnoise, group_time, frame_time)

    ramp = np.asarray([x*100 for x in range(11)])
    ramp_data.data[0, :, 50, 50] = ramp
    # ramp_data.data[1, :, 50, 50] = ramp * 2

    ramp_data2 = copy.deepcopy(ramp_data)
    rnoise2d_2 = rnoise2d.copy()
    gain2d_2 = gain2d.copy()

    save_opt, algo, ncores = False, "OLS", "none"
    oslopes, ocube, ools_opt, ogls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, 'optimal', ncores, test_dq_flags)

    save_opt, algo, ncores = False, "GLS", "none"
    gslopes, gcube, gols_opt, ggls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, 'optimal', ncores, test_dq_flags)

    # print(f"oslopes[0][50, 50] = {oslopes[0][50, 50]}")
    # print(f"gslopes[0][50, 50] = {gslopes[0][50, 50]}")
    # Should be 150 for each.  For OLS it is 150, but for GLS, it is not.
    np.testing.assert_allclose(oslopes[0][50, 50], gslopes[0][50, 50], 1e-6)


@pytest.mark.xfail(reason="GLS code does not [yet] handle single group integrations, nor multiple integrations.")
def test_one_group_two_ints_fit_gls():
    nints, ngroups, nrows, ncols = 2, 1, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1., 1

    ramp_data, gain2d, rnoise2d = setup_inputs(dims, gain, rnoise, group_time, frame_time)

    save_opt, algo, ncores = False, "GLS", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, 'optimal', ncores, test_dq_flags)

    data = slopes[0]
    np.testing.assert_allclose(data[50, 50], 11.0, 1e-6)


def test_nocrs_noflux():
    nints, ngroups, nrows, ncols = 1, 5, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1., 1

    ramp_data, gain2d, rnoise2d = setup_inputs(dims, gain, rnoise, group_time, frame_time)

    save_opt, algo, ncores = False, "GLS", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, 'optimal', ncores, test_dq_flags)

    assert(0 == np.max(slopes[0]))
    assert(0 == np.min(slopes[0]))


@pytest.mark.skip(reason="Getting all NaN's, but expecting all zeros.")
def test_nocrs_noflux_firstrows_are_nan():
    nints, ngroups, nrows, ncols = 1, 5, 103, 102
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    group_time, frame_time = 1., 1

    ramp_data, gain2d, rnoise2d = setup_inputs(dims, gain, rnoise, group_time, frame_time)
    ramp_data.data[0:, 0:12, :] = np.nan

    # save_opt, algo, ncores = False, "OLS", "none"
    save_opt, algo, ncores = False, "GLS", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, 'optimal', ncores, test_dq_flags)

    assert(0 == np.max(slopes[0]))
    assert(0 == np.min(slopes[0]))



'''
Default
dims  -> 1, 10, 103, 102
nints=1,
ngroups=10, 
nrows=103, 
ncols=102, 

readnoise=10, 
gain=1, 

nframes=1, 
grouptime=1.0,
deltatime=1):
'''

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
    int_times = np.zeros((nints,))

    # Set clas arrays
    ramp_class.set_arrays(data, err, groupdq, pixeldq, int_times)

    # Set class meta
    ramp_class.set_meta(
        name="MIRI",
        frame_time=frame_time,
        group_time=group_time,
        groupgap=0,
        nframes=1,
        drop_frames1=0)

    # Set class data quality flags
    ramp_class.set_dqflags(test_dq_flags)

    # Set noise arrays
    gain = np.ones(shape=(nrows, ncols), dtype=np.float64) * gain
    rnoise = np.full((nrows, ncols), rnoise, dtype=np.float32)

    return ramp_class, gain, rnoise

