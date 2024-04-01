import numpy as np
import pytest

from stcal.ramp_fitting.ramp_fit import ramp_fit_class, ramp_fit_data
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
    dark_current = np.zeros(shape=(nrows, ncols), dtype=np.float32)


    # Set clas arrays
    ramp_class.set_arrays(data, err, groupdq, pixeldq, average_dark_current=dark_current)

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
    dark_current = np.zeros(shape=(nrows, ncols), dtype = np.float32)

    ramp_data = RampData()
    ramp_data.set_arrays(data=data, err=err, groupdq=gdq, pixeldq=pixdq, average_dark_current=dark_current)
    ramp_data.set_meta(
        name="NIRSpec",
        frame_time=frame_time,
        group_time=group_time,
        groupgap=groupgap,
        nframes=nframes,
        drop_frames1=None,
    )
    ramp_data.set_dqflags(test_dq_flags)

    gain = np.ones(shape=(nrows, ncols), dtype=np.float64) * gval
    rnoise = np.ones(shape=(nrows, ncols), dtype=np.float64) * rnval

    return ramp_data, gain, rnoise


# -----------------------------------------------------------------------------


def test_basic_ramp():
    nints, ngroups, nrows, ncols = 1, 10, 1, 1
    rnval, gval = 10.0, 5.0
    frame_time, nframes, groupgap = 10.736, 4, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp_data, gain2d, rnoise2d = create_blank_ramp_data(dims, var, tm)

    ramp = np.array(list(range(ngroups))) * 20 + 10
    ramp_data.data[0, :, 0, 0] = ramp

    save_opt, algo, ncores = False, "LIKELY", "none"
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, 512, save_opt, rnoise2d, gain2d, algo, "optimal", ncores, test_dq_flags
    )
