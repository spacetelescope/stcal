import pytest
import numpy as np

from jwst.datamodels import dqflags
from jwst.datamodels import RampModel
from stcal.ramp_fitting.ols_fit import ols_ramp_fit_multi
test_dq_flags = dqflags.pixel

DO_NOT_USE = test_dq_flags["DO_NOT_USE"]
JUMP_DET = test_dq_flags["JUMP_DET"]
SATURATED = test_dq_flags["SATURATED"]

def test_single_group_after_do_not_use():
    model1, gdq, readnoise_2d, pixdq, err, gain_2d = setup_inputs(ngroups=10, gain=1, nrows=1, ncols=3,
                                                         nints=2, readnoise=1)
    model1.data[0, :, 0, 0] = list(range(10))
    model1.data[1, :, 0, 0] = list(range(10))
    model1.data[0, :, 0, 1] = list(range(0, 20, 2))
    model1.data[1, :, 0, 1] = list(range(0, 20, 2))
    model1.data[0, :, 0, 2] = list(range(0, 20, 2))
    model1.data[1, :, 0, 2] = list(range(0, 20, 2))
    gdq[:, 0, :, :] = DO_NOT_USE
    gdq[:,-1, :, :] = DO_NOT_USE
    gdq[1, 0:4, :, :] = DO_NOT_USE
    gdq[1, 5, 0, 0] = JUMP_DET
    gdq[:, :, 0, 2] = SATURATED
    image_info, integ_info, opt_info = ols_ramp_fit_multi(
        model1, 1024 * 30000, True, readnoise_2d, gain_2d, 'optimal', 1)
    slopes = image_info[0]
    assert(slopes[0, 0] == 1)  # pixel with jump in 2nd group after initial skipped groups
    assert(slopes[0, 1] == 2)  # pixel with no jump in 2nd group after initial skipped groups
    assert(slopes[0, 2] == 0)
# Need test for multi-ints near zero with positive and negative slopes
def setup_inputs(ngroups=10, readnoise=10, nints=1,
                 nrows=103, ncols=102, nframes=1, grouptime=1.0, gain=1, deltatime=1):

    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    err = np.ones(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    pixdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint8)
    gain = np.ones(shape=(nrows, ncols), dtype=np.float64) * gain
    rnoise = np.full((nrows, ncols), readnoise, dtype=np.float32)
    int_times = np.zeros((nints,))

    model1 = RampModel(data=data, err=err, pixeldq=pixdq, groupdq=gdq, int_times=int_times)
    model1.meta.instrument.name = 'MIRI'
    model1.meta.instrument.detector = 'MIRIMAGE'
    model1.meta.instrument.filter = 'F480M'
    model1.meta.observation.date = '2015-10-13'
    model1.meta.exposure.type = 'MIR_IMAGE'
    model1.meta.exposure.group_time = deltatime
    model1.meta.subarray.name = 'FULL'
    model1.meta.subarray.xstart = 1
    model1.meta.subarray.ystart = 1
    model1.meta.subarray.xsize = ncols
    model1.meta.subarray.ysize = nrows
    model1.meta.exposure.frame_time = deltatime
    model1.meta.exposure.ngroups = ngroups
    model1.meta.exposure.group_time = deltatime
    model1.meta.exposure.nframes = 1
    model1.meta.exposure.groupgap = 0
    model1.meta.exposure.drop_frames1 = 0
    model1.frame_time = deltatime
    model1.group_time = deltatime
    model1.groupgap = 0
    model1.flags_saturated = False
    model1.flags_jump_det = 4
    model1.flags_do_not_use = 1
    model1.nframes = 1
    model1.instrument_name = "MIRI"
    model1.drop_frames1 = 0

    return model1, gdq, rnoise, pixdq, err, gain
