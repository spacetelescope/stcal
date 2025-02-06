import numpy as np
import pytest
from astropy.io import fits
from stcal.jump.jump_class import JumpData
from stcal.jump.jump import (
    calc_num_slices,
    extend_saturation,
    find_ellipses,
    find_faint_extended,
    flag_large_events,
    point_inside_ellipse,
    find_first_good_group,
    detect_jumps_data
)

DQFLAGS = {
    "GOOD": 0,
    "DO_NOT_USE": 1,
    "SATURATED": 2,
    "JUMP_DET": 4,
    "NO_GAIN_VALUE": 8,
    "REFERENCE_PIXEL": 2147483648
}

GOOD = DQFLAGS["GOOD"]
DNU = DQFLAGS["DO_NOT_USE"]
SAT = DQFLAGS["SATURATED"]
JUMP = DQFLAGS["JUMP_DET"]
NGV = DQFLAGS["NO_GAIN_VALUE"]
REF = DQFLAGS["REFERENCE_PIXEL"]


def create_jump_data(dims, gain, rnoise, tm):
    """
    author: kmacdonald
    date: Nov 20, 2024
    """
    nints, ngroups, nrows, ncols = dims
    data = np.zeros(shape=dims, dtype=np.float32)
    gdq = np.zeros(shape=dims, dtype=np.uint8)

    pdq = np.zeros(shape=(nrows, ncols), dtype=np.uint8)
    gain2d = np.ones(shape=(nrows, ncols), dtype=np.float32) * gain
    rnoise2d = np.ones(shape=(nrows, ncols) , dtype=np.float32) * rnoise

    jump_data = JumpData(gain2d=gain2d, rnoise2d=rnoise2d, dqflags=DQFLAGS)
    jump_data.init_arrays_from_arrays(data, gdq, pdq)

    frame_time, nframes, groupgap = tm
    jump_data.nframes = nframes

    return jump_data


def test_nirspec_saturated_pix():
    """
    This test is based on an actual NIRSpec exposure that has some pixels
    flagged as saturated in one or more groups, which the jump step is
    supposed to ignore, but an old version of the code was setting JUMP flags
    for some of the saturated groups. This is to verify that the saturated
    groups are no longer flagged with jumps.
    """
    nints, ngroups, nrows, ncols = 1, 7, 2, 2
    rnval, gval = 10.7, 1.0
    frame_time, nframes, groupgap = 10.6, 1, 0

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    jump_data = create_jump_data(dims, gval, rnval, tm)

    # Setup the needed input pixel and DQ values
    jump_data.data[0, :, 1, 1] = [639854.75, 4872.451, -17861.791, 14022.15, 22320.176,
                              1116.3828, 1936.9746]
    jump_data.gdq[0, :, 1, 1] = [0, 0, 0, 0, 0, SAT, SAT]
    jump_data.data[0, :, 0, 1] = [8.25666812e+05, -1.10471914e+05, 1.95755371e+02, 1.83118457e+03,
                              1.72250879e+03, 1.81733496e+03, 1.65188281e+03]
    # 2 non-sat groups means only 1 non-sat diff, so no jumps should be flagged
    jump_data.gdq[0, :, 0, 1] = [0, 0, SAT, SAT, SAT, SAT, SAT]
    jump_data.data[0, :, 1, 0] = [1228767., 46392.234, -3245.6553, 7762.413,
                              37190.76, 266611.62, 5072.4434]
    jump_data.gdq[0, :, 1, 0] = [0, 0, 0, 0, 0, 0, SAT]

    jump_data.nframes = nframes
    jump_data.rejection_thresh = 4.0
    jump_data.three_grp_thresh = 5
    jump_data.four_grp_thresh = 6
    jump_data.max_cores = 'none'
    jump_data.max_jump_to_flag_neighbors = 200
    jump_data.min_jump_to_flag_neighbors = 10
    jump_data.flag_4_neighbors = True

    gdq, pdq, total_primary_crs, number_extended_events, stddev = detect_jumps_data(jump_data)

    # Check the results. There should not be any pixels with DQ values of 6, which
    # is saturated (2) plus jump (4). All the DQ's should be either just 2 or just 4.

    np.testing.assert_array_equal(gdq[0, :, 1, 1], [0, 4, 0, 4, 4, 2, 2])

    # assert that no groups are flagged when there's only 1 non-sat. grp
    np.testing.assert_array_equal(gdq[0, :, 0, 1], [0, 0, 2, 2, 2, 2, 2])
    np.testing.assert_array_equal(gdq[0, :, 1, 0], [0, 4, 4, 0, 4, 4, 2])


def test_multiprocessing():
    """
    Basic multiprocessing test.
    """
    nints, ngroups, nrows, ncols = 1, 13, 13, 2
    gval, rnval = 1., 10.
    frame_time, nframes, groupgap = 10.6, 1, 0

    dims = nints, ngroups, nrows, ncols
    tm = frame_time, nframes, groupgap

    jump_data = create_jump_data(dims, gval, rnval, tm)

    jump_data.data[0, 4:, 5, 1] = 2000
    jump_data.gdq[0, 4:, 6, 1] = DNU

    jump_data.max_cores = "1"
    jump_data.rejection_thresh = 5
    jump_data.three_grp_thresh = 6
    jump_data.four_grp_thresh = 7
    jump_data.max_jump_to_flag_neighbors = 10000
    jump_data.min_jump_to_flag_neighbors = 100
    jump_data.flag_4_neighbors = True

    gdq, pdq, total_primary_crs, number_extended_events, stddev = detect_jumps_data(jump_data)

    assert gdq[0, 4, 5, 1] == JUMP
    assert gdq[0, 4, 6, 1] == DNU

    # This section of code will fail without the fixes for PR #239 that prevent
    # the double flagging pixels with jump which already have do_not_use or saturation set.
    jump_data = create_jump_data(dims, gval, rnval, tm)

    jump_data.data[0, 4:, 5, 1] = 2000.
    jump_data.gdq[0, 4:, 6, 1] = DNU

    jump_data.max_cores = "5"
    jump_data.rejection_thresh = 5
    jump_data.three_grp_thresh = 6
    jump_data.four_grp_thresh = 7
    jump_data.max_jump_to_flag_neighbors = 10000
    jump_data.min_jump_to_flag_neighbors = 100
    jump_data.flag_4_neighbors = True

    gdq, pdq, total_primary_crs, number_extended_events, stddev = detect_jumps_data(jump_data)

    assert gdq[0, 4, 5, 1] == JUMP
    assert gdq[0, 4, 6, 1] == DNU  #This value would have been DNU | JUMP without the fix.


def test_multiprocessing_big():
    nints, ngroups, nrows, ncols = 1, 13, 2048, 7
    gval, rnval = 4., 10.
    frame_time, nframes, groupgap = 10.6, 1, 0

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    jump_data = create_jump_data(dims, gval, rnval, tm)

    jump_data.max_cores = "1"
    jump_data.data[0, 4:, 204, 5] = 2000.
    jump_data.gdq[0, 4:, 204, 6] = DNU

    jump_data.rejection_thresh = 5
    jump_data.three_grp_thresh = 6
    jump_data.four_grp_thresh = 7
    jump_data.max_jump_to_flag_neighbors = 10000
    jump_data.min_jump_to_flag_neighbors = 100
    jump_data.flag_4_neighbors = True

    gdq, pdq, total_primary_crs, number_extended_events, stddev = detect_jumps_data(jump_data)

    assert gdq[0, 4, 204, 5] == JUMP
    assert gdq[0, 4, 205, 5] == JUMP
    assert gdq[0, 4, 204, 6] == DNU

    # This section of code will fail without the fixes for PR #239 that prevent
    # the double flagging pixels with jump which already have do_not_use or saturation set.
    gval = 3.

    jump_data = create_jump_data(dims, gval, rnval, tm)

    jump_data.max_cores = "10"
    jump_data.data[0, 4:, 204, 5] = 2000.
    jump_data.gdq[0, 4:, 204, 6] = DNU

    jump_data.rejection_thresh = 5
    jump_data.three_grp_thresh = 6
    jump_data.four_grp_thresh = 7
    jump_data.max_jump_to_flag_neighbors = 10000
    jump_data.min_jump_to_flag_neighbors = 100
    jump_data.flag_4_neighbors = True

    gdq, pdq, total_primary_crs, number_extended_events, stddev = detect_jumps_data(jump_data)

    assert gdq[0, 4, 204, 5] == JUMP
    assert gdq[0, 4, 205, 5] == JUMP
    assert gdq[0, 4, 204, 6] == DNU  #This value would have been 5 without the fix.


def test_find_simple_ellipse():
    plane = np.zeros(shape=(5, 5), dtype=np.uint8)
    plane[2, 2] = JUMP
    plane[3, 2] = JUMP
    plane[1, 2] = JUMP
    plane[2, 3] = JUMP
    plane[2, 1] = JUMP
    plane[1, 3] = JUMP
    plane[2, 4] = JUMP
    plane[3, 3] = JUMP
    ellipse = find_ellipses(plane, JUMP, 1)

    assert ellipse[0][2] == pytest.approx(45.0, 1e-3)  # 90 degree rotation
    assert ellipse[0][0] == pytest.approx((2.5, 2.0))  # center


def test_find_ellipse2():
    plane = np.zeros(shape=(5, 5), dtype=np.uint8)
    plane[1, :] = [0, JUMP, JUMP, JUMP, 0]
    plane[2, :] = [0, JUMP, JUMP, JUMP, 0]
    plane[3, :] = [0, JUMP, JUMP, JUMP, 0]
    ellipses = find_ellipses(plane, JUMP, 1)
    ellipse = ellipses[0]
    assert ellipse[0][0] == 2
    assert ellipse[0][1] == 2
    assert ellipse[1][0] == 2
    assert ellipse[1][1] == 2
    assert ellipse[2] == 90.0


def test_extend_saturation_simple():
    cube = np.zeros(shape=(5, 7, 7), dtype=np.uint8)
    persist_jumps = np.zeros(shape=(7, 7), dtype=np.uint8)
    grp = 1
    min_sat_radius_extend = 1
    cube[1, 3, 3] = SAT
    cube[1, 2, 3] = SAT
    cube[1, 3, 4] = SAT
    cube[1, 4, 3] = SAT
    cube[1, 3, 2] = SAT

    cube[1, 2, 2] = JUMP
    sat_circles = find_ellipses(cube[grp, :, :], SAT, 1)

    jump_data = JumpData(dqflags=DQFLAGS)
    jump_data.min_sat_radius_extend = 1.1


    new_cube, persist_jumps = extend_saturation(
        cube, grp, sat_circles, jump_data, persist_jumps)

    assert new_cube[grp, 2, 2] == SAT
    assert new_cube[grp, 4, 4] == SAT
    assert new_cube[grp, 4, 5] == 0


def test_flag_large_events_nosnowball():
    cube = np.zeros(shape=(1, 5, 7, 7), dtype=np.uint8)

    # cross of saturation with no jump
    cube[0, 0:2, 3, 3] = SAT
    cube[0, 0:2, 2, 3] = SAT
    cube[0, 0:2, 3, 4] = SAT
    cube[0, 0:2, 4, 3] = SAT
    cube[0, 0:2, 3, 2] = SAT

    # cross of saturation surrounding by jump -> snowball but sat core is not new
    # should have no snowball trigger
    cube[0, 2, 3, 3] = SAT
    cube[0, 2, 2, 3] = SAT
    cube[0, 2, 3, 4] = SAT
    cube[0, 2, 4, 3] = SAT
    cube[0, 2, 3, 2] = SAT
    cube[0, 2, 1, 1:6] = JUMP
    cube[0, 2, 5, 1:6] = JUMP
    cube[0, 2, 1:6, 1] = JUMP
    cube[0, 2, 1:6, 5] = JUMP

    jump_data = JumpData(dqflags=DQFLAGS)
    jump_data.min_sat_area = 1
    jump_data.min_jump_area = 6
    jump_data.expand_factor = 1.9
    jump_data.edge_size = 1
    jump_data.sat_required_snowball = True
    jump_data.min_sat_radius_extend = 1
    jump_data.sat_expand = 1.1

    flag_large_events(cube, JUMP, SAT, jump_data)

    assert cube[0, 2, 2, 2] == 0
    assert cube[0, 2, 3, 6] == 0


def test_flag_large_events_withsnowball():
    cube = np.zeros(shape=(1, 5, 7, 7), dtype=np.uint8)
    # cross of saturation surrounding by jump -> snowball
    cube[0, 2, 3, 3] = SAT
    cube[0, 2, 2, 3] = SAT
    cube[0, 2, 3, 4] = SAT
    cube[0, 2, 4, 3] = SAT
    cube[0, 2, 3, 2] = SAT
    cube[0, 2, 1, 1:6] = JUMP
    cube[0, 2, 5, 1:6] = JUMP
    cube[0, 2, 1:6, 1] = JUMP
    cube[0, 2, 1:6, 5] = JUMP

    jump_data = JumpData(dqflags=DQFLAGS)
    jump_data.min_sat_area = 1
    jump_data.min_jump_area = 6
    jump_data.expand_factor = 1.9
    jump_data.edge_size = 0
    jump_data.sat_required_snowball = True
    jump_data.min_sat_radius_extend = 0.5
    jump_data.sat_expand = 1.1

    cube, total_snowballs = flag_large_events(cube, JUMP, SAT, jump_data)

    assert cube[0, 1, 2, 2] == 0
    assert cube[0, 1, 3, 5] == 0
    assert cube[0, 2, 0, 0] == 0
    assert cube[0, 2, 1, 0] == JUMP  # Jump was extended
    assert cube[0, 2, 2, 2] == SAT  # Saturation was extended
    assert cube[0, 2, 3, 6] == JUMP


def test_flag_large_events_groupedsnowball():
    cube = np.zeros(shape=(1, 5, 7, 7), dtype=np.uint8)
    # cross of saturation surrounding by jump -> snowball
#    cube[0, 1, :, :] = JUMP
#    cube[0, 2, :, :] = JUMP
    cube[0, 2, 1:6, 1:6] = JUMP
    cube[0, 1, 1:6, 1:6] = JUMP

    cube[0, 2, 3, 3] = SAT
    cube[0, 2, 2, 3] = SAT
    cube[0, 2, 3, 4] = SAT
    cube[0, 2, 4, 3] = SAT
    cube[0, 2, 3, 2] = SAT

    jump_data = JumpData(dqflags=DQFLAGS)
    jump_data.min_sat_area = 1
    jump_data.min_jump_area = 6
    jump_data.expand_factor = 1.9
    jump_data.edge_size = 0
    jump_data.sat_required_snowball = True
    jump_data.min_sat_radius_extend = 0.5
    jump_data.sat_expand = 1.1

    outgdq, num_snowballs = flag_large_events(cube, JUMP, SAT, jump_data)

    assert outgdq[0, 2, 1, 0] == JUMP  # Jump was extended
    assert outgdq[0, 2, 2, 2] == SAT  # Saturation was extended


def test_flag_large_events_withsnowball_noextension():
    cube = np.zeros(shape=(1, 5, 7, 7), dtype=np.uint8)
    # cross of saturation surrounding by jump -> snowball
    cube[0, 2, 3, 3] = SAT
    cube[0, 2, 2, 3] = SAT
    cube[0, 2, 3, 4] = SAT
    cube[0, 2, 4, 3] = SAT
    cube[0, 2, 3, 2] = SAT
    cube[0, 2, 1, 1:6] = JUMP
    cube[0, 2, 5, 1:6] = JUMP
    cube[0, 2, 1:6, 1] = JUMP
    cube[0, 2, 1:6, 5] = JUMP

    jump_data = JumpData(dqflags=DQFLAGS)
    jump_data.min_sat_area = 1
    jump_data.min_jump_area = 6
    jump_data.expand_factor = 1.9
    jump_data.edge_size = 0
    jump_data.sat_required_snowball = True
    jump_data.min_sat_radius_extend = 0.5
    jump_data.sat_expand = 1.1
    jump_data.max_extended_radius = 1

    cube, num_snowballs = flag_large_events(cube, JUMP, SAT, jump_data)

    assert cube[0, 1, 2, 2] == 0
    assert cube[0, 1, 3, 5] == 0
    assert cube[0, 2, 0, 0] == 0
    assert cube[0, 2, 1, 0] == 0  # Jump was NOT extended due to max_extended_radius=1
    assert cube[0, 2, 2, 2] == 0  # Saturation was NOT extended due to max_extended_radius=1


def test_find_faint_extended(tmp_path):
    nint, ngrps, ncols, nrows = 1, 66, 25, 25

    data = np.zeros(shape=(nint, ngrps, nrows, ncols), dtype=np.float32)
    gdq = np.zeros_like(data, dtype=np.uint32)
    pdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)

    pdq[0, 0] = 1
    pdq[1, 1] = 2147483648

    gain = 4
    readnoise = np.ones(shape=(nrows, ncols), dtype=np.float32) * 6.0 * gain

    # XXX Probably should not generate random data for CI tests.
    rng = np.random.default_rng(12345)
    data[0, 1:, 14:20, 15:20] = 6 * gain * 6.0 * np.sqrt(2)
    data = data + rng.normal(size=(nint, ngrps, nrows, ncols)) * readnoise
    fits.writeto(tmp_path / "data.fits", data, overwrite=True)

    jump_data = JumpData(dqflags=DQFLAGS)
    jump_data.nframes = 1
    jump_data.minimum_sigclip_groups = 100
    jump_data.extend_snr_threshold = 1.2
    jump_data.extend_min_area = 10
    jump_data.extend_inner_radius = 1
    jump_data.extend_outer_radius = 2.6
    jump_data.extend_ellipse_expand_ratio = 1
    jump_data.grps_masked_after_shower = 1
    jump_data.max_shower_amplitude = 10

    readnoise = readnoise * np.sqrt(2)
    gdq, num_showers = find_faint_extended(data, gdq, pdq, readnoise, jump_data)

    #  Check that all the expected samples in group 2 are flagged as jump and
    #  that they are not flagged outside.  This should not be in tests.

    # XXX Why is this write here?
    fits.writeto(tmp_path / "gdq.fits", gdq, overwrite=True)
    # assert num_showers == 1
    assert np.all(gdq[0, 1, 22, 14:23] == 0)
    assert gdq[0, 1, 16, 18] == JUMP
    assert np.all(gdq[0, 1, 11:22, 16:19] == JUMP)
    assert np.all(gdq[0, 1, 22, 16:19] == 0)
    assert np.all(gdq[0, 1, 10, 16:19] == 0)
    #  Check that the same area is flagged in the first group after the event
    assert np.all(gdq[0, 2, 22, 14:23] == 0)
    assert gdq[0, 2, 16, 18] == JUMP
    assert np.all(gdq[0, 2, 11:22, 16:19] == JUMP)
    assert np.all(gdq[0, 2, 22, 16:19] == 0)
    assert np.all(gdq[0, 2, 10, 16:19] == 0)

    assert np.all(gdq[0, 3:, :, :]) == 0

    #  Check that the flags are not applied in the 3rd group after the event
    assert np.all(gdq[0, 4, 12:22, 14:23]) == 0

    def test_find_faint_extended():
        nint, ngrps, ncols, nrows = 1, 66, 5, 5
        data = np.zeros(shape=(nint, ngrps, nrows, ncols), dtype=np.float32)
        gdq = np.zeros_like(data, dtype=np.uint32)
        pdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
        pdq[0, 0] = 1
        pdq[1, 1] = 2147483648
        #    pdq = np.zeros(shape=(data.shape[2], data.shape[3]), dtype=np.uint8)
        gain = 4
        readnoise = np.ones(shape=(nrows, ncols), dtype=np.float32) * 6.0 * gain
        rng = np.random.default_rng(12345)
        data[0, 1:, 14:20, 15:20] = 6 * gain * 6.0 * np.sqrt(2)
        data = data + rng.normal(size=(nint, ngrps, nrows, ncols)) * readnoise

        jump_data = JumpData(dqflags=DQFLAGS)
        jump_data.nframes = 1
        jump_data.minimum_sigclip_groups = 100
        jump_data.extend_snr_threshold = 3
        jump_data.extend_min_area = 10
        jump_data.extend_inner_radius = 1
        jump_data.extend_outer_radius = 2.6
        jump_data.extend_ellipse_expand_ratio = 1.1
        jump_data.grps_masked_after_shower = 0

        readnoise = readnoise * np.sqrt(2),
        gdq, num_showers = find_faint_extended(data, gdq, pdq, readnoise, jump_data)

# No shower is found because the event is identical in all ints
def test_find_faint_extended_sigclip():
    nint, ngrps, ncols, nrows = 101, 6, 30, 30

    data = np.zeros(shape=(nint, ngrps, nrows, ncols), dtype=np.float32)
    gdq = np.zeros_like(data, dtype=np.uint8)
    pdq = np.zeros(shape=(nrows, ncols), dtype=np.uint8)

    gain = 4
    readnoise = np.ones(shape=(nrows, ncols), dtype=np.float32) * 6.0 * gain

    # XXX Probably should not generate random data for CI tests.
    rng = np.random.default_rng(12345)
    data[0, 1:, 14:20, 15:20] = 6 * gain * 1.7
    data = data + rng.normal(size=(nint, ngrps, nrows, ncols)) * readnoise

    jump_data = JumpData(dqflags=DQFLAGS)
    jump_data.nframes = 1
    jump_data.minimum_sigclip_groups = 100
    jump_data.extend_snr_threshold = 1.3
    jump_data.extend_min_area = 20
    jump_data.extend_inner_radius = 1
    jump_data.extend_outer_radius = 2
    jump_data.extend_ellipse_expand_ratio = 1.1
    jump_data.grps_masked_after_shower = 3

    gdq, num_showers = find_faint_extended(data, gdq, pdq, readnoise, jump_data)

    #  Check that all the expected samples in group 2 are flagged as jump and
    #  that they are not flagged outside
    assert num_showers == 0
    assert np.all(gdq[0, 1, 22, 14:23] == 0)
    assert np.all(gdq[0, 1, 21, 16:20] == 0)
    assert np.all(gdq[0, 1, 20, 15:22] == 0)
    assert np.all(gdq[0, 1, 19, 15:23] == 0)
    assert np.all(gdq[0, 1, 18, 14:23] == 0)
    assert np.all(gdq[0, 1, 17, 14:23] == 0)
    assert np.all(gdq[0, 1, 16, 14:23] == 0)
    assert np.all(gdq[0, 1, 15, 14:22] == 0)
    assert np.all(gdq[0, 1, 14, 16:22] == 0)
    assert np.all(gdq[0, 1, 13, 17:21] == 0)
    assert np.all(gdq[0, 1, 12, 14:23] == 0)
    assert np.all(gdq[0, 1, 12:23, 24] == 0)
    assert np.all(gdq[0, 1, 12:23, 13] == 0)

    #  Check that the flags are not applied in the 3rd group after the event
    assert np.all(gdq[0, 4, 12:22, 14:23]) == 0


# No shower is found because the event is identical in all ints
def test_find_faint_extended_sigclip():
    nint, ngrps, ncols, nrows = 101, 6, 30, 30
    data = np.zeros(shape=(nint, ngrps, nrows, ncols), dtype=np.float32)
    gdq = np.zeros_like(data, dtype=np.uint8)
    pdq = np.zeros(shape=(nrows, ncols), dtype=np.int32)
    gain = 4
    readnoise = np.ones(shape=(nrows, ncols), dtype=np.float32) * 6.0 * gain
    rng = np.random.default_rng(12345)
    data[0, 1:, 14:20, 15:20] = 6 * gain * 1.7
    data = data + rng.normal(size=(nint, ngrps, nrows, ncols)) * readnoise

    jump_data = JumpData(dqflags=DQFLAGS)
    jump_data.nframes = 1
    jump_data.minimum_sigclip_groups = 100
    jump_data.extend_snr_threshold = 1.3
    jump_data.extend_min_area = 20
    jump_data.extend_inner_radius = 1
    jump_data.extend_outer_radius = 2
    jump_data.extend_ellipse_expand_ratio = 1.1
    jump_data.grps_masked_after_shower = 3

    # XXX Future collapse using JumpData
    gdq, num_showers = find_faint_extended(data, gdq, pdq, readnoise, jump_data)

    #  Check that all the expected samples in group 2 are flagged as jump and
    #  that they are not flagged outside
    assert (np.all(gdq[0, 1, 22, 14:23] == 0))
    assert (np.all(gdq[0, 1, 21, 16:20] == 0))
    assert (np.all(gdq[0, 1, 20, 15:22] == 0))
    assert (np.all(gdq[0, 1, 19, 15:23] == 0))
    assert (np.all(gdq[0, 1, 18, 14:23] == 0))
    assert (np.all(gdq[0, 1, 17, 14:23] == 0))
    assert (np.all(gdq[0, 1, 16, 14:23] == 0))
    assert (np.all(gdq[0, 1, 15, 14:22] == 0))
    assert (np.all(gdq[0, 1, 14, 16:22] == 0))
    assert (np.all(gdq[0, 1, 13, 17:21] == 0))
    assert (np.all(gdq[0, 1, 12, 14:23] == 0))
    assert (np.all(gdq[0, 1, 12:23, 24] == 0))
    assert (np.all(gdq[0, 1, 12:23, 13] == 0))

    #  Check that the flags are not applied in the 3rd group after the event
    assert (np.all(gdq[0, 4, 12:22, 14:23]) == 0)


def test_inside_ellipse5():
    ellipse = ((0, 0), (1, 2), -10)
    point = (1, 0.6)
    result = point_inside_ellipse(point, ellipse)
    assert result


def test_inside_ellipse4():
    ellipse = ((0, 0), (1, 2), 0)
    point = (1, 0.5)
    result = point_inside_ellipse(point, ellipse)
    assert result


def test_inside_ellipse6():
    ellipse = ((0, 0), (1, 2), 0)
    point = (3, 0.5)
    result = point_inside_ellipse(point, ellipse)
    assert not result


def test_inside_ellipes5():
    point = (1110.5, 870.5)
    ellipse = ((1111.0001220703125, 870.5000610351562), (10.60660171508789, 10.60660171508789), 45.0)
    result = point_inside_ellipse(point, ellipse)
    assert result


def test_calc_num_slices():
    n_rows = 20
    max_available_cores = 10
    assert calc_num_slices(n_rows, "none", max_available_cores) == 1
    assert calc_num_slices(n_rows, "half", max_available_cores) == 5
    assert calc_num_slices(n_rows, "3", max_available_cores) == 3
    assert calc_num_slices(n_rows, "7", max_available_cores) == 7
    assert calc_num_slices(n_rows, "21", max_available_cores) == 10
    assert calc_num_slices(n_rows, "quarter", max_available_cores) == 2
    assert calc_num_slices(n_rows, "7.5", max_available_cores) == 1
    assert calc_num_slices(n_rows, "one", max_available_cores) == 1
    assert calc_num_slices(n_rows, "-5", max_available_cores) == 1
    assert calc_num_slices(n_rows, "all", max_available_cores) == 10
    assert calc_num_slices(n_rows, "3/4", max_available_cores) == 1
    n_rows = 9
    assert calc_num_slices(n_rows, "21", max_available_cores) == 9

