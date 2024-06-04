import numpy as np
import pytest
from astropy.io import fits
from stcal.jump.jump import (
    calc_num_slices,
    extend_saturation,
    find_ellipses,
    find_faint_extended,
    flag_large_events,
    point_inside_ellipse,
    find_first_good_group,
    detect_jumps,
    find_last_grp
)

DQFLAGS = {"JUMP_DET": 4, "SATURATED": 2, "DO_NOT_USE": 1, "GOOD": 0, "NO_GAIN_VALUE": 8,
           "REFERENCE_PIXEL": 2147483648}


@pytest.fixture()
def setup_cube():
    def _cube(ngroups, readnoise=10):
        nints = 1
        nrows = 204
        ncols = 204
        rej_threshold = 3
        nframes = 1
        data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
        read_noise = np.full((nrows, ncols), readnoise, dtype=np.float32)
        gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint32)

        return data, gdq, nframes, read_noise, rej_threshold

    return _cube

def test_nirspec_saturated_pix():
    """
    This test is based on an actual NIRSpec exposure that has some pixels
    flagged as saturated in one or more groups, which the jump step is
    supposed to ignore, but an old version of the code was setting JUMP flags
    for some of the saturated groups. This is to verify that the saturated
    groups are no longer flagged with jumps.
    """
    ingain = 1.0
    inreadnoise = 10.7
    ngroups = 7
    nrows = 2
    ncols = 2
    nints = 1
    nframes = 1
    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    read_noise = np.full((nrows, ncols), inreadnoise, dtype=np.float32)
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint32)
    err = np.zeros(shape=(nrows, ncols), dtype=np.float32)
    pdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    gain = np.ones_like(read_noise) * ingain

    # Setup the needed input pixel and DQ values
    data[0, :, 1, 1] = [639854.75, 4872.451, -17861.791, 14022.15, 22320.176,
                              1116.3828, 1936.9746]
    gdq[0, :, 1, 1] = [0, 0, 0, 0, 0, 2, 2]
    data[0, :, 0, 1] = [8.25666812e+05, -1.10471914e+05, 1.95755371e+02, 1.83118457e+03,
                              1.72250879e+03, 1.81733496e+03, 1.65188281e+03]
    # 2 non-sat groups means only 1 non-sat diff, so no jumps should be flagged
    gdq[0, :, 0, 1] = [0, 0, 2, 2, 2, 2, 2]
    data[0, :, 1, 0] = [1228767., 46392.234, -3245.6553, 7762.413,
                              37190.76, 266611.62, 5072.4434]
    gdq[0, :, 1, 0] = [0, 0, 0, 0, 0, 0, 2]

    # run jump detection
    gdq, pdq, total_primary_crs, number_extended_events, stddev = detect_jumps(nframes, data, gdq, pdq, err,
                                                                               gain, read_noise, rejection_thresh=4.0,
                                                                               three_grp_thresh=5,
                                 four_grp_thresh=6,
                                 max_cores='none', max_jump_to_flag_neighbors=200,
                                 min_jump_to_flag_neighbors=10, flag_4_neighbors=True, dqflags=DQFLAGS)

    # Check the results. There should not be any pixels with DQ values of 6, which
    # is saturated (2) plus jump (4). All the DQ's should be either just 2 or just 4.
    np.testing.assert_array_equal(gdq[0, :, 1, 1], [0, 4, 0, 4, 4, 2, 2])
    # assert that no groups are flagged when there's only 1 non-sat. grp
    np.testing.assert_array_equal(gdq[0, :, 0, 1], [0, 0, 2, 2, 2, 2, 2])
    np.testing.assert_array_equal(gdq[0, :, 1, 0], [0, 4, 4, 0, 4, 4, 2])

def test_multiprocessing():
    nints = 1
    nrows = 13
    ncols = 2
    ngroups = 13
    readnoise = 10
    frames_per_group = 1

    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    readnoise_2d = np.ones((nrows, ncols), dtype=np.float32) * readnoise
    gain_2d = np.ones((nrows, ncols), dtype=np.float32) * 4
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint32)
    pdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    err = np.zeros(shape=(nrows, ncols), dtype=np.float32)
    num_cores = "1"
    data[0, 4:, 5, 1] = 2000
    gdq[0, 4:, 6, 1] = DQFLAGS['DO_NOT_USE']
    gdq, pdq, total_primary_crs, number_extended_events, stddev = detect_jumps(
        frames_per_group, data, gdq, pdq, err, gain_2d, readnoise_2d, rejection_thresh=5, three_grp_thresh=6,
        four_grp_thresh=7, max_cores=num_cores, max_jump_to_flag_neighbors=10000, min_jump_to_flag_neighbors=100,
        flag_4_neighbors=True, dqflags=DQFLAGS)
    print(data[0, 4, :, :])
    print(gdq[0, 4, :, :])
    assert gdq[0, 4, 5, 1] == DQFLAGS['JUMP_DET']
    assert gdq[0, 4, 6, 1] == DQFLAGS['DO_NOT_USE']

    # This section of code will fail without the fixes for PR #239 that prevent
    # the double flagging pixels with jump which already have do_not_use or saturation set.
    num_cores = "5"
    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint32)
    pdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    readnoise_2d = np.ones((nrows, ncols), dtype=np.float32) * readnoise
    gain_2d = np.ones((nrows, ncols), dtype=np.float32) * 3
    err = np.zeros(shape=(nrows, ncols), dtype=np.float32)
    data[0, 4:, 5, 1] = 2000
    gdq[0, 4:, 6, 1] = DQFLAGS['DO_NOT_USE']
    gdq, pdq, total_primary_crs, number_extended_events, stddev = detect_jumps(
        frames_per_group, data, gdq, pdq, err, gain_2d, readnoise_2d, rejection_thresh=5, three_grp_thresh=6,
        four_grp_thresh=7, max_cores=num_cores, max_jump_to_flag_neighbors=10000, min_jump_to_flag_neighbors=100,
        flag_4_neighbors=True, dqflags=DQFLAGS)
    assert gdq[0, 4, 5, 1] == DQFLAGS['JUMP_DET']
    assert gdq[0, 4, 6, 1] == DQFLAGS['DO_NOT_USE'] #This value would have been 5 without the fix.


def test_multiprocessing_big():
    nints = 1
    nrows = 2048
    ncols = 7
    ngroups = 13
    readnoise = 10
    frames_per_group = 1

    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    readnoise_2d = np.ones((nrows, ncols), dtype=np.float32) * readnoise
    gain_2d = np.ones((nrows, ncols), dtype=np.float32) * 4
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint32)
    pdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    err = np.zeros(shape=(nrows, ncols), dtype=np.float32)
    num_cores = "1"
    data[0, 4:, 204, 5] = 2000
    gdq[0, 4:, 204, 6] = DQFLAGS['DO_NOT_USE']
    gdq, pdq, total_primary_crs, number_extended_events, stddev = detect_jumps(
        frames_per_group, data, gdq, pdq, err, gain_2d, readnoise_2d, rejection_thresh=5, three_grp_thresh=6,
        four_grp_thresh=7, max_cores=num_cores, max_jump_to_flag_neighbors=10000, min_jump_to_flag_neighbors=100,
        flag_4_neighbors=True, dqflags=DQFLAGS)
    print(data[0, 4, :, :])
    print(gdq[0, 4, :, :])
    assert gdq[0, 4, 204, 5] == DQFLAGS['JUMP_DET']
    assert gdq[0, 4, 205, 5] == DQFLAGS['JUMP_DET']
    assert gdq[0, 4, 204, 6] == DQFLAGS['DO_NOT_USE']

    # This section of code will fail without the fixes for PR #239 that prevent
    # the double flagging pixels with jump which already have do_not_use or saturation set.
    num_cores = "10"
    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint32)
    pdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    readnoise_2d = np.ones((nrows, ncols), dtype=np.float32) * readnoise
    gain_2d = np.ones((nrows, ncols), dtype=np.float32) * 3
    err = np.zeros(shape=(nrows, ncols), dtype=np.float32)
    data[0, 4:, 204, 5] = 2000
    gdq[0, 4:, 204, 6] = DQFLAGS['DO_NOT_USE']
    gdq, pdq, total_primary_crs, number_extended_events, stddev = detect_jumps(
        frames_per_group, data, gdq, pdq, err, gain_2d, readnoise_2d, rejection_thresh=5, three_grp_thresh=6,
        four_grp_thresh=7, max_cores=num_cores, max_jump_to_flag_neighbors=10000, min_jump_to_flag_neighbors=100,
        flag_4_neighbors=True, dqflags=DQFLAGS)
    assert gdq[0, 4, 204, 5] == DQFLAGS['JUMP_DET']
    assert gdq[0, 4, 205, 5] == DQFLAGS['JUMP_DET']
    assert gdq[0, 4, 204, 6] == DQFLAGS['DO_NOT_USE'] #This value would have been 5 without the fix.


def test_find_simple_ellipse():
    plane = np.zeros(shape=(5, 5), dtype=np.uint8)
    plane[2, 2] = DQFLAGS["JUMP_DET"]
    plane[3, 2] = DQFLAGS["JUMP_DET"]
    plane[1, 2] = DQFLAGS["JUMP_DET"]
    plane[2, 3] = DQFLAGS["JUMP_DET"]
    plane[2, 1] = DQFLAGS["JUMP_DET"]
    plane[1, 3] = DQFLAGS["JUMP_DET"]
    plane[2, 4] = DQFLAGS["JUMP_DET"]
    plane[3, 3] = DQFLAGS["JUMP_DET"]
    ellipse = find_ellipses(plane, DQFLAGS["JUMP_DET"], 1)
    assert ellipse[0][2] == pytest.approx(45.0, 1e-3)  # 90 degree rotation
    assert ellipse[0][0] == pytest.approx((2.5, 2.0))  # center


def test_find_ellipse2():
    plane = np.zeros(shape=(5, 5), dtype=np.uint8)
    plane[1, :] = [0, DQFLAGS["JUMP_DET"], DQFLAGS["JUMP_DET"], DQFLAGS["JUMP_DET"], 0]
    plane[2, :] = [0, DQFLAGS["JUMP_DET"], DQFLAGS["JUMP_DET"], DQFLAGS["JUMP_DET"], 0]
    plane[3, :] = [0, DQFLAGS["JUMP_DET"], DQFLAGS["JUMP_DET"], DQFLAGS["JUMP_DET"], 0]
    ellipses = find_ellipses(plane, DQFLAGS["JUMP_DET"], 1)
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
    cube[1, 3, 3] = DQFLAGS["SATURATED"]
    cube[1, 2, 3] = DQFLAGS["SATURATED"]
    cube[1, 3, 4] = DQFLAGS["SATURATED"]
    cube[1, 4, 3] = DQFLAGS["SATURATED"]
    cube[1, 3, 2] = DQFLAGS["SATURATED"]
    cube[1, 2, 2] = DQFLAGS["JUMP_DET"]
    sat_circles = find_ellipses(cube[grp, :, :], DQFLAGS["SATURATED"], 1)
    new_cube, persist_jumps = extend_saturation(
        cube, grp, sat_circles, DQFLAGS["SATURATED"], DQFLAGS["JUMP_DET"],
        1.1, persist_jumps,
    )

    assert new_cube[grp, 2, 2] == DQFLAGS["SATURATED"]
    assert new_cube[grp, 4, 4] == DQFLAGS["SATURATED"]
    assert new_cube[grp, 4, 5] == 0


def test_flag_large_events_nosnowball():
    cube = np.zeros(shape=(1, 5, 7, 7), dtype=np.uint8)
    # cross of saturation with no jump
    cube[0, 0:2, 3, 3] = DQFLAGS["SATURATED"]
    cube[0, 0:2, 2, 3] = DQFLAGS["SATURATED"]
    cube[0, 0:2, 3, 4] = DQFLAGS["SATURATED"]
    cube[0, 0:2, 4, 3] = DQFLAGS["SATURATED"]
    cube[0, 0:2, 3, 2] = DQFLAGS["SATURATED"]
    # cross of saturation surrounding by jump -> snowball but sat core is not new
    # should have no snowball trigger
    cube[0, 2, 3, 3] = DQFLAGS["SATURATED"]
    cube[0, 2, 2, 3] = DQFLAGS["SATURATED"]
    cube[0, 2, 3, 4] = DQFLAGS["SATURATED"]
    cube[0, 2, 4, 3] = DQFLAGS["SATURATED"]
    cube[0, 2, 3, 2] = DQFLAGS["SATURATED"]
    cube[0, 2, 1, 1:6] = DQFLAGS["JUMP_DET"]
    cube[0, 2, 5, 1:6] = DQFLAGS["JUMP_DET"]
    cube[0, 2, 1:6, 1] = DQFLAGS["JUMP_DET"]
    cube[0, 2, 1:6, 5] = DQFLAGS["JUMP_DET"]
    flag_large_events(
        cube,
        DQFLAGS["JUMP_DET"],
        DQFLAGS["SATURATED"],
        min_sat_area=1,
        min_jump_area=6,
        expand_factor=1.9,
        edge_size=1,
        sat_required_snowball=True,
        min_sat_radius_extend=1,
        sat_expand=1.1,
    )
    assert cube[0, 2, 2, 2] == 0
    assert cube[0, 2, 3, 6] == 0


def test_flag_large_events_withsnowball():
    cube = np.zeros(shape=(1, 5, 7, 7), dtype=np.uint8)
    # cross of saturation surrounding by jump -> snowball
    cube[0, 2, 3, 3] = DQFLAGS["SATURATED"]
    cube[0, 2, 2, 3] = DQFLAGS["SATURATED"]
    cube[0, 2, 3, 4] = DQFLAGS["SATURATED"]
    cube[0, 2, 4, 3] = DQFLAGS["SATURATED"]
    cube[0, 2, 3, 2] = DQFLAGS["SATURATED"]
    cube[0, 2, 1, 1:6] = DQFLAGS["JUMP_DET"]
    cube[0, 2, 5, 1:6] = DQFLAGS["JUMP_DET"]
    cube[0, 2, 1:6, 1] = DQFLAGS["JUMP_DET"]
    cube[0, 2, 1:6, 5] = DQFLAGS["JUMP_DET"]
    cube, total_snowballs = flag_large_events(
        cube,
        DQFLAGS["JUMP_DET"],
        DQFLAGS["SATURATED"],
        min_sat_area=1,
        min_jump_area=6,
        expand_factor=1.9,
        edge_size=0,
        sat_required_snowball=True,
        min_sat_radius_extend=0.5,
        sat_expand=1.1,
    )
    assert cube[0, 1, 2, 2] == 0
    assert cube[0, 1, 3, 5] == 0
    assert cube[0, 2, 0, 0] == 0
    assert cube[0, 2, 1, 0] == DQFLAGS["JUMP_DET"]  # Jump was extended
    assert cube[0, 2, 2, 2] == DQFLAGS["SATURATED"]  # Saturation was extended
    assert cube[0, 2, 3, 6] == DQFLAGS["JUMP_DET"]


def test_flag_large_events_groupedsnowball():
    cube = np.zeros(shape=(1, 5, 7, 7), dtype=np.uint8)
    # cross of saturation surrounding by jump -> snowball
#    cube[0, 1, :, :] = DQFLAGS["JUMP_DET"]
#    cube[0, 2, :, :] = DQFLAGS["JUMP_DET"]
    cube[0, 2, 1:6, 1:6] = DQFLAGS["JUMP_DET"]
    cube[0, 1, 1:6, 1:6] = DQFLAGS["JUMP_DET"]

    cube[0, 2, 3, 3] = DQFLAGS["SATURATED"]
    cube[0, 2, 2, 3] = DQFLAGS["SATURATED"]
    cube[0, 2, 3, 4] = DQFLAGS["SATURATED"]
    cube[0, 2, 4, 3] = DQFLAGS["SATURATED"]
    cube[0, 2, 3, 2] = DQFLAGS["SATURATED"]
    outgdq, num_snowballs = flag_large_events(
        cube,
        DQFLAGS["JUMP_DET"],
        DQFLAGS["SATURATED"],
        min_sat_area=1,
        min_jump_area=6,
        expand_factor=1.9,
        edge_size=0,
        sat_required_snowball=True,
        min_sat_radius_extend=0.5,
        sat_expand=1.1,
    )

    assert outgdq[0, 2, 1, 0] == DQFLAGS["JUMP_DET"]  # Jump was extended
    assert outgdq[0, 2, 2, 2] == DQFLAGS["SATURATED"]  # Saturation was extended


def test_flag_large_events_withsnowball_noextension():
    cube = np.zeros(shape=(1, 5, 7, 7), dtype=np.uint8)
    # cross of saturation surrounding by jump -> snowball
    cube[0, 2, 3, 3] = DQFLAGS["SATURATED"]
    cube[0, 2, 2, 3] = DQFLAGS["SATURATED"]
    cube[0, 2, 3, 4] = DQFLAGS["SATURATED"]
    cube[0, 2, 4, 3] = DQFLAGS["SATURATED"]
    cube[0, 2, 3, 2] = DQFLAGS["SATURATED"]
    cube[0, 2, 1, 1:6] = DQFLAGS["JUMP_DET"]
    cube[0, 2, 5, 1:6] = DQFLAGS["JUMP_DET"]
    cube[0, 2, 1:6, 1] = DQFLAGS["JUMP_DET"]
    cube[0, 2, 1:6, 5] = DQFLAGS["JUMP_DET"]
    cube, num_snowballs = flag_large_events(
        cube,
        DQFLAGS["JUMP_DET"],
        DQFLAGS["SATURATED"],
        min_sat_area=1,
        min_jump_area=6,
        expand_factor=1.9,
        edge_size=0,
        sat_required_snowball=True,
        min_sat_radius_extend=0.5,
        sat_expand=1.1,
        max_extended_radius=1,
    )
    assert cube[0, 1, 2, 2] == 0
    assert cube[0, 1, 3, 5] == 0
    assert cube[0, 2, 0, 0] == 0
    assert cube[0, 2, 1, 0] == 0  # Jump was NOT extended due to max_extended_radius=1
    assert cube[0, 2, 2, 2] == 0  # Saturation was NOT extended due to max_extended_radius=1


def test_find_faint_extended():
    nint, ngrps, ncols, nrows = 1, 66, 25, 25
    data = np.zeros(shape=(nint, ngrps, nrows, ncols), dtype=np.float32)
    gdq = np.zeros_like(data, dtype=np.uint32)
    pdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    pdq[0, 0] = 1
    pdq[1, 1] = 2147483648
    gain = 4
    readnoise = np.ones(shape=(nrows, ncols), dtype=np.float32) * 6.0 * gain
    rng = np.random.default_rng(12345)
    data[0, 1:, 14:20, 15:20] = 6 * gain * 6.0 * np.sqrt(2)
    data = data + rng.normal(size=(nint, ngrps, nrows, ncols)) * readnoise
    fits.writeto("data.fits", data, overwrite=True)
    gdq, num_showers = find_faint_extended(
        data,
        gdq,
        pdq,
        readnoise * np.sqrt(2),
        1,
        100,
        DQFLAGS,
        snr_threshold=1.2,
        min_shower_area=10,
        inner=1,
        outer=2.6,
        sat_flag=2,
        jump_flag=4,
        ellipse_expand=1.,
        num_grps_masked=1,
    )
    #  Check that all the expected samples in group 2 are flagged as jump and
    #  that they are not flagged outside
    fits.writeto("gdq.fits", gdq, overwrite=True)
#    assert num_showers == 1
    assert np.all(gdq[0, 1, 22, 14:23] == 0)
    assert gdq[0, 1, 16, 18] == DQFLAGS['JUMP_DET']
    assert np.all(gdq[0, 1, 11:22, 16:19] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 1, 22, 16:19] == 0)
    assert np.all(gdq[0, 1, 10, 16:19] == 0)
    #  Check that the same area is flagged in the first group after the event
    assert np.all(gdq[0, 2, 22, 14:23] == 0)
    assert gdq[0, 2, 16, 18] == DQFLAGS['JUMP_DET']
    assert np.all(gdq[0, 2, 11:22, 16:19] == DQFLAGS["JUMP_DET"])
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
        gdq, num_showers = find_faint_extended(
            data,
            gdq,
            pdq,
            readnoise * np.sqrt(2),
            1,
            100,
            snr_threshold=3,
            min_shower_area=10,
            inner=1,
            outer=2.6,
            sat_flag=2,
            jump_flag=4,
            ellipse_expand=1.1,
            num_grps_masked=0,
        )


# No shower is found because the event is identical in all ints
def test_find_faint_extended_sigclip():
    nint, ngrps, ncols, nrows = 101, 6, 30, 30
    data = np.zeros(shape=(nint, ngrps, nrows, ncols), dtype=np.float32)
    gdq = np.zeros_like(data, dtype=np.uint8)
    pdq = np.zeros(shape=(nrows, ncols), dtype=np.uint8)
    gain = 4
    readnoise = np.ones(shape=(nrows, ncols), dtype=np.float32) * 6.0 * gain
    rng = np.random.default_rng(12345)
    data[0, 1:, 14:20, 15:20] = 6 * gain * 1.7
    data = data + rng.normal(size=(nint, ngrps, nrows, ncols)) * readnoise
    min_shower_area=20
    gdq, num_showers = find_faint_extended(
        data,
        gdq,
        pdq,
        readnoise,
        1,
        100,
        DQFLAGS,
        snr_threshold=1.3,
        min_shower_area=min_shower_area,
        inner=1,
        outer=2,
        sat_flag=2,
        jump_flag=4,
        ellipse_expand=1.1,
        num_grps_masked=3,
    )
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
    gdq, num_showers = find_faint_extended(data, gdq, pdq, readnoise, 1, 100,
                                           snr_threshold=1.3,
                                           min_shower_area=20, inner=1,
                                           outer=2, sat_flag=2, jump_flag=4,
                                           ellipse_expand=1.1, num_grps_masked=3,
                                           dqflags=DQFLAGS)
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


def test_find_last_grp():
    assert (find_last_grp(grp=5, ngrps=7, num_grps_masked=0) == 6)
    assert (find_last_grp(grp=5, ngrps=7, num_grps_masked=2) == 7)
    assert (find_last_grp(grp=5, ngrps=7, num_grps_masked=3) == 7)
    assert (find_last_grp(grp=5, ngrps=6, num_grps_masked=1) == 6)
    assert (find_last_grp(grp=5, ngrps=6, num_grps_masked=0) == 6)
    assert (find_last_grp(grp=5, ngrps=6, num_grps_masked=2) == 6)
    assert (find_last_grp(grp=5, ngrps=8, num_grps_masked=0) == 6)
    assert (find_last_grp(grp=5, ngrps=8, num_grps_masked=1) == 7)
    assert (find_last_grp(grp=5, ngrps=8, num_grps_masked=2) == 8)
