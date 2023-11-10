import numpy as np
import pytest

from stcal.jump.jump import (
    calc_num_slices,
    extend_saturation,
    find_ellipses,
    find_faint_extended,
    flag_large_events,
    point_inside_ellipse,
)

DQFLAGS = {"JUMP_DET": 4, "SATURATED": 2, "DO_NOT_USE": 1, "GOOD": 0, "NO_GAIN_VALUE": 8}


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
    grp = 1
    min_sat_radius_extend = 1
    cube[1, 3, 3] = DQFLAGS["SATURATED"]
    cube[1, 2, 3] = DQFLAGS["SATURATED"]
    cube[1, 3, 4] = DQFLAGS["SATURATED"]
    cube[1, 4, 3] = DQFLAGS["SATURATED"]
    cube[1, 3, 2] = DQFLAGS["SATURATED"]
    cube[1, 2, 2] = DQFLAGS["JUMP_DET"]
    sat_circles = find_ellipses(cube[grp, :, :], DQFLAGS["SATURATED"], 1)
    new_cube = extend_saturation(
        cube, grp, sat_circles, DQFLAGS["SATURATED"], min_sat_radius_extend, expansion=1.1
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
    flag_large_events(
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
    cube[0, 1, :, :] = DQFLAGS["JUMP_DET"]
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
        edge_size=0,
        sat_required_snowball=True,
        min_sat_radius_extend=0.5,
        sat_expand=1.1,
    )
    #    assert cube[0, 1, 2, 2] == 0
    #    assert cube[0, 1, 3, 5] == 0
    assert cube[0, 2, 0, 0] == 0
    assert cube[0, 2, 1, 0] == DQFLAGS["JUMP_DET"]  # Jump was extended
    assert cube[0, 2, 2, 2] == DQFLAGS["SATURATED"]  # Saturation was extended


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
    flag_large_events(
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
    nint, ngrps, ncols, nrows = 1, 6, 30, 30
    data = np.zeros(shape=(nint, ngrps, nrows, ncols), dtype=np.float32)
    gdq = np.zeros_like(data, dtype=np.uint8)
    gain = 4
    readnoise = np.ones(shape=(nrows, ncols), dtype=np.float32) * 6.0 * gain
    rng = np.random.default_rng(12345)
    data[0, 1:, 14:20, 15:20] = 6 * gain * 1.7
    data = data + rng.normal(size=(nint, ngrps, nrows, ncols)) * readnoise
    gdq, num_showers = find_faint_extended(
        data,
        gdq,
        readnoise,
        1,
        100,
        snr_threshold=1.3,
        min_shower_area=20,
        inner=1,
        outer=2,
        sat_flag=2,
        jump_flag=4,
        ellipse_expand=1.1,
        num_grps_masked=3,
    )
    #  Check that all the expected samples in group 2 are flagged as jump and
    #  that they are not flagged outside
    assert num_showers == 3
    assert np.all(gdq[0, 1, 22, 14:23] == 0)
    assert np.all(gdq[0, 1, 21, 16:20] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 1, 20, 15:22] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 1, 19, 15:23] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 1, 18, 14:23] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 1, 17, 14:23] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 1, 16, 14:23] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 1, 15, 14:22] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 1, 14, 16:22] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 1, 13, 17:21] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 1, 12, 14:23] == 0)
    assert np.all(gdq[0, 1, 12:23, 24] == 0)
    assert np.all(gdq[0, 1, 12:23, 13] == 0)
    #  Check that the same area is flagged in the first group after the event
    assert np.all(gdq[0, 2, 22, 14:23] == 0)
    assert np.all(gdq[0, 2, 21, 16:20] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 2, 20, 15:22] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 2, 19, 15:23] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 2, 18, 14:23] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 2, 17, 14:23] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 2, 16, 14:23] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 2, 15, 14:22] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 2, 14, 16:22] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 2, 13, 17:21] == DQFLAGS["JUMP_DET"])
    assert np.all(gdq[0, 2, 12, 14:23] == 0)
    assert np.all(gdq[0, 2, 12:22, 24] == 0)
    assert np.all(gdq[0, 2, 12:22, 13] == 0)

    #  Check that the flags are not applied in the 3rd group after the event
    assert np.all(gdq[0, 4, 12:22, 14:23]) == 0


# No shower is found because the event is identical in all ints
def test_find_faint_extended_sigclip():
    nint, ngrps, ncols, nrows = 101, 6, 30, 30
    data = np.zeros(shape=(nint, ngrps, nrows, ncols), dtype=np.float32)
    gdq = np.zeros_like(data, dtype=np.uint8)
    gain = 4
    readnoise = np.ones(shape=(nrows, ncols), dtype=np.float32) * 6.0 * gain
    rng = np.random.default_rng(12345)
    data[0, 1:, 14:20, 15:20] = 6 * gain * 1.7
    data = data + rng.normal(size=(nint, ngrps, nrows, ncols)) * readnoise
    gdq, num_showers = find_faint_extended(
        data,
        gdq,
        readnoise,
        1,
        100,
        snr_threshold=1.3,
        min_shower_area=20,
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


def test_inside_ellipse5():
    ellipse = ((0, 0), (1, 2), -10)
    point = (1, 0.6)
    result = point_inside_ellipse(point, ellipse)
    assert not result


def test_inside_ellipse4():
    ellipse = ((0, 0), (1, 2), 0)
    point = (1, 0.5)
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
