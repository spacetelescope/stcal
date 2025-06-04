import numpy as np
import pytest

from stcal.jump.twopoint_difference import calc_med_first_diffs, find_crs
from stcal.jump.twopoint_difference_class import TwoPointParams


DQFLAGS = {"JUMP_DET": 4, "SATURATED": 2, "DO_NOT_USE": 1}


def setup_data(dims, rnoise):
    nints, ngroups, nrows, ncols = dims


    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    read_noise = np.full((nrows, ncols), rnoise, dtype=np.float32)
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint32)

    return data, gdq, read_noise


def default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10):
    twopt_p = TwoPointParams()
    twopt_p.normal_rej_thresh = rej


    twopt_p.two_diff_rej_thresh = 3
    twopt_p.three_diff_rej_thresh = 3
    twopt_p.nframes = 1
            
    twopt_p.flag_4_neighbors = _4n
    twopt_p.max_jump_to_flag_neighbors = mx_flag
    twopt_p.min_jump_to_flag_neighbors = mn_flag

    twopt_p.fl_jump = DQFLAGS["JUMP_DET"]
    twopt_p.fl_sat = DQFLAGS["SATURATED"]
    twopt_p.fl_dnu = DQFLAGS["DO_NOT_USE"]

    twopt_p.after_jump_flag_e1 = 0.
    twopt_p.after_jump_flag_n1 = 0
    twopt_p.after_jump_flag_e2 = 0.
    twopt_p.after_jump_flag_n2 = 0

    twopt_p.minimum_groups = 3
    twopt_p.minimum_sigclip_groups = 100
    twopt_p.only_use_ints = True
    twopt_p.min_diffs_single_pass = 10

    twopt_p.copy_arrs = True

    return twopt_p


def test_varying_groups():
    nints, ngroups, nrows, ncols = 1, 5, 2, 2
    dims = nints, ngroups, nrows, ncols
    rnoise = 8

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, :, 0, 0] = [10, 20, 30, 530, 540]
    data[0, :, 0, 1] = [10, 20, 30, 530, np.nan]
    data[0, :, 1, 0] = [10, 20, 530, np.nan, np.nan]
    data[0, :, 1, 1] = [10, 520, np.nan, np.nan, np.nan]

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.array_equal(out_gdq[0, :, 0, 0], [0, 0, 0, 4, 0])
    assert np.array_equal(out_gdq[0, :, 0, 1], [0, 0, 0, 4, 0])
    assert np.array_equal(out_gdq[0, :, 1, 0], [0, 0, 4, 0, 0])
    assert np.array_equal(out_gdq[0, :, 1, 1], [0, 0, 0, 0, 0])


def test_multint_pixel():
    nints, ngroups, nrows, ncols = 7, 4, 2, 2
    dims = nints, ngroups, nrows, ncols
    rnoise = 8

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, :, 0, 0] = (-24,   -15,     0,    13)
    data[1, :, 0, 0] = (-24,   -11,     6,    21)
    data[2, :, 0, 0] = (-40,   -28,   -24,    -4)
    data[3, :, 0, 0] = (-11,     3,    11,    24)
    data[4, :, 0, 0] = (-43 ,  -24,   -12,     1)
    data[5, :, 0, 0] = (-45,  8537, 17380, 17437)
    data[6, :, 0, 0] = (-178,  -156,  -139,  -125)

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert (np.array_equal([0, 4, 4, 4], out_gdq[5, :, 0, 0]))


def test_nocrs_noflux():
    nints, ngroups, nrows, ncols = 1, 4, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 0  # no CR found


def test_5grps_cr3_noflux():
    nints, ngroups, nrows, ncols = 1, 5, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0:2, 100, 100] = 10.0
    data[0, 2:5, 100, 100] = 1000

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.argmax(out_gdq[0, :, 100, 100]) == 2  # find the CR in the expected group
    data[0, 0, 100, 100] = 10.0
    data[0, 1:6, 100, 100] = 1000


def test_4grps_2ints_cr2_noflux():
    nints, ngroups, nrows, ncols = 2, 5, 2, 2
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 1, 1, 1] = 5
    data[1, 0, 1, 1] = 10.0
    data[1, 1:6, 1, 1] = 1000

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert(4 == np.max(out_gdq))  # a CR was found

    # XXX not sure why this is run a second time
    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.argmax(out_gdq[1, :, 1, 1]) == 1  # find the CR in the expected group
    assert(1 == np.argmax(out_gdq[1, :, 1, 1]))  # find the CR in the expected group


def test_6grps_negative_differences_zeromedian():
    nints, ngroups, nrows, ncols = 1, 6, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 100
    data[0, 1, 100, 100] = 90
    data[0, 2, 100, 100] = 95
    data[0, 3, 100, 100] = 105
    data[0, 4, 100, 100] = 100
    data[0, 5, 100, 100] = 100

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 0  # no CR was found


def test_5grps_cr2_negjumpflux():
    nints, ngroups, nrows, ncols = 1, 5, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 1000.0
    data[0, 1:6, 100, 100] = 10

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.argmax(out_gdq[0, :, 100, 100]) == 1  # find the CR in the expected group


def test_3grps_cr2_noflux():
    nints, ngroups, nrows, ncols = 1, 3, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 10.0
    data[0, 1:4, 100, 100] = 1000
    data[0, 0, 99, 99] = 10.0
    data[0, 2:4, 99, 99] = 1000

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert(np.array_equal([0, 4, 0], out_gdq[0, :, 100, 100]))
    assert (np.array_equal([0, 0, 4], out_gdq[0, :, 99, 99]))


def test_2ints_2grps_noflux():
    nints, ngroups, nrows, ncols = 2, 2, 2, 2
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 1, 1] = 10.0
    data[0, 1:3, 1, 1] = 1000
    data[1, 0, 0, 0] = 10.0
    data[1, 1:3, 0, 0] = 1000

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)
    twopt_p.minimum_groups = 2

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert(np.array_equal([0, 4], out_gdq[0, :, 1, 1]))
    assert (np.array_equal([0, 4], out_gdq[1, :, 0, 0]))


def test_4grps_cr2_noflux():
    nints, ngroups, nrows, ncols = 1, 4, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 10.0
    data[0, 1:4, 100, 100] = 1000

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.argmax(out_gdq[0, :, 100, 100]) == 1  # find the CR in the expected group


def test_6grps_cr2_nframe2():
    nints, ngroups, nrows, ncols = 1, 6, 2, 2
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 1, 1] = 10.0
    data[0, 1, 1, 1] = 500
    data[0, 2, 1, 1] = 1002
    data[0, 3, 1, 1] = 1001
    data[0, 4, 1, 1] = 1005
    data[0, 5, 1, 1] = 1015

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=2, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert (np.array_equal([0, 4, 4, 0, 0, 0], out_gdq[0, :, 1, 1]))
    assert (np.max(out_gdq[0, :, 0, 0]) == 0)
    assert (np.max(out_gdq[0, :, 1, 0]) == 0)
    assert (np.max(out_gdq[0, :, 0, 1]) == 0)


def test_4grps_twocrs_2nd_4th():
    nints, ngroups, nrows, ncols = 1, 4, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    nframes = 1
    data[0, 0, 100, 100] = 10.0
    data[0, 1, 100, 100] = 60
    data[0, 2, 100, 100] = 60
    data[0, 3, 100, 100] = 115

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found


def test_5grps_twocrs_2nd_5th():
    nints, ngroups, nrows, ncols = 1, 5, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 10.0
    data[0, 1, 100, 100] = 60
    data[0, 2, 100, 100] = 60
    data[0, 3, 100, 100] = 60
    data[0, 4, 100, 100] = 115

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.array_equal([0, 4, 0, 0, 4], out_gdq[0, :, 100, 100])


def test_5grps_twocrs_2nd_5thbig():
    nints, ngroups, nrows, ncols = 1, 5, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 10.0
    data[0, 1, 100, 100] = 600
    data[0, 2, 100, 100] = 600
    data[0, 3, 100, 100] = 600
    data[0, 4, 100, 100] = 2115

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.array_equal([0, 4, 0, 0, 4], out_gdq[0, :, 100, 100])


def test_10grps_twocrs_2nd_8th_big():
    nints, ngroups, nrows, ncols = 1, 10, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 10.0
    data[0, 1, 100, 100] = 60
    data[0, 2, 100, 100] = 60
    data[0, 3, 100, 100] = 60
    data[0, 4, 100, 100] = 60
    data[0, 5, 100, 100] = 60
    data[0, 6, 100, 100] = 60
    data[0, 7, 100, 100] = 2115
    data[0, 8, 100, 100] = 2115
    data[0, 9, 100, 100] = 2115

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.array_equal([0, 4, 0, 0, 0, 0, 0, 4, 0, 0], out_gdq[0, :, 100, 100])


def test_10grps_twocrs_10percenthit():
    nints, ngroups, nrows, ncols = 1, 10, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0:200, 0, 100, 100] = 10.0
    data[0:200, 1, 100, 100] = 60
    data[0:200, 2, 100, 100] = 60
    data[0:200, 3, 100, 100] = 60
    data[0:200, 4, 100, 100] = 60
    data[0:200, 5, 100, 100] = 60
    data[0:200, 6, 100, 100] = 60
    data[0:200, 7, 100, 100] = 2115
    data[0:200, 8, 100, 100] = 2115
    data[0:200, 9, 100, 100] = 2115

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=2, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.array_equal([0, 4, 0, 0, 0, 0, 0, 4, 0, 0], out_gdq[0, :, 100, 100])


def test_5grps_twocrs_2nd_5thbig_nframes2():
    nints, ngroups, nrows, ncols = 1, 5, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10 * np.sqrt(2)

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 10.0
    data[0, 1, 100, 100] = 60
    data[0, 2, 100, 100] = 60
    data[0, 3, 100, 100] = 60
    data[0, 4, 100, 100] = 2115

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=2, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.array_equal([0, 4, 0, 0, 4], out_gdq[0, :, 100, 100])


def test_6grps_twocrs_2nd_5th():
    nints, ngroups, nrows, ncols = 1, 6, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10 * np.sqrt(2)

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 10.0
    data[0, 1, 100, 100] = 60
    data[0, 2, 100, 100] = 60
    data[0, 3, 100, 100] = 60
    data[0, 4, 100, 100] = 115
    data[0, 5, 100, 100] = 115

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.array_equal([0, 4, 0, 0, 4, 0], out_gdq[0, :, 100, 100])


def test_6grps_twocrs_2nd_5th_nframes2():
    nints, ngroups, nrows, ncols = 1, 6, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10 * np.sqrt(2)

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 10.0
    data[0, 1, 100, 100] = 60
    data[0, 2, 100, 100] = 60
    data[0, 3, 100, 100] = 60
    data[0, 4, 100, 100] = 115
    data[0, 5, 100, 100] = 115

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=2, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.array_equal([0, 4, 0, 0, 4, 0], out_gdq[0, :, 100, 100])


def test_6grps_twocrs_twopixels_nframes2():
    nints, ngroups, nrows, ncols = 1, 6, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10 * np.sqrt(2)

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 10.0
    data[0, 1, 100, 100] = 60
    data[0, 2, 100, 100] = 60
    data[0, 3, 100, 100] = 60
    data[0, 4, 100, 100] = 115
    data[0, 5, 100, 100] = 115
    data[0, 0, 200, 100] = 10.0
    data[0, 1, 200, 100] = 10.0
    data[0, 2, 200, 100] = 60
    data[0, 3, 200, 100] = 60
    data[0, 4, 200, 100] = 115
    data[0, 5, 200, 100] = 115

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=2, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.array_equal([0, 4, 0, 0, 4, 0], out_gdq[0, :, 100, 100])
    assert np.array_equal([0, 0, 4, 0, 4, 0], out_gdq[0, :, 200, 100])


def test_5grps_cr2_negslope():
    nints, ngroups, nrows, ncols = 1, 5, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 1, 1] = 100.0
    data[0, 1, 1, 1] = 0
    data[0, 2, 1, 1] = -200
    data[0, 3, 1, 1] = -260
    data[0, 4, 1, 1] = -360

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.array_equal([0, 0, 4, 0, 0], out_gdq[0, :, 1, 1])


def test_6grps_1cr():
    nints, ngroups, nrows, ncols = 1, 6, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 0
    data[0, 1, 100, 100] = 10
    data[0, 2, 100, 100] = 21
    data[0, 3, 100, 100] = 33
    data[0, 4, 100, 100] = 46
    data[0, 5, 100, 100] = 1146

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert out_gdq[0, 5, 100, 100] == 4


def test_7grps_1cr():
    nints, ngroups, nrows, ncols = 1, 7, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 0
    data[0, 1, 100, 100] = 10
    data[0, 2, 100, 100] = 21
    data[0, 3, 100, 100] = 33
    data[0, 4, 100, 100] = 46
    data[0, 5, 100, 100] = 60
    data[0, 6, 100, 100] = 1160

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert out_gdq[0, 6, 100, 100] == 4


def test_8grps_1cr():
    nints, ngroups, nrows, ncols = 1, 8, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 0
    data[0, 1, 100, 100] = 10
    data[0, 2, 100, 100] = 21
    data[0, 3, 100, 100] = 33
    data[0, 4, 100, 100] = 46
    data[0, 5, 100, 100] = 60
    data[0, 6, 100, 100] = 1160
    data[0, 7, 100, 100] = 1175

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert out_gdq[0, 6, 100, 100] == 4


def test_9grps_1cr_1sat():
    nints, ngroups, nrows, ncols = 1, 9, 2, 2
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 1, 1] = 0
    data[0, 1, 1, 1] = 10
    data[0, 2, 1, 1] = 21
    data[0, 3, 1, 1] = 33
    data[0, 4, 1, 1] = 46
    data[0, 5, 1, 1] = 60
    data[0, 6, 1, 1] = 1160
    data[0, 7, 1, 1] = 1175
    data[0, 8, 1, 1] = 6175
    gdq[0, 8, 1, 1] = DQFLAGS["SATURATED"]

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert out_gdq[0, 6, 1, 1] == 4


def test_10grps_1cr_2sat():
    nints, ngroups, nrows, ncols = 1, 10, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 0
    data[0, 1, 100, 100] = 10
    data[0, 2, 100, 100] = 21
    data[0, 3, 100, 100] = 33
    data[0, 4, 100, 100] = 46
    data[0, 5, 100, 100] = 60
    data[0, 6, 100, 100] = 1160
    data[0, 7, 100, 100] = 1175
    data[0, 8, 100, 100] = 6175
    data[0, 9, 100, 100] = 6175
    gdq[0, 8, 100, 100] = DQFLAGS["SATURATED"]
    gdq[0, 9, 100, 100] = DQFLAGS["SATURATED"]

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert out_gdq[0, 6, 100, 100] == 4


def test_11grps_1cr_3sat():
    nints, ngroups, nrows, ncols = 1, 11, 2, 2
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 1, 1] = 0
    data[0, 1, 1, 1] = 20
    data[0, 2, 1, 1] = 39
    data[0, 3, 1, 1] = 57
    data[0, 4, 1, 1] = 74
    data[0, 5, 1, 1] = 90
    data[0, 6, 1, 1] = 1160
    data[0, 7, 1, 1] = 1175
    data[0, 8, 1, 1] = 6175
    data[0, 9, 1, 1] = 6175
    data[0, 10, 1, 1] = 6175
    gdq[0, 8, 1, 1] = DQFLAGS["SATURATED"]
    gdq[0, 9, 1, 1] = DQFLAGS["SATURATED"]
    gdq[0, 10, 1, 1] = DQFLAGS["SATURATED"]

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert out_gdq[0, 6, 1, 1] == 4


def test_11grps_0cr_3donotuse():
    nints, ngroups, nrows, ncols = 1, 11, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 0
    data[0, 1, 100, 100] = 18
    data[0, 2, 100, 100] = 39
    data[0, 3, 100, 100] = 57
    data[0, 4, 100, 100] = 74
    data[0, 5, 100, 100] = 90
    data[0, 6, 100, 100] = 115
    data[0, 7, 100, 100] = 131
    data[0, 8, 100, 100] = 150
    data[0, 9, 100, 100] = 6175
    data[0, 10, 100, 100] = 6175
    gdq[0, 0, 100, 100] = DQFLAGS["DO_NOT_USE"]
    gdq[0, 9, 100, 100] = DQFLAGS["DO_NOT_USE"]
    gdq[0, 10, 100, 100] = DQFLAGS["DO_NOT_USE"]

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.array_equal([0, 0, 0, 0, 0, 0, 0, 0], out_gdq[0, 1:-2, 100, 100])


@pytest.mark.skip("Copied, but checks nothing and is named wrong")
def test_5grps_nocr():
    nints, ngroups, nrows, ncols = 1, 6, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 0
    data[0, 1, 100, 100] = 10
    data[0, 2, 100, 100] = 21
    data[0, 3, 100, 100] = 33
    data[0, 4, 100, 100] = 46

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)


@pytest.mark.skip("Copied, but checks nothing")
def test_6grps_nocr():
    nints, ngroups, nrows, ncols = 1, 6, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 10

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 0
    data[0, 1, 100, 100] = 10
    data[0, 2, 100, 100] = 21
    data[0, 3, 100, 100] = 33
    data[0, 4, 100, 100] = 46
    data[0, 5, 100, 100] = 60

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)


def test_10grps_cr2_gt3sigma():
    crmag = 16
    nints, ngroups, nrows, ncols = 1, 10, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 5

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 0
    data[0, 1:11, 100, 100] = crmag

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.array_equal([0, 4, 0, 0, 0, 0, 0, 0, 0, 0], out_gdq[0, :, 100, 100])


def test_10grps_cr2_3sigma_nocr():
    crmag = 15
    nints, ngroups, nrows, ncols = 1, 10, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 5

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 0
    data[0, 1:11, 100, 100] = crmag

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=1, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 0  # a CR was found
    assert np.array_equal([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], out_gdq[0, :, 100, 100])


@pytest.mark.skip("Fails for some reason")
def test_10grps_cr2_gt3sigma_2frames():
    crmag = 16
    nints, ngroups, nrows, ncols = 1, 10, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 5 * np.sqrt(2)

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 100] = 0
    data[0, 1:11, 100, 100] = crmag

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=2, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.array_equal([0, 4, 0, 0, 0, 0, 0, 0, 0, 0], out_gdq[0, :, 100, 100])


@pytest.mark.skip("Fails for some reason")
def test_10grps_cr2_gt3sigma_2frames_offdiag():
    crmag = 16
    nints, ngroups, nrows, ncols = 1, 10, 204, 204
    dims = nints, ngroups, nrows, ncols
    rnoise = 5 * np.sqrt(2)

    data, gdq, read_noise = setup_data(dims, rnoise)
    data[0, 0, 100, 110] = 0
    data[0, 1:11, 100, 110] = crmag

    twopt_p = default_twopt_p(
        rej=3, _1drej=3, _3drej=3, nframes=2, _4n=False, mx_flag=200, mn_flag=10)

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    assert np.max(out_gdq) == 4  # a CR was found
    assert np.array_equal([0, 4, 0, 0, 0, 0, 0, 0, 0, 0], out_gdq[0, :, 100, 110])


def sigclip_twopt_p():
    twopt_p = TwoPointParams()

    twopt_p.normal_rej_thresh = 6.0
    twopt_p.two_diff_rej_thresh = 6.0
    twopt_p.three_diff_rej_thresh = 5.0
    twopt_p.nframes = (1,)
            
    twopt_p.flag_4_neighbors = False
    twopt_p.max_jump_to_flag_neighbors = 200.0
    twopt_p.min_jump_to_flag_neighbors = 10.0

    twopt_p.fl_jump = DQFLAGS["JUMP_DET"]
    twopt_p.fl_sat = DQFLAGS["SATURATED"]
    twopt_p.fl_dnu = DQFLAGS["DO_NOT_USE"]

    twopt_p.after_jump_flag_e1 = 1619.335205078125
    twopt_p.after_jump_flag_n1 = 16
    twopt_p.after_jump_flag_e2 = 0.0
    twopt_p.after_jump_flag_n2 = 0

    twopt_p.minimum_groups = 10
    twopt_p.minimum_sigclip_groups = 10
    # twopt_p.only_use_ints = True
    twopt_p.only_use_ints = False
    twopt_p.min_diffs_single_pass = 10

    twopt_p.copy_arrs = False

    return twopt_p


def set_sigma_clip_data(data):
    base_arr = np.array([
       [ 10.731703 ,  27.203512 , 106.22826  , 151.0139   , 212.16364  ,
        258.99268  , 288.5518   , 347.3305   , 415.43622  ],
       [ 81.00812  , 126.25662  , 182.64836  , 222.33426  , 271.6754   ,
        313.7703   , 340.5042   , 383.5762   , 440.76724  ],
       [ 25.141754 ,  44.785824 ,  90.29829  , 118.28577  , 209.94017  ,
        251.54932  , 259.02567  , 332.81836  , 390.73328  ],
       [ 42.45757  ,  87.34434  , 120.69542  , 181.82999  , 230.85397  ,
        283.2989   , 322.58997  , 359.593    , 413.067    ],
       [-20.901432 ,  21.450695 ,  74.89305  , 143.93784  , 171.35698  ,
        230.34262  , 292.6677   , 326.13788  , 376.32413  ],
       [-45.09379  ,   3.33447  ,  75.925735 , 117.744316 , 141.13106  ,
        216.48933  , 242.42192  , 298.9422   , 360.00977  ],
       [ 17.00263  ,  76.606064 ,  99.686714 , 134.43599  , 222.63557  ,
        262.86624  , 273.09332  , 314.8982   , 374.4669   ],
       [ 35.92628  , 100.91746  , 146.84523  , 191.37482  , 223.67198  ,
        284.9848   , 321.36685  , 405.71707  , 423.96967  ],
       [ 49.854065 ,  52.88707  , 101.63217  , 152.14273  , 185.52617  ,
        261.91086  , 283.9805   , 348.89255  , 395.10388  ],
       [ 46.19449  ,  95.038574 , 136.92563  , 213.52617  , 255.4488   ,
        285.599    , 353.41556  , 398.4853   , 450.24988  ],
       [ 20.631287 ,   6.7477093,  87.05258  , 102.348785 , 138.26501  ,
        199.50543  , 232.0937   , 271.54175  , 318.06076  ],
       [ 44.94166  ,  79.9948   , 152.88118  , 189.57803  , 232.15485  ,
        284.44135  , 295.0025   , 362.4202   , 408.2485   ],
       [ 49.948082 ,  76.186874 , 114.14371  , 181.95833  , 215.66791  ,
        263.49374  , 318.52133  , 375.40717  , 424.171    ],
       [ 33.684372 ,  57.108395 ,  99.87311  , 158.9294   , 207.78914  ,
        310.25403  , 349.13037  , 380.33728  , 423.91754  ],
       [  0.8763512,  57.26083  ,  98.34808  , 149.68803  , 200.28212  ,
        258.59805  , 300.5317   , 331.96344  , 372.13153  ]],
      dtype=np.float32)
    _, _, nrows, ncols = data.shape
    for row in range(nrows):
        for col in range(ncols):
            data[:, :, row, col] = base_arr
    return data


def test_det_jump_sigma_clipping():
    """
    Test jump detection for sigma clipping.
    """
    nints, ngroups, nrows, ncols = 15, 9, 5, 5
    dims = nints, ngroups, nrows, ncols
    rnoise = 16.613169
    crmag = 1000.0

    # Example taken from pixel dat[15:, :, 100, 204]
    # jw01366001001_04101_00001-seg001_nis_dark_current.fits
    data, gdq, read_noise = setup_data(dims, rnoise)
    data = set_sigma_clip_data(data)

    data[7, 3:, 3, 3] += crmag

    twopt_p = sigclip_twopt_p()

    out_gdq, row_below_gdq, rows_above_gdq, total_crs, stddev = find_crs(
        data, gdq, read_noise, twopt_p)

    jump = DQFLAGS["JUMP_DET"]
    check = np.array([0, 0, 0, jump, 0, 0, 0, 0, 0], dtype=np.uint32)
    assert np.array_equal(check, out_gdq[7, :, 3, 3])
