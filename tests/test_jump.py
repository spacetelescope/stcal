import numpy as np
import pytest
from astropy.io import fits

from stcal.jump.jump import flag_large_events, find_ellipses, extend_saturation, \
    point_inside_ellipse, flag_large_events, detect_jumps, find_faint_extended

DQFLAGS = {'JUMP_DET': 4, 'SATURATED': 2, 'DO_NOT_USE': 1, 'GOOD': 0, 'NO_GAIN_VALUE': 8}

try:
    import cv2 as cv # noqa: F401

    OPENCV_INSTALLED = True
except ImportError:
    OPENCV_INSTALLED = False


@pytest.fixture(scope='function')
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


@pytest.mark.skipif(not OPENCV_INSTALLED, reason="`opencv-python` not installed")
def test_find_simple_ellipse():
    plane = np.zeros(shape=(5, 5), dtype=np.uint8)
    plane[2, 2] = DQFLAGS['JUMP_DET']
    plane[3, 2] = DQFLAGS['JUMP_DET']
    plane[1, 2] = DQFLAGS['JUMP_DET']
    plane[2, 3] = DQFLAGS['JUMP_DET']
    plane[2, 1] = DQFLAGS['JUMP_DET']
    plane[1, 3] = DQFLAGS['JUMP_DET']
    plane[2, 4] = DQFLAGS['JUMP_DET']
    plane[3, 3] = DQFLAGS['JUMP_DET']
    ellipse = find_ellipses(plane, DQFLAGS['JUMP_DET'], 1)
    assert ellipse[0][2] == pytest.approx(45.0, 1e-3)  # 90 degree rotation
    assert ellipse[0][0] == pytest.approx((2.5, 2.0))  # center


def test_find_ellipse2():
    plane = np.zeros(shape=(5, 5), dtype=np.uint8)
    plane[1,:] = [0,  DQFLAGS['JUMP_DET'], DQFLAGS['JUMP_DET'],DQFLAGS['JUMP_DET'], 0]
    plane[2,:] = [0, DQFLAGS['JUMP_DET'], DQFLAGS['JUMP_DET'], DQFLAGS['JUMP_DET'], 0]
    plane[3,:] = [0, DQFLAGS['JUMP_DET'], DQFLAGS['JUMP_DET'], DQFLAGS['JUMP_DET'], 0]
    ellipses = find_ellipses(plane, DQFLAGS['JUMP_DET'], 1)
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
    cube[1, 3, 3] = DQFLAGS['SATURATED']
    cube[1, 2, 3] = DQFLAGS['SATURATED']
    cube[1, 3, 4] = DQFLAGS['SATURATED']
    cube[1, 4, 3] = DQFLAGS['SATURATED']
    cube[1, 3, 2] = DQFLAGS['SATURATED']
    cube[1, 2, 2] = DQFLAGS['JUMP_DET']
    sat_circles = find_ellipses(cube[grp, :, :], DQFLAGS['SATURATED'], 1)
    new_cube = extend_saturation(cube, grp, sat_circles, DQFLAGS['SATURATED'], DQFLAGS['JUMP_DET'],
                                 min_sat_radius_extend, expansion=1.1)
    assert new_cube[grp, 2, 2] == DQFLAGS['SATURATED']
    assert new_cube[grp, 3, 5] == DQFLAGS['SATURATED']
    assert new_cube[grp, 0, 5] == 0


def test_flag_large_events():
    cube = np.zeros(shape=(1, 5, 7, 7), dtype=np.uint8)
    grp = 1
    min_sat_radius_extend = 1
    cube[0, 1, 3, 3] = DQFLAGS['SATURATED']
    cube[0, 1, 2, 3] = DQFLAGS['SATURATED']
    cube[0, 1, 3, 4] = DQFLAGS['SATURATED']
    cube[0, 1, 4, 3] = DQFLAGS['SATURATED']
    cube[0, 1, 3, 2] = DQFLAGS['SATURATED']
    cube[0, 2, 3, 4] = DQFLAGS['SATURATED']
    cube[0, 2, 2, 4] = DQFLAGS['SATURATED']
    cube[0, 2, 3, 4] = DQFLAGS['SATURATED']
    cube[0, 2, 4, 4] = DQFLAGS['SATURATED']
    cube[0, 2, 3, 4] = DQFLAGS['SATURATED']
    flag_large_events(cube, DQFLAGS['JUMP_DET'], DQFLAGS['SATURATED'], min_sat_area=1,
                      min_jump_area=6,
                      expand_factor=1.9, use_ellipses=False,
                      sat_required_snowball=True, min_sat_radius_extend=1, sat_expand=1.1)
    assert cube[0, 1, 2, 2] == DQFLAGS['SATURATED']
    assert cube[0, 1, 3, 5] == DQFLAGS['SATURATED']
    assert cube[0, 1, 1, 6] == DQFLAGS['JUMP_DET']


def test_find_faint_extended():
    nint, ngrps, ncols, nrows = 1, 6, 30, 30
    data = np.zeros(shape=(nint, ngrps, nrows, ncols), dtype=np.float32)
    gdq = np.zeros_like(data, dtype=np.uint8)
    gain = 4
    readnoise = np.ones(shape=(nrows, ncols), dtype=np.float32) * 6.0 * gain
    rng = np.random.default_rng(12345)
    data[0, 1:, 14:20, 15:20] = 6 * gain * 1.7
    data = data + rng.normal(size=(nint, ngrps, nrows, ncols)) * readnoise
    gdq, num_showers = find_faint_extended(data, gdq, readnoise, 1, snr_threshold=1.3, min_shower_area=20, inner=1,
                              outer=2, sat_flag=2, jump_flag=4, ellipse_expand=1.1, num_grps_masked=3)
    #  Check that all the expected samples in group 2 are flagged as jump and that they are not flagged outside
    assert (np.all(gdq[0, 1, 22, 14:23] == 0))
    assert (np.all(gdq[0, 1, 21, 16:20] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 1, 20, 15:22] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 1, 19, 15:23] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 1, 18, 14:23] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 1, 17, 14:23] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 1, 16, 14:23] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 1, 15, 14:22] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 1, 14, 16:22] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 1, 13, 17:21] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 1, 12, 14:23] == 0))
    assert (np.all(gdq[0, 1, 12:23, 24] == 0))
    assert (np.all(gdq[0, 1, 12:23, 13] == 0))
    #  Check that the same area is flagged in the first group after the event
    assert (np.all(gdq[0, 2, 22, 14:23] == 0))
    assert (np.all(gdq[0, 2, 21, 16:20] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 2, 20, 15:22] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 2, 19, 15:23] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 2, 18, 14:23] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 2, 17, 14:23] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 2, 16, 14:23] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 2, 15, 14:22] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 2, 14, 16:22] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 2, 13, 17:21] == DQFLAGS['JUMP_DET']))
    assert (np.all(gdq[0, 2, 12, 14:23] == 0))
    assert (np.all(gdq[0, 2, 12:22, 24] == 0))
    assert (np.all(gdq[0, 2, 12:22, 13] == 0))

    #  Check that the flags are not applied in the 3rd group after the event
    assert (np.all(gdq[0, 4, 12:22, 14:23]) == 0)


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


def test_inputjumpall():
    testcube = fits.getdata('data/large_event_input_dq_cube2.fits')
    flag_large_events(testcube, DQFLAGS['JUMP_DET'], DQFLAGS['SATURATED'], min_sat_area=1,
                      min_jump_area=6,
                      expand_factor=2.0, use_ellipses=False,
                      sat_required_snowball=True, min_sat_radius_extend=2.5, sat_expand=2)
    snowball_1 = testcube[0, 1,  260:280, 310:331]
    correct_snowball_1 = fits.getdata('data/snowball1.fits')
    snowball_diff = snowball_1 - correct_snowball_1
    assert (np.all(snowball_diff == 0))


def test_inputjump_sat_star():
    testcube = fits.getdata('data/input_gdq_flarge.fits')
    flag_large_events(testcube, DQFLAGS['JUMP_DET'], DQFLAGS['SATURATED'], min_sat_area=1,
                      min_jump_area=6,
                      expand_factor=2.0, use_ellipses=False,
                      sat_required_snowball=True, min_sat_radius_extend=2.5, sat_expand=2)
