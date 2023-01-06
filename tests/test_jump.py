import numpy as np
import pytest
from astropy.io import fits

from stcal.jump.jump import flag_large_events, find_circles, find_ellipses, extend_saturation, \
    point_inside_ellipse, point_inside_rectangle, flag_large_events

DQFLAGS = {'JUMP_DET': 4, 'SATURATED': 2, 'DO_NOT_USE': 1}

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
def test_find_simple_circle():
    plane = np.zeros(shape=(5, 5), dtype=np.uint8)
    plane[2, 2] = DQFLAGS['JUMP_DET']
    plane[3, 2] = DQFLAGS['JUMP_DET']
    plane[1, 2] = DQFLAGS['JUMP_DET']
    plane[2, 3] = DQFLAGS['JUMP_DET']
    plane[2, 1] = DQFLAGS['JUMP_DET']
    circle = find_circles(plane, DQFLAGS['JUMP_DET'], 1)
    assert circle[0][1] == pytest.approx(1.0, 1e-3)


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
    fits.writeto("start_sat_extend.fits", cube, overwrite=True)
    sat_circles = find_circles(cube[grp, :, :], DQFLAGS['SATURATED'], 1)
    new_cube = extend_saturation(cube, grp, sat_circles, DQFLAGS['SATURATED'], DQFLAGS['JUMP_DET'],
                                 min_sat_radius_extend, expansion=1)
    assert cube[grp, 2, 2] == DQFLAGS['SATURATED']
    assert cube[grp, 3, 5] == DQFLAGS['SATURATED']
    assert cube[grp, 3, 6] == 0
    fits.writeto("out_sat_extend.fits", cube, overwrite=True)



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
    fits.writeto("start_sat_extend2.fits", cube, overwrite=True)
    flag_large_events(cube, DQFLAGS['JUMP_DET'], DQFLAGS['SATURATED'], min_sat_area=1,
                      min_jump_area=6,
                      expand_factor=1.9, use_ellipses=False,
                      sat_required_snowball=True, min_sat_radius_extend=1, sat_expand=1)
    fits.writeto("out_flag_large_events.fits", cube, overwrite=True)
    assert cube[0, 1, 2, 2] == DQFLAGS['SATURATED']
    assert cube[0, 1, 3, 5] == DQFLAGS['SATURATED']
    assert cube[0, 1, 3, 6] == 0
    fits.writeto("out_flag_large_events.fits", cube, overwrite=True)

@pytest.mark.skip(reason="only for local testing")
def test_single_group():
    inplane = fits.getdata("jumppix.fits")
    indq = np.zeros(shape=(1, 1, inplane.shape[0], inplane.shape[1]), dtype=np.uint8)
    indq[0, 0, :, :] = inplane
    flag_large_events(indq, DQFLAGS['JUMP_DET'], DQFLAGS['SATURATED'], min_sat_area=1,
                      min_jump_area=15, expand_factor=1.9, use_ellipses=False,
                      sat_required_snowball=True, min_sat_radius_extend=1)
    fits.writeto("jumppix_expand.new.fits", indq, overwrite=True)

def test_inside_ellipse3():
        ellipse = ((0, 0), (1, 2), -45)
        point = (1, 2)
        result = point_inside_rectangle(point, ellipse)
        assert not result


def test_inside_ellipse5():
    ellipse = ((0, 0), (1, 2), -10)
    point = (1, 0.6)
    result = point_inside_ellipse(point, ellipse)
    assert not result

def test_inside_ellipse4():
    ellipse = ((0, 0), (1, 2), 0)
    point = (1, 0.5)
    result = point_inside_ellipse(point, ellipse)
    assert result

def test_inside_ellipes5():
    point = (1110.5, 870.5)
    ellipse = ((1111.0001220703125, 870.5000610351562), (10.60660171508789, 10.60660171508789), 45.0)
    result = point_inside_ellipse(point, ellipse)
    assert result

def test_plane23():
    incube = fits.getdata('input_jump_cube.fits')
    testcube = incube[:, 22:24, :, :]

    flag_large_events(testcube, DQFLAGS['JUMP_DET'], DQFLAGS['SATURATED'], min_sat_area=1,
                          min_jump_area=6,
                          expand_factor=2.0, use_ellipses=False,
                          sat_required_snowball=True, min_sat_radius_extend=2.5, sat_expand=2)
    fits.writeto("output_jump_cube23.fits", testcube, overwrite=True)

def test_plane13():
        incube = fits.getdata('input_jump_cube.fits')
        testcube = incube[:, 13:15, :, :]

        flag_large_events(testcube, DQFLAGS['JUMP_DET'], DQFLAGS['SATURATED'], min_sat_area=1,
                          min_jump_area=6,
                          expand_factor=2.0, use_ellipses=False,
                          sat_required_snowball=True, min_sat_radius_extend=2.5, sat_expand=2)
        fits.writeto("output_jump_cube13.fits", testcube, overwrite=True)
