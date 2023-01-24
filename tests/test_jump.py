import numpy as np
import pytest
from astropy.io import fits

from stcal.jump.jump import flag_large_events, find_circles, find_ellipses, extend_saturation, \
    point_inside_ellipse, point_inside_rectangle, flag_large_events, detect_jumps, find_faint_extended

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


def test_find_ellipse2():
    plane = np.zeros(shape=(5, 5), dtype=np.uint8)
    plane[1,:] = [0,  DQFLAGS['JUMP_DET'], DQFLAGS['JUMP_DET'],DQFLAGS['JUMP_DET'], 0]
    plane[2,:] = [0, DQFLAGS['JUMP_DET'], DQFLAGS['JUMP_DET'], DQFLAGS['JUMP_DET'], 0]
    plane[3,:] = [0, DQFLAGS['JUMP_DET'], DQFLAGS['JUMP_DET'], DQFLAGS['JUMP_DET'], 0]
    ellipse = find_ellipses(plane, DQFLAGS['JUMP_DET'], 1)
    print(ellipse)
    assert ellipse == 1


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


def test_5580_plane8():
    incube = fits.getdata('input5580_jump_cube.fits')
    testcube = incube[:, 2:5, :, :]

    flag_large_events(testcube, DQFLAGS['JUMP_DET'], DQFLAGS['SATURATED'], min_sat_area=1,
                      min_jump_area=6,
                      expand_factor=2.0, use_ellipses=False,
                      sat_required_snowball=True, min_sat_radius_extend=2.5, sat_expand=2)
    fits.writeto("output_jump_cube8.fits", testcube, overwrite=True)

def test_2333_plane25():
    incube = fits.getdata('input_jump_cube23-33.fits')
    testcube = incube[:, 0:3, :, :]

    flag_large_events(testcube, DQFLAGS['JUMP_DET'], DQFLAGS['SATURATED'], min_sat_area=1,
                      min_jump_area=7,
                      expand_factor=2.5, use_ellipses=False,
                      sat_required_snowball=True, min_sat_radius_extend=2.5, sat_expand=2)
    fits.writeto("output_jump_cube2.fits", testcube, overwrite=True)

def test_edgeflage_130140():
    incube = fits.getdata('input_jump_cube130140.fits')
    testcube = incube[:, :-1, :, :]

    flag_large_events(testcube, DQFLAGS['JUMP_DET'], DQFLAGS['SATURATED'], min_sat_area=1,
                      min_jump_area=7,
                      expand_factor=2.5, use_ellipses=False,
                      sat_required_snowball=True, min_sat_radius_extend=2.5, sat_expand=3)
    fits.writeto("output_jump_cube2.fits", testcube, overwrite=True)

def test_miri_input():
    incube = fits.getdata('input_jump_cube_miri_01.fits')
    testcube = incube[:, 1:5, :, :]
    testcube = incube

    flag_large_events(testcube, DQFLAGS['JUMP_DET'], DQFLAGS['SATURATED'], min_sat_area=1,
                      min_jump_area=7,
                      expand_factor=2.5, use_ellipses=True,
                      sat_required_snowball=True, min_sat_radius_extend=2.5, sat_expand=3)
    fits.writeto("output_jump_cube_miri.fits", testcube, overwrite=True)

def test_inputjumpall():
    testcube = fits.getdata('input_jump_cube.fits')

    flag_large_events(testcube, DQFLAGS['JUMP_DET'], DQFLAGS['SATURATED'], min_sat_area=1,
                      min_jump_area=6,
                      expand_factor=2.0, use_ellipses=False,
                      sat_required_snowball=True, min_sat_radius_extend=2.5, sat_expand=2)
    fits.writeto("output_jump_cube2.fits", testcube, overwrite=True)


def test_detect_jumps_runaway():
    testcube = fits.getdata('smalldark2232_00_dark_current.fits')
    hdl = fits.open('smalldark2232_00_dark_current.fits')
    gdq = hdl['GROUPDQ'].data
    pdq = hdl['pixeldq'].data
    err = np.ones_like(pdq).astype('float64')
    gain_2d = fits.getdata('jwst_nirspec_gain_0023.fits')
    readnoise_2d = fits.getdata('jwst_nirspec_readnoise_0038.fits')

    detect_jumps(1, testcube, gdq, pdq, err,
                     gain_2d, readnoise_2d, 4,
                     5, 6, 'half', 1000,
                     10, True, DQFLAGS,
                     after_jump_flag_dn1=0.0,
                     after_jump_flag_n1=0,
                     after_jump_flag_dn2=0.0,
                     after_jump_flag_n2=0,
                     min_sat_area=1,
                     min_jump_area=5,
                     expand_factor=2.5,
                     use_ellipses=False,
                     sat_required_snowball=True,
                     expand_large_events=True)
    fits.writeto("output_gdq.fits", gdq, overwrite=True)


def test_extended1():
    incube = fits.getdata("dark_imager_wthexp_00_jump.fits")
    testcube = np.expand_dims(incube[0, 230:240, :, :], 0)
    testcube = np.expand_dims(incube[0, 0:50, :, :], 0)
    hdl = fits.open("dark_imager_wthexp_00_jump.fits")
    gdq = hdl['GROUPDQ'].data
    testgdq = np.expand_dims(gdq[0, 230:240, :, :], 0)
    testgdq = np.expand_dims(gdq[0, 0:50, :, :], 0)

    pdq = hdl['pixeldq'].data

    readnoise_2d = fits.getdata('MIRI_IMAGER_FASTR1_FULL_READNOISE_09.00.04.fits')
    gain_2d = np.zeros_like(readnoise_2d)
    gain_2d[:, :] = 3.9
    print("readnoise shape", readnoise_2d.shape)
    newgdq = find_faint_extended(testcube, testgdq, readnoise_2d, 1, snr_threshold=1.2, min_area=60,
                                 inner=1., outer=2.2)
    fits.writeto("newgdall.fits", newgdq, overwrite=True)
#    [print(cv.contourArea(con)) for con in contours]
##    [print('moment', cv.moments(con)['m10']/cv.moments(con)['m00'], cv.moments(con)['m01']/cv.moments(con)['m00']) for con in contours]

def test_extended_all():
    incube = fits.getdata("dark_imager_wthexp_00_jump.fits")
    testcube = np.expand_dims(incube[0, 230:232, :, :], 0)
    hdl = fits.open("dark_imager_wthexp_00_jump.fits")
    gdq = hdl['GROUPDQ'].data
    testgdq = np.expand_dims(gdq[0, 230:232, :, :], 0)
    pdq = hdl['pixeldq'].data
 #   gain_2d = fits.getdata('jwst_nirspec_gain_0023.fits')
    readnoise_2d = fits.getdata('MIRI_IMAGER_FASTR1_FULL_READNOISE_09.00.04.fits')
    print("readnoise shape", readnoise_2d.shape)
    newgdq = find_faint_extended(incube, gdq, readnoise_2d, 1, snr_threshold=1.2,
                                 min_area=70, inner=1,
                                 outer=2.2)
    fits.writeto("newgdq_big_th1.2.fits", newgdq, overwrite=True)