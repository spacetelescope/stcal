import numpy as np
import pytest
from astropy.io import fits

from stcal.jump.jump import flag_large_events, find_circles, find_ellipses

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


@pytest.mark.skip(reason="only for local testing")
def test_single_group():
    inplane = fits.getdata("jumppix.fits")
    indq = np.zeros(shape=(1, 1, inplane.shape[0], inplane.shape[1]), dtype=np.uint8)
    indq[0, 0, :, :] = inplane
    flag_large_events(indq, DQFLAGS['JUMP_DET'], DQFLAGS['SATURATED'], min_sat_area=1,
                      min_jump_area=15, max_offset=1, expand_factor=1.1, use_ellipses=True,
                      sat_required_snowball=False)
    fits.writeto("jumppix_expand.fits", indq, overwrite=True)
