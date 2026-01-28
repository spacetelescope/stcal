import numpy as np
import numpy.testing
import pytest

from stcal.skymatch import SkyStats


@pytest.mark.parametrize("skystat", ["mean", "median", "mode"])
def test_skystat(skystat):
    data = np.full((10, 10), 1.0, dtype="f8")
    stats = SkyStats(skystat=skystat)
    sky_value, npix = stats.calc_sky(data)
    numpy.testing.assert_allclose(sky_value, 1.0)
    assert npix == 100


def test_midpt():
    data = np.array([0, 1, 2, 3], dtype="f8")
    stats = SkyStats(skystat="midpt")
    sky_value, npix = stats.calc_sky(data)
    # TODO shouldn't this be closer to 1.5
    numpy.testing.assert_allclose(sky_value, 1.032796, rtol=1e-6)
    assert npix == 4


def test_limit():
    # TODO why does this fail for lower=0.5 upper=1.5 and data=1.0?
    data = np.full((10, 10), 10.0, dtype="f8")
    data[:5, :] = 0
    stats = SkyStats(skystat="mean", lower=5.0, upper=15.0)
    sky_value, npix = stats.calc_sky(data)
    numpy.testing.assert_allclose(sky_value, 10.0)
    assert npix == 50
