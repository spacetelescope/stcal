import copy

import gwcs
import numpy as np
import pytest
from astropy.modeling.models import Scale, Shift

from stcal.skymatch import SkyGroup, SkyImage, SkyStats, skymatch

IMAGE_SIZE = (10, 10)
FILL_VALUES = [21, 42]


@pytest.fixture()
def images():
    # a simple 1 arcminute FOV wcs
    scales = [1 / (s * 60) for s in IMAGE_SIZE]
    # shifted 1 degree to avoid dealing with RA wrapping
    wcs = gwcs.WCS(
        [["pixel", (Scale(scales[0]) & Scale(scales[1])) | (Shift(1) & Shift(1))], ["world", None]]
    )
    im = SkyImage(
        np.zeros(IMAGE_SIZE, dtype="f4"),
        np.ones(IMAGE_SIZE, dtype=bool),
        wcs.forward_transform,
        wcs.backward_transform,
        SkyStats("mean"),
    )
    images = [im, SkyGroup([copy.deepcopy(im)])]
    images[0].image[:] = FILL_VALUES[0]
    images[1][0].image[:] = FILL_VALUES[1]
    return images


@pytest.mark.parametrize(
    "method, match_down, skys",
    [
        ("local", True, FILL_VALUES),
        ("local", False, FILL_VALUES),
        ("match", True, [0, 21]),
        ("match", False, [-21, 0]),
        ("global", True, [21, 21]),
        ("global", False, [21, 21]),
        ("global+match", True , FILL_VALUES),
        ("global+match", False , FILL_VALUES),
    ],
)
@pytest.mark.parametrize("subtract", [True, False])
def test_skymatch(images, method, match_down, skys, subtract):
    skymatch(images, method, match_down, subtract)
    assert np.allclose([i.sky for i in images], skys)
    if subtract:
        assert np.allclose(images[0].image, FILL_VALUES[0] - skys[0])
        assert np.allclose(images[1][0].image, FILL_VALUES[1] - skys[1])
    else:
        assert np.allclose(images[0].image, FILL_VALUES[0])
        assert np.allclose(images[1][0].image, FILL_VALUES[1])


@pytest.mark.parametrize(
    "input_images, method, error_class, error_match",
    [
        ([], "foo", ValueError, "Unsupported 'skymethod'"),
        ([1, 2], "global", TypeError, "Each element of the 'images' must be"),
        ([], "global", ValueError, "Argument 'images' must contain at least one image"),
    ],
)
def test_skymatch_errors(input_images, method, error_class, error_match):
    with pytest.raises(error_class, match=error_match):
        skymatch(input_images, method)
