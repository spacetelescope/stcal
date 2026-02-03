import copy

import gwcs
import numpy as np
import pytest
from astropy.modeling.models import Scale, Shift

from stcal.skymatch import SkyGroup, SkyImage, SkyStats

IMAGE_SIZE = (10, 10)


@pytest.fixture
def wcs():
    # a simple 1 arcminute FOV wcs
    scales = [1 / (s * 60) for s in IMAGE_SIZE]
    return gwcs.WCS([["pixel", Scale(scales[0]) & Scale(scales[1])], ["world", None]])


@pytest.fixture
def image():
    return np.zeros(IMAGE_SIZE, dtype="f4")


@pytest.fixture
def mask(image):
    return np.ones(IMAGE_SIZE, dtype=bool)


@pytest.fixture
def meta():
    return {}


@pytest.fixture
def skystats():
    return SkyStats(skystat="mean")


@pytest.fixture
def skyimage(image, mask, wcs, skystats, meta):
    return SkyImage(image, mask, wcs.forward_transform, wcs.backward_transform, skystats, meta=meta)


@pytest.fixture
def skygroup(skyimage):
    return SkyGroup([skyimage])


@pytest.fixture(params=["skyimage", "skygroup"])
def skyinstance(request):
    yield request.getfixturevalue(request.param)


def test_sky_init(skyinstance):
    assert skyinstance.sky == 0.0


@pytest.mark.parametrize("value", [1, -1, 42])
def test_set_sky(skyinstance, value):
    skyinstance.sky = value
    assert skyinstance.sky == value
    if isinstance(skyinstance, SkyGroup):
        assert skyinstance[0].sky == value


@pytest.mark.parametrize("attr", ["meta", "image", "mask"])
def test_passthrough_attr(skyimage, attr, request):
    """Test __init__ arguments that become attributes are correctly assigned"""
    assert getattr(skyimage, attr) is request.getfixturevalue(attr)


def test_skyimage_is_sky_valid(skyimage):
    assert not skyimage.is_sky_valid


@pytest.mark.parametrize(
    "fill_value, initial_sky, expected_sky, delta",
    [
        (42, 0, 42, False),
        (42, 21, 42, False),
        (42, 21, 21, True),
    ],
)
def test_calc_sky(skyinstance, fill_value, initial_sky, expected_sky, delta):
    if isinstance(skyinstance, SkyImage):
        skyinstance.image[:] = fill_value
        expected_npix = skyinstance.image.size
    else:
        skyinstance[0].image[:] = fill_value
        expected_npix = skyinstance[0].image.size
    skyinstance.sky = initial_sky
    skyval, npix, polyarea = skyinstance.calc_sky(delta=delta)
    assert skyval == expected_sky
    assert npix == expected_npix
    assert np.isclose(polyarea, skyinstance._polygon.area())


@pytest.mark.parametrize("other_class", [SkyImage, SkyGroup])
def test_calc_sky_overlap(skyinstance, wcs, other_class):
    # make a partially overlapping skyimage
    offset_wcs = gwcs.WCS(
        (Shift(1) & Shift(0)) | wcs.forward_transform,
        input_frame="pixel",
        output_frame="word",
    )
    if isinstance(skyinstance, SkyGroup):
        reference = skyinstance[0]
    else:
        reference = skyinstance
    other = SkyImage(
        reference.image.copy(),
        reference.mask.copy(),
        offset_wcs.forward_transform,
        offset_wcs.backward_transform,
        copy.copy(reference.skystat),
    )
    # fill half image with one value
    reference.image[:, 1:] = 42
    other.image[:] = 42
    if other_class is SkyGroup:
        other = SkyGroup([other])
    skyval, npix, _ = skyinstance.calc_sky(overlap=other)
    assert skyval == 42
    assert npix == (IMAGE_SIZE[1] - 1) * IMAGE_SIZE[0]


def test_no_images_skygroup():
    with pytest.raises(ValueError, match="SkyGroup requires a list of images"):
        SkyGroup([])


def test_skygroup_len(skygroup):
    assert len(skygroup) == len(skygroup._images)


def test_skygroup_iter(skygroup):
    for i0, i1 in zip(skygroup, skygroup._images, strict=True):
        assert i0 is i1


def test_skygroup_getitem(skygroup):
    for i in range(len(skygroup)):
        assert skygroup[i] is skygroup._images[i]
