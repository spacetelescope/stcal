"""

Unit tests for saturation flagging

"""

import numpy as np

from stcal.saturation.saturation import flag_saturated_pixels

# dictionary with required DQ flags
DQFLAGS = {"DO_NOT_USE": 1, "SATURATED": 2, "AD_FLOOR": 64, "NO_SAT_CHECK": 2097152}
ATOD_LIMIT = 65535.0  # Hard DN limit of 16-bit A-to-D converter


def test_basic_saturation_flagging():
    # Create inputs, data, and saturation maps
    data = np.zeros((1, 5, 20, 20)).astype("float32")
    gdq = np.zeros((1, 5, 20, 20)).astype("uint32")
    pdq = np.zeros((20, 20)).astype("uint32")
    sat_thresh = np.ones((20, 20)) * 100000.0
    sat_dq = np.zeros((20, 20)).astype("uint32")

    # Add ramp values up to the saturation limit
    data[0, 0, 5, 5] = 0
    data[0, 1, 5, 5] = 20000
    data[0, 2, 5, 5] = 40000
    data[0, 3, 5, 5] = 60000  # Signal reaches saturation limit
    data[0, 4, 5, 5] = 62000

    # Set saturation value in the saturation model
    satvalue = 60000
    sat_thresh[5, 5] = satvalue

    gdq, pdq, _ = flag_saturated_pixels(data, gdq, pdq, sat_thresh, sat_dq, ATOD_LIMIT, DQFLAGS)

    # Make sure that groups with signal > saturation limit get flagged
    satindex = np.argmax(data[0, :, 5, 5] == satvalue)
    assert np.all(gdq[0, satindex:, 5, 5] == DQFLAGS["SATURATED"])


def test_read_pattern_saturation_flagging():
    """Check that the saturation threshold varies depending on how the reads
    are allocated into resultants."""

    # Create inputs, data, and saturation maps
    data = np.zeros((1, 5, 20, 20)).astype("float32")
    gdq = np.zeros((1, 5, 20, 20)).astype("uint32")
    pdq = np.zeros((20, 20)).astype("uint32")
    sat_thresh = np.ones((20, 20)) * 100000.0
    sat_dq = np.zeros((20, 20)).astype("uint32")

    # Add ramp values up to the saturation limit
    data[0, 0, 5, 5] = 0
    data[0, 1, 5, 5] = 20000
    data[0, 2, 5, 5] = 40000
    data[0, 3, 5, 5] = 60000  # Signal reaches saturation limit
    data[0, 4, 5, 5] = 62000

    # Set saturation value in the saturation model
    satvalue = 60000
    sat_thresh[5, 5] = satvalue

    # set read_pattern to have many reads in the third resultant, so that
    # its mean exposure time is much smaller than its last read time
    # (in this case, the ratio is 13 / 20).
    # This means that the effective saturation for the third resultant
    # is 60000 * 13 / 20 = 39000 and the third resultant should be marked
    # saturated.
    read_pattern = [[1], [2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]

    gdq, pdq, _ = flag_saturated_pixels(
        data, gdq, pdq, sat_thresh, sat_dq, ATOD_LIMIT, DQFLAGS, read_pattern=read_pattern
    )

    # Make sure that groups after the third get flagged
    assert np.all(gdq[0, 2:, 5, 5] == DQFLAGS["SATURATED"])


def test_no_sat_check_at_limit():
    """Test to verify that pixels at the A-to-D limit (65535), but flagged with
    NO_SAT_CHECK do NOT get flagged as saturated, and that their neighbors
    also do NOT get flagged."""

    # Create inputs, data, and saturation maps
    data = np.zeros((1, 5, 10, 10)).astype("float32")
    gdq = np.zeros((1, 5, 10, 10)).astype("uint32")
    pdq = np.zeros((10, 10)).astype("uint32")
    sat_thresh = np.ones((10, 10)) * 50000.0
    sat_dq = np.zeros((10, 10)).astype("uint32")

    # Add ramp values that are flat-lined at the A-to-D limit,
    # which is well above the sat_thresh of 50,000.
    data[0, 0, 5, 5] = ATOD_LIMIT
    data[0, 1, 5, 5] = ATOD_LIMIT
    data[0, 2, 5, 5] = ATOD_LIMIT
    data[0, 3, 5, 5] = ATOD_LIMIT
    data[0, 4, 5, 5] = ATOD_LIMIT

    # Set a DQ value of NO_SAT_CHECK
    sat_dq[5, 5] = DQFLAGS["NO_SAT_CHECK"]

    # Run the saturation flagging
    gdq, pdq, _ = flag_saturated_pixels(data, gdq, pdq, sat_thresh, sat_dq, ATOD_LIMIT, DQFLAGS, 1)

    # Make sure that no groups for the flat-lined pixel and all
    # of its neighbors are flagged as saturated.
    # Also make sure that NO_SAT_CHECK has been propagated to the
    # pixeldq array.
    assert np.all(gdq[0, :, 4:6, 4:6] != DQFLAGS["SATURATED"])
    assert pdq[5, 5] == DQFLAGS["NO_SAT_CHECK"]


def test_adjacent_pixel_flagging():
    """Test to see if specified number of adjacent pixels next to a saturated
    pixel are also flagged, and that the edges of the dq array are treated
    correctly when this is done."""

    # Create inputs, data, and saturation maps
    data = np.ones((1, 2, 5, 5)).astype("float32")
    gdq = np.zeros((1, 2, 5, 5)).astype("uint32")
    pdq = np.zeros((5, 5)).astype("uint32")
    sat_thresh = np.ones((5, 5)) * 60000  # sat. thresh is 60000
    sat_dq = np.zeros((5, 5)).astype("uint32")

    nints, ngroups, nrows, ncols = data.shape

    # saturate a few pixels just in the first group
    # (0, 0) and (1, 1) to test adjacent pixels
    data[0, 0, 0, 0] = 62000
    data[0, 0, 0, 1] = 62000
    data[0, 0, 3, 3] = 62000

    gdq, pdq, _ = flag_saturated_pixels(data, gdq, pdq, sat_thresh, sat_dq, ATOD_LIMIT, DQFLAGS)

    sat_locs = np.where(np.bitwise_and(gdq, DQFLAGS["SATURATED"]) == DQFLAGS["SATURATED"])

    """
    print(f"dims = {dims}")
    print(f"len(sat_locs = {len(sat_locs)})")
    for k in range(len(sat_locs)):
        ostr = np.array2string(sat_locs[k], separator=", ")
        print(f"sat_locs[{k}] = {ostr}")
    """
    # return

    assert sat_locs[0].all() == 0
    assert np.all(
        sat_locs[1]
        == np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )
    )
    assert np.all(
        sat_locs[2]
        == np.array(
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        )
    )
    assert np.all(
        sat_locs[3]
        == np.array(
            [0, 1, 2, 0, 1, 2, 2, 3, 4, 2, 3, 4, 2, 3, 4, 0, 1, 2, 0, 1, 2, 2, 3, 4, 2, 3, 4, 2, 3, 4]
        )
    )


def test_zero_frame():
    """
    Pixel 0 has fully saturated ramp with saturated frame 0.
    Pixel 1 has fully saturated ramp with good frame 0.
    Pixel 2 has a good ramp with good frame 0.

    The second integration has the ZERORAME swapped for pixels
    0 and 1, so the resulting zeroed out ZEROFRAME pixel are
    swapped.
    """
    darr1 = [11800.0, 11793.0, 11823.0, 11789.0, 11857.0]
    darr2 = [11800.0, 11793.0, 11823.0, 11789.0, 11857.0]
    darr3 = [10579.0, 10594.0, 10620.0, 10583.0, 10621.0]
    zarr = [11800.0, 10500.0, 10579.0]
    rarr = [11795.0, 11795.0, 60501.0]

    nints, ngroups, nrows, ncols = 2, len(darr1), 1, len(zarr)
    dims = nints, ngroups, nrows, ncols

    # Create inputs, data, and saturation maps
    data = np.zeros(dims, dtype=float)
    gdq = np.zeros(dims, dtype=np.uint32)
    pdq = np.zeros((nrows, ncols), dtype=np.uint32)
    zfrm = np.zeros((nints, nrows, ncols), dtype=float)
    ref = np.zeros((nrows, ncols), dtype=float)
    rdq = np.zeros((nrows, ncols), dtype=np.uint32)

    data[0, :, 0, 0] = np.array(darr1)
    data[0, :, 0, 1] = np.array(darr2)
    data[0, :, 0, 2] = np.array(darr3)

    data[1, :, 0, 0] = np.array(darr1)
    data[1, :, 0, 1] = np.array(darr2)
    data[1, :, 0, 2] = np.array(darr3)

    zfrm[0, 0, :] = np.array(zarr)
    zfrm[1, 0, :] = np.array([zarr[1], zarr[0], zarr[2]])
    ref[0, :] = np.array(rarr)

    # dictionary with required DQ flags
    dqflags = {"DO_NOT_USE": 1, "SATURATED": 2, "AD_FLOOR": 64, "NO_SAT_CHECK": 2097152}

    atod_limit = 65535.0  # Hard DN limit of 16-bit A-to-D converter

    gdq, pdq, zframe = flag_saturated_pixels(data, gdq, pdq, ref, rdq, atod_limit, dqflags, 0, zfrm)

    # Check DQ flags
    cdq = np.array([dqflags["SATURATED"]] * ngroups)
    z = np.array([0] * ngroups)
    check = np.zeros(gdq.shape, dtype=gdq.dtype)
    check[0, :, 0, 0] = cdq
    check[0, :, 0, 1] = cdq
    check[0, :, 0, 2] = z
    check[1, :, 0, 0] = cdq
    check[1, :, 0, 1] = cdq
    check[1, :, 0, 2] = z

    np.testing.assert_array_equal(check, gdq)

    # Check ZEROFRAME flagged elements are zeroed out.
    assert zframe[0, 0, 0] == 0.0
    assert zframe[1, 0, 1] == 0.0
