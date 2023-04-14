import numpy

from stcal.saturation.saturation import flag_saturated_pixels

DQFLAGS = {"DO_NOT_USE": 1, "SATURATED": 2, "AD_FLOOR": 64, "NO_SAT_CHECK": 2097152}
ATOD_LIMIT = 65535.0  # Hard DN limit of 16-bit A-to-D converter


def time_basic_saturation_flagging():
    # Create inputs, data, and saturation maps
    data = numpy.zeros((1, 5, 20, 20)).astype("float32")
    gdq = numpy.zeros((1, 5, 20, 20)).astype("uint32")
    pdq = numpy.zeros((20, 20)).astype("uint32")
    sat_thresh = numpy.ones((20, 20)) * 100000.0
    sat_dq = numpy.zeros((20, 20)).astype("uint32")

    # Add ramp values up to the saturation limit
    data[0, 0, 5, 5] = 0
    data[0, 1, 5, 5] = 20000
    data[0, 2, 5, 5] = 40000
    data[0, 3, 5, 5] = 60000  # Signal reaches saturation limit
    data[0, 4, 5, 5] = 62000

    # Set saturation value in the saturation model
    satvalue = 60000
    sat_thresh[5, 5] = satvalue

    flag_saturated_pixels(data, gdq, pdq, sat_thresh, sat_dq, ATOD_LIMIT, DQFLAGS)


def time_no_sat_check_at_limit():
    """Test to verify that pixels at the A-to-D limit (65535), but flagged with
    NO_SAT_CHECK do NOT get flagged as saturated, and that their neighbors
    also do NOT get flagged."""

    # Create inputs, data, and saturation maps
    data = numpy.zeros((1, 5, 10, 10)).astype("float32")
    gdq = numpy.zeros((1, 5, 10, 10)).astype("uint32")
    pdq = numpy.zeros((10, 10)).astype("uint32")
    sat_thresh = numpy.ones((10, 10)) * 50000.0
    sat_dq = numpy.zeros((10, 10)).astype("uint32")

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
    flag_saturated_pixels(data, gdq, pdq, sat_thresh, sat_dq, ATOD_LIMIT, DQFLAGS, 1)


def time_adjacent_pixel_flagging():
    """Test to see if specified number of adjacent pixels next to a saturated
    pixel are also flagged, and that the edges of the dq array are treated
    correctly when this is done."""

    # Create inputs, data, and saturation maps
    data = numpy.ones((1, 2, 5, 5)).astype("float32")
    gdq = numpy.zeros((1, 2, 5, 5)).astype("uint32")
    pdq = numpy.zeros((5, 5)).astype("uint32")
    sat_thresh = numpy.ones((5, 5)) * 60000  # sat. thresh is 60000
    sat_dq = numpy.zeros((5, 5)).astype("uint32")

    # saturate a few pixels just in the first group
    # (0, 0) and (1, 1) to test adjacent pixels
    data[0, 0, 0, 0] = 62000
    data[0, 0, 0, 1] = 62000
    data[0, 0, 3, 3] = 62000

    flag_saturated_pixels(data, gdq, pdq, sat_thresh, sat_dq, ATOD_LIMIT, DQFLAGS)


def time_zero_frame():
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
    data = numpy.zeros(dims, dtype=float)
    gdq = numpy.zeros(dims, dtype=numpy.uint32)
    pdq = numpy.zeros((nrows, ncols), dtype=numpy.uint32)
    zfrm = numpy.zeros((nints, nrows, ncols), dtype=float)
    ref = numpy.zeros((nrows, ncols), dtype=float)
    rdq = numpy.zeros((nrows, ncols), dtype=numpy.uint32)

    data[0, :, 0, 0] = numpy.array(darr1)
    data[0, :, 0, 1] = numpy.array(darr2)
    data[0, :, 0, 2] = numpy.array(darr3)

    data[1, :, 0, 0] = numpy.array(darr1)
    data[1, :, 0, 1] = numpy.array(darr2)
    data[1, :, 0, 2] = numpy.array(darr3)

    zfrm[0, 0, :] = numpy.array(zarr)
    zfrm[1, 0, :] = numpy.array([zarr[1], zarr[0], zarr[2]])
    ref[0, :] = numpy.array(rarr)

    # dictionary with required DQ flags
    dqflags = {"DO_NOT_USE": 1, "SATURATED": 2, "AD_FLOOR": 64, "NO_SAT_CHECK": 2097152}

    atod_limit = 65535.0  # Hard DN limit of 16-bit A-to-D converter

    flag_saturated_pixels(data, gdq, pdq, ref, rdq, atod_limit, dqflags, 0, zfrm)
