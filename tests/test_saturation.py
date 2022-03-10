"""

Unit tests for saturation flagging

"""

import numpy as np

from stcal.saturation.saturation import flag_saturated_pixels


def test_basic_saturation_flagging():

    # Create inputs, data, and saturation maps
    data = np.zeros((1, 5, 20, 20)).astype('float32')
    gdq = np.zeros((1, 5, 20, 20)).astype('uint32')
    pdq = np.zeros((20, 20)).astype('uint32')
    sat_thresh = np.ones((20, 20)) * 100000.
    sat_dq = np.zeros((20, 20)).astype('uint32')

    # Add ramp values up to the saturation limit
    data[0, 0, 5, 5] = 0
    data[0, 1, 5, 5] = 20000
    data[0, 2, 5, 5] = 40000
    data[0, 3, 5, 5] = 60000   # Signal reaches saturation limit
    data[0, 4, 5, 5] = 62000

    # Set saturation value in the saturation model
    satvalue = 60000
    sat_thresh[5, 5] = satvalue

    # dictionary with required DQ flags
    dqflags = {'DO_NOT_USE': 1, 'SATURATED': 2, 'AD_FLOOR': 64,
               'NO_SAT_CHECK': 2097152}

    atod_limit = 65535.  # Hard DN limit of 16-bit A-to-D converter

    gdq, pdq = flag_saturated_pixels(data, gdq, pdq, sat_thresh, sat_dq,
                                     atod_limit, dqflags)

    # Make sure that groups with signal > saturation limit get flagged
    satindex = np.argmax(data[0, :, 5, 5] == satvalue)
    assert np.all(gdq[0, satindex:, 5, 5] == dqflags['SATURATED'])


def test_zero_frame():
    """
    Pixel 0 has fully saturated ramp with saturated frame 0.
    Pixel 1 has fully saturated ramp with good frame 0.
    Pixel 2 has a good ramp with good frame 0.

    The second integration has the ZERORAME swapped for pixels
    0 and 1, so the resulting zeroed out ZEROFRAME pixel are
    swapped.
    """
    darr1 = [11800., 11793., 11823., 11789., 11857.]
    darr2 = [11800., 11793., 11823., 11789., 11857.]
    darr3 = [10579., 10594., 10620., 10583., 10621.]
    zarr = [11800., 10500., 10579.]
    rarr = [11795., 11795., 60501.]

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
    dqflags = {'DO_NOT_USE': 1, 'SATURATED': 2, 'AD_FLOOR': 64,
               'NO_SAT_CHECK': 2097152}

    atod_limit = 65535.  # Hard DN limit of 16-bit A-to-D converter

    gdq, pdq = flag_saturated_pixels(
        data, gdq, pdq, ref, rdq, atod_limit, dqflags, zfrm)

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
    assert(zfrm[0, 0, 0] == 0.)
    assert(zfrm[1, 0, 1] == 0.)
