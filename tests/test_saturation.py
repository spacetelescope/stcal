"""

Unit tests for saturation flagging

"""

import numpy as np

from stcal.saturation.saturation import flag_saturated_pixels

# dictionary with required DQ flags
DQFLAGS = {'DO_NOT_USE': 1, 'SATURATED': 2, 'AD_FLOOR': 64,
           'NO_SAT_CHECK': 2097152}
ATOD_LIMIT = 65535.  # Hard DN limit of 16-bit A-to-D converter


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

    gdq, pdq = flag_saturated_pixels(data, gdq, pdq, sat_thresh, sat_dq,
                                     ATOD_LIMIT, DQFLAGS)
    print(gdq[:,:, 2:6, 2:6])

    # Make sure that groups with signal > saturation limit get flagged
    satindex = np.argmax(data[0, :, 5, 5] == satvalue)
    assert np.all(gdq[0, satindex:, 5, 5] == DQFLAGS['SATURATED'])


def test_adjacent_pixel_flagging():
    """ Test to see if specified number of adjacent pixels next to a saturated
        pixel are also flagged, and that the edges of the dq array are treated
        correctly when this is done. """

    # Create inputs, data, and saturation maps
    data = np.ones((1, 2, 5, 5)).astype('float32')
    gdq = np.zeros((1, 2, 5, 5)).astype('uint32')
    pdq = np.zeros((5, 5)).astype('uint32')
    sat_thresh = np.ones((5, 5)) * 60000   # sat. thresh is 60000
    sat_dq = np.zeros((5, 5)).astype('uint32')

    # saturate a few pixels just in the first group
    # (0, 0) and (1, 1) to test adjacent pixels
    data[0, 0, 0, 0] = 62000
    data[0, 0, 0, 1] = 62000
    data[0, 0, 3, 3] = 62000

    gdq, pdq = flag_saturated_pixels(data, gdq, pdq, sat_thresh, sat_dq,
                                     ATOD_LIMIT, DQFLAGS)

    sat_locs = np.where(np.bitwise_and(gdq, DQFLAGS['SATURATED']) ==
                        DQFLAGS['SATURATED'])

    assert sat_locs[0].all() == 0
    assert np.all(sat_locs[1] == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                           1, 1, 1, 1, 1, 1]))
    assert np.all(sat_locs[2] == np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                                           4, 4, 4, 0, 0, 0, 1, 1, 1, 2, 2, 2,
                                           3, 3, 3, 4, 4, 4]))
    assert np.all(sat_locs[3] == np.array([0, 1, 2, 0, 1, 2, 2, 3, 4, 2, 3, 4,
                                           2, 3, 4, 0, 1, 2, 0, 1, 2, 2, 3, 4,
                                           2, 3, 4, 2, 3, 4]))
