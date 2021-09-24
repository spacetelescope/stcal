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
