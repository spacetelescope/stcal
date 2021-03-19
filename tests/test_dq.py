from stcal import dqflags

PIXEL = {'GOOD':             0,      # No bits set, all is good
         'DO_NOT_USE':       2**0,   # Bad pixel. Do not use.
         'SATURATED':        2**1,   # Pixel saturated during exposure
         'JUMP_DET':         2**2,   # Jump detected during exposure
         'DROPOUT':          2**3,   # Data lost in transmission
         'OUTLIER':          2**4,   # Flagged by outlier detection (was RESERVED_1)
         'PERSISTENCE':      2**5,   # High persistence (was RESERVED_2)
         'AD_FLOOR':         2**6,   # Below A/D floor (0 DN, was RESERVED_3)
         'RESERVED_4':       2**7,   #
         'UNRELIABLE_ERROR': 2**8,   # Uncertainty exceeds quoted error
         'NON_SCIENCE':      2**9,   # Pixel not on science portion of detector
         'DEAD':             2**10,  # Dead pixel
         'HOT':              2**11,  # Hot pixel
         'WARM':             2**12,  # Warm pixel
         'LOW_QE':           2**13,  # Low quantum efficiency
         'RC':               2**14,  # RC pixel
         'TELEGRAPH':        2**15,  # Telegraph pixel
         'NONLINEAR':        2**16,  # Pixel highly nonlinear
         'BAD_REF_PIXEL':    2**17,  # Reference pixel cannot be used
         'NO_FLAT_FIELD':    2**18,  # Flat field cannot be measured
         'NO_GAIN_VALUE':    2**19,  # Gain cannot be measured
         'NO_LIN_CORR':      2**20,  # Linearity correction not available
         'NO_SAT_CHECK':     2**21,  # Saturation check not available
         'UNRELIABLE_BIAS':  2**22,  # Bias variance large
         'UNRELIABLE_DARK':  2**23,  # Dark variance large
         'UNRELIABLE_SLOPE': 2**24,  # Slope variance large (i.e., noisy pixel)
         'UNRELIABLE_FLAT':  2**25,  # Flat variance large
         'OPEN':             2**26,  # Open pixel (counts move to adjacent pixels)
         'ADJ_OPEN':         2**27,  # Adjacent to open pixel
         'UNRELIABLE_RESET': 2**28,  # Sensitive to reset anomaly
         'MSA_FAILED_OPEN':  2**29,  # Pixel sees light from failed-open shutter
         'OTHER_BAD_PIXEL':  2**30,  # A catch-all flag
         'REFERENCE_PIXEL':  2**31,  # Pixel is a reference pixel
         }


def test_dqflags():
    assert dqflags.dqflags_to_mnemonics(1, PIXEL) == {'DO_NOT_USE'}
    assert dqflags.dqflags_to_mnemonics(7, PIXEL) == {'JUMP_DET', 'DO_NOT_USE', 'SATURATED'}
    assert dqflags.interpret_bit_flags('DO_NOT_USE + WARM', mnemonic_map=PIXEL) == 4097
