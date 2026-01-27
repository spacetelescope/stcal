"""
Test script for set_velocity_aberration.py
"""

from numpy import isclose

from stcal.velocity_aberration import compute_va_effects

# Testing constants
GOOD_VELOCITY = (100.0, 100.0, 100.0)
GOOD_POS = (359.0, -2.0)
GOOD_SCALE_FACTOR = 1.000316017905845
GOOD_APPARENT_RA = 359.01945099823
GOOD_APPARENT_DEC = -1.980247580394956


def test_compute_va_effects_valid():
    scale_factor, va_ra, va_dec = compute_va_effects(*GOOD_VELOCITY, *GOOD_POS)
    assert isclose(scale_factor, GOOD_SCALE_FACTOR)
    assert isclose(va_ra, GOOD_APPARENT_RA)
    assert isclose(va_dec, GOOD_APPARENT_DEC)


def test_compute_va_effects_zero_velocity():
    scale_factor, va_ra, va_dec = compute_va_effects(0.0, 0.0, 0.0, *GOOD_POS)
    assert isclose(scale_factor, 1.0, atol=1e-16)
    assert isclose(va_ra, GOOD_POS[0], atol=1e-16)
    assert isclose(va_dec, GOOD_POS[1], atol=1e-16)
