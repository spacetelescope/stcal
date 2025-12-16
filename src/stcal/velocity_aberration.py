"""
Utilities for velocity aberration correction.
"""

import logging

import numpy as np
from gwcs.geometry import CartesianToSpherical, SphericalToCartesian
from scipy.constants import speed_of_light

# Configure logging
logger = logging.getLogger(__name__)

SPEED_OF_LIGHT = speed_of_light / 1000  # km / s

__all__ = ["compute_va_effects_vector", "compute_va_effects"]


def compute_va_effects_vector(velocity_x, velocity_y, velocity_z, u):
    """
    Compute velocity aberration effects scale factor.

    Computes constant scale factor due to velocity aberration as well as
    corrected ``RA`` and ``DEC`` values, in vector form.

    Parameters
    ----------
    velocity_x, velocity_y, velocity_z : float
        The components of the velocity of JWST, in km / s with respect to
        the Sun.  These are celestial coordinates, with x toward the
        vernal equinox, y toward right ascension 90 degrees and declination
        0, z toward the north celestial pole.

    u : numpy.ndarray
        The vector form (``[u0, u1, u2]``) of right ascension and declination of the target (or some other
        point, such as the center of a detector) in the barycentric coordinate
        system.  The equator and equinox should be the same as the coordinate
        system for the velocity.

    Returns
    -------
    scale_factor : float
        Multiply the nominal image scale (e.g., in degrees per pixel) by
        this value to obtain the image scale corrected for the "aberration
        of starlight" due to the velocity of JWST with respect to the Sun.

    u_corr : numpy.ndarray
        Apparent position vector (``[ua0, ua1, ua2]``) in the moving telescope frame.
    """
    beta = np.array([velocity_x, velocity_y, velocity_z]) / SPEED_OF_LIGHT
    beta2 = np.dot(beta, beta)  # |beta|^2
    if beta2 == 0.0:
        logger.warning("Observatory speed is zero. Setting VA scale to 1.0")
        return 1.0, u

    u_beta = np.dot(u, beta)
    igamma = np.sqrt(1.0 - beta2)  # inverse of usual gamma
    scale_factor = (1.0 + u_beta) / igamma

    # Algorithm below is from Colin Cox notebook.
    # Also see: Instrument Science Report OSG-CAL-97-06 by Colin Cox (1997).
    u_corr = (igamma * u + beta * (1.0 + (1.0 - igamma) * u_beta / beta2)) / (1.0 + u_beta)

    return scale_factor, u_corr


def compute_va_effects(velocity_x, velocity_y, velocity_z, ra, dec):
    """
    Compute velocity aberration effects.

    Computes constant scale factor due to velocity aberration as well as
    corrected ``RA`` and ``DEC`` values.

    Parameters
    ----------
    velocity_x, velocity_y, velocity_z : float
        The components of the velocity of JWST, in km / s with respect to
        the Sun.  These are celestial coordinates, with x toward the
        vernal equinox, y toward right ascension 90 degrees and declination
        0, z toward the north celestial pole.

    ra, dec : float
        The right ascension and declination of the target (or some other
        point, such as the center of a detector) in the barycentric coordinate
        system.  The equator and equinox should be the same as the coordinate
        system for the velocity.

    Returns
    -------
    scale_factor : float
        Multiply the nominal image scale (e.g., in degrees per pixel) by
        this value to obtain the image scale corrected for the "aberration
        of starlight" due to the velocity of JWST with respect to the Sun.

    apparent_ra : float
        Apparent star position in the moving telescope frame.

    apparent_dec : float
        Apparent star position in the moving telescope frame.
    """
    u = np.asanyarray(SphericalToCartesian()(ra, dec))
    scale_factor, u_corr = compute_va_effects_vector(velocity_x, velocity_y, velocity_z, u)
    apparent_ra, apparent_dec = CartesianToSpherical()(*u_corr)
    return scale_factor, apparent_ra, apparent_dec
