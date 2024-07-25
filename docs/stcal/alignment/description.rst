Description
============

This sub-package contains all the modules common to all missions.

WCS Info Dictionary
-------------------
Many of the functions in this submodule require a ``wcsinfo`` dictionary.
This dictionary contains information about the spacecraft pointing, and
requires at least the following keys:

- ``'ra_ref'``: The right ascension at the reference point in degrees.
- ``'dec_ref'``: The declination at the reference point in degrees.
- ``'v2_ref'``: The V2 reference point in arcseconds,
  with ``'v3_ref'`` maps to ``'ra_ref'`` and ``'dec_ref'``.
- ``'v3_ref'``: The V3 reference point in arcseconds,
  with ``'v2_ref'`` maps to ``'ra_ref'`` and ``'dec_ref'``.
- ``'roll_ref'``: Local roll angle associated with each aperture in degrees.
- ``'v3yangle'``: The angle between V3 and North in degrees.
- ``'vparity'``: The "V-parity" of the observation, which is 1 if the
  V3 axis is parallel to the detector Y-axis, and -1 if the V3 axis is
  parallel to the detector X-axis.
