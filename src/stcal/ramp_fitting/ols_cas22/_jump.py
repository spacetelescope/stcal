# cython: language_level=3str


"""
This module contains all the functions needed to execute jump detection for the
    Castentano+22 ramp fitting algorithm.

    The _ramp module contains the actual ramp fitting algorithm, this module
    contains a driver for the algorithm and detection of jumps/splitting ramps.

Structs
-------
Thresh : struct
    intercept - constant * log10(slope)
        - intercept : float
            The intercept of the jump threshold
        - constant : float
            The constant of the jump threshold

JumpFits : struct
    All the data on a given pixel's ramp fit with (or without) jump detection
        - average : RampFit
            The average of all the ramps fit for the pixel
        - jumps : vector[int]
            The indices of the resultants which were detected as jumps
        - fits : vector[RampFit]
            All of the fits for each ramp fit for the pixel
        - index : RampQueue
            The RampIndex representations corresponding to each fit in fits

Enums
-----
FixedOffsets : enum
    Enumerate the different pieces of information computed for jump detection
        which only depend on the read pattern.

PixelOffsets : enum
    Enumerate the different pieces of information computed for jump detection
        which only depend on the given pixel (independent of specific ramp).

JUMP_DET : value
    A the fixed value for the jump detection dq flag.

(Public) Functions
------------------
fill_fixed_values : function
    Pre-compute all the values needed for jump detection for a given read_pattern,
        this is independent of the pixel involved.

fit_jumps : function
    Compute all the ramps for a single pixel using the Casertano+22 algorithm
        with jump detection. This is a driver for the ramp fit algorithm in general
        meaning it automatically handles splitting ramps across dq flags in addition
        to splitting across detected jumps (if jump detection is turned on).
"""
