# cython: language_level=3str

"""
This module contains all the functions needed to execute the Casertano+22 ramp
    fitting algorithm on its own without jump detection.

    The _jump module contains a driver function which calls the `fit_ramp` function
    from this module iteratively. This evvetively handles dq flags and detected
    jumps simultaneously.

Structs
-------
RampIndex : struct
    - start : int
        Index of the first resultant in the ramp
    - end : int
        Index of the last resultant in the ramp (so indexing of ramp requires end + 1)

RampFit : struct
    - slope : float
        The slope fit to the ramp
    - read_var : float
        The read noise variance for the fit
    - poisson_var : float
        The poisson variance for the fit

RampQueue : vector[RampIndex]
    Vector of RampIndex objects (convenience typedef)

Classes
-------
ReadPattern :
    Container class for all the metadata derived from the read pattern, this
    is just a temporary object to allow us to return multiple memory views from
    a single function.

(Public) Functions
------------------
init_ramps : function
    Create the initial ramp "queue" for each pixel in order to handle any initial
    "dq" flags passed in from outside.

from_read_pattern : function
    Derive the input data from the the read pattern
        This is faster than using __init__ or __cinit__ to construct the object with
        these calls.

fit_ramps : function
    Implementation of running the Casertano+22 algorithm on a (sub)set of resultants
    listed for a single pixel
"""
