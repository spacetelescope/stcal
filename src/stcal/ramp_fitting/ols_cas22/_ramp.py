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
import cython
from cython.cimports.libc.math import INFINITY, NAN, fabs, fmaxf, sqrt
from cython.cimports.libcpp.vector import vector
from cython.cimports.stcal.ramp_fitting.ols_cas22._ramp import RampFit, RampIndex, RampQueue


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.inline
@cython.ccall
def _fill_metadata(
    t_bar: cython.float[:],
    tau: cython.float[:],
    n_reads: cython.int[:],
    read_pattern: vector[vector[cython.int]],
    read_time: cython.float,
    n_resultants: cython.int,
) -> cython.void:
    n_read: cython.int

    i: cython.int
    j: cython.int
    resultant: vector[cython.int]
    for i in range(n_resultants):
        resultant = read_pattern[i]
        n_read = resultant.size()

        n_reads[i] = n_read
        t_bar[i] = 0
        tau[i] = 0

        for j in range(n_read):
            t_bar[i] += read_time * resultant[j]
            tau[i] += (2 * (n_read - j) - 1) * resultant[j]

        t_bar[i] /= n_read
        tau[i] *= read_time / n_read**2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.inline
@cython.ccall
def init_ramps(dq: cython.int[:], n_resultants: cython.int) -> RampQueue:
    """
    Create the initial ramp "queue" for each pixel
        if dq[index_resultant, index_pixel] == 0, then the resultant is in a ramp
        otherwise, the resultant is not in a ramp.

    Parameters
    ----------
    dq : int[n_resultants]
        DQ array
    n_resultants : int
        Number of resultants

    Returns
    -------
    RampQueue
        vector of RampIndex objects
            - vector with entry for each ramp found (last entry is last ramp found)
            - RampIndex with start and end indices of the ramp in the resultants
    """
    ramps: RampQueue = RampQueue()

    # Note: if start/end are -1, then no value has been assigned
    # ramp.start == -1 means we have not started a ramp
    # dq[index_resultant, index_pixel] == 0 means resultant is in ramp
    ramp: RampIndex = RampIndex(-1, -1)
    index_resultant: cython.int
    for index_resultant in range(n_resultants):
        # Checking for start of ramp
        if ramp.start == -1:
            if dq[index_resultant] == 0:
                # This resultant is in the ramp
                # => We have found the start of a ramp!
                ramp.start = index_resultant

        # This resultant cannot be the start of a ramp
        # => Checking for end of ramp
        elif dq[index_resultant] != 0:
            # This pixel is not in the ramp
            # => index_resultant - 1 is the end of the ramp
            ramp.end = index_resultant - 1

            # Add completed ramp to the queue and reset ramp
            ramps.push_back(ramp)
            ramp = RampIndex(-1, -1)

    # Handle case where last resultant is in ramp (so no end has been set)
    if ramp.start != -1 and ramp.end == -1:
        # Last resultant is end of the ramp => set then add to stack
        ramp.end = n_resultants - 1
        ramps.push_back(ramp)

    return ramps


# Casertano+2022, Table 2
_P_TABLE = cython.declare(
    cython.float[6][2],
    [
        [-INFINITY, 5, 10, 20, 50, 100],
        [0, 0.4, 1, 3, 6, 10],
    ],
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.inline
@cython.cfunc
@cython.exceptval(check=False)
def _get_power(signal: cython.float) -> cython.float:
    """
    Return the power from Casertano+22, Table 2.

    Parameters
    ----------
    signal: float
        signal from the resultants

    Returns
    -------
    signal power from Table 2
    """
    i: cython.int
    for i in range(6):
        if signal < _P_TABLE[0][i]:
            return _P_TABLE[1][i - 1]

    return _P_TABLE[1][i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.inline
@cython.cfunc
def fit_ramp(
    resultants_: cython.float[:],
    t_bar_: cython.float[:],
    tau_: cython.float[:],
    n_reads_: cython.int[:],
    read_noise: cython.float,
    ramp: RampIndex,
) -> RampFit:
    """
    Fit a single ramp using Casertano+22 algorithm.

    Parameters
    ----------
    resultants_ : float[:]
        All of the resultants for the pixel
    t_bar_ : float[:]
        All the t_bar values
    tau_ : float[:]
        All the tau values
    n_reads_ : int[:]
        All the n_reads values
    read_noise : float
        The read noise for the pixel
    ramp : RampIndex
        Struct for start and end of ramp to fit

    Returns
    -------
    RampFit
        struct containing
        - slope
        - read_var
        - poisson_var
    """
    n_resultants: cython.int = ramp.end - ramp.start + 1

    # Special case where there is no or one resultant, there is no fit and
    # we bail out before any computations.
    #    Note that in this case, we cannot compute the slope or the variances
    #    because these computations require at least two resultants. Therefore,
    #    this case is degernate and we return NaNs for the values.
    if n_resultants <= 1:
        return RampFit(NAN, NAN, NAN)

    # Compute the fit
    i: cython.int = 0
    j: cython.int = 0

    # Setup data for fitting (work over subset of data) to make things cleaner
    #    Recall that the RampIndex contains the index of the first and last
    #    index of the ramp. Therefore, the Python slice needed to get all the
    #    data within the ramp is:
    #         ramp.start:ramp.end + 1
    resultants: cython.float[:] = resultants_[ramp.start : ramp.end + 1]
    t_bar: cython.float[:] = t_bar_[ramp.start : ramp.end + 1]
    tau: cython.float[:] = tau_[ramp.start : ramp.end + 1]
    n_reads: cython.int[:] = n_reads_[ramp.start : ramp.end + 1]

    # Compute mid point time
    end: cython.int = n_resultants - 1
    t_bar_mid: cython.float = (t_bar[0] + t_bar[end]) / 2

    # Casertano+2022 Eq. 44
    #    Note we've departed from Casertano+22 slightly;
    #    there s is just resultants[ramp.end].  But that doesn't seem good if, e.g.,
    #    a CR in the first resultant has boosted the whole ramp high but there
    #    is no actual signal.
    power: cython.float = fmaxf(resultants[end] - resultants[0], 0)
    power = power / sqrt(read_noise**2 + power)
    power = _get_power(power)

    # It's easy to use up a lot of dynamic range on something like
    # (tbar - tbarmid) ** 10.  Rescale these.
    t_scale: cython.float = (t_bar[end] - t_bar[0]) / 2
    t_scale = 1 if t_scale == 0 else t_scale

    # Initialize the fit loop
    #   it is faster to generate a c++ vector than a numpy array
    weights: vector[cython.float] = vector[float](n_resultants)
    coeffs: vector[cython.float] = vector[float](n_resultants)
    ramp_fit: RampFit = RampFit(0, 0, 0)
    f0: cython.float = 0
    f1: cython.float = 0
    f2: cython.float = 0
    coeff: cython.float

    # Issue when tbar[] == tbarmid causes exception otherwise
    with cython.cpow(True):
        for i in range(n_resultants):
            # Casertano+22, Eq. 45
            weights[i] = (((1 + power) * n_reads[i]) / (1 + power * n_reads[i])) * fabs(
                (t_bar[i] - t_bar_mid) / t_scale
            ) ** power

            # Casertano+22 Eq. 35
            f0 += weights[i]
            f1 += weights[i] * t_bar[i]
            f2 += weights[i] * t_bar[i] ** 2

    # Casertano+22 Eq. 36
    det: cython.float = f2 * f0 - f1**2
    if det == 0:
        return ramp_fit

    for i in range(n_resultants):
        # Casertano+22 Eq. 37
        coeff = (f0 * t_bar[i] - f1) * weights[i] / det
        coeffs[i] = coeff

        # Casertano+22 Eq. 38
        ramp_fit.slope += coeff * resultants[i]

        # Casertano+22 Eq. 39
        ramp_fit.read_var += coeff**2 * read_noise**2 / n_reads[i]

        # Casertano+22 Eq 40
        #    Note that this is an inversion of the indexing from the equation;
        #    however, commutivity of addition results in the same answer. This
        #    makes it so that we don't have to loop over all the resultants twice.
        ramp_fit.poisson_var += coeff**2 * tau[i]
        for j in range(i):
            ramp_fit.poisson_var += 2 * coeff * coeffs[j] * t_bar[j]

    return ramp_fit
