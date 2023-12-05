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
import numpy as np

cimport numpy as cnp
from cython cimport boundscheck, cdivision, cpow, wraparound
from libc.math cimport INFINITY, NAN, fabs, fmaxf, sqrt
from libcpp.vector cimport vector

from stcal.ramp_fitting.ols_cas22._ramp cimport RampFit, RampIndex, RampQueue, ReadPattern

# Initialize numpy for cython use in this module
cnp.import_array()


cdef class ReadPattern:
    """
    Class to contain the read pattern derived metadata
        This exists only to allow us to output multiple memory views at the same time
        from the same cython function. This is needed because neither structs nor unions
        can contain memory views.

        In the case of this code memory views are the fastest "safe" array data structure.
        This class will immediately be unpacked into raw memory views, so that we avoid
        any further overhead of switching between python and cython.

    Attributes:
    ----------
    t_bar : np.ndarray[float_t, ndim=1]
        The mean time of each resultant
    tau : np.ndarray[float_t, ndim=1]
        The variance in time of each resultant
    n_reads : np.ndarray[cnp.int32_t, ndim=1]
        The number of reads in each resultant
    """

    def _to_dict(ReadPattern self):
        """
        This is a private method to convert the ReadPattern object to a dictionary,
            so that attributes can be directly accessed in python. Note that this
            is needed because class attributes cannot be accessed on cython classes
            directly in python. Instead they need to be accessed or set using a
            python compatible method. This method is a pure puthon method bound
            to to the cython class and should not be used by any cython code, and
            only exists for testing purposes.
        """
        return dict(t_bar=np.array(self.t_bar, dtype=np.float32),
                    tau=np.array(self.tau, dtype=np.float32),
                    n_reads=np.array(self.n_reads, dtype=np.int32))


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef ReadPattern from_read_pattern(list[list[int]] read_pattern, float read_time, int n_resultants):
    """
    Derive the input data from the the read pattern
        This is faster than using __init__ or __cinit__ to construct the object with
        these calls.

    Parameters
    ----------
    read pattern: list[list[int]]
        read pattern for the image
    read_time : float
        Time to perform a readout.
    n_resultants : int
        Number of resultants in the image

    Returns
    -------
    ReadPattern
        Contains:
        - t_bar
        - tau
        - n_reads
    """

    cdef ReadPattern data = ReadPattern()
    data.t_bar = np.empty(n_resultants, dtype=np.float32)
    data.tau = np.empty(n_resultants, dtype=np.float32)
    data.n_reads = np.empty(n_resultants, dtype=np.int32)

    cdef int index, n_reads
    cdef list[int] resultant
    for index, resultant in enumerate(read_pattern):
        n_reads = len(resultant)

        data.n_reads[index] = n_reads
        data.t_bar[index] = read_time * np.mean(resultant)
        data.tau[index] = (np.sum((2 * (n_reads - np.arange(n_reads)) - 1) * resultant) *
                           read_time / n_reads**2)

    return data


@boundscheck(False)
@wraparound(False)
cpdef inline RampQueue init_ramps(int[:] dq, int n_resultants):
    """
    Create the initial ramp "queue" for each pixel
        if dq[index_resultant, index_pixel] == 0, then the resultant is in a ramp
        otherwise, the resultant is not in a ramp

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
    cdef RampQueue ramps = RampQueue()

    # Note: if start/end are -1, then no value has been assigned
    # ramp.start == -1 means we have not started a ramp
    # dq[index_resultant, index_pixel] == 0 means resultant is in ramp
    cdef RampIndex ramp = RampIndex(-1, -1)
    cdef int index_resultant
    for index_resultant in range(n_resultants):
        if ramp.start == -1:
            # Looking for the start of a ramp
            if dq[index_resultant] == 0:
                # We have found the start of a ramp!
                ramp.start = index_resultant
            else:
                # This is not the start of the ramp yet
                continue
        else:
            # Looking for the end of a ramp
            if dq[index_resultant] == 0:
                # This pixel is in the ramp do nothing
                continue
            else:
                # This pixel is not in the ramp
                # => index_resultant - 1 is the end of the ramp
                ramp.end = index_resultant - 1

                # Add completed ramp to stack and reset ramp
                ramps.push_back(ramp)
                ramp = RampIndex(-1, -1)

    # Handle case where last resultant is in ramp (so no end has been set)
    if ramp.start != -1 and ramp.end == -1:
        # Last resultant is end of the ramp => set then add to stack
        ramp.end = n_resultants - 1
        ramps.push_back(ramp)

    return ramps

# Keeps the static type checker/highlighter happy this has no actual effect
ctypedef float[6] _row

# Casertano+2022, Table 2
cdef _row[2] _PTABLE = [[-INFINITY, 5,   10, 20, 50, 100],
                        [0,         0.4, 1,  3,  6,  10]]


@boundscheck(False)
@wraparound(False)
cdef inline float _get_power(float signal):
    """
    Return the power from Casertano+22, Table 2

    Parameters
    ----------
    signal: float
        signal from the resultants

    Returns
    -------
    signal power from Table 2
    """
    cdef int i
    for i in range(6):
        if signal < _PTABLE[0][i]:
            return _PTABLE[1][i - 1]

    return _PTABLE[1][i]


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef inline RampFit fit_ramp(float[:] resultants_,
                             float[:] t_bar_,
                             float[:] tau_,
                             int[:] n_reads_,
                             float read_noise,
                             RampIndex ramp):
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
    cdef int n_resultants = ramp.end - ramp.start + 1

    # Special case where there is no or one resultant, there is no fit and
    # we bail out before any computations.
    #    Note that in this case, we cannot compute the slope or the variances
    #    because these computations require at least two resultants. Therefore,
    #    this case is degernate and we return NaNs for the values.
    if n_resultants <= 1:
        return RampFit(NAN, NAN, NAN)

    # Compute the fit
    cdef int i = 0, j = 0

    # Setup data for fitting (work over subset of data) to make things cleaner
    #    Recall that the RampIndex contains the index of the first and last
    #    index of the ramp. Therefore, the Python slice needed to get all the
    #    data within the ramp is:
    #         ramp.start:ramp.end + 1
    cdef float[:] resultants = resultants_[ramp.start:ramp.end + 1]
    cdef float[:] t_bar = t_bar_[ramp.start:ramp.end + 1]
    cdef float[:] tau = tau_[ramp.start:ramp.end + 1]
    cdef int[:] n_reads = n_reads_[ramp.start:ramp.end + 1]

    # Compute mid point time
    cdef int end = n_resultants - 1
    cdef float t_bar_mid = (t_bar[0] + t_bar[end]) / 2

    # Casertano+2022 Eq. 44
    #    Note we've departed from Casertano+22 slightly;
    #    there s is just resultants[ramp.end].  But that doesn't seem good if, e.g.,
    #    a CR in the first resultant has boosted the whole ramp high but there
    #    is no actual signal.
    cdef float power = fmaxf(resultants[end] - resultants[0], 0)
    power = power / sqrt(read_noise**2 + power)
    power = _get_power(power)

    # It's easy to use up a lot of dynamic range on something like
    # (tbar - tbarmid) ** 10.  Rescale these.
    cdef float t_scale = (t_bar[end] - t_bar[0]) / 2
    t_scale = 1 if t_scale == 0 else t_scale

    # Initialize the fit loop
    #   it is faster to generate a c++ vector than a numpy array
    cdef vector[float] weights = vector[float](n_resultants)
    cdef vector[float] coeffs = vector[float](n_resultants)
    cdef RampFit ramp_fit = RampFit(0, 0, 0)
    cdef float f0 = 0, f1 = 0, f2 = 0
    cdef float coeff

    # Issue when tbar[] == tbarmid causes exception otherwise
    with cpow(True):
        for i in range(n_resultants):
            # Casertano+22, Eq. 45
            weights[i] = ((((1 + power) * n_reads[i]) / (1 + power * n_reads[i])) *
                          fabs((t_bar[i] - t_bar_mid) / t_scale) ** power)

            # Casertano+22 Eq. 35
            f0 += weights[i]
            f1 += weights[i] * t_bar[i]
            f2 += weights[i] * t_bar[i]**2

    # Casertano+22 Eq. 36
    cdef float det = f2 * f0 - f1 ** 2
    if det == 0:
        return ramp_fit

    for i in range(n_resultants):
        # Casertano+22 Eq. 37
        coeff = (f0 * t_bar[i] - f1) * weights[i] / det
        coeffs[i] = coeff

        # Casertano+22 Eq. 38
        ramp_fit.slope += coeff * resultants[i]

        # Casertano+22 Eq. 39
        ramp_fit.read_var += (coeff ** 2 * read_noise ** 2 / n_reads[i])

        # Casertano+22 Eq 40
        #    Note that this is an inversion of the indexing from the equation;
        #    however, commutivity of addition results in the same answer. This
        #    makes it so that we don't have to loop over all the resultants twice.
        ramp_fit.poisson_var += coeff ** 2 * tau[i]
        for j in range(i):
            ramp_fit.poisson_var += (2 * coeff * coeffs[j] * t_bar[j])

    return ramp_fit
