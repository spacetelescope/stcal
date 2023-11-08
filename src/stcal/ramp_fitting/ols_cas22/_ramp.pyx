import numpy as np

cimport cython
cimport numpy as cnp

from libc.math cimport sqrt, fabs, INFINITY, NAN, fmaxf
from libcpp.vector cimport vector

from stcal.ramp_fitting.ols_cas22._ramp cimport RampIndex, RampQueue, RampFit
from stcal.ramp_fitting.ols_cas22._fixed cimport FixedValues
from stcal.ramp_fitting.ols_cas22._pixel cimport Pixel


cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline RampQueue init_ramps(int[:, :] dq, int n_resultants, int index_pixel):
    """
    Create the initial ramp stack for each pixel
        if dq[index_resultant, index_pixel] == 0, then the resultant is in a ramp
        otherwise, the resultant is not in a ramp

    Parameters
    ----------
    dq : int[n_resultants, n_pixel]
        DQ array
    n_resultants : int
        Number of resultants
    index_pixel : int
        The index of the pixel to find ramps for

    Returns
    -------
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
            if dq[index_resultant, index_pixel] == 0:
                # We have found the start of a ramp!
                ramp.start = index_resultant
            else:
                # This is not the start of the ramp yet
                continue
        else:
            # Looking for the end of a ramp
            if dq[index_resultant, index_pixel] == 0:
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

# Keeps the static type checker/highligher happy this has no actual effect
ctypedef float[6] _row

# Casertano+2022, Table 2
cdef _row[2] PTABLE = [[-INFINITY, 5,   10, 20, 50, 100],
                       [ 0,        0.4, 1,  3,  6,  10 ]]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float get_power(float signal):
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
        if signal < PTABLE[0][i]:
            return PTABLE[1][i - 1]

    return PTABLE[1][i]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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
    ramp : RampIndex
        Struct for start and end of ramp to fit

    Returns
    -------
    RampFit struct of slope, read_var, poisson_var
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

    # Setup data for fitting (work over subset of data)
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
    # Note we've departed from Casertano+22 slightly;
    # there s is just resultants[ramp.end].  But that doesn't seem good if, e.g.,
    # a CR in the first resultant has boosted the whole ramp high but there
    # is no actual signal.
    cdef float power = fmaxf(resultants[end] - resultants[0], 0)
    power = power / sqrt(read_noise**2 + power)
    power = get_power(power)

    # It's easy to use up a lot of dynamic range on something like
    # (tbar - tbarmid) ** 10.  Rescale these.
    cdef float t_scale = (t_bar[end] - t_bar[0]) / 2
    t_scale = 1 if t_scale == 0 else t_scale

    # Initalize the fit loop
    cdef vector[float] weights = vector[float](n_resultants)
    cdef vector[float] coeffs = vector[float](n_resultants)
    cdef RampFit ramp_fit = RampFit(0, 0, 0)
    cdef float f0 = 0, f1 = 0, f2 = 0

    # Issue when tbar[] == tbarmid causes exception otherwise
    with cython.cpow(True):
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
        coeffs[i] = (f0 * t_bar[i] - f1) * weights[i] / det

    for i in range(n_resultants):
        # Casertano+22 Eq. 38
        ramp_fit.slope += coeffs[i] * resultants[i]

        # Casertano+22 Eq. 39
        ramp_fit.read_var += (coeffs[i] ** 2 * read_noise ** 2 / n_reads[i])

        # Casertano+22 Eq 40
        ramp_fit.poisson_var += coeffs[i] ** 2 * tau[i]
        for j in range(i + 1, n_resultants):
            ramp_fit.poisson_var += (2 * coeffs[i] * coeffs[j] * t_bar[i])

    return ramp_fit
