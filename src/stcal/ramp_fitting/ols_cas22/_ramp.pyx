cimport cython
cimport numpy as cnp

from stcal.ramp_fitting.ols_cas22._ramp cimport RampIndex, RampQueue


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