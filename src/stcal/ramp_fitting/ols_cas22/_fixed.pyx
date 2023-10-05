"""
Define the data which is fixed for all pixels to compute the CAS22 algorithm with
    jump detection

Objects
-------
Fixed : class
    Class to contain the data fixed for all pixels and commonly referenced
    universal values for jump detection

Functions
---------
make_fixed : function
    Fast constructor for Fixed class
"""
import numpy as np
cimport numpy as np
cimport cython

from stcal.ramp_fitting.ols_cas22._core cimport Thresh, ReadPatternMetadata, Diff
from stcal.ramp_fitting.ols_cas22._fixed cimport Fixed

cdef class Fixed:
    """
    Class to contain the data fixed for all pixels and commonly referenced
    universal values for jump detection

    Parameters
    ----------
    t_bar : float[:]
        mean times of resultants (data input)
    tau : float[:]
        variance weighted mean times of resultants (data input)
    n_reads : float[:]
        number of reads contributing to reach resultant (data input)

    use_jump : bool
        flag to indicate whether to use jump detection (user input)

    t_bar_diff : float[:, :]
        single differences of t_bar:
            t_bar_diff[0, :] = (t_bar[i+1] - t_bar[i])
        double differences of t_bar:
            t_bar_diff[1, :] = (t_bar[i+2] - t_bar[i])
    recip : float[:, :]
        single sum of reciprocal n_reads:
            recip[0, :] = ((1/n_reads[i+1]) + (1/n_reads[i]))
        double sum of reciprocal n_reads:
            recip[1, :] = ((1/n_reads[i+2]) + (1/n_reads[i]))
    slope_var : float[:, :]
        single of slope variance term:
            slope_var[0, :] = ([tau[i] + tau[i+1] - min(t_bar[i], t_bar[i+1]))
        double of slope variance term:
            slope_var[1, :] = ([tau[i] + tau[i+2] - min(t_bar[i], t_bar[i+2]))

    Notes
    -----
    - t_bar_diff, recip, slope_var are only computed if use_jump is True.  These
      values represent reused computations for jump detection which are used by
      every pixel for jump detection.  They are computed once and stored in the
      Fixed for reuse by all pixels.
    - The computations are done using vectorized operations for some performance
      increases. However, this is marginal compaired with the performance increase
      from pre-computing the values and reusing them.
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline float[:, :] t_bar_diff_val(Fixed self):
        """
        Compute the difference offset of t_bar

        Returns
        -------
        [
            <t_bar[i+1] - t_bar[i]>,
            <t_bar[i+2] - t_bar[i]>,
        ]
        """
        # Cast vector to memory view
        #    This way of doing it is potentially memory unsafe because the memory
        #    can outlive the vector. However, this is much faster (no copies) and
        #    much simpler than creating an intermediate wrapper which can pretend
        #    to be a memory view. In this case, I make sure that the memory view
        #    stays local to the function (numpy operations create brand new objects)
        cdef float[:] t_bar = <float [:self.data.t_bar.size()]> self.data.t_bar.data()
        cdef int end = len(t_bar)

        cdef np.ndarray[float, ndim=2] t_bar_diff = np.zeros((2, self.data.t_bar.size() - 1), dtype=np.float32)

        t_bar_diff[Diff.single, :] = np.subtract(t_bar[1:], t_bar[:end - 1]) 
        t_bar_diff[Diff.double, :end - 2] = np.subtract(t_bar[2:], t_bar[:end - 2])
        t_bar_diff[Diff.double, end - 2] = np.nan  # last double difference is undefined

        return t_bar_diff

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline float[:, :] recip_val(Fixed self):
        """
        Compute the reciprical sum values

        Returns
        -------
        [
            <(1/n_reads[i+1] + 1/n_reads[i])>,
            <(1/n_reads[i+2] + 1/n_reads[i])>,
        ]

        """
        # Cast vector to memory view
        #    This way of doing it is potentially memory unsafe because the memory
        #    can outlive the vector. However, this is much faster (no copies) and
        #    much simpler than creating an intermediate wrapper which can pretend
        #    to be a memory view. In this case, I make sure that the memory view
        #    stays local to the function (numpy operations create brand new objects)
        cdef int[:] n_reads = <int [:self.data.n_reads.size()]> self.data.n_reads.data()
        cdef int end = len(n_reads)

        cdef np.ndarray[float, ndim=2] recip = np.zeros((2, self.data.n_reads.size() - 1), dtype=np.float32)

        recip[Diff.single, :] = (np.divide(1.0, n_reads[1:], dtype=np.float32) +
                                 np.divide(1.0, n_reads[:end - 1], dtype=np.float32))
        recip[Diff.double, :end - 2] = (np.divide(1.0, n_reads[2:], dtype=np.float32) +
                                        np.divide(1.0, n_reads[:end - 2], dtype=np.float32))
        recip[Diff.double, end - 2] = np.nan  # last double difference is undefined

        return recip


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline float[:, :] slope_var_val(Fixed self):
        """
        Compute slope part of the variance

        Returns
        -------
        [
            <(tau[i] + tau[i+1] - min(t_bar[i], t_bar[i+1])) * correction(i, i+1)>,
            <(tau[i] + tau[i+2] - min(t_bar[i], t_bar[i+2])) * correction(i, i+2)>,
        ]
        """
        # Cast vectors to memory views
        #    This way of doing it is potentially memory unsafe because the memory
        #    can outlive the vector. However, this is much faster (no copies) and
        #    much simpler than creating an intermediate wrapper which can pretend
        #    to be a memory view. In this case, I make sure that the memory view
        #    stays local to the function (numpy operations create brand new objects)
        cdef float[:] t_bar = <float [:self.data.t_bar.size()]> self.data.t_bar.data()
        cdef float[:] tau = <float [:self.data.tau.size()]> self.data.tau.data()
        cdef int end = len(t_bar)

        cdef np.ndarray[float, ndim=2] slope_var = np.zeros((2, self.data.t_bar.size() - 1), dtype=np.float32)

        slope_var[Diff.single, :] = (np.add(tau[1:], tau[:end - 1]) - np.minimum(t_bar[1:], t_bar[:end - 1]))
        slope_var[Diff.double, :end - 2] = (np.add(tau[2:], tau[:end - 2]) - np.minimum(t_bar[2:], t_bar[:end - 2]))
        slope_var[Diff.double, end - 2] = np.nan  # last double difference is undefined

        return slope_var


cdef inline Fixed make_fixed(ReadPatternMetadata data, Thresh threshold, bool use_jump):
    """
    Fast constructor for Fixed class
        Use this instead of an __init__ because it does not incure the overhead of
        switching back and forth to python

    Parameters
    ----------
    data : DerivedData
        derived data object created from MA table (input data)
    threshold : Thresh
        threshold object (user input)
    use_jump : bool
        flag to indicate whether to use jump detection (user input)

    Returns
    -------
    Fixed parameters object (with pre-computed values if use_jump is True)
    """
    cdef Fixed fixed = Fixed()

    # Fill in input information for all pixels
    fixed.use_jump = use_jump
    fixed.threshold = threshold

    # Cast vector to a c array
    fixed.data = data

    # Pre-compute jump detection computations shared by all pixels
    if use_jump:
        fixed.t_bar_diff = fixed.t_bar_diff_val()
        fixed.recip = fixed.recip_val()
        fixed.slope_var = fixed.slope_var_val()

    return fixed
