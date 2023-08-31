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

from stcal.ramp_fitting.ols_cas22._core cimport Thresh, DerivedData
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

    t_bar_1 : float[:]
        single differences of t_bar:
            (t_bar[i+1] - t_bar[i])
    t_bar_1_sq : float[:]
        squared single differences of t_bar:
            (t_bar[i+1] - t_bar[i])**2
    t_bar_2 : float[:]
        double differences of t_bar:
            (t_bar[i+2] - t_bar[i])
    t_bar_2_sq: float[:]
        squared double differences of t_bar:
            (t_bar[i+2] - t_bar[i])**2
    recip_1 : vector[float]
        single sum of reciprocal n_reads:
            ((1/n_reads[i+1]) + (1/n_reads[i]))
    recip_2 : vector[float]
        double sum of reciprocal n_reads:
            ((1/n_reads[i+2]) + (1/n_reads[i]))
    slope_var_1 : vector[float]
        single of slope variance term:
            ([tau[i] + tau[i+1] - min(t_bar[i], t_bar[i+1]))
    slope_var_2 : vector[float]
        double of slope variance term:
            ([tau[i] + tau[i+2] - min(t_bar[i], t_bar[i+2]))

    Notes
    -----
    - t_bar_*, t_bar_*_sq, recip_*, slope_var_* are only computed if use_jump is True.
      These values represent reused computations for jump detection which are used by
      every pixel for jump detection.  They are computed once and stored in the Fixed
      for reuse by all pixels.
    - The computations are done using vectorized operations for some performance
      increases. However, this is marginal compaired with the performance increase
      from pre-computing the values and reusing them.
    """

    cdef inline float[:] t_bar_diff(Fixed self, int offset):
        """
        Compute the difference offset of t_bar

        Parameters
        ----------
        offset : int
            index offset to compute difference

        Returns
        -------
        t_bar[i+offset] - t_bar[i]
        """
        cdef float[:] t_bar = self.t_bar

        return np.subtract(t_bar[offset:], t_bar[:-offset])

    cdef inline float[:] t_bar_diff_sq(Fixed self, int offset):
        """
        Compute the square difference offset of t_bar

        Parameters
        ----------
        offset : int
            index offset

        Returns
        -------
        (t_bar[i+offset] - t_bar[i])**2
        """
        return np.array(self.t_bar_diff(offset)) ** 2

    cdef inline float[:] recip_val(Fixed self, int offset):
        """
        Compute the recip values
            (1/n_reads[i+offset] + 1/n_reads[i])

        Parameters
        ----------
        offset : int
            index offset

        Returns
        -------
        (1/n_reads[i+offset] + 1/n_reads[i])
        """
        cdef int[:] n_reads = self.n_reads

        return (np.divide(1.0, n_reads[offset:], dtype=np.float32) +
                np.divide(1.0, n_reads[:-offset], dtype=np.float32))


    cdef inline float[:] slope_var_val(Fixed self, int offset):
        """
        Compute the sigma values

        Parameters
        ----------
        offset : int
            index offset

        Returns
        -------
        (tau[i] + tau[i+offset] - min(t_bar[i], t_bar[i+offset])) *
            correction(i, i+offset)
        """
        cdef float[:] t_bar = self.t_bar
        cdef float[:] tau = self.tau

        return (np.add(tau[offset:], tau[:-offset]) -
                np.minimum(t_bar[offset:], t_bar[:-offset]))


cdef inline Fixed make_fixed(DerivedData data, Thresh threshold, bool use_jump):
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
    fixed.t_bar = <float [:data.t_bar.size()]> data.t_bar.data()
    fixed.tau = <float [:data.tau.size()]> data.tau.data()
    fixed.n_reads = <int [:data.n_reads.size()]> data.n_reads.data()

    # Pre-compute jump detection computations shared by all pixels
    if use_jump:
        fixed.t_bar_1 = fixed.t_bar_diff(1)
        fixed.t_bar_2 = fixed.t_bar_diff(2)

        fixed.t_bar_1_sq = fixed.t_bar_diff_sq(1)
        fixed.t_bar_2_sq = fixed.t_bar_diff_sq(2)

        fixed.recip_1 = fixed.recip_val(1)
        fixed.recip_2 = fixed.recip_val(2)

        fixed.slope_var_1 = fixed.slope_var_val(1)
        fixed.slope_var_2 = fixed.slope_var_val(2)

    return fixed
