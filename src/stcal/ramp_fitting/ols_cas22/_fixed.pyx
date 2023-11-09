"""
Define the data which is fixed for all pixels to compute the CAS22 algorithm with
    jump detection

Objects
-------
FixedValues : class
    Class to contain the data fixed for all pixels and commonly referenced
    universal values for jump detection

Functions
---------
    fixed_values_from_metadata : function
        Fast constructor for FixedValues from the read pattern metadata
            - cpdef gives a python wrapper, but the python version of this method
              is considered private, only to be used for testing
"""
import numpy as np
cimport numpy as cnp

from cython cimport boundscheck, wraparound

from libc.math cimport NAN
from libcpp cimport bool

from stcal.ramp_fitting.ols_cas22._core cimport Diff
from stcal.ramp_fitting.ols_cas22._fixed cimport FixedValues
from stcal.ramp_fitting.ols_cas22._read_pattern cimport ReadPattern

cnp.import_array()


cdef class FixedValues:
    """
    Class to contain all the values which are fixed for all pixels for a given
    read pattern.
        This class is used to pre-compute these values once so that they maybe
        reused for all pixels.  This is done for performance reasons.

    Parameters
    ----------
    use_jump : bool
        flag to indicate whether to use jump detection (user input)

    data : ReadPatternMetadata
        Metadata struct created from a read pattern

    threshold : Thresh
        Parameterization struct for threshold function

    t_bar_diffs : float[:, :]
        These are the differences of t_bar used for jump detection.
            single differences of t_bar:
                t_bar_diffs[Diff.single, :] = (t_bar[i+1] - t_bar[i])
            double differences of t_bar:
                t_bar_diffs[Diff.double, :] = (t_bar[i+2] - t_bar[i])
    t_bar_diff_sqrs : float[:, :]
        These are the squared differnences of t_bar used for jump detection.
            single differences of t_bar:
                t_bar_diff_sqrs[Diff.single, :] = (t_bar[i+1] - t_bar[i])**2
            double differences of t_bar:
                t_bar_diff_sqrs[Diff.double, :] = (t_bar[i+2] - t_bar[i])**2
    read_recip_coeffs : float[:, :]
        Coefficients for the read noise portion of the variance used to compute
        the jump detection statistics. These are formed from the reciprocal sum
        of the number of reads.
            single sum of reciprocal n_reads:
                read_recip_coeffs[Diff.single, :] = ((1/n_reads[i+1]) + (1/n_reads[i]))
            double sum of reciprocal n_reads:
                read_recip_coeffs[Diff.double, :] = ((1/n_reads[i+2]) + (1/n_reads[i]))
    var_slope_coeffs : float[:, :]
        Coefficients for the slope portion of the variance used to compute the
        jump detection statistics, which happend to be fixed for any given ramp
        fit.
            single of slope variance term:
                var_slope_coeffs[Diff.single, :] = (tau[i] + tau[i+1]
                                                    - 2 * min(t_bar[i], t_bar[i+1]))
            double of slope variance term:
                var_slope_coeffs[Diff.double, :] = (tau[i] + tau[i+2]
                                                    - 2 * min(t_bar[i], t_bar[i+2]))

    Notes
    -----
    - t_bar_diffs, t_bar_diff_sqrs, read_recip_coeffs, var_slope_coeffs are only
      computed if use_jump is True.  These values represent reused computations
      for jump detection which are used by every pixel for jump detection. They
      are computed once and stored in the FixedValues for reuse by all pixels.
    - The computations are done using vectorized operations for some performance
      increases. However, this is marginal compaired with the performance increase
      from pre-computing the values and reusing them.
    """

    def _to_dict(FixedValues self):
        """
        This is a private method to convert the FixedValues object to a dictionary,
            so that attributes can be directly accessed in python. Note that this
            is needed because class attributes cannot be accessed on cython classes
            directly in python. Instead they need to be accessed or set using a
            python compatible method. This method is a pure puthon method bound
            to to the cython class and should not be used by any cython code, and
            only exists for testing purposes.
        """
        cdef cnp.ndarray[float, ndim=2] t_bar_diffs
        cdef cnp.ndarray[float, ndim=2] t_bar_diff_sqrs
        cdef cnp.ndarray[float, ndim=2] read_recip_coeffs
        cdef cnp.ndarray[float, ndim=2] var_slope_coeffs

        if self.use_jump:
            t_bar_diffs = np.array(self.t_bar_diffs, dtype=np.float32)
            t_bar_diff_sqrs = np.array(self.t_bar_diff_sqrs, dtype=np.float32)
            read_recip_coeffs = np.array(self.read_recip_coeffs, dtype=np.float32)
            var_slope_coeffs = np.array(self.var_slope_coeffs, dtype=np.float32)
        else:
            try:
                self.t_bar_diffs
            except AttributeError:
                t_bar_diffs = np.array([[np.nan],[np.nan]], dtype=np.float32)
            else:
                raise AttributeError("t_bar_diffs should not exist")

            try:
                self.t_bar_diff_sqrs
            except AttributeError:
                t_bar_diff_sqrs = np.array([[np.nan],[np.nan]], dtype=np.float32)
            else:
                raise AttributeError("t_bar_diff_sqrs should not exist")

            try:
                self.read_recip_coeffs
            except AttributeError:
                read_recip_coeffs = np.array([[np.nan],[np.nan]], dtype=np.float32)
            else:
                raise AttributeError("read_recip_coeffs should not exist")

            try:
                self.var_slope_coeffs
            except AttributeError:
                var_slope_coeffs = np.array([[np.nan],[np.nan]], dtype=np.float32)
            else:
                raise AttributeError("var_slope_coeffs should not exist")

        return dict(data=self.data._to_dict(),
                    t_bar_diffs=t_bar_diffs,
                    t_bar_diff_sqrs=t_bar_diff_sqrs,
                    read_recip_coeffs=read_recip_coeffs,
                    var_slope_coeffs=var_slope_coeffs)


cpdef enum FixedOffsets:
    single_t_bar_diff
    double_t_bar_diff
    single_t_bar_diff_sqr
    double_t_bar_diff_sqr
    single_read_recip
    double_read_recip
    single_var_slope_val
    double_var_slope_val
    n_fixed_offsets


@boundscheck(False)
@wraparound(False)
cdef inline float[:, :] fill_fixed_values(float[:] t_bar,
                                          float[:] tau,
                                          int[:] n_reads,
                                          int end):
    """
    Compute the difference offset of t_bar

    Returns
    -------
    [
        <t_bar[i+1] - t_bar[i]>,
        <t_bar[i+2] - t_bar[i]>,
    ]
    [
        <t_bar[i+1] - t_bar[i]> ** 2,
        <t_bar[i+2] - t_bar[i]> ** 2,
    ]
    [
        <(1/n_reads[i+1] + 1/n_reads[i])>,
        <(1/n_reads[i+2] + 1/n_reads[i])>,
    ]
    [
        <(tau[i] + tau[i+1] - 2 * min(t_bar[i], t_bar[i+1]))>,
        <(tau[i] + tau[i+2] - 2 * min(t_bar[i], t_bar[i+2]))>,
    ]
    """

    cdef int single_t_bar_diff = FixedOffsets.single_t_bar_diff
    cdef int double_t_bar_diff = FixedOffsets.double_t_bar_diff
    cdef int single_t_bar_diff_sqr = FixedOffsets.single_t_bar_diff_sqr
    cdef int double_t_bar_diff_sqr = FixedOffsets.double_t_bar_diff_sqr
    cdef int single_read_recip = FixedOffsets.single_read_recip
    cdef int double_read_recip = FixedOffsets.double_read_recip
    cdef int single_var_slope_val = FixedOffsets.single_var_slope_val
    cdef int double_var_slope_val = FixedOffsets.double_var_slope_val

    cdef float[:, :] pre_compute = np.empty((n_fixed_offsets, end - 1), dtype=np.float32)

    # Coerce division to be using floats
    cdef float num = 1

    cdef int i
    for i in range(end - 1):
        pre_compute[single_t_bar_diff, i] = t_bar[i + 1] - t_bar[i]
        pre_compute[single_t_bar_diff_sqr, i] = pre_compute[single_t_bar_diff, i] ** 2
        pre_compute[single_read_recip, i] = (num / n_reads[i + 1]) + (num / n_reads[i])
        pre_compute[single_var_slope_val, i] = tau[i + 1] + tau[i] - 2 * min(t_bar[i + 1], t_bar[i])

        if i < end - 2:
            pre_compute[double_t_bar_diff, i] = t_bar[i + 2] - t_bar[i]
            pre_compute[double_t_bar_diff_sqr, i] = pre_compute[double_t_bar_diff, i] ** 2
            pre_compute[double_read_recip, i] = (num / n_reads[i + 2]) + (num / n_reads[i])
            pre_compute[double_var_slope_val, i] = tau[i + 2] + tau[i] - 2 * min(t_bar[i + 2], t_bar[i])
        else:
            # Last double difference is undefined
            pre_compute[double_t_bar_diff, i] = NAN
            pre_compute[double_t_bar_diff_sqr, i] = NAN
            pre_compute[double_read_recip, i] = NAN
            pre_compute[double_var_slope_val, i] = NAN

    return pre_compute


@boundscheck(False)
@wraparound(False)
cdef inline float[:, :] var_slope_vals(ReadPattern data):
    """
    Compute slope part of the jump statistic variances

    Returns
    -------
    """
    cdef int end = len(data.t_bar)

    cdef cnp.ndarray[float, ndim=2] var_slope_vals = np.zeros((2, end - 1), dtype=np.float32)

    var_slope_vals[Diff.single, :] = (np.add(data.tau[1:], data.tau[:end - 1]) - 2 * np.minimum(data.t_bar[1:], data.t_bar[:end - 1]))
    var_slope_vals[Diff.double, :end - 2] = (np.add(data.tau[2:], data.tau[:end - 2]) - 2 * np.minimum(data.t_bar[2:], data.t_bar[:end - 2]))
    var_slope_vals[Diff.double, end - 2] = np.nan  # last double difference is undefined

    return var_slope_vals


@boundscheck(False)
@wraparound(False)
cpdef inline FixedValues fixed_values_from_metadata(ReadPattern data, bool use_jump):
    """
    Fast constructor for FixedValues class
        Use this instead of an __init__ because it does not incure the overhead
        of switching back and forth to python

    Parameters
    ----------
    data : ReadPatternMetadata
        metadata object created from the read pattern (user input)
    threshold : Thresh
        threshold object (user input)
    use_jump : bool
        flag to indicate whether to use jump detection (user input)

    Returns
    -------
    FixedValues object (with pre-computed values for jump detection if use_jump
    is True)
    """
    cdef FixedValues fixed = FixedValues()

    # Fill in input information for all pixels
    fixed.use_jump = use_jump

    # Cast vector to a c array
    fixed.data = data

    # Pre-compute jump detection computations shared by all pixels
    cdef float[:, :] pre_compute
    if use_jump:
        pre_compute = fill_fixed_values(data.t_bar, data.tau, data.n_reads, data.n_resultants)
        fixed.t_bar_diffs = pre_compute[:FixedOffsets.double_t_bar_diff + 1, :]
        fixed.t_bar_diff_sqrs = pre_compute[FixedOffsets.single_t_bar_diff_sqr:FixedOffsets.double_t_bar_diff_sqr + 1, :]
        fixed.read_recip_coeffs = pre_compute[FixedOffsets.single_read_recip:FixedOffsets.double_read_recip + 1, :]
        fixed.var_slope_coeffs = pre_compute[FixedOffsets.single_var_slope_val:FixedOffsets.double_var_slope_val + 1, :]

    return fixed
