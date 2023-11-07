"""
Define the C class for the Cassertano22 algorithm for fitting ramps with jump detection

Objects
-------
Pixel : class
    Class to handle ramp fit with jump detection for a single pixel
    Provides fits method which fits all the ramps for a single pixel

Functions
---------
    make_pixel : function
        Fast constructor for a Pixel class from input data.
            - cpdef gives a python wrapper, but the python version of this method
              is considered private, only to be used for testing
"""
from libc.math cimport sqrt, fabs
from libcpp.vector cimport vector

import numpy as np
cimport numpy as cnp
cimport cython


from stcal.ramp_fitting.ols_cas22._core cimport Diff
from stcal.ramp_fitting.ols_cas22._fixed cimport FixedValues
from stcal.ramp_fitting.ols_cas22._jump cimport get_power
from stcal.ramp_fitting.ols_cas22._pixel cimport Pixel
from stcal.ramp_fitting.ols_cas22._ramp cimport RampIndex, RampQueue


cdef class Pixel:
    """
    Class to contain the data to fit ramps for a single pixel.
        This data is drawn from for all ramps for a single pixel.
        This class pre-computes jump detection values shared by all ramps
        for a given pixel.

    Parameters
    ----------
    fixed : FixedValues
        The object containing all the values and metadata which is fixed for a
        given read pattern>
    read_noise : float
        The read noise for the given pixel
    resultants : float [:]
        Resultants input for the given pixel

    local_slopes : float [:, :]
        These are the local slopes between the resultants for the pixel.
            single difference local slope:
                local_slopes[Diff.single, :] = (resultants[i+1] - resultants[i])
                                                / (t_bar[i+1] - t_bar[i])
            double difference local slope:
                local_slopes[Diff.double, :] = (resultants[i+2] - resultants[i])
                                                / (t_bar[i+2] - t_bar[i])
    var_read_noise : float [:, :]
        The read noise variance term of the jump statistics
            single difference read noise variance:
                var_read_noise[Diff.single, :] = read_noise * ((1/n_reads[i+1]) + (1/n_reads[i]))
            double difference read_noise variance:
                var_read_noise[Diff.doule, :] = read_noise * ((1/n_reads[i+2]) + (1/n_reads[i]))

    Notes
    -----
    - local_slopes and var_read_noise are only computed if use_jump is True. 
      These values represent reused computations for jump detection which are
      used by every ramp for the given pixel for jump detection. They are
      computed once and stored for reuse by all ramp computations for the pixel.
    - The computations are done using vectorized operations for some performance
      increases. However, this is marginal compaired with the performance increase
      from pre-computing the values and reusing them.

    Methods
    -------
    fit_ramp (ramp_index) : method
        Compute the ramp fit for a single ramp defined by an inputed RampIndex
    fit_ramps (ramp_stack) : method
        Compute all the ramps for a single pixel using the Casertano+22 algorithm
        with jump detection.
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline float[:, :] local_slope_vals(Pixel self):
        """
        Compute the local slopes between resultants for the pixel

        Returns
        -------
        [
            <(resultants[i+1] - resultants[i])> / <(t_bar[i+1] - t_bar[i])>,
            <(resultants[i+2] - resultants[i])> / <(t_bar[i+2] - t_bar[i])>,
        ]
        """
        cdef float[:] resultants = self.resultants
        cdef int end = len(resultants)

        # Read the t_bar_diffs into a local variable to avoid calling through Python
        #    multiple times
        cdef cnp.ndarray[float, ndim=2] t_bar_diffs = np.array(self.fixed.t_bar_diffs, dtype=np.float32)

        cdef cnp.ndarray[float, ndim=2] local_slope_vals = np.zeros((2, end - 1), dtype=np.float32)

        local_slope_vals[Diff.single, :] = (np.subtract(resultants[1:], resultants[:end - 1])
                                            / t_bar_diffs[Diff.single, :]).astype(np.float32)
        local_slope_vals[Diff.double, :end - 2] = (np.subtract(resultants[2:], resultants[:end - 2])
                                                   / t_bar_diffs[Diff.double, :end-2]).astype(np.float32)
        local_slope_vals[Diff.double, end - 2] = np.nan  # last double difference is undefined

        return local_slope_vals

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline RampFit fit_ramp(Pixel self, RampIndex ramp):
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
            return RampFit(np.nan, np.nan, np.nan)

        # Start computing the fit

        # Setup data for fitting (work over subset of data)
        #    Recall that the RampIndex contains the index of the first and last
        #    index of the ramp. Therefore, the Python slice needed to get all the
        #    data within the ramp is:
        #         ramp.start:ramp.end + 1
        cdef float[:] resultants = self.resultants[ramp.start:ramp.end + 1]
        cdef float[:] t_bar = self.fixed.data.t_bar[ramp.start:ramp.end + 1]
        cdef float[:] tau = self.fixed.data.tau[ramp.start:ramp.end + 1]
        cdef int[:] n_reads = self.fixed.data.n_reads[ramp.start:ramp.end + 1]

        # Reference read_noise as a local variable to avoid calling through Python
        # every time it is accessed.
        cdef float read_noise = self.read_noise

        # Compute mid point time
        cdef int end = len(resultants) - 1
        cdef float t_bar_mid = (t_bar[0] + t_bar[end]) / 2

        # Casertano+2022 Eq. 44
        # Note we've departed from Casertano+22 slightly;
        # there s is just resultants[ramp.end].  But that doesn't seem good if, e.g.,
        # a CR in the first resultant has boosted the whole ramp high but there
        # is no actual signal.
        cdef float s = max(resultants[end] - resultants[0], 0)
        s = s / sqrt(read_noise**2 + s)
        cdef float power = get_power(s)

        # It's easy to use up a lot of dynamic range on something like
        # (tbar - tbarmid) ** 10.  Rescale these.
        cdef float t_scale = (t_bar[end] - t_bar[0]) / 2
        t_scale = 1 if t_scale == 0 else t_scale

        # Initalize the fit loop
        cdef int i = 0, j = 0
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

    def _to_dict(Pixel self):
        """
        This is a private method to convert the Pixel object to a dictionary, so
            that attributes can be directly accessed in python. Note that this is
            needed because class attributes cannot be accessed on cython classes
            directly in python. Instead they need to be accessed or set using a
            python compatible method. This method is a pure puthon method bound
            to to the cython class and should not be used by any cython code, and
            only exists for testing purposes.
        """

        cdef cnp.ndarray[float, ndim=1] resultants_ = np.array(self.resultants, dtype=np.float32)

        cdef cnp.ndarray[float, ndim=2] local_slopes
        cdef cnp.ndarray[float, ndim=2] var_read_noise

        if self.fixed.use_jump:
            local_slopes = np.array(self.local_slopes, dtype=np.float32)
            var_read_noise = np.array(self.var_read_noise, dtype=np.float32)
        else:
            try:
                self.local_slopes
            except AttributeError:
                local_slopes = np.array([[np.nan],[np.nan]], dtype=np.float32)
            else:
                raise AttributeError("local_slopes should not exist")

            try:
                self.var_read_noise
            except AttributeError:
                var_read_noise = np.array([[np.nan],[np.nan]], dtype=np.float32)
            else:
                raise AttributeError("var_read_noise should not exist")

        return dict(fixed=self.fixed._to_dict(),
                    resultants=resultants_,
                    read_noise=self.read_noise,
                    local_slopes=local_slopes,
                    var_read_noise=var_read_noise)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline Pixel make_pixel(FixedValues fixed, float read_noise, float [:] resultants):
    """
    Fast constructor for the Pixel C class.
        This creates a Pixel object for a single pixel from the input data.

    This is signifantly faster than using the `__init__` or `__cinit__`
        this is because this does not have to pass through the Python as part
        of the construction.

    Parameters
    ----------
    fixed : FixedValues
        Fixed values for all pixels
    read_noise : float
        read noise for the single pixel
    resultants : float [:]
        array of resultants for the single pixel
            - memoryview of a numpy array to avoid passing through Python

    Return
    ------
    Pixel C-class object (with pre-computed values if use_jump is True)
    """
    cdef Pixel pixel = Pixel()

    # Fill in input information for pixel
    pixel.fixed = fixed
    pixel.read_noise = read_noise
    pixel.resultants = resultants

    # Pre-compute values for jump detection shared by all pixels for this pixel
    if fixed.use_jump:
        pixel.local_slopes = pixel.local_slope_vals()
        pixel.var_read_noise = (read_noise ** 2) * np.array(fixed.read_recip_coeffs)

    return pixel
