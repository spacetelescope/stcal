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
import numpy as np
cimport numpy as cnp
cimport cython


from stcal.ramp_fitting.ols_cas22._core cimport Diff
from stcal.ramp_fitting.ols_cas22._fixed cimport FixedValues
from stcal.ramp_fitting.ols_cas22._pixel cimport Pixel


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
