# import numpy as np
# cimport numpy as np
# from libc.math cimport sqrt, log10
# from libcpp.vector cimport vector
# from libcpp.stack cimport stack
# cimport cython

# from stcal.ramp_fitting.ols_cas22._core cimport Ramp


# cdef struct RampIndex:
#     int start
#     int end


# cdef struct Thresh:
#     float intercept
#     float constant


# cdef class Jump(Ramp):

#     """
#     Class to contain the data for a single ramp fit with jump detection
#     """
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     @cython.cdivision(True)
#     cdef inline float[:] stats(Jump self, float slope, int start, int end):
#         cdef np.ndarray[float] delta_1 = np.array(self.delta_1[start:end-1]) - slope
#         cdef np.ndarray[float] delta_2 = np.array(self.delta_2[start:end-1]) - slope

#         cdef np.ndarray[float] var_1 = ((np.array(self.sigma_1[start:end-1]) +
#                                          slope * np.array(self.slope_var_1[start:end-1])) /
#                                         self.fixed.t_bar_1_sq[start:end-1]).astype(np.float32)
#         cdef np.ndarray[float] var_2 = ((np.array(self.sigma_2[start:end-1]) +
#                                          slope * np.array(self.slope_var_2[start:end-1])) /
#                                         self.fixed.t_bar_2_sq[start:end-1]).astype(np.float32)

#         cdef np.ndarray[float] stats_1 = delta_1 / sqrt(var_1)
#         cdef np.ndarray[float] stats_2 = delta_2 / sqrt(var_2)

#         return np.maximum(stats_1, stats_2)

#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     @cython.cdivision(True)
#     cdef inline (stack[float], stack[float], stack[float]) fits(Jump self, stack[RampIndex] ramps, Thresh thresh):
#         cdef stack[float] slopes, read_vars, poisson_vars
#         cdef RampIndex ramp
#         cdef float slope = 0, read_var = 0, poisson_var = 0
#         cdef float [:] stats
#         cdef int split

#         while not ramps.empty():
#             ramp = ramps.top()
#             ramps.pop()
#             slope, read_var, poisson_var = self.fit(ramp.start, ramp.end)
#             stats = self.stats(slope, ramp.start, ramp.end)
            
#             if max(stats) > threshold(thresh, slope):
#                 split = np.argmax(stats)

#                 ramps.push(RampIndex(ramp.start, ramp.start + split))
#                 ramps.push(RampIndex(ramp.start + split + 2, ramp.end))
#             else:
#                 slopes.push(slope)
#                 read_vars.push(read_var)
#                 poisson_vars.push(poisson_var)

#         return slopes, read_vars, poisson_vars
    

# cdef float threshold(Thresh thresh, float slope):
#     return thresh.intercept - thresh.constant * log10(slope)


# # cdef inline Jump make_ramp(Fixed fixed, float read_noise, float [:] resultants):
# #     """
# #     Fast constructor for the Jump C class.

# #     This is signifantly faster than using the `__init__` or `__cinit__`
# #         this is because this does not have to pass through the Python as part
# #         of the construction.

# #     Parameters
# #     ----------
# #     fixed : Fixed
# #         Fixed values for all pixels
# #     resultants : float [:]
# #         array of resultants for single pixel
# #             - memoryview of a numpy array to avoid passing through Python

# #     Return
# #     ------
# #     ramp : Jump
# #         Jump C-class object
# #     """

# #     cdef Jump jump = Jump()

# #     jump.start = start
# #     jump.end = end

# #     jump.resultants = resultants
# #     jump.t_bar = t_bar
# #     jump.tau = tau

# #     jump.read_noise = read_noise

# #     jump.n_reads = n_reads

# #     return jump
