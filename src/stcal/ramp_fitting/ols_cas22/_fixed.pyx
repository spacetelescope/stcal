import numpy as np
cimport numpy as np

from stcal.ramp_fitting.ols_cas22._fixed cimport Fixed

cdef class Fixed:
    """
    Class to contain the data fixed for all pixels and commonly referenced
    universal values for jump detection

    Parameters
    ----------
    t_bar : vector[float]
        mean times of resultants
    tau : vector[float]
        variance weighted mean times of resultants
    n_reads : vector[int]
        number of reads contributing to reach resultant

    t_bar_1 : vector[float]
        single differences of t_bar (t_bar[i+1] - t_bar[i])
    t_bar_1_sq : vector[float]
        squared single differences of t_bar (t_bar[i+1] - t_bar[i])**2
    t_bar_2 : vector[float]
        double differences of t_bar (t_bar[i+2] - t_bar[i])
    t_bar_2_sq: vector[float]
        squared double differences of t_bar (t_bar[i+2] - t_bar[i])**2
    sigma_1 : vector[float]
        single of sigma term read_noise * ((1/n_reads[i+1]) + (1/n_reads[i]))
    sigma_2 : vector[float]
        double of sigma term read_noise * ((1/n_reads[i+1]) + (1/n_reads[i]))
    slope_var_1 : vector[float]
        single of slope variance term
        ([tau[i] + tau[i+1] - min(t_bar[i], t_bar[i+1])) * correction(i, i+1)
    slope_var_2 : vector[float]
        double of slope variance term
        ([tau[i] + tau[i+2] - min(t_bar[i], t_bar[i+2])) * correction(i, i+2)
    """

    cdef inline float[:] t_bar_diff(Fixed self, int offset):
        """
        Compute the difference offset of t_bar

        Parameters
        ----------
        offset : int
            index offset to compute difference
        """
        cdef int n_diff = len(self.t_bar) - offset
        cdef float[:] diff = (np.roll(self.t_bar, -offset) - self.t_bar)[:n_diff]

        return diff

    cdef inline float[:] t_bar_diff_sq(Fixed self, int offset):
        """
        Compute the square difference offset of t_bar

        Parameters
        ----------
        offset : int
            index offset to compute difference
        """
        cdef int n_diff = len(self.t_bar) - offset
        cdef float[:] diff = (np.roll(self.t_bar, -offset) - self.t_bar)[:n_diff] ** 2

        return diff

    cdef inline float[:] recip_val(Fixed self, int offset):
        """
        Compute the recip values
            (1/n_reads[i+offset] + 1/n_reads[i])

        Parameters
        ----------
        offset : int
            index offset to compute difference
        """
        cdef int n_diff = len(self.t_bar) - offset
        
        cdef float[:] recip = ((1 / np.roll(self.n_reads, -offset)).astype(np.float32) +
                               (1 / np.array(self.n_reads)).astype(np.float32))[:n_diff]

        return recip

    cdef inline float correction(Fixed self, int i, int j):
        """Compute the correction factor

        Parameters
        ----------
        i : int
            The index of the first read in the segment
        j : int
            The index of the last read in the segment
        """
        cdef float denom = self.t_bar[self.n_reads[i] - 1] - self.t_bar[0]

        if j - i == 1:
            return (1 - (self.t_bar[i + 1] - self.t_bar[i]) / denom) ** 2
        else:
            return (1 - 0.75 * (self.t_bar[i + 2] - self.t_bar[i]) / denom) ** 2

    cdef inline float[:] slope_var_val(Fixed self, int offset):
        """
        Compute the sigma values
            (tau[i] + tau[i+offset] - min(t_bar[i], t_bar[i+offset])) *
                correction(i, i+offset)

        Parameters
        ----------
        offset : int
            index offset to compute difference
        """
        cdef int n_diff = len(self.t_bar) - offset

        cdef float[:] slope_var_val = (
            (self.tau + np.roll(self.tau, -offset) -
             np.minimum(self.t_bar, np.roll(self.t_bar, -offset))) *
            self.correction(0, offset))[:n_diff]

        return slope_var_val


cdef inline Fixed make_fixed(float[:] t_bar, float[:] tau, int[:] n_reads, bool use_jump):

    cdef Fixed fixed = Fixed()

    fixed.use_jump = use_jump
    fixed.t_bar = t_bar
    fixed.tau = tau
    fixed.n_reads = n_reads

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