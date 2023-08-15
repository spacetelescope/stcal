cdef class Ramp:
    # """
    # Class to contain the data for a single pixel ramp to be fit

    # This has to be a class rather than a struct in order to contain memory views

    # Parameters
    # ----------
    # resultants : float [:]
    #     array of resultants for single pixel
    # start : int
    #     starting point of portion to fit within this pixel
    # end : int
    #     ending point of portion to fit within this pixel
    # read_noise : float
    #     read noise for this pixel
    # t_bar : float [:]
    #     mean times of resultants
    # tau : float [:]
    #     variance weighted mean times of resultants
    # n_reads : int [:]
    #     number of reads contributing to reach resultant
    # """
    cdef public:
        int start, end
        float read_noise
        float [:] resultants, t_bar, tau
        int [:] n_reads

cdef struct Fit:
    # """
    # Output of a single fit

    # Parameters
    # ----------
    # slope : float
    #     fit slope
    # slope_read_var : float
    #     read noise induced variance in slope
    # slope_poisson_var : float
    #     coefficient of Poisson-noise induced variance in slope
    #     multiply by true flux to get actual Poisson variance.
    # """

    float slope, slope_read_var, slope_poisson_var

cdef Ramp make_ramp(
    float [:] resultants, int start, int end, float read_noise,
    float [:] t_bar, float [:] tau, int [:] n_reads)