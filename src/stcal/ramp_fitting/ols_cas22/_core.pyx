from stcal.ramp_fitting.ols_cas22._core cimport Ramp, Fit


cdef inline Ramp make_ramp(
    float [:] resultants, int start, int end, float read_noise,
    float [:] t_bar, float [:] tau, int [:] n_reads):
    """
    Fast constructor for the Ramp C class.

    This is signifantly faster than using the `__init__` or `__cinit__`
        this is because this does not have to pass through the Python as part
        of the construction.

    Parameters
    ----------
    resultants : float [:]
        array of resultants for single pixel
    start : int
        starting point of portion to fit within this pixel
    end : int
        ending point of portion to fit within this pixel
    read_noise : float
        read noise for this pixel
    t_bar : float [:]
        mean times of resultants
    tau : float [:]
        variance weighted mean times of resultants
    n_reads : int [:]
        number of reads contributing to reach resultant

    Return
    ------
    ramp : Ramp
        Ramp C-class object
    """

    cdef Ramp ramp = Ramp()

    ramp.start = start
    ramp.end = end

    ramp.resultants = resultants
    ramp.t_bar = t_bar
    ramp.tau = tau

    ramp.read_noise = read_noise

    ramp.n_reads = n_reads

    return ramp

