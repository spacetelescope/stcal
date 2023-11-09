"""
Define the basic types and functions for the CAS22 algorithm with jump detection

Structs
-------
    RampIndex
        int start: starting index of the ramp in the resultants
        int end: ending index of the ramp in the resultants

            Note that the Python range would be [start:end+1] for any ramp index.
    RampFit
        float slope: slope of a single ramp
        float read_var: read noise variance of a single ramp
        float poisson_var: poisson noise variance of single ramp
    RampFits
        vector[RampFit] fits: ramp fits (in time order) for a single pixel
        vector[RampIndex] index: ramp indices (in time order) for a single pixel
        RampFit average: average ramp fit for a single pixel
    ReadPatternMetata
        vector[float] t_bar: mean time of each resultant
        vector[float] tau: variance time of each resultant
        vector[int] n_reads: number of reads in each resultant

            Note that these are entirely computed from the read_pattern and
            read_time (which should be constant for a given telescope) for the
            given observation.
    Thresh
        float intercept: intercept of the threshold
        float constant: constant of the threshold

Enums
-----
    Diff
        This is the enum to track the index for single vs double difference related
        computations.

        single: single difference
        double: double difference

    Parameter
        This is the enum to track the index of the computed fit parameters for
        the ramp fit.

        intercept: the intercept of the ramp fit
        slope: the slope of the ramp fit

    Variance
        This is the enum to track the index of the computed variance values for
        the ramp fit.

        read_var: read variance computed
        poisson_var: poisson variance computed
        total_var: total variance computed (read_var + poisson_var)

    RampJumpDQ
        This enum is to specify the DQ flags for Ramp/Jump detection

        JUMP_DET: jump detected

Functions
---------
    get_power
        Return the power from Casertano+22, Table 2
    threshold
        Compute jump threshold
        - cpdef gives a python wrapper, but the python version of this method
          is considered private, only to be used for testing
    init_ramps
        Find initial ramps for each pixel, accounts for DQ flags
        - A python wrapper, _init_ramps_list, that adjusts types so they can
          be directly inspected in python exists for testing purposes only.
    metadata_from_read_pattern
        Read the read pattern and derive the baseline metadata parameters needed
        - cpdef gives a python wrapper, but the python version of this method
          is considered private, only to be used for testing
"""
from stcal.ramp_fitting.ols_cas22._core cimport Parameter, Variance, RampJumpDQ
