import numpy

from stcal.ramp_fitting.ramp_fit import ramp_fit_data
from stcal.ramp_fitting.ramp_fit_class import RampData

dqflags = {
    "GOOD": 0,  # Good pixel.
    "DO_NOT_USE": 2**0,  # Bad pixel. Do not use.
    "SATURATED": 2**1,  # Pixel saturated during exposure.
    "JUMP_DET": 2**2,  # Jump detected during exposure.
    "NO_GAIN_VALUE": 2**19,  # Gain cannot be measured.
    "UNRELIABLE_SLOPE": 2**24,  # Slope variance large (i.e., noisy pixel).
}

GOOD = dqflags["GOOD"]
DNU = dqflags["DO_NOT_USE"]
SAT = dqflags["SATURATED"]
JUMP = dqflags["JUMP_DET"]


def base_neg_med_rates_single_integration():
    """
    Creates single integration data for testing ensuring negative median rates.
    """
    nints, ngroups, nrows, ncols = 1, 10, 1, 1
    rnoise_val, gain_val = 10.0, 1.0
    nframes, gtime, dtime = 1, 1.0, 1
    dims = (nints, ngroups, nrows, ncols)
    var = (rnoise_val, gain_val)
    tm = (nframes, gtime, dtime)
    ramp_data, rnoise, gain = setup_inputs(dims, var, tm)

    # Set up negative ramp
    neg_ramp = numpy.array([k + 1 for k in range(ngroups)])
    nslope = -0.5
    neg_ramp = neg_ramp * nslope
    ramp_data.data[0, :, 0, 0] = neg_ramp

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    slopes, cube, optional, gls_dummy = ramp_fit_data(
        ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags
    )

    return slopes, cube, optional, gls_dummy


def base_neg_med_rates_multi_integrations():
    """
    Creates multi-integration data for testing ensuring negative median rates.
    """
    nints, ngroups, nrows, ncols = 3, 10, 1, 1
    rnoise_val, gain_val = 10.0, 1.0
    nframes, gtime, dtime = 1, 1.0, 1
    dims = (nints, ngroups, nrows, ncols)
    var = (rnoise_val, gain_val)
    tm = (nframes, gtime, dtime)
    ramp_data, rnoise, gain = setup_inputs(dims, var, tm)

    # Set up negative ramp
    neg_ramp = numpy.array([k + 1 for k in range(ngroups)])
    nslope = -0.5
    neg_ramp = neg_ramp * nslope
    ramp_data.data[0, :, 0, 0] = neg_ramp
    for k in range(1, nints):
        n = k + 1
        ramp_data.data[k, :, 0, 0] = neg_ramp * n

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    slopes, cube, optional, gls_dummy = ramp_fit_data(
        ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags
    )

    return slopes, cube, optional, gls_dummy, dims


def base_neg_med_rates_single_integration_multi_segment():
    """
    Creates single integration, multi-segment data for testing ensuring
    negative median rates.
    """
    nints, ngroups, nrows, ncols = 1, 15, 2, 1
    rnoise_val, gain_val = 10.0, 1.0
    nframes, gtime, dtime = 1, 1.0, 1
    dims = (nints, ngroups, nrows, ncols)
    var = (rnoise_val, gain_val)
    tm = (nframes, gtime, dtime)
    ramp_data, rnoise, gain = setup_inputs(dims, var, tm)

    # Set up negative ramp
    neg_ramp = numpy.array([k + 1 for k in range(ngroups)])
    nslope = -0.5
    neg_ramp = neg_ramp * nslope
    ramp_data.data[0, :, 0, 0] = neg_ramp

    ramp_data.data[0, 5:, 1, 0] = ramp_data.data[0, 5:, 1, 0] + 50
    ramp_data.groupdq[0, 5, 1, 0] = dqflags["JUMP_DET"]
    ramp_data.data[0, 10:, 1, 0] = ramp_data.data[0, 10:, 1, 0] + 50
    ramp_data.groupdq[0, 10, 1, 0] = dqflags["JUMP_DET"]

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    slopes, cube, optional, gls_dummy = ramp_fit_data(
        ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags
    )

    return slopes, cube, optional, gls_dummy, dims


def setup_inputs(dims, var, tm):
    """
    Given dimensions, variances, and timing data, this creates test data to
    be used for unit tests.
    """
    nints, ngroups, nrows, ncols = dims
    rnoise, gain = var
    nframes, gtime, dtime = tm

    data = numpy.zeros(shape=(nints, ngroups, nrows, ncols), dtype=numpy.float32)
    err = numpy.ones(shape=(nints, ngroups, nrows, ncols), dtype=numpy.float32)
    pixdq = numpy.zeros(shape=(nrows, ncols), dtype=numpy.uint32)
    gdq = numpy.zeros(shape=(nints, ngroups, nrows, ncols), dtype=numpy.uint8)

    base_array = numpy.array([k + 1 for k in range(ngroups)])
    base, inc = 1.5, 1.5
    for row in range(nrows):
        for col in range(ncols):
            data[0, :, row, col] = base_array * base
            base = base + inc

    for c_int in range(1, nints):
        data[c_int, :, :, :] = data[0, :, :, :].copy()

    ramp_data = RampData()
    ramp_data.set_arrays(data=data, err=err, groupdq=gdq, pixeldq=pixdq)
    ramp_data.set_meta(
        name="MIRI",
        frame_time=dtime,
        group_time=gtime,
        groupgap=0,
        nframes=nframes,
        drop_frames1=None,
    )
    ramp_data.set_dqflags(dqflags)

    gain = numpy.ones(shape=(nrows, ncols), dtype=numpy.float64) * gain
    rnoise = numpy.full((nrows, ncols), rnoise, dtype=numpy.float32)

    return ramp_data, rnoise, gain


def jp_2326_test_setup():
    """
    Sets up data for MIRI testing DO_NOT_USE flags at the beginning of ramps.
    """
    # Set up ramp data
    ramp = numpy.array(
        [
            120.133545,
            117.85222,
            87.38832,
            66.90588,
            51.392555,
            41.65941,
            32.15081,
            24.25277,
            15.955284,
            9.500946,
        ]
    )
    dnu = dqflags["DO_NOT_USE"]
    dq = numpy.array([dnu, 0, 0, 0, 0, 0, 0, 0, 0, dnu])

    nints, ngroups, nrows, ncols = 1, len(ramp), 1, 1
    data = numpy.zeros((nints, ngroups, nrows, ncols))
    gdq = numpy.zeros((nints, ngroups, nrows, ncols), dtype=numpy.uint8)
    err = numpy.zeros((nints, ngroups, nrows, ncols))
    pdq = numpy.zeros((nrows, ncols), dtype=numpy.uint32)

    data[0, :, 0, 0] = ramp.copy()
    gdq[0, :, 0, 0] = dq.copy()

    ramp_data = RampData()
    ramp_data.set_arrays(data=data, err=err, groupdq=gdq, pixeldq=pdq)
    ramp_data.set_meta(
        name="MIRI",
        frame_time=2.77504,
        group_time=2.77504,
        groupgap=0,
        nframes=1,
        drop_frames1=None,
    )
    ramp_data.set_dqflags(dqflags)

    # Set up gain and read noise
    gain = numpy.ones(shape=(nrows, ncols), dtype=numpy.float32) * 5.5
    rnoise = numpy.ones(shape=(nrows, ncols), dtype=numpy.float32) * 1000.0

    return ramp_data, gain, rnoise


def run_one_group_ramp_suppression(nints, suppress):
    """
    Forms the base of the one group suppression tests.  Create three ramps
    using three pixels with two integrations.  In the first integration:
        The first ramp has no good groups.
        The second ramp has one good groups.
        The third ramp has all good groups.

    In the second integration all pixels have all good groups.
    """
    # Define the data.
    ngroups, nrows, ncols = 5, 1, 3
    dims = (nints, ngroups, nrows, ncols)
    rnoise, gain = 10, 1
    nframes, group_time, frame_time = 1, 5.0, 1
    var = rnoise, gain
    tm = nframes, group_time, frame_time

    # Using the above create the classes and arrays.
    ramp_data, rnoise2d, gain2d = setup_inputs(dims, var, tm)

    arr = numpy.array([k + 1 for k in range(ngroups)], dtype=float)

    sat = ramp_data.flags_saturated
    sat_dq = numpy.array([sat] * ngroups, dtype=ramp_data.groupdq.dtype)
    zdq = numpy.array([0] * ngroups, dtype=ramp_data.groupdq.dtype)

    ramp_data.data[0, :, 0, 0] = arr
    ramp_data.data[0, :, 0, 1] = arr
    ramp_data.data[0, :, 0, 2] = arr

    ramp_data.groupdq[0, :, 0, 0] = sat_dq  # All groups sat
    ramp_data.groupdq[0, :, 0, 1] = sat_dq  # 0th good, all others sat
    ramp_data.groupdq[0, 0, 0, 1] = 0
    ramp_data.groupdq[0, :, 0, 2] = zdq  # All groups good

    if nints > 1:
        ramp_data.data[1, :, 0, 0] = arr
        ramp_data.data[1, :, 0, 1] = arr
        ramp_data.data[1, :, 0, 2] = arr

        # All good ramps
        ramp_data.groupdq[1, :, 0, 0] = zdq
        ramp_data.groupdq[1, :, 0, 1] = zdq
        ramp_data.groupdq[1, :, 0, 2] = zdq

    ramp_data.suppress_one_group_ramps = suppress

    algo = "OLS"
    save_opt, ncores, bufsize = False, "none", 1024 * 30000
    slopes, cube, ols_opt, gls_opt = ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise2d, gain2d, algo, "optimal", ncores, dqflags
    )

    return slopes, cube, dims


def create_zero_frame_data():
    """
    A two integration three pixel image.

    The first integration:
    1. Good first group with the remainder of the ramp being saturated.
    2. Saturated ramp.
    3. Saturated ramp with ZEROFRAME data used for the first group.

    The second integration has all good groups with half the data values.
    """
    # Create meta data.
    frame_time, nframes, groupgap = 10.736, 4, 1
    group_time = (nframes + groupgap) * frame_time
    nints, ngroups, nrows, ncols = 2, 5, 1, 3
    rnval, gval = 10.0, 5.0

    # Create arrays for RampData.
    data = numpy.zeros(shape=(nints, ngroups, nrows, ncols), dtype=numpy.float32)
    err = numpy.ones(shape=(nints, ngroups, nrows, ncols), dtype=numpy.float32)
    pixdq = numpy.zeros(shape=(nrows, ncols), dtype=numpy.uint32)
    gdq = numpy.zeros(shape=(nints, ngroups, nrows, ncols), dtype=numpy.uint8)
    zframe = numpy.ones(shape=(nints, nrows, ncols), dtype=numpy.float32)

    # Create base ramps for each pixel in each integration.
    base_slope = 2000.0
    base_arr = [8000.0 + k * base_slope for k in range(ngroups)]
    base_ramp = numpy.array(base_arr, dtype=numpy.float32)

    data[0, :, 0, 0] = base_ramp
    data[0, :, 0, 1] = base_ramp
    data[0, :, 0, 2] = base_ramp
    data[1, :, :, :] = data[0, :, :, :] / 2.0

    # ZEROFRAME data.
    fdn = (data[0, 1, 0, 0] - data[0, 0, 0, 0]) / (nframes + groupgap)
    dummy = data[0, 0, 0, 2] - (fdn * 2.5)
    zframe[0, 0, :] *= dummy
    zframe[0, 0, 1] = 0.0  # ZEROFRAME is saturated too.
    fdn = (data[1, 1, 0, 0] - data[1, 0, 0, 0]) / (nframes + groupgap)
    dummy = data[1, 0, 0, 2] - (fdn * 2.5)
    zframe[1, 0, :] *= dummy

    # Set up group DQ array.
    gdq[0, :, :, :] = dqflags["SATURATED"]
    gdq[0, 0, 0, 0] = dqflags["GOOD"]

    # Create RampData for testing.
    ramp_data = RampData()
    ramp_data.set_arrays(data=data, err=err, groupdq=gdq, pixeldq=pixdq)
    ramp_data.set_meta(
        name="NIRCam",
        frame_time=frame_time,
        group_time=group_time,
        groupgap=groupgap,
        nframes=nframes,
        drop_frames1=None,
    )
    ramp_data.set_dqflags(dqflags)

    ramp_data.suppress_one_group_ramps = False
    ramp_data.zeroframe = zframe

    # Create variance arrays
    gain = numpy.ones((nrows, ncols), numpy.float32) * gval
    rnoise = numpy.ones((nrows, ncols), numpy.float32) * rnval

    return ramp_data, gain, rnoise


def create_blank_ramp_data(dims, var, tm):
    """
    Create empty RampData classes, as well as gain and read noise arrays,
    based on dimensional, variance, and timing input.
    """
    nints, ngroups, nrows, ncols = dims
    rnval, gval = var
    frame_time, nframes, groupgap = tm
    group_time = (nframes + groupgap) * frame_time

    data = numpy.zeros(shape=(nints, ngroups, nrows, ncols), dtype=numpy.float32)
    err = numpy.ones(shape=(nints, ngroups, nrows, ncols), dtype=numpy.float32)
    pixdq = numpy.zeros(shape=(nrows, ncols), dtype=numpy.uint32)
    gdq = numpy.zeros(shape=(nints, ngroups, nrows, ncols), dtype=numpy.uint8)

    ramp_data = RampData()
    ramp_data.set_arrays(data=data, err=err, groupdq=gdq, pixeldq=pixdq)
    ramp_data.set_meta(
        name="NIRSpec",
        frame_time=frame_time,
        group_time=group_time,
        groupgap=groupgap,
        nframes=nframes,
        drop_frames1=None,
    )
    ramp_data.set_dqflags(dqflags)

    gain = numpy.ones(shape=(nrows, ncols), dtype=numpy.float64) * gval
    rnoise = numpy.ones(shape=(nrows, ncols), dtype=numpy.float64) * rnval

    return ramp_data, gain, rnoise


def time_neg_med_rates_single_integration_slope():
    """
    Make sure the negative ramp has negative slope, the Poisson variance
    is zero, readnoise is non-zero  and the ERR array is a function of
    only RNOISE.
    """
    base_neg_med_rates_single_integration()


def time_neg_med_rates_single_integration_integ():
    """
    Make sure that for the single integration data the single integration
    is the same as the slope data.
    """
    base_neg_med_rates_single_integration()


def time_neg_med_rates_single_integration_optional():
    """
    Make sure that for the single integration data the optional results
    is the same as the slope data.
    """
    base_neg_med_rates_single_integration()


def time_neg_med_rates_multi_integrations_slopes():
    """
    Test computing median rates of a ramp with multiple integrations.
    """
    base_neg_med_rates_multi_integrations()


def time_neg_med_rates_multi_integration_integ():
    """
    Make sure that for the multi-integration data with a negative slope
    results in zero Poisson info and the ERR array a function of only
    RNOISE.
    """
    base_neg_med_rates_multi_integrations()


def time_neg_med_rates_multi_integration_optional():
    """
    Make sure that for the multi-integration data with a negative slope with
    one segment has only one segment in the optional results product as well
    as zero Poisson variance.
    """
    base_neg_med_rates_multi_integrations()


def time_neg_med_rates_single_integration_multi_segment_optional():
    """
    Test a ramp with multiple segments to make sure the right number of
    segments are created and to make sure all Poisson segements are set to
    zero.
    """
    base_neg_med_rates_single_integration_multi_segment()


def time_utils_dq_compress_final():
    """
    If there is any integration that has usable data, the DO_NOT_USE flag
    should not be set in the final DQ flag, even if it is set for one or more
    integrations.

    Set up a multi-integration 3 pixel data array each ramp as the following:
    1. Both integrations having all groups saturated.
        - Since all groups are saturated in all integrations the final DQ value
          for this pixel should have the DO_NOT_USE flag set.  Ramp fitting
          will flag a pixel as DO_NOT_USE in an integration if all groups in
          that integration are saturated.
    2. Only one integration with all groups saturated.
        - Since all groups are saturated in only one integration the final DQ
          value for this pixel should not have the DO_NOT_USE flag set, even
          though it is set in one of the integrations.
    3. No group saturated in any integration.
        - This is a "normal" pixel where there is usable information in both
          integrations.  Neither integration should have the DO_NOT_SET flag
          set, nor should it be set in the final DQ.
    """
    nints, ngroups, nrows, ncols = 2, 5, 1, 3
    rnoise_val, gain_val = 10.0, 1.0
    nframes, gtime, dtime = 1, 1.0, 1
    dims = (nints, ngroups, nrows, ncols)
    var = (rnoise_val, gain_val)
    tm = (nframes, gtime, dtime)
    ramp_data, rnoise, gain = setup_inputs(dims, var, tm)

    ramp_data.groupdq[0, :, 0, 0] = numpy.array([dqflags["SATURATED"]] * ngroups)
    ramp_data.groupdq[1, :, 0, 0] = numpy.array([dqflags["SATURATED"]] * ngroups)

    ramp_data.groupdq[0, :, 0, 1] = numpy.array([dqflags["SATURATED"]] * ngroups)

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    ramp_fit_data(
        ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags
    )


def time_miri_ramp_dnu_at_ramp_beginning():
    """
    Tests a MIRI ramp with DO_NOT_USE in the first two groups and last group.
    This test ensures these groups are properly excluded.
    """
    ramp_data, gain, rnoise = jp_2326_test_setup()
    ramp_data.groupdq[0, 1, 0, 0] = dqflags["DO_NOT_USE"]

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    ramp_fit_data(
        ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags
    )


def time_miri_ramp_dnu_and_jump_at_ramp_beginning():
    """
    Tests a MIRI ramp with DO_NOT_USE in the first and last group, with a
    JUMP_DET in the second group. This test ensures the DO_NOT_USE groups are
    properly excluded, while the JUMP_DET group is included.
    """
    ramp_data, gain, rnoise = jp_2326_test_setup()
    ramp_data.groupdq[0, 1, 0, 0] = dqflags["JUMP_DET"]

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    ramp_fit_data(
        ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags
    )


def time_2_group_cases():
    """
    Tests the special cases of 2 group ramps.  Create multiple pixel ramps
    with two groups to test the various DQ cases.
    """
    base_group = [-12328.601, -4289.051]
    base_err = [0.0, 0.0]
    gain_val = 0.9699
    rnoise_val = 9.4552

    possibilities = [
        # Both groups are good
        [GOOD, GOOD],
        # Both groups are bad.  Note saturated 0th group kills group 1.
        [SAT, GOOD],
        [DNU | SAT, GOOD],
        [DNU, SAT],
        # One group is bad, while the other group is good.
        [DNU, GOOD],
        [GOOD, DNU],
        [GOOD, DNU | SAT],
    ]
    nints, ngroups, nrows, ncols = 1, 2, 1, len(possibilities)
    dims = nints, ngroups, nrows, ncols
    npix = nrows * ncols

    # Create the ramp data with a pixel for each of the possibilities above.
    # Set the data to the base data of 2 groups and set up the group DQ flags
    # are taken from the 'possibilities' list above.

    # Resize gain and read noise arrays.
    rnoise = numpy.ones((1, npix)) * rnoise_val
    gain = numpy.ones((1, npix)) * gain_val
    pixeldq = numpy.zeros((1, npix), dtype=numpy.uint32)

    data = numpy.zeros(dims, dtype=numpy.float32)  # Science data
    for k in range(npix):
        data[0, :, 0, k] = numpy.array(base_group)

    err = numpy.zeros(dims, dtype=numpy.float32)  # Error data
    for k in range(npix):
        err[0, :, 0, k] = numpy.array(base_err)

    groupdq = numpy.zeros(dims, dtype=numpy.uint8)  # Group DQ
    for k in range(npix):
        groupdq[0, :, 0, k] = numpy.array(possibilities[k])

    # Setup the RampData class to run ramp fitting on.
    ramp_data = RampData()

    ramp_data.set_arrays(data, err, groupdq, pixeldq)

    ramp_data.set_meta(
        name="NIRSPEC",
        frame_time=14.58889,
        group_time=14.58889,
        groupgap=0,
        nframes=1,
        drop_frames1=None,
    )

    ramp_data.set_dqflags(dqflags)

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    ramp_fit_data(
        ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags
    )


def time_one_group_ramp_suppressed_one_integration():
    """
    Tests one group ramp fitting where suppression turned on.
    """
    run_one_group_ramp_suppression(1, True)


def time_one_group_ramp_not_suppressed_one_integration():
    """
    Tests one group ramp fitting where suppression turned off.
    """
    run_one_group_ramp_suppression(1, False)


def time_one_group_ramp_suppressed_two_integrations():
    """
    Test one good group ramp and two integrations with
    suppression suppression turned on.
    """
    run_one_group_ramp_suppression(2, True)


def time_one_group_ramp_not_suppressed_two_integrations():
    """
    Test one good group ramp and two integrations with
    suppression suppression turned off.
    """
    run_one_group_ramp_suppression(2, False)


def time_zeroframe():
    """
    A two integration three pixel image.

    The first integration:
    1. Good first group with the remainder of the ramp being saturated.
    2. Saturated ramp.
    3. Saturated ramp with ZEROFRAME data used for the first group.

    The second integration has all good groups with half the data values.
    """
    ramp_data, gain, rnoise = create_zero_frame_data()

    algo, save_opt, ncores, bufsize = "OLS", False, "none", 1024 * 30000
    ramp_fit_data(
        ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )


def time_all_sat():
    """
    Test all ramps in all integrations saturated.
    """
    nints, ngroups, nrows, ncols = 2, 5, 1, 3
    rnval, gval = 10.0, 5.0
    frame_time, nframes, groupgap = 10.736, 4, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp, gain, rnoise = create_blank_ramp_data(dims, var, tm)
    ramp.groupdq[:, 0, :, :] = ramp.flags_saturated

    algo, save_opt, ncores, bufsize = "OLS", False, "none", 1024 * 30000
    ramp_fit_data(
        ramp, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )


def time_dq_multi_int_dnu():
    """
    Tests to make sure that integration DQ flags get set when all groups
    in an integration are set to DO_NOT_USE.
    """
    nints, ngroups, nrows, ncols = 2, 5, 1, 1
    rnval, gval = 10.0, 5.0
    frame_time, nframes, groupgap = 10.736, 4, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp, gain, rnoise = create_blank_ramp_data(dims, var, tm)
    base_arr = [(k + 1) * 100 for k in range(ngroups)]
    dq_arr = [ramp.flags_do_not_use] * ngroups

    ramp.data[0, :, 0, 0] = numpy.array(base_arr)
    ramp.data[1, :, 0, 0] = numpy.array(base_arr)
    ramp.groupdq[0, :, 0, 0] = numpy.array(dq_arr)

    algo, save_opt, ncores, bufsize = "OLS", False, "none", 1024 * 30000
    ramp_fit_data(
        ramp, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags
    )
