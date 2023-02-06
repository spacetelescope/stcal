import numpy

from stcal.ramp_fitting.ramp_fit import ramp_fit_data
from stcal.ramp_fitting.ramp_fit_class import RampData
from tests.test_ramp_fitting import base_neg_med_rates_single_integration, base_neg_med_rates_multi_integrations, \
    base_neg_med_rates_single_integration_multi_segment, setup_inputs, dqflags, jp_2326_test_setup, DNU, GOOD, SAT, \
    run_one_group_ramp_suppression, create_zero_frame_data, create_blank_ramp_data


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
    rnoise_val, gain_val = 10., 1.
    nframes, gtime, dtime = 1, 1., 1
    dims = (nints, ngroups, nrows, ncols)
    var = (rnoise_val, gain_val)
    tm = (nframes, gtime, dtime)
    ramp_data, rnoise, gain = setup_inputs(dims, var, tm)

    ramp_data.groupdq[0, :, 0, 0] = numpy.array([dqflags["SATURATED"]] * ngroups)
    ramp_data.groupdq[1, :, 0, 0] = numpy.array([dqflags["SATURATED"]] * ngroups)

    ramp_data.groupdq[0, :, 0, 1] = numpy.array([dqflags["SATURATED"]] * ngroups)

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    ramp_fit_data(ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags)


def time_miri_ramp_dnu_at_ramp_beginning():
    """
    Tests a MIRI ramp with DO_NOT_USE in the first two groups and last group.
    This test ensures these groups are properly excluded.
    """
    ramp_data, gain, rnoise = jp_2326_test_setup()
    ramp_data.groupdq[0, 1, 0, 0] = dqflags["DO_NOT_USE"]

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    ramp_fit_data(ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags)


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
    ramp_fit_data(ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags)


def time_2_group_cases():
    """
    Tests the special cases of 2 group ramps.  Create multiple pixel ramps
    with two groups to test the various DQ cases.
    """
    base_group = [-12328.601, -4289.051]
    base_err = [0., 0.]
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
        drop_frames1=None)

    ramp_data.set_dqflags(dqflags)

    # Run ramp fit on RampData
    buffsize, save_opt, algo, wt, ncores = 512, True, "OLS", "optimal", "none"
    ramp_fit_data(ramp_data, buffsize, save_opt, rnoise, gain, algo, wt, ncores, dqflags)


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
    ramp_fit_data(ramp_data, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags)


def time_all_sat():
    """
    Test all ramps in all integrations saturated.
    """
    nints, ngroups, nrows, ncols = 2, 5, 1, 3
    rnval, gval = 10., 5.
    frame_time, nframes, groupgap = 10.736, 4, 1

    dims = nints, ngroups, nrows, ncols
    var = rnval, gval
    tm = frame_time, nframes, groupgap

    ramp, gain, rnoise = create_blank_ramp_data(dims, var, tm)
    ramp.groupdq[:, 0, :, :] = ramp.flags_saturated

    algo, save_opt, ncores, bufsize = "OLS", False, "none", 1024 * 30000
    ramp_fit_data(ramp, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags)


def time_dq_multi_int_dnu():
    """
    Tests to make sure that integration DQ flags get set when all groups
    in an integration are set to DO_NOT_USE.
    """
    nints, ngroups, nrows, ncols = 2, 5, 1, 1
    rnval, gval = 10., 5.
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
    ramp_fit_data(ramp, bufsize, save_opt, rnoise, gain, algo, "optimal", ncores, dqflags)
