#! /usr/bin/env python
#
# utils.py: utility functions
import logging
import warnings

import numpy as np


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Replace zero or negative variances with this:
LARGE_VARIANCE = 1.0e8
LARGE_VARIANCE_THRESHOLD = 0.01 * LARGE_VARIANCE


class OptRes:
    """
    Object to hold optional results for all good pixels for
    y-intercept, slope, uncertainty for y-intercept, uncertainty for
    slope, inverse variance, first frame (for pedestal image), and
    cosmic ray magnitude.
    """

    def __init__(self, n_int, imshape, max_seg, nreads, save_opt):
        """
        Initialize the optional attributes. These are 4D arrays for the
        segment-specific values of the y-intercept, the slope, the uncertainty
        associated with both, the weights, the approximate cosmic ray
        magnitudes, and the inverse variance.  These are 3D arrays for the
        integration-specific first frame and pedestal values.

        Parameters
        ----------
        n_int : int
            number of integrations in data set

        imshape : tuple
            shape of 2D image

        max_seg : int
            maximum number of segments fit

        nreads : int
            number of reads in an integration

        save_opt : bool
           save optional fitting results
        """
        self.slope_seg = np.zeros((n_int, max_seg, *imshape), dtype=np.float32)
        if save_opt:
            self.yint_seg = np.zeros((n_int, max_seg, *imshape), dtype=np.float32)
            self.sigyint_seg = np.zeros((n_int, max_seg, *imshape), dtype=np.float32)
            self.sigslope_seg = np.zeros((n_int, max_seg, *imshape), dtype=np.float32)
            self.inv_var_seg = np.zeros((n_int, max_seg, *imshape), dtype=np.float32)
            self.firstf_int = np.zeros((n_int, *imshape), dtype=np.float32)
            self.ped_int = np.zeros((n_int, *imshape), dtype=np.float32)
            self.cr_mag_seg = np.zeros((n_int, nreads, *imshape), dtype=np.float32)
            self.var_p_seg = np.zeros((n_int, max_seg, *imshape), dtype=np.float32)
            self.var_r_seg = np.zeros((n_int, max_seg, *imshape), dtype=np.float32)

    def reshape_res(self, num_int, rlo, rhi, sect_shape, ff_sect, save_opt):
        """
        Loop over the segments and copy the reshaped 2D segment-specific
        results for the current data section to the 4D output arrays.

        Parameters
        ----------
        num_int : int
            integration number

        rlo : int
            first column of section

        rhi : int
            last column of section

        sect_sect : tuple
            shape of section image

        ff_sect : ndarray
            first frame data, 2-D float

        save_opt : bool
            save optional fitting results
        """
        for ii_seg in range(self.slope_seg.shape[1]):
            self.slope_seg[num_int, ii_seg, rlo:rhi, :] = self.slope_2d[ii_seg, :].reshape(sect_shape)

            if save_opt:
                self.yint_seg[num_int, ii_seg, rlo:rhi, :] = self.interc_2d[ii_seg, :].reshape(sect_shape)
                self.slope_seg[num_int, ii_seg, rlo:rhi, :] = self.slope_2d[ii_seg, :].reshape(sect_shape)
                self.sigyint_seg[num_int, ii_seg, rlo:rhi, :] = self.siginterc_2d[ii_seg, :].reshape(
                    sect_shape
                )
                self.sigslope_seg[num_int, ii_seg, rlo:rhi, :] = self.sigslope_2d[ii_seg, :].reshape(
                    sect_shape
                )
                self.inv_var_seg[num_int, ii_seg, rlo:rhi, :] = self.inv_var_2d[ii_seg, :].reshape(sect_shape)
                self.firstf_int[num_int, rlo:rhi, :] = ff_sect

    def append_arr(self, num_seg, g_pix, intercept, slope, sig_intercept, sig_slope, inv_var, save_opt):
        """
        Add the fitting results for the current segment to the 2d arrays.

        Parameters
        ----------
        num_seg : ndarray
            counter for segment number within the section, 1-D int

        g_pix : ndarray
            pixels having fitting results in current section, 1-D int

        intercept : ndarray
            intercepts for pixels in current segment and section, 1-D float

        slope : ndarray
            slopes for pixels in current segment and section, 1-D float

        sig_intercept : ndarray
            uncertainties of intercepts for pixels in current segment
            and section, 1-D float

        sig_slope : ndarray
            uncertainties of slopes for pixels in current segment and
            section, 1-D float

        inv_var : ndarray
            reciprocals of variances for fits of pixels in current
            segment and section, 1-D float

        save_opt : bool
            save optional fitting results

        Returns
        -------
        None
        """
        '''
        if False:
            print("=" * 80)
            dbg_print(f"slope         = {slope}")
            dbg_print(f"intercept     = {intercept}")
            dbg_print(f"inv_var       = {inv_var}")
            dbg_print(f"sig_intercept = {sig_intercept}")
            dbg_print(f"sig_slope     = {sig_slope}")
            print("=" * 80)
        '''

        self.slope_2d[num_seg[g_pix], g_pix] = slope[g_pix]

        if save_opt:
            self.interc_2d[num_seg[g_pix], g_pix] = intercept[g_pix]
            self.siginterc_2d[num_seg[g_pix], g_pix] = sig_intercept[g_pix]
            self.sigslope_2d[num_seg[g_pix], g_pix] = sig_slope[g_pix]
            self.inv_var_2d[num_seg[g_pix], g_pix] = inv_var[g_pix]

    def output_optional(self, group_time):
        """
        These results are the cosmic ray magnitudes in the
        segment-specific results for the count rates, y-intercept,
        uncertainty in the slope, uncertainty in the y-intercept,
        pedestal image, fitting weights, and the uncertainties in
        the slope due to poisson noise only and read noise only, and
        the integration-specific results for the pedestal image.  The
        slopes are divided by the effective integration time here to
        yield the count rates. Any variance values that are a large fraction
        of the default value LARGE_VARIANCE correspond to non-existent segments,
        so will be set to 0 here before output.

        Parameters
        ----------
        group_time : float
            effective integration time for a single group

        Returns
        -------
        opt_info : tuple
            The tuple of computed optional results arrays for fitting.
        """
        self.var_p_seg[self.var_p_seg > LARGE_VARIANCE_THRESHOLD] = 0.0
        self.var_r_seg[self.var_r_seg > LARGE_VARIANCE_THRESHOLD] = 0.0

        # Suppress, then re-enable, arithmetic warnings
        warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
        warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)

        # Tiny 'weights' values correspond to non-existent segments, so set to 0.
        self.weights[1.0 / self.weights > LARGE_VARIANCE_THRESHOLD] = 0.0
        warnings.resetwarnings()

        return (
            self.slope_seg,
            self.sigslope_seg,
            self.var_p_seg,
            self.var_r_seg,
            self.yint_seg,
            self.sigyint_seg,
            self.ped_int,
            self.weights,
            self.cr_mag_seg,
        )

    def print_full(self):  # pragma: no cover
        """
        Diagnostic function for printing optional output arrays; most
        useful for tiny datasets.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print("Will now print all optional output arrays - ")
        print(" yint_seg: ")
        print(self.yint_seg)
        print("  ")
        print(" slope_seg: ")
        print(self.slope_seg)
        print("  ")
        print(" sigyint_seg: ")
        print(self.sigyint_seg)
        print("  ")
        print(" sigslope_seg: ")
        print(self.sigslope_seg)
        print("  ")
        print(" inv_var_2d: ")
        print(self.inv_var_2d)
        print("  ")
        print(" firstf_int: ")
        print(self.firstf_int)
        print("  ")
        print(" ped_int: ")
        print(self.ped_int)
        print("  ")
        print(" cr_mag_seg: ")
        print(self.cr_mag_seg)
#END class OptRes


def output_integ(ramp_data, slope_int, dq_int, var_p3, var_r3, var_both3):
    """
    For the OLS algorithm, construct the output integration-specific results.
    Any variance values that are a large fraction of the default value
    LARGE_VARIANCE correspond to non-existent segments, so will be set to 0
    here before output.

    Parameters
    ----------
    ramp_data : RampData
        Contains flag information.

    model : instance of Data Model
       DM object for input

    slope_int : ndarray
       Data cube of weighted slopes for each integration, 3-D float

    dq_int : ndarray
       Data cube of DQ arrays for each integration, 3-D int

    var_p3 : ndarray
        Cube of integration-specific values for the slope variance due to
        Poisson noise only, 3-D float

    var_r3 : ndarray
        Cube of integration-specific values for the slope variance due to
        read noise only, 3-D float

    var_both3 : ndarray
        Cube of integration-specific values for the slope variance due to
        read noise and Poisson noise, 3-D float

    Returns
    -------
    integ_info : tuple
        The tuple of computed integration ramp fitting arrays.

    """
    # Suppress harmless arithmetic warnings for now
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)

    var_p3[var_p3 > LARGE_VARIANCE_THRESHOLD] = 0.0
    var_r3[var_r3 > LARGE_VARIANCE_THRESHOLD] = 0.0
    var_both3[var_both3 > LARGE_VARIANCE_THRESHOLD] = 0.0

    data = slope_int
    invalid_data = ramp_data.flags_saturated | ramp_data.flags_do_not_use
    data[np.bitwise_and(dq_int, invalid_data).astype(bool)] = np.nan

    err = np.sqrt(var_both3)
    dq = dq_int
    var_poisson = var_p3

    var_rnoise = var_r3
    integ_info = (data, dq, var_poisson, var_rnoise, err)

    # Reset the warnings filter to its original state
    warnings.resetwarnings()

    return integ_info


def get_efftim_ped(ramp_data):
    """
    XXX - Work to remove this function.
    Calculate the effective integration time for a single group, and return the
    number of frames per group, and the number of frames dropped between groups.

    Parameters
    ----------
    ramp_data: RampClass
        Object for input data.

    Returns
    -------
    effintim : float
        effective integration time for a single group

    nframes : int
        number of frames averaged per group; from the NFRAMES keyword.

    groupgap : int
        number of frames dropped between groups; from the GROUPGAP keyword.

    dropframes1 : int
        number of frames dropped at the beginning of every integration; from
        the DRPFRMS1 keyword, or 0 if the keyword is missing
    """
    groupgap = ramp_data.groupgap
    nframes = ramp_data.nframes
    frame_time = ramp_data.frame_time
    dropframes1 = ramp_data.drop_frames1

    if dropframes1 is None:  # set to default if missing
        dropframes1 = 0
        log.debug("Missing keyword DRPFRMS1, so setting to default value of 0")

    try:
        effintim = (nframes + groupgap) * frame_time
    except TypeError:
        log.exception("Can not retrieve values needed to calculate integ. time")

    log.debug("Calculating effective integration time for a single group using:")
    log.debug(" groupgap: %s", groupgap)
    log.debug(" nframes: %s", nframes)
    log.debug(" frame_time: %s", frame_time)
    log.debug(" dropframes1: %s", dropframes1)
    log.info("Effective integration time per group: %s", effintim)

    return effintim, nframes, groupgap, dropframes1


def get_dataset_info(ramp_data):
    """
    Extract values for the number of groups, the number of pixels, dataset
    shapes, the number of integrations, the instrument name, the frame time,
    and the observation time.

    Parameters
    ----------
    ramp_data: RampClass
       Object for input data.

    Returns
    -------
    nreads : int
       number of reads in input dataset

    npix : int
       number of pixels in 2D array

    imshape : tuple
       shape of 2D image

    cubeshape : tuple
       shape of input dataset

    n_int : int
       number of integrations

    instrume : str
       instrument

    frame_time : float
       integration time from TGROUP

    ngroups : int
        number of groups per integration

    group_time : float
        Time increment between groups, in seconds.
    """
    instrume = ramp_data.instrument_name
    frame_time = ramp_data.frame_time
    ngroups = ramp_data.data.shape[1]
    group_time = ramp_data.group_time

    n_int = ramp_data.data.shape[0]
    nreads = ramp_data.data.shape[1]
    asize2 = ramp_data.data.shape[2]
    asize1 = ramp_data.data.shape[3]

    npix = asize2 * asize1  # number of pixels in 2D array
    imshape = (asize2, asize1)
    cubeshape = (nreads, *imshape)

    return (nreads, npix, imshape, cubeshape, n_int, instrume, frame_time, ngroups, group_time)


def get_more_info(ramp_data, saturated_flag, jump_flag):  # pragma: no cover
    """
    Get information used by GLS algorithm.

    Parameters
    ----------
    ramp_data: RampClass
        Object for input data.

    Returns
    -------
    group_time : float
        Time increment between groups, in seconds.

    nframes_used : int
        Number of frames that were averaged together to make a group,
        i.e. excluding skipped frames.

    saturated_flag : int
        Group data quality flag that indicates a saturated pixel.

    jump_flag : int
        Group data quality flag that indicates a cosmic ray hit.
    """
    group_time = ramp_data.group_time
    nframes_used = ramp_data.nframes
    saturated_flag = ramp_data.flags_saturated
    jump_flag = ramp_data.flags_jump_det

    return group_time, nframes_used, saturated_flag, jump_flag


def get_max_num_cr(gdq_cube, jump_flag):  # pragma: no cover
    """
    Find the maximum number of cosmic-ray hits in any one pixel.

    Parameters
    ----------
    gdq_cube : ndarray
        The group data quality array, 3-D flag

    jump_flag : int
        The data quality flag indicating a cosmic-ray hit.

    Returns
    -------
    max_num_cr : int
        The maximum number of cosmic-ray hits for any pixel.
    """
    cr_flagged = np.empty(gdq_cube.shape, dtype=np.uint8)
    cr_flagged[:] = np.where(np.bitwise_and(gdq_cube, jump_flag), 1, 0)
    max_num_cr = cr_flagged.sum(axis=0, dtype=np.int32).max()
    del cr_flagged

    return max_num_cr


def remove_bad_singles(segs_beg_3):
    """
    For the current integration and data section, remove all segments having only
    a single group if there are other segments in the ramp.  This method allows
    for the possibility that a ramp can have multiple (necessarily consecutive)
    1-group segments, which in principle could occur if there are consecutive
    cosmic rays.

    Parameters
    ----------
    segs_beg_3 : ndarray
        lengths of all segments for all ramps in the given data section and
        integration; some of these ramps may contain segments having a single
        group, and another segment, 3-D int

    Returns
    -------
    segs_beg_3 : ndarray
        lengths of all segments for all ramps in the given data section and
        integration; segments having a single group, and another segment
        will be removed, 3-D int
    """
    max_seg = segs_beg_3.shape[0]

    # get initial number of ramps having single-group segments
    tot_num_single_grp_ramps = len(np.where((segs_beg_3 == 1) & (segs_beg_3.sum(axis=0) > 1))[0])

    while tot_num_single_grp_ramps > 0:
        # until there are no more single-group segments
        for ii_0 in range(max_seg):
            slice_0 = segs_beg_3[ii_0, :, :]

            for ii_1 in range(max_seg):  # correctly includes EARLIER segments
                if ii_0 == ii_1:  # don't compare with itself
                    continue

                slice_1 = segs_beg_3[ii_1, :, :]

                # Find ramps of a single-group segment and another segment
                # either earlier or later
                wh_y, wh_x = np.where((slice_0 == 1) & (slice_1 > 0))

                if len(wh_y) == 0:
                    # Are none, so go to next pair of segments to check
                    continue

                # Remove the 1-group segment
                segs_beg_3[ii_0:-1, wh_y, wh_x] = segs_beg_3[ii_0 + 1 :, wh_y, wh_x]

                # Zero the last segment entry for the ramp, which would otherwise
                # remain non-zero due to the shift
                segs_beg_3[-1, wh_y, wh_x] = 0

                del wh_y, wh_x

                tot_num_single_grp_ramps = len(np.where((segs_beg_3 == 1) & (segs_beg_3.sum(axis=0) > 1))[0])

    return segs_beg_3


def do_all_sat(ramp_data, pixeldq, groupdq, imshape, n_int, save_opt):
    """
    For an input exposure where all groups in all integrations are saturated,
    the DQ in the primary and integration-specific output products are updated,
    and the other arrays in all output products are populated with zeros.

    Parameters
    ----------
    model : instance of Data Model
       DM object for input

    imshape : (int, int) tuple
       shape of 2D image

    n_int : int
       number of integrations

    save_opt : bool
       save optional fitting results

    Returns
    -------
    image_info : tuple
        The tuple of computed ramp fitting arrays.

    integ_info : tuple
        The tuple of computed integration fitting arrays.

    opt_info : tuple
        The tuple of computed optional results arrays for fitting.
    """
    # Create model for the primary output. Flag all pixels in the pixiel DQ
    #   extension as SATURATED and DO_NOT_USE.
    pixeldq = np.bitwise_or(pixeldq, ramp_data.flags_saturated)
    pixeldq = np.bitwise_or(pixeldq, ramp_data.flags_do_not_use)

    data = np.zeros(imshape, dtype=np.float32)
    dq = pixeldq
    var_poisson = np.zeros(imshape, dtype=np.float32)
    var_rnoise = np.zeros(imshape, dtype=np.float32)
    err = np.zeros(imshape, dtype=np.float32)
    image_info = (data, dq, var_poisson, var_rnoise, err)

    # Create model for the integration-specific output. The 3D group DQ created
    #   is based on the 4D group DQ of the model, and all pixels in all
    #   integrations will be flagged here as DO_NOT_USE (they are already flagged
    #   as SATURATED). The INT_TIMES extension will be left as None.
    if n_int > 1:
        m_sh = groupdq.shape  # (integ, grps/integ, y, x )
        groupdq_3d = np.zeros((m_sh[0], m_sh[2], m_sh[3]), dtype=np.uint32)

        for ii in range(n_int):  # add SAT flag to existing groupdq in each slice
            groupdq_3d[ii, :, :] = np.bitwise_or.reduce(groupdq[ii, :, :, :], axis=0)

        groupdq_3d = np.bitwise_or(groupdq_3d, ramp_data.flags_do_not_use)

        data = np.zeros((n_int, *imshape), dtype=np.float32)
        dq = groupdq_3d
        var_poisson = np.zeros((n_int, *imshape), dtype=np.float32)
        var_rnoise = np.zeros((n_int, *imshape), dtype=np.float32)
        err = np.zeros((n_int, *imshape), dtype=np.float32)

        integ_info = (data, dq, var_poisson, var_rnoise, err)
    else:
        integ_info = None

    # Create model for the optional output
    if save_opt:
        new_arr = np.zeros((n_int, 1, *imshape), dtype=np.float32)

        slope = new_arr
        sigslope = new_arr
        var_poisson = new_arr
        var_rnoise = new_arr
        yint = new_arr
        sigyint = new_arr
        pedestal = np.zeros((n_int, *imshape), dtype=np.float32)
        weights = new_arr
        crmag = new_arr

        opt_info = (slope, sigslope, var_poisson, var_rnoise, yint, sigyint, pedestal, weights, crmag)

    else:
        opt_info = None

    log.info("All groups of all integrations are saturated.")

    return image_info, integ_info, opt_info


def log_stats(c_rates):
    """
    Optionally log statistics of detected cosmic rays.

    Parameters
    ----------
    c_rates : ndarray
       weighted count rate, 2-D float

    Returns
    -------
    None
    """
    wh_c_0 = np.where(c_rates == 0.0)  # insuff data or no signal

    log.debug("The number of pixels having insufficient data")
    log.debug("due to excessive CRs or saturation %d:", len(wh_c_0[0]))
    log.debug(
        "Count rates - min, mean, max, std: %f, %f, %f, %f",
        c_rates.min(),
        c_rates.mean(),
        c_rates.max(),
        c_rates.std(),
    )


def compute_num_slices(max_cores, nrows, max_available):
    """
    Computes the number of slices to be created for multiprocessing.

    Parameters
    ----------
    max_cores : str
        Number of cores to use for multiprocessing. If set to 'none' (the default),
        then no multiprocessing will be done. The other allowable values are 'quarter',
        'half', and 'all' and string integers. This is the fraction of cores
        to use for multi-proc.
    nrows : int
        The number of rows that will be used across all process. This is the
        maximum number of slices to make sure that each process has some data.
    max_available: int
        This is the total number of cores available. The total number of cores
        includes the SMT cores (Hyper Threading for Intel).

    Returns
    -------
    number_slices : int
        The number of slices for multiprocessing.
    """
    number_slices = 1
    if max_cores.isnumeric():
        number_slices = int(max_cores)
    elif max_cores.lower() == "none" or max_cores.lower() == "one":
        number_slices = 1
    elif max_cores == "quarter":
        number_slices = max_available // 4 or 1
    elif max_cores == "half":
        number_slices = max_available // 2 or 1
    elif max_cores == "all":
        number_slices = max_available
    # Make sure we don't have more slices than rows or available cores.
    return min([nrows, number_slices, max_available])


def set_if_total_integ(final_dq, integ_dq, flag, set_flag):
    """
    Set set_flag in final_dq if flag is present in all integrations.

    Parameters
    ----------
    final_dq : ndarray
        2-D array (nrows, ncols) of the final pixel DQ.

    integ_dq : ndarray
        3-D array (nints, nrows, ncols) of the integration level DQ.

    flag : int
        Flag to check in each integration.

    set_flag : int
        Flag to set if flag is found in each integration.
    """
    nints = integ_dq.shape[0]

    # Find where flag is set
    test_dq = np.zeros(integ_dq.shape, dtype=np.uint32)
    test_dq[np.bitwise_and(integ_dq, flag).astype(bool)] = 1

    # Sum over all integrations
    test_sum = test_dq.sum(axis=0)
    all_set = np.where(test_sum == nints)

    # If flag is set in all integrations, then set the set_flag
    final_dq[all_set] = np.bitwise_or(final_dq[all_set], set_flag)


def set_if_total_ramp(pixeldq_sect, gdq_sect, flag, set_flag):
    """
    Set set_flag in final_dq if flag is present in all integrations.

    Parameters
    ----------
    pixeldq_sect: ndarray
        2-D array (nrows, ncols) of the integration DQ.

    gdq_dq : ndarray
        3-D array (ngroups, nrows, ncols) of the integration level DQ.

    flag : int
        Flag to check in each integration.

    set_flag : int
        Flag to set if flag is found in each integration.
    """
    # Checking for all groups is the same as checking for all integrations
    # because in both we are checking cubes.  For the integration check the
    # first dimension is the number of integrations, for the ramp check the
    # first dimension is the number of groups.
    set_if_total_integ(pixeldq_sect, gdq_sect, flag, set_flag)


def groups_saturated_in_integration(intdq, sat_flag, num_sat_groups):
    """
    Find the ramps in an integration that have num_sat_groups saturated.

    Parameters
    ----------
    intdq : ndarray
        DQ flags for an integration

    sat_flag : uint
        The data quality flag for SATURATED

    num_sat_groups : int
        The number of saturated groups in an integration of interest.
    """
    sat_groups = np.zeros(intdq.shape, dtype=int)
    sat_groups[np.bitwise_and(intdq, sat_flag).astype(bool)] = 1
    nsat_groups = sat_groups.sum(axis=0)
    return np.where(nsat_groups == num_sat_groups)
