#! /usr/bin/env python
#
# utils.py: utility functions
import logging
import multiprocessing
import numpy as np
import warnings


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Replace zero or negative variances with this:
LARGE_VARIANCE = 1.e8


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
        self.slope_seg = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32)
        if save_opt:
            self.yint_seg = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32)
            self.sigyint_seg = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32)
            self.sigslope_seg = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32)
            self.inv_var_seg = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32)
            self.firstf_int = np.zeros((n_int,) + imshape, dtype=np.float32)
            self.ped_int = np.zeros((n_int,) + imshape, dtype=np.float32)
            self.cr_mag_seg = np.zeros((n_int,) + (nreads,) + imshape, dtype=np.float32)
            self.var_p_seg = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32)
            self.var_r_seg = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32)

    def init_2d(self, npix, max_seg, save_opt):
        """
        Initialize the 2D segment-specific attributes for the current data
        section.

        Parameters
        ----------
        npix : integer
            number of pixels in section of 2D array

        max_seg : integer
            maximum number of segments that will be fit within an
            integration, calculated over all pixels and all integrations

        save_opt : bool
            save optional fitting results

        Returns
        -------
        None
        """
        self.slope_2d = np.zeros((max_seg, npix), dtype=np.float32)
        if save_opt:
            self.interc_2d = np.zeros((max_seg, npix), dtype=np.float32)
            self.siginterc_2d = np.zeros((max_seg, npix), dtype=np.float32)
            self.sigslope_2d = np.zeros((max_seg, npix), dtype=np.float32)
            self.inv_var_2d = np.zeros((max_seg, npix), dtype=np.float32)
            self.firstf_2d = np.zeros((max_seg, npix), dtype=np.float32)
            self.var_s_2d = np.zeros((max_seg, npix), dtype=np.float32)
            self.var_r_2d = np.zeros((max_seg, npix), dtype=np.float32)

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

        Returns
        -------
        """
        for ii_seg in range(0, self.slope_seg.shape[1]):
            self.slope_seg[num_int, ii_seg, rlo:rhi, :] = \
                self.slope_2d[ii_seg, :].reshape(sect_shape)

            if save_opt:
                self.yint_seg[num_int, ii_seg, rlo:rhi, :] = \
                    self.interc_2d[ii_seg, :].reshape(sect_shape)
                self.slope_seg[num_int, ii_seg, rlo:rhi, :] = \
                    self.slope_2d[ii_seg, :].reshape(sect_shape)
                self.sigyint_seg[num_int, ii_seg, rlo:rhi, :] = \
                    self.siginterc_2d[ii_seg, :].reshape(sect_shape)
                self.sigslope_seg[num_int, ii_seg, rlo:rhi, :] = \
                    self.sigslope_2d[ii_seg, :].reshape(sect_shape)
                self.inv_var_seg[num_int, ii_seg, rlo:rhi, :] = \
                    self.inv_var_2d[ii_seg, :].reshape(sect_shape)
                self.firstf_int[num_int, rlo:rhi, :] = ff_sect

    def append_arr(self, num_seg, g_pix, intercept, slope, sig_intercept,
                   sig_slope, inv_var, save_opt):
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
        self.slope_2d[num_seg[g_pix], g_pix] = slope[g_pix]

        if save_opt:
            self.interc_2d[num_seg[g_pix], g_pix] = intercept[g_pix]
            self.siginterc_2d[num_seg[g_pix], g_pix] = sig_intercept[g_pix]
            self.sigslope_2d[num_seg[g_pix], g_pix] = sig_slope[g_pix]
            self.inv_var_2d[num_seg[g_pix], g_pix] = inv_var[g_pix]

    def shrink_crmag(self, n_int, dq_cube, imshape, nreads, jump_det):
        """
        Compress the 4D cosmic ray magnitude array for the current
        integration, removing all groups whose cr magnitude is 0 for
        pixels having at least one group with a non-zero magnitude. For
        every integration, the depth of the array is equal to the
        maximum number of cosmic rays flagged in all pixels in all
        integrations.  This routine currently involves a loop over all
        pixels having at least 1 group flagged; if this algorithm takes
        too long for datasets having an overabundance of cosmic rays,
        this routine will require further optimization.

        Parameters
        ----------
        n_int : int
            number of integrations in dataset

        dq_cube : ndarray
            input data quality array, 4-D flag

        imshape : tuple
            shape of a single input image

        nreads : int
            number of reads in an integration

        Returns
        ----------
        None

        """
        # Loop over data integrations to find max num of crs flagged per pixel
        # (this could exceed the maximum number of segments fit)
        max_cr = 0
        for ii_int in range(0, n_int):
            dq_int = dq_cube[ii_int, :, :, :]
            dq_cr = np.bitwise_and(jump_det, dq_int)
            max_cr_int = (dq_cr > 0.).sum(axis=0).max()
            max_cr = max(max_cr, max_cr_int)

        # Allocate compressed array based on max number of crs
        cr_com = np.zeros((n_int,) + (max_cr,) + imshape, dtype=np.float32)

        # Loop over integrations and groups: for those pix having a cr, add
        #    the magnitude to the compressed array
        for ii_int in range(0, n_int):
            cr_mag_int = self.cr_mag_seg[ii_int, :, :, :]
            cr_int_has_cr = np.where(cr_mag_int.sum(axis=0) != 0)

            # Initialize number of crs for each image pixel for this integration
            end_cr = np.zeros(imshape, dtype=np.int16)

            for k_rd in range(nreads):
                # loop over pixels having a CR
                for nn in range(len(cr_int_has_cr[0])):
                    y, x = cr_int_has_cr[0][nn], cr_int_has_cr[1][nn]

                    if cr_mag_int[k_rd, y, x] > 0.:
                        cr_com[ii_int, end_cr[y, x], y, x] = cr_mag_int[k_rd, y, x]
                        end_cr[y, x] += 1

        max_num_crs = end_cr.max()
        if max_num_crs == 0:
            max_num_crs = 1
            self.cr_mag_seg = np.zeros(shape=(n_int, 1, imshape[0], imshape[1]))
        else:
            self.cr_mag_seg = cr_com[:, :max_num_crs, :, :]

    def output_optional(self, effintim):
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
        effintim : float
            effective integration time for a single group

        Returns
        -------
        opt_info : tuple
            The tuple of computed optional results arrays for fitting.
        """
        self.var_p_seg[self.var_p_seg > 0.4 * LARGE_VARIANCE] = 0.
        self.var_r_seg[self.var_r_seg > 0.4 * LARGE_VARIANCE] = 0.

        # Suppress, then re-enable, arithmetic warnings
        warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
        warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)

        # Tiny 'weights' values correspond to non-existent segments, so set to 0.
        self.weights[1. / self.weights > 0.4 * LARGE_VARIANCE] = 0.
        warnings.resetwarnings()

        self.slope_seg /= effintim

        opt_info = (self.slope_seg, self.sigslope_seg, self.var_p_seg,
                    self.var_r_seg, self.yint_seg, self.sigyint_seg,
                    self.ped_int, self.weights, self.cr_mag_seg)

        return opt_info

    def print_full(self):  # pragma: no cover
        """
        Diagnostic function for printing optional output arrays; most
        useful for tiny datasets

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print('Will now print all optional output arrays - ')
        print(' yint_seg: ')
        print((self.yint_seg))
        print('  ')
        print(' slope_seg: ')
        print(self.slope_seg)
        print('  ')
        print(' sigyint_seg: ')
        print(self.sigyint_seg)
        print('  ')
        print(' sigslope_seg: ')
        print(self.sigslope_seg)
        print('  ')
        print(' inv_var_2d: ')
        print((self.inv_var_2d))
        print('  ')
        print(' firstf_int: ')
        print((self.firstf_int))
        print('  ')
        print(' ped_int: ')
        print((self.ped_int))
        print('  ')
        print(' cr_mag_seg: ')
        print((self.cr_mag_seg))


def alloc_arrays_1(n_int, imshape):
    """
    Allocate arrays for integration-specific results and segment-specific
    results and variances.

    Parameters
    ----------
    n_int : int
        number of integrations

    imshape : tuple
        shape of a single image

    Returns
    -------
    dq_int : ndarray
        Cube of integration-specific group data quality values, 3-D flag

    median_diffs_2d : ndarray
        Estimated median slopes, 2-D float

    num_seg_per_int : ndarray
        Cube of numbers of segments for all integrations and pixels, 3-D int

    sat_0th_group_int : ndarray
        Integration-specific slice whose value for a pixel is 1 if the initial
        group of the ramp is saturated, 3-D uint8
    """
    dq_int = np.zeros((n_int,) + imshape, dtype=np.uint32)
    num_seg_per_int = np.zeros((n_int,) + imshape, dtype=np.uint8)

    # for estimated median slopes
    sat_0th_group_int = np.zeros((n_int,) + imshape, dtype=np.uint8)

    return dq_int, num_seg_per_int, sat_0th_group_int


def alloc_arrays_2(n_int, imshape, max_seg):
    """
    Allocate arrays for integration-specific results and segment-specific
    results and variances.

    Parameters
    ----------
    n_int : int
        number of integrations

    imshape : tuple
        shape of a single image

    max_seg : int
        maximum number of segments fit

    Returns
    -------
    var_p3 : ndarray
        Cube of integration-specific values for the slope variance due to
        Poisson noise only, 3-D float

    var_r3 : ndarray
        Cube of integration-specific values for the slope variance due to
        readnoise only, 3-D float

    var_p4 : ndarray
        Hypercube of segment- and integration-specific values for the slope
        variance due to Poisson noise only, 4-D float

    var_r4 : ndarray
        Hypercube of segment- and integration-specific values for the slope
        variance due to read noise only, 4-D float

    var_both4 : ndarray
        Hypercube of segment- and integration-specific values for the slope
        variance due to combined Poisson noise and read noise, 4-D float

    var_both3 : ndarray
        Cube of segment- and integration-specific values for the slope
        variance due to combined Poisson noise and read noise, 3-D float

    inv_var_both4 : ndarray
        Hypercube of reciprocals of segment- and integration-specific values for
        the slope variance due to combined Poisson noise and read noise, 4-D float

    s_inv_var_p3 : ndarray
        Cube of reciprocals of segment- and integration-specific values for
        the slope variance due to Poisson noise only, summed over segments, 3-D float

    s_inv_var_r3 : ndarray
        Cube of reciprocals of segment- and integration-specific values for
        the slope variance due to read noise only, summed over segments, 3-D float

    s_inv_var_both3 : ndarray
        Cube of reciprocals of segment- and integration-specific values for
        the slope variance due to combined Poisson noise and read noise, summed
        over segments, 3-D float

    segs_4 : ndarray
        Hypercube of lengths of segments for all integrations and pixels, 4-D
        int
    """
    # Initialize variances so that non-existing ramps and segments will have
    #   negligible contributions
    # Integration-specific:
    var_p3 = np.zeros((n_int,) + imshape, dtype=np.float32) + LARGE_VARIANCE
    var_r3 = var_p3.copy()
    var_both3 = var_p3.copy()
    s_inv_var_p3 = np.zeros_like(var_p3)
    s_inv_var_r3 = np.zeros_like(var_p3)
    s_inv_var_both3 = np.zeros_like(var_p3)

    # Segment-specific:
    var_p4 = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32) + LARGE_VARIANCE
    var_r4 = var_p4.copy()
    var_both4 = var_p4.copy()
    inv_var_both4 = np.zeros_like(var_p4)

    # number of segments
    segs_4 = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.uint8)

    return (var_p3, var_r3, var_p4, var_r4, var_both4, var_both3,
            inv_var_both4, s_inv_var_p3, s_inv_var_r3,
            s_inv_var_both3, segs_4)


def calc_slope_vars(ramp_data, rn_sect, gain_sect, gdq_sect, group_time, max_seg):
    """
    Calculate the segment-specific variance arrays for the given
    integration.

    Parameters
    ----------
    ramp_data : ramp_fit_class.RampData
        Contains the DQ flags needed for this function

    rn_sect : ndarray
        read noise values for all pixels in data section, 2-D float

    gain_sect : ndarray
        gain values for all pixels in data section, 2-D float

    gdq_sect : ndarray
        data quality flags for pixels in section, 3-D int

    group_time : float
        Time increment between groups, in seconds.

    max_seg : int
        maximum number of segments fit

    Returns
    -------
    den_r3 : ndarray
        for a given integration, the reciprocal of the denominator of the
        segment-specific variance of the segment's slope due to read noise, 3-D float

    den_p3 : ndarray
        for a given integration, the reciprocal of the denominator of the
        segment-specific variance of the segment's slope due to Poisson noise, 3-D float

    num_r3 : ndarray
        numerator of the segment-specific variance of the segment's slope
        due to read noise, 3-D float

    segs_beg_3 : ndarray
        lengths of segments for all pixels in the given data section and
        integration, 3-D int
    """
    (nreads, asize2, asize1) = gdq_sect.shape
    npix = asize1 * asize2
    imshape = (asize2, asize1)

    # Create integration-specific sections of input arrays for determination
    #   of the variances.
    gdq_2d = gdq_sect[:, :, :].reshape((nreads, npix))
    gain_1d = gain_sect.reshape(npix)
    gdq_2d_nan = gdq_2d.copy()  # group dq with SATS will be replaced by nans
    gdq_2d_nan = gdq_2d_nan.astype(np.float32)

    wh_sat = np.where(np.bitwise_and(gdq_2d, ramp_data.flags_saturated))
    if len(wh_sat[0]) > 0:
        gdq_2d_nan[wh_sat] = np.nan  # set all SAT groups to nan

    del wh_sat

    # Get lengths of semiramps for all pix [number_of_semiramps, number_of_pix]
    segs = np.zeros_like(gdq_2d)

    # Counter of semiramp for each pixel
    sr_index = np.zeros(npix, dtype=np.uint8)
    pix_not_done = np.ones(npix, dtype=bool)  # initialize to True

    i_read = 0
    # Loop over reads for all pixels to get segments (segments per pixel)
    while i_read < nreads and np.any(pix_not_done):
        gdq_1d = gdq_2d_nan[i_read, :]
        wh_good = np.where(gdq_1d == 0)  # good groups

        # if this group is good, increment those pixels' segments' lengths
        if len(wh_good[0]) > 0:
            segs[sr_index[wh_good], wh_good] += 1
        del wh_good

        # Locate any CRs that appear before the first SAT group...
        wh_cr = np.where(
            gdq_2d_nan[i_read, :].astype(np.int32) & ramp_data.flags_jump_det > 0)

        # ... but not on final read:
        if len(wh_cr[0]) > 0 and (i_read < nreads - 1):
            sr_index[wh_cr[0]] += 1
            segs[sr_index[wh_cr], wh_cr] += 1

        del wh_cr

        # If current group is a NaN, this pixel is done (pix_not_done is False)
        wh_nan = np.where(np.isnan(gdq_2d_nan[i_read, :]))
        if len(wh_nan[0]) > 0:
            pix_not_done[wh_nan[0]] = False

        del wh_nan

        i_read += 1

    segs = segs.astype(np.uint8)
    segs_beg = segs[:max_seg, :]  # the leading nonzero lengths

    # Create reshaped version [ segs, y, x ] to simplify computation
    segs_beg_3 = segs_beg.reshape(max_seg, imshape[0], imshape[1])
    segs_beg_3 = remove_bad_singles(segs_beg_3)

    # Create a version 1 less for later calculations for the variance due to
    #   Poisson, with a floor=1 to handle single-group segments
    wh_pos_3 = np.where(segs_beg_3 > 1)
    segs_beg_3_m1 = segs_beg_3.copy()
    segs_beg_3_m1[wh_pos_3] -= 1
    segs_beg_3_m1[segs_beg_3_m1 < 1] = 1

    # For a segment, the variance due to Poisson noise
    #   = slope/(tgroup * gain * (ngroups-1)),
    #   where slope is the estimated median slope, tgroup is the group time,
    #   and ngroups is the number of groups in the segment.
    #   Here the denominator of this quantity will be computed, which will be
    #   later multiplied by the estimated median slope.

    # Suppress, then re-enable, harmless arithmetic warnings, as NaN will be
    #   checked for and handled later
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    den_p3 = 1. / (group_time * gain_1d.reshape(imshape) * segs_beg_3_m1)
    if ramp_data.zframe_locs:
        for pix in ramp_data.zframe_locs[ramp_data.current_integ]:
            frame_time = ramp_data.frame_time
            row, col = pix
            den_p3[0, row, col] = 1. / (frame_time * gain_sect[row, col])
    warnings.resetwarnings()

    # For a segment, the variance due to readnoise noise
    # = 12 * readnoise**2 /(ngroups_seg**3. - ngroups_seg)/( tgroup **2.)
    num_r3 = 12. * (rn_sect / group_time)**2.  # always >0
    if ramp_data.zframe_locs:
        for pix in ramp_data.zframe_locs[ramp_data.current_integ]:
            row, col = pix
            num_r3[row, col] = 12. * (rn_sect[row, col] / frame_time)**2.

    # Reshape for every group, every pixel in section
    num_r3 = np.dstack([num_r3] * max_seg)
    num_r3 = np.transpose(num_r3, (2, 0, 1))

    # Denominator den_r3 = 1./(segs_beg_3 **3.-segs_beg_3). The minimum number
    #   of allowed groups is 2, which will apply if there is actually only 1
    #   group; in this case den_r3 = 1/6. This covers the case in which there is
    #   only one good group at the beginning of the integration, so it will be
    #   be compared to the plane of (near) zeros resulting from the reset. For
    #   longer segments, this value is overwritten below.
    den_r3 = num_r3.copy() * 0. + 1. / 6
    wh_seg_pos = np.where(segs_beg_3 > 1)

    # Suppress, then, re-enable harmless arithmetic warnings, as NaN will be
    #   checked for and handled later
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    # overwrite where segs>1
    den_r3[wh_seg_pos] = 1. / (segs_beg_3[wh_seg_pos] ** 3. - segs_beg_3[wh_seg_pos])
    warnings.resetwarnings()

    return den_r3, den_p3, num_r3, segs_beg_3


def calc_pedestal(ramp_data, num_int, slope_int, firstf_int, dq_first, nframes,
                  groupgap, dropframes1):
    """
    The pedestal is calculated by extrapolating the final slope for each pixel
    from its value at the first sample in the integration to an exposure time
    of zero; this calculation accounts for the values of nframes and groupgap.
    Any pixel that is saturated on the 1st group is given a pedestal value of 0.

    Parameters
    ----------
    num_int : int
        integration number

    slope_int : ndarray
        cube of integration-specific slopes, 3-D float

    firstf_int : ndarray
        integration-specific first frame array, 3-D float

    dq_first : ndarray
        DQ of the initial group for all ramps in the given integration, 2-D
        flag

    nframes : int
        number of frames averaged per group; from the NFRAMES keyword. Does
        not contain the groupgap.

    groupgap : int
        number of frames dropped between groups, from the GROUPGAP keyword.

    dropframes1 : int
        number of frames dropped at the beginning of every integration, from
        the DRPFRMS1 keyword.

    Returns
    -------
    ped : ndarray
        pedestal image, 2-D float
    """
    ff_all = firstf_int[num_int, :, :].astype(np.float32)
    ped = ff_all - slope_int[num_int, ::] * \
        (((nframes + 1.) / 2. + dropframes1) / (nframes + groupgap))

    sat_flag = ramp_data.flags_saturated
    ped[np.bitwise_and(dq_first, sat_flag) == sat_flag] = 0
    ped[np.isnan(ped)] = 0.

    return ped


def output_integ(slope_int, dq_int, effintim, var_p3, var_r3, var_both3):
    """
    For the OLS algorithm, construct the output integration-specific results.
    Any variance values that are a large fraction of the default value
    LARGE_VARIANCE correspond to non-existent segments, so will be set to 0
    here before output.

    Parameters
    ----------
    model : instance of Data Model
       DM object for input

    slope_int : ndarray
       Data cube of weighted slopes for each integration, 3-D float

    dq_int : ndarray
       Data cube of DQ arrays for each integration, 3-D int

    effintim : float
       Effective integration time per integration

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

    var_p3[var_p3 > 0.4 * LARGE_VARIANCE] = 0.
    var_r3[var_r3 > 0.4 * LARGE_VARIANCE] = 0.
    var_both3[var_both3 > 0.4 * LARGE_VARIANCE] = 0.

    data = slope_int / effintim
    err = np.sqrt(var_both3)
    dq = dq_int
    var_poisson = var_p3
    var_rnoise = var_r3
    integ_info = (data, dq, var_poisson, var_rnoise, err)

    # Reset the warnings filter to its original state
    warnings.resetwarnings()

    return integ_info


def gls_pedestal(first_group, slope_int, s_mask,
                 frame_time, nframes_used):  # pragma: no cover
    """
    Calculate the pedestal for the GLS case.

    The pedestal is the first group, but extrapolated back to zero time
    using the slope obtained by the fit to the whole ramp.  The time of
    the first group is the frame time multiplied by (M + 1) / 2, where M
    is the number of frames per group, not including the number (if any)
    of skipped frames.

    The input arrays and output pedestal are slices of the full arrays.
    They are just the relevant data for the current integration (assuming
    that this function is called within a loop over integrations), and
    they may include only a subset of image lines.  For example, this
    function might be called with slope_int argument given as:
    `slope_int[num_int, rlo:rhi, :]`.  The input and output parameters
    are in electrons.

    Parameters
    ----------
    first_group : ndarray
        A slice of the first group in the ramp, 2-D float

    slope_int : ndarray
        The slope obtained by GLS ramp fitting.  This is a slice for the
        current integration and a subset of the image lines, 2-D float

    s_mask : ndarray
        True for ramps that were saturated in the first group, 2-D bool

    frame_time : float
        The time to read one frame, in seconds.

    nframes_used : int
        Number of frames that were averaged together to make a group.
        Exludes the groupgap.

    Returns
    -------
    pedestal : ndarray
        This is a slice of the full pedestal array, and it's for the
        current integration, 2-D float
    """
    M = float(nframes_used)
    pedestal = first_group - slope_int * frame_time * (M + 1.) / 2.
    if s_mask.any():
        pedestal[s_mask] = 0.

    return pedestal


def shift_z(a, off):
    """
    Shift input 3D array by requested offset in z-direction, padding shifted
    array (of the same size) by leading or trailing zeros as needed.

    Parameters
    ----------
    a : ndarray
        input array, 3-D float

    off : int
        offset in z-direction

    Returns
    -------
    b : ndarray
        shifted array, 3-D float

    """
    # set initial and final indices along z-direction for original and
    #    shifted 3D arrays
    ai_z = int((abs(off) + off) / 2)
    af_z = a.shape[0] + int((-abs(off) + off) / 2)

    bi_z = a.shape[0] - af_z
    bf_z = a.shape[0] - ai_z

    b = a * 0
    b[bi_z:bf_z, :, :] = a[ai_z:af_z, :, :]

    return b


def get_efftim_ped(ramp_data):
    """
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

    if dropframes1 is None:    # set to default if missing
        dropframes1 = 0
        log.debug('Missing keyword DRPFRMS1, so setting to default value of 0')

    try:
        effintim = (nframes + groupgap) * frame_time
    except TypeError:
        log.error('Can not retrieve values needed to calculate integ. time')

    log.debug('Calculating effective integration time for a single group using:')
    log.debug(' groupgap: %s' % (groupgap))
    log.debug(' nframes: %s' % (nframes))
    log.debug(' frame_time: %s' % (frame_time))
    log.debug(' dropframes1: %s' % (dropframes1))
    log.info('Effective integration time per group: %s' % (effintim))

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
    cubeshape = (nreads,) + imshape

    return (nreads, npix, imshape, cubeshape, n_int, instrume,
            frame_time, ngroups, group_time)


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


def reset_bad_gain(ramp_data, pdq, gain):
    """
    For pixels in the gain array that are either non-positive or NaN, reset the
    the corresponding pixels in the pixel DQ array to NO_GAIN_VALUE and
    DO_NOT_USE so that they will be ignored.

    Parameters
    ----------
    pdq : ndarray
        pixel dq array of input model, 2-D int

    gain : ndarray
        gain array from reference file, 2-D float

    Returns
    -------
    pdq : ndarray
        pixleldq array of input model, reset to NO_GAIN_VALUE and DO_NOT_USE
        for pixels in the gain array that are either non-positive or NaN., 2-D
        flag
    """
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value.*", RuntimeWarning)
        wh_g = np.where(gain <= 0.)
    '''
    wh_g = np.where(gain <= 0.)
    if len(wh_g[0]) > 0:
        pdq[wh_g] = np.bitwise_or(pdq[wh_g], ramp_data.flags_no_gain_val)
        pdq[wh_g] = np.bitwise_or(pdq[wh_g], ramp_data.flags_do_not_use)

    wh_g = np.where(np.isnan(gain))
    if len(wh_g[0]) > 0:
        pdq[wh_g] = np.bitwise_or(pdq[wh_g], ramp_data.flags_no_gain_val)
        pdq[wh_g] = np.bitwise_or(pdq[wh_g], ramp_data.flags_do_not_use)

    return pdq


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
    tot_num_single_grp_ramps = len(np.where((segs_beg_3 == 1) &
                                            (segs_beg_3.sum(axis=0) > 1))[0])

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
                segs_beg_3[ii_0:-1, wh_y, wh_x] = segs_beg_3[ii_0 + 1:, wh_y, wh_x]

                # Zero the last segment entry for the ramp, which would otherwise
                # remain non-zero due to the shift
                segs_beg_3[-1, wh_y, wh_x] = 0

                del wh_y, wh_x

                tot_num_single_grp_ramps = len(np.where((segs_beg_3 == 1) &
                                                        (segs_beg_3.sum(axis=0) > 1))[0])

    return segs_beg_3


def fix_sat_ramps(ramp_data, sat_0th_group_int, var_p3, var_both3, slope_int, dq_int):
    """
    For ramps within an integration that are saturated on the initial group,
    reset the integration-specific variances and slope so they will have no
    contribution.

    Parameters
    ----------
    sat_0th_group_int : ndarray
        Integration-specific slice whose value for a pixel is 1 if the initial
        group of the ramp is saturated, 3-D uint8

    var_p3 : ndarray
        Cube of integration-specific values for the slope variance due to
        Poisson noise only; some ramps may be saturated in the initial group,
        3-D float

    var_both3 : ndarray
        Cube of segment- and integration-specific values for the slope
        variance due to combined Poisson noise and read noise; some ramps may
        be saturated in the initial group, 3-D float

    slope_int : ndarray
        Cube of integration-specific slopes. Some ramps may be saturated in the
        initial group, 3-D float

    dq_int : ndarray
        Cube of integration-specific DQ flags, 3-D flag

    Returns
    -------
    var_p3 : ndarray
        Cube of integration-specific values for the slope variance due to
        Poisson noise only; for ramps that are saturated in the initial group,
        this variance has been reset to a huge value to minimize the ramps
        contribution, 3-D float

    var_both3 : ndarray
        Cube of segment- and integration-specific values for the slope
        variance due to combined Poisson noise and read noise; for ramps that
        are saturated in the initial group, this variance has been reset to a
        huge value to minimize the ramps contribution, 3-D float

    slope_int : ndarray
        Cube of integration-specific slopes; for ramps that are saturated in
        the initial group, this variance has been reset to a huge value to
        minimize the ramps contribution, 3-D float

    dq_int : ndarray
        Cube of integration-specific DQ flags. For ramps that are saturated in
        the initial group, the flag 'DO_NOT_USE' is added, 3-D flag
    """
    var_p3[sat_0th_group_int > 0] = LARGE_VARIANCE
    var_both3[sat_0th_group_int > 0] = LARGE_VARIANCE
    slope_int[sat_0th_group_int > 0] = 0.
    dq_int[sat_0th_group_int > 0] = np.bitwise_or(
        dq_int[sat_0th_group_int > 0], ramp_data.flags_do_not_use)

    return var_p3, var_both3, slope_int, dq_int


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
            groupdq_3d[ii, :, :] = np.bitwise_or.reduce(groupdq[ii, :, :, :],
                                                        axis=0)

        groupdq_3d = np.bitwise_or(groupdq_3d, ramp_data.flags_do_not_use)

        data = np.zeros((n_int,) + imshape, dtype=np.float32)
        dq = groupdq_3d
        var_poisson = np.zeros((n_int,) + imshape, dtype=np.float32)
        var_rnoise = np.zeros((n_int,) + imshape, dtype=np.float32)
        err = np.zeros((n_int,) + imshape, dtype=np.float32)

        integ_info = (data, dq, var_poisson, var_rnoise, err)
    else:
        integ_info = None

    # Create model for the optional output
    if save_opt:
        new_arr = np.zeros((n_int,) + (1,) + imshape, dtype=np.float32)

        slope = new_arr
        sigslope = new_arr
        var_poisson = new_arr
        var_rnoise = new_arr
        yint = new_arr
        sigyint = new_arr
        pedestal = np.zeros((n_int,) + imshape, dtype=np.float32)
        weights = new_arr
        crmag = new_arr

        opt_info = (slope, sigslope, var_poisson, var_rnoise,
                    yint, sigyint, pedestal, weights, crmag)

    else:
        opt_info = None

    log.info('All groups of all integrations are saturated.')

    return image_info, integ_info, opt_info


def log_stats(c_rates):
    """
    Optionally log statistics of detected cosmic rays

    Parameters
    ----------
    c_rates : ndarray
       weighted count rate, 2-D float

    Returns
    -------
    None
    """
    wh_c_0 = np.where(c_rates == 0.)  # insuff data or no signal

    log.debug('The number of pixels having insufficient data')
    log.debug('due to excessive CRs or saturation %d:', len(wh_c_0[0]))
    log.debug('Count rates - min, mean, max, std: %f, %f, %f, %f'
              % (c_rates.min(), c_rates.mean(), c_rates.max(), c_rates.std()))


def compute_slices(max_cores):
    """
    Computes the number of slices to be created for multiprocessing.

    Parameters
    ----------
    max_cores : str
        Number of cores to use for multiprocessing. If set to 'none' (the default),
        then no multiprocessing will be done. The other allowable values are 'quarter',
        'half', and 'all'. This is the fraction of cores to use for multi-proc. The
        total number of cores includes the SMT cores (Hyper Threading for Intel).

    Returns
    -------
    number_slices : int
        The number of slices for multiprocessing.
    """
    if max_cores == 'none':
        number_slices = 1
    else:
        num_cores = multiprocessing.cpu_count()
        log.debug(f'Found {num_cores} possible cores to use for ramp fitting')
        if max_cores == 'quarter':
            number_slices = num_cores // 4 or 1
        elif max_cores == 'half':
            number_slices = num_cores // 2 or 1
        elif max_cores == 'all':
            number_slices = num_cores
        else:
            number_slices = 1
    return number_slices


def dq_compress_final(dq_int, ramp_data):
    """
    From the integration level DQ flags, set the final pixel DQ flags.

    Parameters
    ----------
    dq_int : ndarray
        The integration level DQ flags, 3-D (nints, nrows, ncols).

    ramp_data : RampData
        Contains the DQ flag information.

    Return
    ------
    final_dq : ndarray
        The final 2-D (nrows, ncols) pixel DQ array.
    """
    final_dq = dq_int[0, :, :]
    nints = dq_int.shape[0]
    for integ in range(1, nints):
        final_dq = np.bitwise_or(final_dq, dq_int[integ, :, :])

    dnu, sat = ramp_data.flags_do_not_use, ramp_data.flags_saturated

    # Remove DO_NOT_USE and SATURATED because they need special handling.
    # These flags are not set in the final pixel DQ array by simply being set
    # in one of the integrations.
    not_sat_or_dnu = np.uint32(~(dnu | sat))
    final_dq = np.bitwise_and(final_dq, not_sat_or_dnu)

    # If all integrations are DO_NOT_USE or SATURATED, then set DO_NOT_USE.
    set_if_total_integ(final_dq, dq_int, dnu | sat, dnu)

    # If all integrations have SATURATED, then set DO_NOT_USE and SATURATED.
    set_if_total_integ(final_dq, dq_int, sat, dnu | sat)

    return final_dq


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
    test_dq[np.where(np.bitwise_and(integ_dq, flag))] = 1

    # Sum over all integrations
    test_sum = test_dq.sum(axis=0)
    all_set = np.where(test_sum == nints)

    # If flag is set in all integrations, then set the set_flag
    final_dq[all_set] = np.bitwise_or(final_dq[all_set], set_flag)


def dq_compress_sect(ramp_data, num_int, gdq_sect, pixeldq_sect):
    """
    This sets the integration level flags for DO_NOT_USE, JUMP_DET and
    SATURATED.  If any ramp has a jump, this flag will be set for the
    integration.  If all groups in a ramp are flagged as DO_NOT_USE, then the
    integration level DO_NOT_USE flag will be set.  If a ramp is saturated in
    group 0, then the integration level flag is marked as SATURATED.  Also, if
    all groups are marked as DO_NOT_USE or SATURATED (as in suppressed one
    groups), then the DO_NOT_USE flag is set.

    Parameters
    ----------
    ramp_data : RampData
        Contains the DQ flag information.

    num_int : int
        The current integration number.

    gdq_sect : ndarray
        The current 3-D (ngroups, nrows, ncols) integration DQ array.

    pixeldq_sect : ndarray
        The 2-D (nrows, ncols) pixel DQ flags for the current integration.

    Return
    ------
    pixeldq_sect : ndarray
        The 2-D (nrows, ncols) pixel DQ flags for the current integration.
    """
    sat = ramp_data.flags_saturated
    jump = ramp_data.flags_jump_det
    dnu = ramp_data.flags_do_not_use
    ngroups, nrows, ncols = gdq_sect.shape

    # Check total SATURATED or DO_NOT_USE
    set_if_total_ramp(pixeldq_sect, gdq_sect, sat | dnu, dnu)

    # Assume total saturation if group 0 is SATURATED.
    gdq0_sat = np.bitwise_and(gdq_sect[0], sat)
    pixeldq_sect[gdq0_sat != 0] = np.bitwise_or(
        pixeldq_sect[gdq0_sat != 0], sat | dnu)

    # If jump occurs mark the appropriate flag.
    jump_loc = np.bitwise_and(gdq_sect, jump)
    jump_check = np.where(jump_loc.sum(axis=0) > 0)
    pixeldq_sect[jump_check] = np.bitwise_or(pixeldq_sect[jump_check], jump)

    return pixeldq_sect


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


def compute_median_rates(ramp_data):
    """
    Compute the median rates for each pixel.

    Parameter
    ---------
    ramp_data : RampData
        Contains the data needed to compute the median rates.

    Return
    ------
    median_rates : ndarray
        The computed median rates for each pixel.
    """
    nints, ngroups, nrows, ncols = ramp_data.data.shape
    imshape = (nrows, ncols)

    group_time = ramp_data.group_time
    frame_time = ramp_data.frame_time
    adjustment = group_time / frame_time
    median_diffs_2d = np.zeros(imshape, dtype=np.float32)

    for integ in range(nints):
        data_sect = ramp_data.data[integ, :, :, :].copy()
        gdq_sect = ramp_data.groupdq[integ, :, :, :]

        # Reset all saturated groups in the input data array to NaN
        where_sat = np.where(np.bitwise_and(gdq_sect, ramp_data.flags_saturated))

        data_sect[where_sat] = np.NaN
        del where_sat

        data_sect = data_sect / group_time
        if ramp_data.zframe_locs is not None:
            for pixel in ramp_data.zframe_locs[integ]:
                row, col = pixel
                # This makes the division by frame_time, instead of group_time
                data_sect[0, row, col] = data_sect[0, row, col] * adjustment

        # Compute the first differences of all groups
        first_diffs_sect = np.diff(data_sect, axis=0)

        # If the dataset has only 1 group/integ, assume the 'previous group'
        #   is all zeros, so just use data as the difference
        if first_diffs_sect.shape[0] == 0:
            first_diffs_sect = data_sect.copy()
        else:
            # Similarly, for datasets having >1 group/integ and having
            #   single-group segments, just use the data as the difference
            wh_nan = np.where(np.isnan(first_diffs_sect[0, :, :]))

            if len(wh_nan[0]) > 0:
                first_diffs_sect[0, :, :][wh_nan] = data_sect[0, :, :][wh_nan]

            del wh_nan

            # Mask all the first differences that are affected by a CR,
            #   starting at group 1.  The purpose of starting at index 1 is
            #   to shift all the indices down by 1, so they line up with the
            #   indices in first_diffs.
            i_group, i_yy, i_xx, = np.where(np.bitwise_and(
                gdq_sect[1:, :, :], ramp_data.flags_jump_det))
            first_diffs_sect[i_group, i_yy, i_xx] = np.NaN

            del i_group, i_yy, i_xx

            # Check for pixels in which there is good data in 0th group, but
            #   all first_diffs for this ramp are NaN because there are too
            #   few good groups past the 0th. Due to the shortage of good
            #   data, the first_diffs will be set here equal to the data in
            #   the 0th group.
            wh_min = np.where(np.logical_and(
                np.isnan(first_diffs_sect).all(axis=0), np.isfinite(data_sect[0, :, :])))
            if len(wh_min[0] > 0):
                first_diffs_sect[0, :, :][wh_min] = data_sect[0, :, :][wh_min]

            del wh_min

        # All first differences affected by saturation and CRs have been set
        #  to NaN, so compute the median of all non-NaN first differences.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN.*", RuntimeWarning)
            nan_med = np.nanmedian(first_diffs_sect, axis=0)

        nan_med[np.isnan(nan_med)] = 0.  # if all first_diffs_sect are nans
        median_diffs_2d[:, :] += nan_med

    # Compute the final 2D array of differences; create rate array
    median_rates = median_diffs_2d / nints
    del median_diffs_2d
    del data_sect

    return median_rates


def use_zeroframe_for_saturated_ramps(ramp_data):
    """
    For saturated ramps, if there is good data in the ZEROFRAME, replace
    group zero with the data in ZEROFRAME to use the ramp as a one group
    ramp.

    Parameters
    ----------
    model : data model
        input data model, assumed to be of type RampModel

    Return
    ------
    zframe_locs : list
        A 2D list for the location of the ramps using ZEROFRAME data.
        zframe_locs[k] is the list of pixels in the kth integration.
    """
    nints, ngroups, nrows, ncols = ramp_data.data.shape
    sat_flag = ramp_data.flags_saturated
    good_flag = 0
    dq = ramp_data.groupdq

    zframe_locs = [None] * nints

    cnt = 0
    for integ in range(nints):
        zframe_locs[integ] = []
        intdq = dq[integ, :, :, :]

        # Find ramps with a good zeroeth group, but saturated in
        # the remainder of the ramp.
        wh_sat = groups_saturated_in_integration(intdq, sat_flag, ngroups)

        whs_rows = wh_sat[0]
        whs_cols = wh_sat[1]
        for n in range(len(whs_rows)):
            row = whs_rows[n]
            col = whs_cols[n]

            # For ramps completely saturated look for data in the ZEROFRAME
            # that is non-zero.  If it is non-zero, replace group zero in the
            # ramp with the data in ZEROFRAME.
            if ramp_data.zeroframe[integ, row, col] != 0:
                zframe_locs[integ].append((row, col))
                ramp_data.data[integ, 0, row, col] = ramp_data.zeroframe[integ, row, col]
                ramp_data.groupdq[integ, 0, row, col] = good_flag
                cnt = cnt + 1

    return zframe_locs, cnt


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
    sat_groups[np.where(np.bitwise_and(intdq, sat_flag))] = 1
    nsat_groups = sat_groups.sum(axis=0)
    wh_nsat_groups = np.where(nsat_groups == num_sat_groups)

    return wh_nsat_groups
