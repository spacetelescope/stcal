import numpy as np


def linearity_correction(
        data, gdq, pdq, lin_coeffs, lin_dq, dqflags, zframe=None):
    """
    Apply linearity correction to individual groups in `data` to pixels that
    haven't already been flagged as saturated.

    Pixels having at least one correction coefficient equal to NaN will not
    have the linearity correction applied and the DQ flag `NO_LIN_CORR` is
    added to the science exposure PIXELDQ array. Pixels that have the
    `NO_LIN_CORR` flag set in the DQ array of the linearity reference file will
    not have the correction applied and the “NO_LIN_CORR” flag is added to the
    science exposure `PIXELDQ` array. Pixel values that have the `SATURATED`
    flag set in a particular group of the science exposure GROUPDQ array will
    not have the linearity correction applied to that group. Any groups for
    that pixel that are not flagged as saturated will be corrected.

    The corrected data and updated `PIXELDQ` arrays are returned.
    Reference:
    http://www.stsci.edu/hst/wfc3/documents/handbooks/currentDHB/wfc3_Ch606.html
    WFC3 Data Handbook, Chapter 6, WFC3-IR Error Sources, 6.5 Detector
    Nonlinearity issues, 2.1 May 2011

    Parameters
    ----------
    data : ndarray
        The 4D array for the SCI data.

    gdq : ndarray
        The 4D group dq array.

    pdq : ndarray
        The 2D pixel dq array.

    lin_coeffs : ndarray.
        The 3D array containing pixel-by-pixel linearity coefficient
        values for each term in the polynomial fit.

    dq_flags : dict
        A dictionary with at least the following keywords:
        SATURATED, NO_LIN_CORR.

    zframe : ndarray or None
        The ZEROFRAME array with dimensions (nints, nrows, ncols) or None

    Returns
    -------
    data : ndarray
        The 4D linearly corrected SCI data.

    new_pdq : ndarray
        The updated 2D pixel data quality flags.

    zframe : ndarray or None
        If not None, the 3D linearly corrected ZEROFRAME.
    """
    if zframe is not None:
        # Save off data that gets transformed during linearity processing.
        zlin_coeffs = lin_coeffs.copy()
        zlin_dq = lin_dq.copy()

    # Do linear correction on SCI data
    data, new_pdq = linearity_correction_branch(
        data, gdq, pdq, lin_coeffs, lin_dq, dqflags, False)

    zdata = None  # zframe needs to be returned, so initialize it to None.
    if zframe is not None:
        # Do linear correction on ZEROFRAME
        # As ZEROFRAME gets processed through the pipeline, since it has no
        # corresponding DQ array, when data is found to be bad, the data is
        # set to ZERO.  Since zero ZEROFRAME values indicates bad data,
        # remember where this happens.  Make a dummy ZEROFRAME DQ array and
        # mark zeroed data as saturated.
        wh_zero = np.where(zframe[:, :, :] == 0.)
        zdq = np.zeros(zframe.shape, dtype=gdq.dtype)
        zdq[zframe == 0.] = dqflags["SATURATED"]
        zpdq = np.zeros(zframe.shape[-2:], dtype=pdq.dtype)

        # Linearly correct ZEROFRAME
        zdata, _ = linearity_correction_branch(
            zframe, zdq, zpdq, zlin_coeffs, zlin_dq, dqflags, True)

        # Ensure bad data remains bad.
        zdata[wh_zero] = 0.

    return data, new_pdq, zdata


def linearity_correction_branch(
        data, gdq, pdq, lin_coeffs, lin_dq, dqflags, zframe):
    """
    Parameters
    ----------
    data : `np.array`
        The data to be linearly corrected.

    gdq : `np.array`
        Group dq array.

    pdq : `np.array`
        Pixel dq array.

    lin_coeffs : `np.array`
        Array containing pixel-by-pixel linearity coefficient values
        for each term in the polynomial fit.

    dq_flags : dict
        A dictionary with at least the following keywords:
        SATURATED, NO_LIN_CORR.

    zframe : ndarray or None
        The ZEROFRAME array with dimensions (nints, nrows, ncols) or None

    Returns
    -------
    data : 3D array
        Updated array of correction coefficients in reference file.

    new_pdq : `np.array`
        Updated pixel data quality flags.
    """
    # Retrieve the ramp data cube characteristics
    if zframe:
        nints, nrows, ncols = data.shape
    else:
        nints, ngroups, nrows, ncols = data.shape

    # Number of coeffs is equal to the number of planes in coeff cube
    ncoeffs = lin_coeffs.shape[0]

    # Combine the DQ arrays using bitwise_or
    new_pdq = np.bitwise_or(pdq, lin_dq)

    # Check for NO_LIN_CORR flags in the DQ extension of the ref file
    lin_coeffs = correct_for_flag(lin_coeffs, lin_dq, dqflags)

    # Check for NaNs in the COEFFS extension of the ref file
    lin_coeffs, new_pdq = correct_for_NaN(lin_coeffs, new_pdq, dqflags)

    # Check when all the Linearity COEFFS = 0. Set DQ flag to NO_LIN_COEFF
    lin_coeffs, new_pdq = correct_for_zero(lin_coeffs, new_pdq, dqflags)

    # Apply the linearity correction one integration at a time.
    for ints in range(nints):
        if not zframe:
            # Apply the linearity correction one group at a time
            for plane in range(ngroups):
                dataplane = data[ints, plane]
                gdqplane = gdq[ints, plane]
                linear_correct_plane(
                    dataplane, gdqplane, lin_coeffs, ncoeffs, dqflags)

        else:
            # ZEROFRAME processing
            dataplane = data[ints]
            gdqplane = gdq[ints]
            linear_correct_plane(
                dataplane, gdqplane, lin_coeffs, ncoeffs, dqflags)

    return data, new_pdq


def linear_correct_plane(dataplane, gdqplane, lin_coeffs, ncoeffs, dqflags):
    """
    dataplane : ndarray
        The 2D array of the frame/group plane of pixels to linearly correct.

    gdqplane : ndarray
        The DQ flags for the plane.

    lin_coeffs : ndarray
        The linearity coefficients to apply to the pixels.

    ncoeffs : int
        The number of linearity coefficients.

    dqflags : dict
        The dictionary of DQ flags.
    """
    # Accumulate the polynomial terms into the corrected counts
    scorr = lin_coeffs[ncoeffs - 1] * dataplane
    for j in range(ncoeffs - 2, 0, -1):
        scorr = (scorr + lin_coeffs[j]) * dataplane
    scorr = lin_coeffs[0] + scorr
    # Only use the corrected signal where the original signal value
    # has not been flagged by the saturation step.
    # Otherwise use the original signal.
    dataplane[:, :] = np.where(np.bitwise_and(gdqplane[:, :], dqflags['SATURATED']),
                               dataplane[:, :],
                               scorr)


def correct_for_NaN(lin_coeffs, pixeldq, dqflags):
    """
    Check for NaNs in the COEFFS extension of the ref file in case there are
    pixels that should have been (but were not) flagged there as NO_LIN_CORR
    (linearity correction not determined for pixel).

    For such pixels, update the
    coefficients so that there is effectively no correction, and flag their
    pixeldq values in place as NO_LIN_CORR in the step output.

    Parameters
    ----------
    lin_coeffs: 3D array
        array of correction coefficients in reference file

    input: data model object
        science data model to be corrected in place

    Returns
    -------
    lin_coeffs: 3D array
        updated array of correction coefficients in reference file
    """

    wh_nan = np.where(np.isnan(lin_coeffs))
    znan, ynan, xnan = wh_nan[0], wh_nan[1], wh_nan[2]
    num_nan = 0

    nan_array = np.zeros((lin_coeffs.shape[1], lin_coeffs.shape[2]),
                         dtype=np.uint32)

    # If there are NaNs as the correction coefficients, update those
    # coefficients so that those SCI values will be unchanged.
    if len(znan) > 0:
        ben_cor = ben_coeffs(lin_coeffs)  # get benign coefficients
        num_nan = len(znan)

        for ii in range(num_nan):
            lin_coeffs[:, ynan[ii], xnan[ii]] = ben_cor
            nan_array[ynan[ii], xnan[ii]] = dqflags['NO_LIN_CORR']

        # Include these pixels in the output pixeldq
        pixeldq = np.bitwise_or(pixeldq, nan_array)

    return lin_coeffs, pixeldq


def correct_for_zero(lin_coeffs, pixeldq, dqflags):
    """
    Check when the linear term in the linearity coefficients is zero.  For such pixels, update the
    coefficients so that there is effectively no correction, and flag their
    pixeldq values in place as NO_LIN_CORR in the step output.

    Parameters
    ----------
    lin_coeffs: 3D array
        array of correction coefficients in reference file

    input: data model object
        science data model to be corrected in place

    Returns
    -------
    lin_coeffs: 3D array
        updated array of correction coefficients in reference file
    """

    # The critcal coefficient that should not be zero is the linear term other terms are fine to be zero
    linear_term = lin_coeffs[1,:,:]
    wh_zero = np.where(linear_term == 0)
    yzero, xzero = wh_zero[0], wh_zero[1]
    num_zero = 0
    lin_dq_array = np.zeros((lin_coeffs.shape[1], lin_coeffs.shape[2]),
                            dtype=np.uint32)

    # If there are linearity linear term equal to zero,
    # update the coefficients so the SCI values will be unchanged.
    if len(yzero) > 0:
        ben_cor = ben_coeffs(lin_coeffs)  # get benign coefficients
        num_zero = len(yzero)

        for ii in range(num_zero):
            lin_coeffs[:, yzero[ii], xzero[ii]] = ben_cor
            lin_dq_array[yzero[ii], xzero[ii]] = dqflags['NO_LIN_CORR']

        # Include these pixels in the output pixeldq
        pixeldq = np.bitwise_or(pixeldq, lin_dq_array)

    return lin_coeffs, pixeldq


def correct_for_flag(lin_coeffs, lin_dq, dqflags):
    """
    Short Summary
    -------------
    Check for pixels that are flagged as NO_LIN_CORR
    ('No linearity correction available') in the DQ extension of the ref data.
    For such pixels, update the coefficients so that there is effectively no
    correction. Because these are already flagged in the ref file, they will
    also be flagged in the output dq.

    Parameters
    ----------
    lin_coeffs: 3D array
        array of correction coefficients in reference file

    lin_dq: 2D array
        array of data quality flags in reference file

    Returns
    -------
    lin_coeffs: 3D array
        updated array of correction coefficients in reference file
    """

    wh_flag = np.bitwise_and(lin_dq, dqflags['NO_LIN_CORR'])
    num_flag = len(np.where(wh_flag > 0)[0])

    wh_lin = np.where(wh_flag == dqflags['NO_LIN_CORR'])
    yf, xf = wh_lin[0], wh_lin[1]

    # If there are pixels flagged as 'NO_LIN_CORR', update the corresponding
    #     coefficients so that those SCI values will be unchanged.
    if num_flag > 0:
        ben_cor = ben_coeffs(lin_coeffs)  # get benign coefficients

        for ii in range(num_flag):
            lin_coeffs[:, yf[ii], xf[ii]] = ben_cor

    return lin_coeffs


def ben_coeffs(lin_coeffs):
    """
    Short Summary
    -------------
    For pixels having at least 1 NaN coefficient, reset the coefficients to be
    benign, which will effectively leave the SCI values unaffected.

    Parameters
    ----------
    lin_coeffs: 3D array
        array of correction coefficients in reference file

    Returns
    -------
    ben_cor: 1D array
        benign coefficients - all ben_cor[:] = 0.0 except ben_cor[1] = 1.0
    """
    ben_cor = np.zeros(lin_coeffs.shape[0])
    ben_cor[1] = 1.0

    return ben_cor
