import numpy as np


def linearity_correction(
    data,
    gdq,
    pdq,
    lin_coeffs,
    lin_dq,
    dqflags,
    zframe=None,
    ilin_coeffs=None,
    additional_correction=None,
    read_pattern=None,
    satval=None,
):
    """
    Apply linearity correction.

    Apply linearity correction to individual groups in `data` to pixels that
    haven't already been flagged as saturated.

    Pixels having at least one correction coefficient equal to NaN will not
    have the linearity correction applied and the DQ flag `NO_LIN_CORR` is
    added to the science exposure PIXELDQ array. Pixels that have the
    `NO_LIN_CORR` flag set in the DQ array of the linearity reference file will
    not have the correction applied and the "NO_LIN_CORR" flag is added to the
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
        The 4D array for the SCI data (nints, ngroups, nrows, ncols).

    gdq : ndarray
        The 4D group dq array.

    pdq : ndarray
        The 2D pixel dq array.

    lin_coeffs : ndarray.
        The 3D array containing pixel-by-pixel linearity coefficient
        values for each term in the polynomial fit.

    lin_dq : ndarray
        The 2D DQ array from the linearity reference file.

    dqflags : dict
        A dictionary with at least the following keywords:
        SATURATED, NO_LIN_CORR.

    zframe : ndarray or None
        The ZEROFRAME array with dimensions (nints, nrows, ncols) or None

    ilin_coeffs : ndarray or None
        The 3D array containing inverse linearity coefficients. If provided,
        enables read-level linearity correction that accounts for averaging
        of multiple reads into resultants. Requires read_pattern to also be
        provided.

    additional_correction : callable or None
        A callable that takes the non-linear counts array and returns a correction
        array to be added to it. This allows mission-specific corrections (e.g.,
        integral non-linearity) to be applied. The callable should accept a 3D
        array (nreads, nrows, ncols) and return a correction array of the same
        shape.

    read_pattern : list of lists or None
        The pattern of reads entering into groups. Each element is a list of
        1-indexed read indices that are averaged together to form the
        corresponding resultant. Example: [[1], [2], [3, 4], [5, 6, 7]] means
        resultant 0 is read 1, resultant 1 is read 2, resultant 2 is the
        average of reads 3 and 4, etc.

    satval : ndarray or None
        2D array of saturation values for each pixel.

    Returns
    -------
    data : ndarray
        The 4D linearly corrected SCI data.

    new_pdq : ndarray
        The updated 2D pixel data quality flags.

    zframe : ndarray or None
        If not None, the 3D linearly corrected ZEROFRAME.
    """
    # Check if we should use read-level correction
    read_level_correction = ilin_coeffs is not None

    if read_level_correction:
        # Validate that all required parameters are provided
        if read_pattern is None:
            raise ValueError("When ilin_coeffs is provided, read_pattern must also be provided")

    # Prepare coefficients
    nints = data.shape[0]
    lin_coeffs, new_pdq = prepare_coefficients(lin_coeffs, lin_dq, pdq, dqflags)
    if read_level_correction:
        ilin_coeffs, new_pdq = prepare_coefficients(ilin_coeffs, lin_dq, new_pdq, dqflags)

    # Apply linearity correction to each integration
    for i in range(nints):
        data[i] = linearity_correction_int(
            data[i],
            gdq[i],
            lin_coeffs,
            dqflags,
            ilin_coeffs=ilin_coeffs,
            additional_correction=additional_correction,
            read_pattern=read_pattern,
            satval=satval,
        )

    zdata = None  # zframe needs to be returned, so initialize it to None.
    if zframe is not None:
        # Do linear correction on ZEROFRAME
        # As ZEROFRAME gets processed through the pipeline, since it has no
        # corresponding DQ array, when data is found to be bad, the data is
        # set to ZERO.  Since zero ZEROFRAME values indicates bad data,
        # remember where this happens.  Make a dummy ZEROFRAME DQ array and
        # mark zeroed data as saturated.
        wh_zero = np.where(zframe[:, :, :] == 0.0)

        # Add a groups axis to make zframe 4D like regular data
        zframe = zframe[:, np.newaxis, :, :]
        zdq = np.zeros(zframe.shape, dtype=gdq.dtype)
        zdq[zframe == 0.0] = dqflags["SATURATED"]

        # Linearly correct ZEROFRAME
        # Note: this reuses lin_coeffs, which have been changed to avoid
        # problematic values. But this is okay since we want to do that for
        # the zero frame anyway.
        zdata = np.zeros_like(zframe)
        for i in range(zframe.shape[0]):
            zdata[i] = linearity_correction_int(zframe[i], zdq[i], lin_coeffs, dqflags)

        # Remove the groups axis and ensure bad data remains bad
        zdata = zdata[:, 0, :, :]
        zdata[wh_zero] = 0.0

    return data, new_pdq, zdata


def correct_for_NaN(lin_coeffs, pixeldq, dqflags):  # noqa: N802
    """
    Check for NaNs in the COEFFS extension of the ref file.

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

    nan_array = np.zeros((lin_coeffs.shape[1], lin_coeffs.shape[2]), dtype=np.uint32)

    # If there are NaNs as the correction coefficients, update those
    # coefficients so that those SCI values will be unchanged.
    if len(znan) > 0:
        ben_cor = ben_coeffs(lin_coeffs)  # get benign coefficients
        num_nan = len(znan)

        for ii in range(num_nan):
            lin_coeffs[:, ynan[ii], xnan[ii]] = ben_cor
            nan_array[ynan[ii], xnan[ii]] = dqflags["NO_LIN_CORR"]

        # Include these pixels in the output pixeldq
        pixeldq = np.bitwise_or(pixeldq, nan_array)

    return lin_coeffs, pixeldq


def correct_for_zero(lin_coeffs, pixeldq, dqflags):
    """
    Check when the linear term in the linearity coefficients is zero.

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
    # The critical coefficient that should not be zero is the linear term
    # other terms are fine to be zero
    linear_term = lin_coeffs[1, :, :]
    wh_zero = np.where(linear_term == 0)
    yzero, xzero = wh_zero[0], wh_zero[1]
    num_zero = 0
    lin_dq_array = np.zeros((lin_coeffs.shape[1], lin_coeffs.shape[2]), dtype=np.uint32)

    # If there are linearity linear term equal to zero,
    # update the coefficients so the SCI values will be unchanged.
    if len(yzero) > 0:
        ben_cor = ben_coeffs(lin_coeffs)  # get benign coefficients
        num_zero = len(yzero)

        for ii in range(num_zero):
            lin_coeffs[:, yzero[ii], xzero[ii]] = ben_cor
            lin_dq_array[yzero[ii], xzero[ii]] = dqflags["NO_LIN_CORR"]

        # Include these pixels in the output pixeldq
        pixeldq = np.bitwise_or(pixeldq, lin_dq_array)

    return lin_coeffs, pixeldq


def correct_for_flag(lin_coeffs, lin_dq, dqflags):
    """
    Check for NO_LIN_CORR flagged pixels.

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
    wh_flag = np.bitwise_and(lin_dq, dqflags["NO_LIN_CORR"])
    num_flag = len(np.where(wh_flag > 0)[0])

    wh_lin = np.where(wh_flag == dqflags["NO_LIN_CORR"])
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
    Compute benign coefficients.

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


def prepare_coefficients(coeffs, lin_dq, pdq, dqflags):
    """
    Prepare linearity coefficients by checking for and handling bad values.

    This function checks for NO_LIN_CORR flags, NaN values, and zero coefficients,
    updating the coefficients and pixel DQ flags as needed.

    Parameters
    ----------
    coeffs : ndarray
        The linearity coefficients (ncoeffs, nrows, ncols).

    lin_dq : ndarray
        The 2D DQ array from the linearity reference file.

    pdq : ndarray
        The 2D pixel dq array.

    dqflags : dict
        Dictionary of DQ flags.

    Returns
    -------
    coeffs : ndarray
        The updated coefficients.

    new_pdq : ndarray
        The updated pixel DQ flags.
    """
    # Combine the DQ arrays using bitwise_or
    new_pdq = np.bitwise_or(pdq, lin_dq)

    # Check for NO_LIN_CORR flags in the DQ extension of the ref file
    coeffs = correct_for_flag(coeffs, lin_dq, dqflags)

    # Check for NaNs in the COEFFS extension of the ref file
    coeffs, new_pdq = correct_for_NaN(coeffs, new_pdq, dqflags)

    # Check when all the Linearity COEFFS = 0
    coeffs, new_pdq = correct_for_zero(coeffs, new_pdq, dqflags)

    return coeffs, new_pdq


def apply_polynomial(data, coeffs, gdq=None, dqflags=None):
    """
    Apply a polynomial correction to data using the provided coefficients.

    The polynomial is evaluated using Horner's method as:
    result = coeffs[0] + coeffs[1]*data + coeffs[2]*data^2 + ... + coeffs[n]*data^n

    Parameters
    ----------
    data : ndarray
        The data to which the polynomial will be applied.

    coeffs : ndarray
        The polynomial coefficients. Shape should be (ncoeffs, ...) where
        the remaining dimensions match the data shape.

    gdq : ndarray or None
        Group DQ flags. If provided along with dqflags, saturated pixels will
        keep their original values instead of being corrected.

    dqflags : dict or None
        Dictionary of DQ flags. Required if gdq is provided.

    Returns
    -------
    result : ndarray
        The polynomial-corrected data.
    """
    # Determine number of coefficients from array shape
    ncoeffs = coeffs.shape[0]

    # Accumulate the polynomial terms using Horner's method
    result = coeffs[ncoeffs - 1] * data
    for j in range(ncoeffs - 2, 0, -1):
        result = (result + coeffs[j]) * data
    result = coeffs[0] + result

    # Optionally respect saturation flags
    if gdq is not None and dqflags is not None:
        result = np.where(np.bitwise_and(gdq, dqflags["SATURATED"]), data, result)

    return result


def linearity_correction_int(
    data,
    gdq,
    lin_coeffs,
    dqflags,
    ilin_coeffs=None,
    additional_correction=None,
    read_pattern=None,
    satval=None,
):
    """
    Apply linearity correction to a single integration.

    If ilin_coeffs is provided along with read_pattern, performs
    read-level correction that accounts for averaging of multiple reads into
    resultants. Otherwise, performs simple group-by-group correction.

    Parameters
    ----------
    data : ndarray
        The 3D array for a single integration (ngroups, nrows, ncols).

    gdq : ndarray
        The 3D group dq array for a single integration.

    lin_coeffs : ndarray
        The linearity coefficients (ncoeffs, nrows, ncols).

    dqflags : dict
        Dictionary of DQ flags.

    ilin_coeffs : ndarray or None
        The inverse linearity coefficients (ncoeffs, nrows, ncols).
        If None, simple group-by-group correction is used.

    additional_correction : callable or None
        A callable that takes the non-linear counts array and returns a correction
        array to be added to it. Only used for read-level correction.

    read_pattern : list of lists or None
        The pattern of 1-indexed reads entering into groups. Required for
        read-level correction.

    satval : ndarray or None
        2D array of saturation values for each pixel. Used for read-level
        correction.

    Returns
    -------
    data_linearized : ndarray
        The corrected data (ngroups, nrows, ncols).
    """
    ngroups, nrows, ncols = data.shape

    # If no inverse linearity coefficients, do simple correction in place
    if ilin_coeffs is None:
        for plane in range(ngroups):
            data[plane] = apply_polynomial(data[plane], lin_coeffs, gdq[plane], dqflags)
        return data

    # Calculate mean read number of each resultant (e.g. reads [4, 5, 6, 7] is 5.5)
    mean_read_resultant = np.array([np.mean(reads) for reads in read_pattern])

    # Identify saturated pixels and count unsaturated resultants
    is_saturated = (gdq & dqflags["SATURATED"]) != 0
    # additionally treat the last-unsaturated resultant as saturated just to be
    # conservative.
    n_unsaturated = np.sum(~is_saturated, axis=0) - 1

    # Avoid divide-by-zero errors for pixels with < 2 unsaturated resultants
    n_unsaturated[n_unsaturated < 2] = 2

    # Calculate count rate using first and last unsaturated resultants
    firstread_lin = apply_polynomial(data[0], lin_coeffs, gdq[0], dqflags)

    lastvalidresultant = n_unsaturated - 1
    iy, ix = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")

    # Create index tuple for last valid resultant of each pixel
    last_idx = (lastvalidresultant[iy, ix], iy, ix)

    # Linearize last valid resultant for each pixel
    lastread_lin = apply_polynomial(data[last_idx], lin_coeffs, gdq[last_idx], dqflags)

    d_reads = mean_read_resultant[last_idx[0]] - mean_read_resultant[0]
    # countrate is in units of counts/read
    countrate = (lastread_lin - firstread_lin) / d_reads

    # Process each resultant
    for i in range(ngroups):
        reads_for_resultant = read_pattern[i]

        reads_since_first = np.array(read_pattern[i]) - mean_read_resultant[0]
        # Reconstruct linearized counts for these reads
        reads_linearized = (
            firstread_lin[None, :, :]
            + countrate[None, :, :] * reads_since_first[:, None, None])

        # Convert back to uncorrected counts using inverse linearity
        reads_unlinearized = apply_polynomial(reads_linearized, ilin_coeffs)

        # Adjust to match this resultant
        predicted_cts = np.mean(reads_unlinearized, axis=0)
        offset = data[i] - predicted_cts
        reads_unlinearized += offset[np.newaxis, :, :]

        # Apply additional correction if provided
        if additional_correction is not None:
            reads_unlinearized += additional_correction(reads_unlinearized)

        # Cap values at 1.2 * saturation
        # Note this means that we run the linearity correction on values up to
        # 1.2x saturation.  This is fine since these points should have already
        # been flagged as saturated, and below we only correct unsaturated
        # resultants
        if satval is not None:
            np.putmask(
                reads_unlinearized,
                reads_unlinearized > 1.2 * satval[np.newaxis, :, :],
                1.2 * satval[np.newaxis, :, :],
            )

        # Apply classic linearity to the reads
        reads_corrected = np.zeros_like(reads_unlinearized)
        for j in range(reads_corrected.shape[0]):
            reads_corrected[j] = apply_polynomial(reads_unlinearized[j], lin_coeffs)

        # Average and write back, respecting saturation
        resultant_saturated = (gdq[i] & dqflags["SATURATED"]) != 0
        corrected_resultant = np.mean(reads_corrected, axis=0)
        data[i] = np.where(resultant_saturated, data[i], corrected_resultant)

    return data
