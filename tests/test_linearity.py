"""

Unit tests for linearity correction

"""

import numpy as np

from stcal.linearity.linearity import linearity_correction, apply_polynomial

DQFLAGS = {"GOOD": 0, "DO_NOT_USE": 1, "SATURATED": 2, "DEAD": 1024, "HOT": 2048, "NO_LIN_CORR": 1048576}

DELIM = "-" * 80


def test_coeff_dq():
    """
    Test linearity algorithm with random data ramp (does
    algorithm match expected algorithm) also test a variety
    of dq flags and expected output
    """

    # size of integration
    nints = 1
    ngroups = 160
    xsize = 103
    ysize = 102

    # Create data array and group/pixel dq arrays
    data = np.ones((nints, ngroups, ysize, xsize)).astype(np.float32)
    pdq = np.zeros((ysize, xsize)).astype(np.uint32)
    gdq = np.zeros((nints, ngroups, ysize, xsize)).astype(np.uint32)

    # Create reference file data and dq arrays
    numcoeffs = 5
    lin_coeffs = np.zeros((numcoeffs, ysize, xsize))

    # Set coefficient values in reference file to check the algorithm
    # Equation is DNcorr = L0 + L1*DN(i) + L2*DN(i)^2 + L3*DN(i)^3 + L4*DN(i)^4
    # DN(i) = signal in pixel, Ln = coefficient from ref file
    # L0 = 0 for all pixels for CDP6
    L0 = 0.0e00
    L1 = 0.85
    L2 = 4.62e-6
    L3 = -6.16e-11
    L4 = 7.23e-16

    coeffs = np.asarray([L0, L1, L2, L3, L4], dtype="float")

    # pixels we are testing using above coefficients
    lin_coeffs[:, 30, 50] = coeffs
    lin_coeffs[:, 35, 36] = coeffs
    lin_coeffs[:, 35, 35] = coeffs

    lin_dq = np.zeros((ysize, xsize), dtype=np.uint32)

    # check behavior with NaN coefficients: should not alter pixel values
    coeffs2 = np.asarray([L0, np.nan, L2, L3, L4], dtype="float")

    lin_coeffs[:, 20, 50] = coeffs2
    data[0, 50, 20, 50] = 500.0

    # test case where all coefficients are zero
    lin_coeffs[:, 25, 25] = 0.0
    data[0, 50, 25, 25] = 600.0

    tgroup = 2.775

    # set pixel values (DN) for specific pixels up the ramp
    data[0, :, 30, 50] = np.arange(ngroups) * 100 * tgroup

    scival = 40000.0
    data[0, 45, 30, 50] = scival  # to check linearity multiplication is done correctly
    data[0, 30, 35, 36] = 35  # pixel to check that dq=2 meant no correction was applied

    # check if dq flags in pixeldq are correctly populated in output
    pdq[50, 40] = DQFLAGS["DO_NOT_USE"]
    pdq[50, 41] = DQFLAGS["SATURATED"]
    pdq[50, 42] = DQFLAGS["DEAD"]
    pdq[50, 43] = DQFLAGS["HOT"]

    # set dq flags in DQ of reference file
    lin_dq[35, 35] = DQFLAGS["DO_NOT_USE"]
    lin_dq[35, 36] = DQFLAGS["NO_LIN_CORR"]
    lin_dq[30, 50] = DQFLAGS["GOOD"]

    np.bitwise_or(pdq, lin_dq)

    # run linearity correction
    output_data, output_pdq, _ = linearity_correction(data, gdq, pdq, lin_coeffs, lin_dq, DQFLAGS)

    # check that multiplication of polynomial was done correctly for specified pixel
    outval = L0 + (L1 * scival) + (L2 * scival**2) + (L3 * scival**3) + (L4 * scival**4)
    assert np.isclose(output_data[0, 45, 30, 50], outval, rtol=0.00001)

    # check that dq value was handled correctly

    assert output_pdq[35, 35] == DQFLAGS["DO_NOT_USE"]
    assert output_pdq[35, 36] == DQFLAGS["NO_LIN_CORR"]
    # NO_LIN_CORR, sci value should not change
    assert output_data[0, 30, 35, 36] == 35
    # NaN coefficient should not change data value
    assert output_data[0, 50, 20, 50] == 500.0
    # dq for pixel with all zero lin coeffs should be NO_LIN_CORR
    assert output_pdq[25, 25] == DQFLAGS["NO_LIN_CORR"]


def create_science_data(dims, ncoeffs):
    """
    Create science data arrays with specific dimensions.

    dims : tuple
        The dimensions of the science data (nints, ngroups, nrows, ncols).

    ncoeffs : int
        The number of coefficients for the linear correction.
    """
    nints, ngroups, nrows, ncols = dims
    image_shape = (nrows, ncols)
    coeffs_shape = (ncoeffs, nrows, ncols)

    data = np.zeros(dims, dtype=float)
    gdq = np.zeros(dims, dtype=np.uint8)
    pdq = np.zeros(image_shape, dtype=np.uint8)
    zframe = np.zeros((nints, nrows, ncols), dtype=float)

    lin_coeffs = np.zeros(coeffs_shape, dtype=float)
    lin_dq = np.zeros(image_shape, dtype=np.uint32)

    return data, gdq, pdq, lin_coeffs, lin_dq, zframe


def test_zero_frame():
    """
    Check to make sure the ZEROFRAME properly gets corrected.
    """

    nints, ngroups, nrows, ncols = 1, 5, 1, 2
    ncoeffs = 5
    dims = nints, ngroups, nrows, ncols
    ncoeffs = 5

    data, gdq, pdq, lin_coeffs, lin_dq, zframe = create_science_data(dims, ncoeffs)

    base = 31.459
    data[0, :, 0, 0] = np.array([(k + 1) * base for k in range(ngroups)], dtype=float)
    zframe[0, 0, :] = np.array([data[0, 0, 0, 0] * 0.666666, 0.0])

    lin_base = 2.718 / (base * 10.0)
    coeffs = np.array([lin_base ** (k) for k in range(ncoeffs)], dtype=float)
    lin_coeffs[:, 0, 0] = coeffs

    output_data, output_pdq, new_zframe = linearity_correction(
        data, gdq, pdq, lin_coeffs, lin_dq, DQFLAGS, zframe
    )

    zcheck = np.zeros((nints, nrows, ncols), dtype=float)
    zcheck[0, 0, :] = np.array([1.22106063, 0.0])
    np.testing.assert_almost_equal(new_zframe, zcheck, decimal=5)


def test_read_level_correction():
    """
    Test read-level linearity correction that accounts for averaging of
    multiple reads into resultants.
    """
    # Set up test data
    nreads_per_group = 5
    ngroups = 4
    nreads = ngroups * nreads_per_group
    nints, nrows, ncols = 1, 1, 1
    data = np.ones((nints, ngroups, nrows, ncols), dtype=np.float32)
    gdq = np.zeros((nints, ngroups, nrows, ncols), dtype=np.uint32)
    pdq = np.zeros((nrows, ncols), dtype=np.uint32)

    sat = 65000
    # ramp that nearly saturated in the last read
    # this is the true number of electrons in the ramp
    reads = np.arange(nreads) * (sat - 1) / (nreads - 1)
    # some made up inverse linearity coefficients
    # these are somewhat steep; DN / electron changes from
    # ~0.7 at 50k electrons to 0.4 at 65k electrons
    ilin_coeffs_flat = np.array(
        [ 0,  1, -1.e-07, -1.e-12, -5.e-16], dtype='f8')
    # linearity coefficients corresponding to the above;
    # these were computed separately
    lin_coeffs_flat = np.array(
        [ 0,  1, -3.34319810e-07,  1.24117346e-10,
          -1.11872923e-14,  4.92169172e-19, -9.50437757e-24,  7.04201969e-29],
        dtype='f8')

    # Set up linearity & inverse linearity coefficients
    nlcoeffs = len(lin_coeffs_flat)
    lin_coeffs = np.zeros((nlcoeffs, nrows, ncols), dtype=np.float32)
    lin_coeffs[:, 0, 0] = lin_coeffs_flat

    nicoeffs = len(ilin_coeffs_flat)
    ilin_coeffs = np.zeros((nicoeffs, nrows, ncols), dtype=np.float32)
    ilin_coeffs[:, 0, 0] = ilin_coeffs_flat

    lin_dq = np.zeros((nrows, ncols), dtype=np.uint32)

    read_pattern = (np.arange(nreads) + 1).reshape(ngroups, nreads_per_group)

    # Set up saturation values
    satval = np.full((nrows, ncols), sat, dtype=np.float32)

    nl_reads = apply_polynomial(reads, ilin_coeffs_flat)
    true_groups = np.average(
        reads.reshape(ngroups, nreads_per_group), axis=1)
    nl_groups = np.average(
        nl_reads.reshape(ngroups, nreads_per_group), axis=1)
    # we now have what we want after linearity correction (true groups)
    # and what is observed (true_nl_groups)
    # success if the linearity correction with the new mode accounting
    # for the read pattern better reproduces true_groups than the old
    # mode

    data[0, :, 0, 0] = nl_groups

    # Test with read-level correction
    corrected_with_read_pattern, output_pdq, _ = linearity_correction(
        data.copy(), gdq, pdq, lin_coeffs, lin_dq, DQFLAGS,
        ilin_coeffs=ilin_coeffs,
        read_pattern=read_pattern,
        satval=satval
    )
    corrected_without_read_pattern, _, _ = linearity_correction(
        data.copy(), gdq, pdq, lin_coeffs, lin_dq, DQFLAGS)

    fracdiff_read_pattern = (
        corrected_with_read_pattern[0, :, 0, 0] / true_groups) - 1
    fracdiff = (
        corrected_without_read_pattern[0, :, 0, 0] / true_groups) - 1
    worstdiff_read_pattern = np.max(np.abs(fracdiff_read_pattern))
    worstdiff = np.max(np.abs(fracdiff))

    assert worstdiff_read_pattern < 0.004
    assert worstdiff < 0.007
    assert worstdiff_read_pattern < worstdiff * 0.5

    # Basic checks: output should have same shape and be corrected
    assert corrected_with_read_pattern.shape == data.shape
    assert output_pdq.shape == pdq.shape
    return true_groups, corrected_with_read_pattern, corrected_without_read_pattern
