"""

Unit tests for linearity correction

"""

import numpy as np

from stcal.linearity.linearity import linearity_correction

DQFLAGS = {
    'GOOD': 0,
    'DO_NOT_USE': 1,
    'SATURATED': 2,
    'DEAD': 1024,
    'HOT': 2048,
    'NO_LIN_CORR': 1048576}

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
    L0 = 0.0e+00
    L1 = 0.85
    L2 = 4.62E-6
    L3 = -6.16E-11
    L4 = 7.23E-16

    coeffs = np.asfarray([L0, L1, L2, L3, L4])

    # pixels we are testing using above coefficients
    lin_coeffs[:, 30, 50] = coeffs
    lin_coeffs[:, 35, 36] = coeffs
    lin_coeffs[:, 35, 35] = coeffs

    lin_dq = np.zeros((ysize, xsize), dtype=np.uint32)

    # check behavior with NaN coefficients: should not alter pixel values
    coeffs2 = np.asfarray([L0, np.nan, L2, L3, L4])

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
    pdq[50, 40] = DQFLAGS['DO_NOT_USE']
    pdq[50, 41] = DQFLAGS['SATURATED']
    pdq[50, 42] = DQFLAGS['DEAD']
    pdq[50, 43] = DQFLAGS['HOT']

    # set dq flags in DQ of reference file
    lin_dq[35, 35] = DQFLAGS['DO_NOT_USE']
    lin_dq[35, 36] = DQFLAGS['NO_LIN_CORR']
    lin_dq[30, 50] = DQFLAGS['GOOD']

    np.bitwise_or(pdq, lin_dq)

    # run linearity correction
    output_data, output_pdq, _ = linearity_correction(
        data, gdq, pdq, lin_coeffs, lin_dq, DQFLAGS)

    # check that multiplication of polynomial was done correctly for specified pixel
    outval = L0 + (L1 * scival) + (L2 * scival**2) + (L3 * scival**3) + (L4 * scival**4)
    assert(np.isclose(output_data[0, 45, 30, 50], outval, rtol=0.00001))

    # check that dq value was handled correctly

    assert output_pdq[35, 35] == DQFLAGS['DO_NOT_USE']
    assert output_pdq[35, 36] == DQFLAGS['NO_LIN_CORR']
    # NO_LIN_CORR, sci value should not change
    assert output_data[0, 30, 35, 36] == 35
    # NaN coefficient should not change data value
    assert output_data[0, 50, 20, 50] == 500.0
    # dq for pixel with all zero lin coeffs should be NO_LIN_CORR
    assert output_pdq[25, 25] == DQFLAGS['NO_LIN_CORR']


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
    zframe[0, 0, :] = np.array([data[0, 0, 0, 0] * 0.666666, 0.])

    lin_base = 2.718 / (base * 10.)
    coeffs = np.array([lin_base**(k) for k in range(ncoeffs)], dtype=float)
    lin_coeffs[:, 0, 0] = coeffs

    output_data, output_pdq, new_zframe = linearity_correction(
        data, gdq, pdq, lin_coeffs, lin_dq, DQFLAGS, zframe)

    zcheck = np.zeros((nints, nrows, ncols), dtype=float)
    zcheck[0, 0, :] = np.array([1.22106063, 0.])
    np.testing.assert_almost_equal(new_zframe, zcheck, decimal=5)
