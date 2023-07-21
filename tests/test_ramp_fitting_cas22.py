
"""
Unit tests for ramp-fitting functions.  Tested routines:
* ma_table_to_tbar
* ma_table_to_tau
* construct_covar
* construct_ramp_fitting_matrices
* construct_ki_and_variances
* ki_and_variance_grid
* RampFitInterpolator
  * __init__
  * ki
  * variances
  * fit_ramps
* resultants_to_differences
* simulate_many_ramps
"""
import pytest

import numpy as np

from stcal.ramp_fitting import ols_cas22_fit as ramp
from stcal.ramp_fitting.ols_cas22_util import matable_to_readpattern, readpattern_to_matable

# Read Time in seconds
#   For Roman, the read time of the detectors is a fixed value and is currently
#   backed into code. Will need to refactor to consider the more general case.
#   Used to deconstruct the MultiAccum tables into integration times.
ROMAN_READ_TIME = 3.04


def test_matable_to_readpattern():
    """Test conversion from read pattern to multi-accum table"""
    ma_table = [[1, 1], [2, 2], [4, 1], [5, 4], [9,2], [11,1]]
    expected = [[1], [2, 3], [4], [5, 6, 7, 8], [9, 10], [11]]

    result = matable_to_readpattern(ma_table)

    assert result == expected


def test_readpattern_to_matable():
    """Test conversion from read pattern to multi-accum table"""
    pattern = [[1], [2, 3], [4], [5, 6, 7, 8], [9, 10], [11]]
    expected = [[1, 1], [2, 2], [4, 1], [5, 4], [9,2], [11,1]]

    result = readpattern_to_matable(pattern)

    assert result == expected


def test_simulated_ramps():
    romanisim_ramp = pytest.importorskip('romanisim.ramp')
    ntrial = 100000
    ma_table, flux, read_noise, resultants = romanisim_ramp.simulate_many_ramps(
        ntrial=ntrial)

    par, var = ramp.fit_ramps_casertano(
        resultants, resultants * 0, read_noise, ROMAN_READ_TIME, ma_table=ma_table)
    chi2dof_slope = np.sum((par[:, 1] - flux)**2 / var[:, 2, 1, 1]) / ntrial
    assert np.abs(chi2dof_slope - 1) < 0.03

    # now let's mark a bunch of the ramps as compromised.
    bad = np.random.uniform(size=resultants.shape) > 0.7
    dq = resultants * 0 + bad
    par, var = ramp.fit_ramps_casertano(
        resultants, dq, read_noise, ROMAN_READ_TIME, ma_table=ma_table)
    # only use okay ramps
    # ramps passing the below criterion have at least two adjacent valid reads
    # i.e., we can make a measurement from them.
    m = np.sum((dq[1:, :] == 0) & (dq[:-1, :] == 0), axis=0) != 0
    chi2dof_slope = np.sum((par[m, 1] - flux)**2 / var[m, 2, 1, 1]) / np.sum(m)
    assert np.abs(chi2dof_slope - 1) < 0.03
    assert np.all(par[~m, 1] == 0)
    assert np.all(var[~m, 1] == 0)
