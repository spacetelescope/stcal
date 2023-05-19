"""Utility routines for Mutli-Accum Ramp Fitting
"""
import numpy as np

# Read Time in seconds
#   For Roman, the read time of the detectors is a fixed value and is currently
#   backed into code. Will need to refactor to consider the more general case.
#   Used to deconstruct the MultiAccum tables into integration times.
READ_TIME = 3.04

__all__ = ['ma_table_to_tau', 'ma_table_to_tbar']


def ma_table_to_tau(ma_table, read_time=READ_TIME):
    """Construct the tau for each resultant from an ma_table.

    .. math:: \\tau = \\overline{t} - (n - 1)(n + 1)\\delta t / 6n

    following Casertano (2022).

    Parameters
    ----------
    ma_table : list[list]
        List of lists specifying the first read and the number of reads in each
        resultant.

    Returns
    -------
    :math:`\\tau`
        A time scale appropriate for computing variances.
    """

    meantimes = ma_table_to_tbar(ma_table)
    nreads = np.array([x[1] for x in ma_table])
    return meantimes - (nreads - 1) * (nreads + 1) * read_time / 6 / nreads


def ma_table_to_tbar(ma_table, read_time=READ_TIME):
    """Construct the mean times for each resultant from an ma_table.

    Parameters
    ----------
    ma_table : list[list]
        List of lists specifying the first read and the number of reads in each
        resultant.

    Returns
    -------
    tbar : np.ndarray[n_resultant] (float)
        The mean time of the reads of each resultant.
    """
    firstreads = np.array([x[0] for x in ma_table])
    nreads = np.array([x[1] for x in ma_table])
    meantimes = read_time * firstreads + read_time * (nreads - 1) / 2
    # at some point I need to think hard about whether the first read has
    # slightly less exposure time than all other reads due to the read/reset
    # time being slightly less than the read time.
    return meantimes
