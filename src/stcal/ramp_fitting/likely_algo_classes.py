import numpy as np


class IntegInfo:
    """Storage for the integration information for ramp fitting computations."""

    def __init__(self, nints, nrows, ncols):
        """
        Initialize output arrays.

        Parameters
        ----------
        nints : int
            The number of integrations in the data.

        nrows : int
            The number of rows in the data.

        ncols : int
            The number of columns in the data.
        """
        dims = (nints, nrows, ncols)
        self.data = np.zeros(shape=dims, dtype=np.float32)

        self.dq = np.zeros(shape=dims, dtype=np.uint32)

        self.var_poisson = np.zeros(shape=dims, dtype=np.float32)
        self.var_rnoise = np.zeros(shape=dims, dtype=np.float32)

        self.err = np.zeros(shape=dims, dtype=np.float32)

    def prepare_info(self):
        """Arrange output arrays as a tuple, which the ramp fit step expects."""
        #return (self.data, self.dq, self.var_poisson, self.var_rnoise, self.err)
        return {'data':self.data, 'dq':self.dq, 'var_p':self.var_poisson, 'var_r':self.var_rnoise, 'err':self.err}

    def get_results(self, result, integ, row):
        """
        Capture the ramp fitting computation.

        Parameters
        ----------
        result : RampResult
            Holds computed ramp fitting information.

        integ : int
            The current integration being operated on.

        row : int
            The current row being operated on.
        """
        self.data[integ, row, :] = result.countrate
        self.err[integ, row, :] = result.uncert
        self.var_poisson[integ, row, :] = result.var_poisson
        self.var_rnoise[integ, row, :] = result.var_rnoise


class RampResult:  # noqa: D101
    def __init__(self):
        """Contains the ramp fitting results."""
        self.countrate = None
        self.chisq = None
        self.uncert = None
        self.var_poisson = None
        self.var_rnoise = None
        self.weights = None

        self.countrate_two_omit = None
        self.chisq_two_omit = None
        self.uncert_two_omit = None

        self.countrate_one_omit = None
        self.jumpval_one_omit = None
        self.jumpsig_one_omit = None
        self.chisq_one_omit = None
        self.uncert_one_omit = None

    def __repr__(self):
        """Return string of information about the class."""
        ostring = f"countrate = \n{self.countrate}"
        ostring += f"\nchisq = \n{self.chisq}"
        ostring += f"\nucert = \n{self.uncert}"
        """
        ostring += f"\nweights = \n{self.weights}"

        ostring += f"\ncountrate_two_omit = \n{self.countrate_two_omit}"
        ostring += f"\nchisq_two_omit = \n{self.chisq_two_omit}"
        ostring += f"\nuncert_two_omit = \n{self.uncert_two_omit}"

        ostring += f"\ncountrate_one_omit = \n{self.countrate_one_omit}"
        ostring += f"\njumpval_one_omit = \n{self.jumpval_one_omit}"
        ostring += f"\njumpsig_one_omit = \n{self.jumpsig_one_omit}"
        ostring += f"\nchisq_one_omit = \n{self.chisq_one_omit}"
        ostring += f"\nuncert_one_omit = \n{self.uncert_one_omit}"
        """

        return ostring

    def fill_masked_reads(self, diffs2use):
        """
        Mask groups to use for ramp fitting.

        Replace countrates, uncertainties, and chi squared values that
        are NaN because resultant differences were doubly omitted.
        For these cases, revert to the corresponding values in with
        fewer omitted resultant differences to get the correct values
        without double-counting omissions.

        This function replaces the relevant entries of
        self.countrate_two_omit, self.chisq_two_omit,
        self.uncert_two_omit, self.countrate_one_omit, and
        self.chisq_one_omit in place.  It does not return a value.

        Parameters
        ----------
        diffs2use : ndarray
            A 2D array matching self.countrate_one_omit in shape with zero
            for resultant differences that were masked and one for
            differences that were not masked.
        """
        # replace entries that would be nan (from trying to
        # doubly exclude read differences) with the global fits.
        omit = diffs2use == 0
        ones = np.ones(diffs2use.shape)

        self.countrate_one_omit[omit] = (self.countrate * ones)[omit]
        self.chisq_one_omit[omit] = (self.chisq * ones)[omit]
        self.uncert_one_omit[omit] = (self.uncert * ones)[omit]

        omit = diffs2use[1:] == 0

        self.countrate_two_omit[omit] = (self.countrate_one_omit[:-1])[omit]
        self.chisq_two_omit[omit] = (self.chisq_one_omit[:-1])[omit]
        self.uncert_two_omit[omit] = (self.uncert_one_omit[:-1])[omit]

        omit = diffs2use[:-1] == 0

        self.countrate_two_omit[omit] = (self.countrate_one_omit[1:])[omit]
        self.chisq_two_omit[omit] = (self.chisq_one_omit[1:])[omit]
        self.uncert_two_omit[omit] = (self.uncert_one_omit[1:])[omit]


class Covar:
    """
    Covar class.

    class Covar holding read and photon noise components of alpha and
    beta and the time intervals between the resultant midpoints.
    """

    def __init__(self, readtimes):
        """
        Compute alpha and beta.

        These are the diagonal and off-diagonal elements of
        the covariance matrix of the resultant differences, and the time
        intervals between the resultant midpoints.

        Parameters
        ----------
        readtimes : list
            List of values or lists for the times of reads.  If a list of
            lists, times for reads that are averaged together to produce
            a resultant.
        """
        # Equations (4) and (11) in paper 1.
        mean_t, tau, n_reads, delta_t = self._compute_means_and_taus(readtimes)

        self.delta_t = delta_t
        self.mean_t = mean_t
        self.tau = tau
        self.Nreads = n_reads

        # Equations (28) and (29) in paper 1.
        self._compute_alphas_and_betas(mean_t, tau, n_reads, delta_t)

    def _compute_means_and_taus(self, readtimes):
        """
        Compute the means and taus of defined in EQNs 4 and 11 in paper 1.

        Parameters
        ----------
        readtimes : list
            List of values or lists for the times of reads.  If a list of
            lists, times for reads that are averaged together to produce
            a resultant.
        """
        mean_t = []  # mean time of the resultant as defined in the paper
        tau = []  # variance-weighted mean time of the resultant
        n_reads = []  # Number of reads per resultant

        for times in readtimes:
            mean_t.append(np.mean(times))

            if hasattr(times, "__len__"):
                # eqn 11
                length = len(times)
                n_reads.append(length)
                k = np.arange(1, length + 1)
                weight = (2 * length + 1) - (2 * k)
                tau.append(np.sum(weight * np.array(times)) / length**2)

            else:
                tau.append(times)
                n_reads.append(1)

        # readtimes is a list of lists, so mean_t is the list of each
        # mean of each list.
        mean_t = np.array(mean_t)
        tau = np.array(tau)
        n_reads = np.array(n_reads)
        delta_t = mean_t[1:] - mean_t[:-1]

        return mean_t, tau, n_reads, delta_t

    def _compute_alphas_and_betas(self, mean_t, tau, N, delta_t):  # noqa: N803
        """
        Compute the means and taus defined in EQNs 28 and 29 in paper 1.

        Parameters
        ----------
        mean_t : ndarray
            The means of the reads for each group.

        tau : ndarray
            Intermediate computation.

        N : ndarray
            The number of reads in each group.

        delta_t : ndarray
            The group differences of integration ramps.
        """
        self.alpha_readnoise = (1 / N[:-1] + 1 / N[1:]) / delta_t**2
        self.beta_readnoise = -1 / (N[1:-1] * delta_t[1:] * delta_t[:-1])

        self.alpha_phnoise = (tau[:-1] + tau[1:] - 2 * mean_t[:-1]) / delta_t**2
        self.beta_phnoise = (mean_t[1:-1] - tau[1:-1]) / (delta_t[1:] * delta_t[:-1])
