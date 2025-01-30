import numpy as np
from scipy import special


class IntegInfo:
    """
    Storage for the integration information for ramp fitting computations.
    """
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
        """
        Arrange output arrays as a tuple, which the ramp fit step expects.
        """
        return (self.data, self.dq, self.var_poisson, self.var_rnoise, self.err)

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


class RampResult:
    def __init__(self):
        """
        Contains the ramp fitting results.
        """
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
        """
        Return string of information about the class.
        """
        ostring = f"countrate = \n{self.countrate}"
        ostring += f"\nchisq = \n{self.chisq}"
        ostring += f"\nucert = \n{self.uncert}"
        '''
        ostring += f"\nweights = \n{self.weights}"

        ostring += f"\ncountrate_two_omit = \n{self.countrate_two_omit}"
        ostring += f"\nchisq_two_omit = \n{self.chisq_two_omit}"
        ostring += f"\nuncert_two_omit = \n{self.uncert_two_omit}"

        ostring += f"\ncountrate_one_omit = \n{self.countrate_one_omit}"
        ostring += f"\njumpval_one_omit = \n{self.jumpval_one_omit}"
        ostring += f"\njumpsig_one_omit = \n{self.jumpsig_one_omit}"
        ostring += f"\nchisq_one_omit = \n{self.chisq_one_omit}"
        ostring += f"\nuncert_one_omit = \n{self.uncert_one_omit}"
        '''

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
    class Covar holding read and photon noise components of alpha and
    beta and the time intervals between the resultant midpoints
    """
    def __init__(self, readtimes):
        """
        Compute alpha and beta, the diagonal and off-diagonal elements of
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
        mean_t, tau, N, delta_t = self._compute_means_and_taus(readtimes)

        self.delta_t = delta_t
        self.mean_t = mean_t
        self.tau = tau
        self.Nreads = N

        # Equations (28) and (29) in paper 1.
        self._compute_alphas_and_betas(mean_t, tau, N, delta_t)

    def _compute_means_and_taus(self, readtimes):
        """
        Computes the means and taus of defined in EQNs 4 and 11 in paper 1.

        Parameters
        ----------
        readtimes : list
            List of values or lists for the times of reads.  If a list of
            lists, times for reads that are averaged together to produce
            a resultant.
        """
        mean_t = []  # mean time of the resultant as defined in the paper
        tau = []  # variance-weighted mean time of the resultant
        N = []  # Number of reads per resultant

        for times in readtimes:
            mean_t += [np.mean(times)]

            if hasattr(times, "__len__"):
                # eqn 11
                N += [len(times)]
                k = np.arange(1, N[-1] + 1)
                if False:
                    tau += [
                        1
                        / N[-1] ** 2
                        * np.sum((2 * N[-1] + 1 - 2 * k) * np.array(times))
                    ]
                    # tau += [(np.sum((2*N[-1] + 1 - 2*k)*np.array(times))) / N[-1]**2]
                else:
                    length = N[-1]
                    tmp0 = (2 * length + 1) - (2 * k)
                    sm = np.sum(tmp0 * np.array(times))
                    tmp = sm / length**2
                    tau.append(tmp)
            else:
                tau += [times]
                N += [1]

        # readtimes is a list of lists, so mean_t is the list of each
        # mean of each list.
        mean_t = np.array(mean_t)
        tau = np.array(tau)
        N = np.array(N)
        delta_t = mean_t[1:] - mean_t[:-1]

        return mean_t, tau, N, delta_t

    def _compute_alphas_and_betas(self, mean_t, tau, N, delta_t):
        """
        Computes the means and taus defined in EQNs 28 and 29 in paper 1.

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

    def calc_bias(self, countrates, sig, cvec, da=1e-7):
        """
        Calculate the bias in the best-fit count rate from estimating the covariance matrix.

        Section 5 of paper 1.

        Arguments:
        Parameters
        ----------
        countrates : ndarray
            Array of count rates at which the bias is desired.

        sig : float
            Single read noise.

        cvec : ndarray
            Weight vector on resultant differences for initial estimation
            of count rate for the covariance matrix. Will be renormalized
            inside this function.

        da : float
            Fraction of the count rate plus sig**2 to use for finite difference
            estimate of the derivative.  Optional parameter.  Default 1e-7.

        Returns
        -------
        bias : ndarray
            Bias of the best-fit count rate from using cvec plus the observed
            resultants to estimate the covariance matrix.
        """
        raise NotImplementedError('Bias calculations are not implemented.')

        alpha = countrates[np.newaxis, :] * self.alpha_phnoise[:, np.newaxis]
        alpha += sig**2 * self.alpha_readnoise[:, np.newaxis]
        beta = countrates[np.newaxis, :] * self.beta_phnoise[:, np.newaxis]
        beta += sig**2 * self.beta_readnoise[:, np.newaxis]

        # we only want the weights; it doesn't matter what the count rates are.
        n = alpha.shape[0]
        z = np.zeros((len(cvec), len(countrates)))
        result_low_a = fit_ramps(z, self, sig, countrateguess=countrates)

        # try to avoid problems with roundoff error
        da_incr = da * (countrates[np.newaxis, :] + sig**2)

        dalpha = da_incr * self.alpha_phnoise[:, np.newaxis]
        dbeta = da_incr * self.beta_phnoise[:, np.newaxis]
        result_high_a = fit_ramps(z, self, sig, countrateguess=countrates + da_incr)
        # finite difference approximation to dw/da

        dw_da = (result_high_a.weights - result_low_a.weights) / da_incr

        bias = np.zeros(len(countrates))
        c = cvec / np.sum(cvec)

        for i in range(len(countrates)):

            C = np.zeros((n, n))
            for j in range(n):
                C[j, j] = alpha[j, i]
            for j in range(n - 1):
                C[j + 1, j] = C[j, j + 1] = beta[j, i]

            bias[i] = np.linalg.multi_dot([c[np.newaxis, :], C, dw_da[:, i]])

            sig_a = np.sqrt(
                np.linalg.multi_dot([c[np.newaxis, :], C, c[:, np.newaxis]])
            )
            bias[i] *= 0.5 * (1 + special.erf(countrates[i] / sig_a / 2**0.5))

        return bias
