"""
`skystatistics` module provides statistics computation class.

Used by :py:func:`~stcal.skymatch.skymatch.skymatch`
and :py:class:`~stcal.skymatch.skyimage.SkyImage`.
"""

from stsci.imagestats import ImageStats

__all__ = ["SkyStats"]


class SkyStats:
    """
    Class built on top of :py:class:`stsci.imagestats.ImageStats`.

    Deligates its functionality to calls to the ``ImageStats`` object. Compared
    to :py:class:`stsci.imagestats.ImageStats`, `SkyStats` has "persistent settings"
    in the sense that object's parameters need to be set once and these settings
    will be applied to all subsequent computations on different data.

    """

    def __init__(self, skystat="mean", lower=None, upper=None, nclip=5, lsig=4.0, usig=4.0, binwidth=0.1):
        """Initialize the SkyStats object.

        Parameters
        ----------
        skystat : optional
            possible values are 'mean', 'median', 'mode', 'midpt".
            Sets the statistics that will be returned by `~stcal.skymatch.skystatistics.SkyStats.calc_sky`.
            The following statistics are supported: 'mean', 'mode', 'midpt',
            and 'median'. First three statistics have the same meaning as in
            `stsdas.toolbox.imgtools.gstatistics <http://stsdas.stsci.edu/\
cgi-bin/gethelp.cgi?gstatistics>`_
            while 'median' will compute the median of the distribution.

        lower : float, None, optional
            Lower limit of usable pixel values for computing the sky.
            This value should be specified in the units of the input image(s).

        upper : float, None, optional
            Upper limit of usable pixel values for computing the sky.
            This value should be specified in the units of the input image(s).

        nclip : int, optional
            A non-negative number of clipping iterations to use when computing
            the sky value.

        lsig : float, optional
            Lower clipping limit, in sigma, used when computing the sky value.

        usig : float, optional
            Upper clipping limit, in sigma, used when computing the sky value.

        binwidth : float, optional
            Bin width, in sigma, used to sample the distribution of pixel
            brightness values in order to compute the sky background
            statistics.
        """
        self.npix = None
        self.skyval = None

        self._kwargs = {
            "fields": f"npix,{skystat}",
            "lower": lower,
            "upper": upper,
            "nclip": nclip,
            "lsig": lsig,
            "usig": usig,
            "binwidth": binwidth,
        }

        self._skystat = skystat

    def calc_sky(self, data):
        """Compute statistics on data.

        Parameters
        ----------
        data : numpy.ndarray
            A numpy array of values for which the statistics needs to be
            computed.

        Returns
        -------
        statistics : tuple
            A tuple of two values: (`skyvalue`, `npix`), where `skyvalue` is
            the statistics specified by the `skystat` parameter during the
            initialization of the `SkyStats` object and `npix` is the number
            of pixels used in computing the statistics reported in `skyvalue`.

        """
        imstat = ImageStats(image=data, **(self._kwargs))
        return getattr(imstat, self._skystat), imstat.npix

    def __call__(self, data):  # noqa: D102
        return self.calc_sky(data)
