.. _jump_algorithm:

Algorithm
---------
This routine detects jumps by looking for outliers in the up-the-ramp signal
for each pixel.  On output, the GROUPDQ array is updated with the DQ flag
"JUMP_DET" to indicate the location of each jump that was found. In addition,
any pixels that have non-positive or NaN values in the gain reference file will
have DQ flags "NO_GAIN_VALUE" and "DO_NOT_USE" set in the output PIXELDQ array.
The SCI array of the input data is not modified.

Jumps in the ramps of a given pixel are detected using statistics of the
two-point differences between successive groups to identify outlying values.
Depending on the ramp length and parameter configuration one of four different
methods can be used.

1) Astropy sigma clipping across integrations for each group difference
in the ramp (e.g., sigma clip groups 3-2 for all integrations, then sigma clip
groups 4-3 for all integrations, etc).  The appropriate value of sigma to use is
determined empirically from the ensemble of group differences.

2) Astropy sigma clipping across all group differences and all integrations
simultaneously (e.g., treat all group differences within an integration and in other
integrations equally).  The appropriate value of sigma to use is
determined empirically from the ensemble of group differences.

3) Sigma clipping between all group differences within a given integration using
a single pass of the method described by 
`Anderson & Gordon (2011) <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1237A>`_
(see below).  The appropriate value of sigma to use is determined
using the estimated read noise plus poisson noise for each pixel.

4) Sigma clipping using the
`Anderson & Gordon (2011) <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1237A>`_
method in which the median and rejection parameters are iteratively recalculated
after each group difference is rejected.
This is an iterative approach that loops over all
first group differences, :math:`(ngroups-1) * nints`, where :math:`ngroups` is
the number of groups in each integration (the :math:`-1` is used because the
operations are on the first differences) and :math:`nints` is the number of
integrations.

Method 1 is used if ``only_use_ints`` is ``True`` and the number of usable integrations
is greater than ``minimum_sigclip_groups``, which has a default of 100.  This is thus
the method typically used for many time-series observations.

Method 2 is used if ``only_use_ints`` is ``False`` and the number of usable differences
across all groups and integrations (i.e., approximately ngroups*nints)
is greater than ``minimum_sigclip_groups``.  This is not typically used by the default
pipeline.

Method 3 is used if neither Methods 1 or 2 were selected, but there are a sufficient number
of usable group differences in each integration (``min_diffs_single_pass`` is 10 by default)
to find all outliers in a single calculation.  This is the default method used
by the pipeline for most non-time-series observations in which ngroups is greater than 10.

Method 4 is used if neither Methods 1 or 2 were selected, and there are too few usable
group differences in each integration to find all outliers at once.
This method is more robust for short ramps, although the iterative
rejection increases the step runtime.

In all cases, if flagging of the 4 neighbors is requested, then the 4 adjacent pixels will
have ramp jumps flagged in the same group as the central pixel as long as it has
a jump between the min and max requested levels for this option.
Likewise, if flagging of groups after a ramp jump is requested, then the groups in the
requested time since a detected ramp jump will be flagged as ramp jumps if
the ramp jump is above the requested threshold.  Two thresholds and times are
possible for this option.
Note that any ramp groups flagged as SATURATED in the input GROUPDQ array
are not used in any of the above calculations and hence will never be
marked as containing a jump.

If the ramps are extremely short with the number of usable groups less than
``minimum_groups`` (default value of 3) no jump detection is performed.

Anderson & Gordon Method
^^^^^^^^^^^^^^^^^^^^^^^^

The full iterative method described by `Anderson & Gordon (2011) <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1237A>`_ is as follows:

#. Compute the first differences for each pixel (the difference between
   adjacent groups)
#. Compute the clipped median (dropping the largest difference) of the first
   differences for each pixel. If there are only three first difference values
   (four groups), no clipping is performed when computing the median.
#. Use the median to estimate the Poisson noise for each group and combine it
   with the read noise to arrive at an estimate of the total expected noise for
   each difference.
#. Compute the "difference ratio" as the difference between the first differences
   of each group and the median, divided by the expected noise.
#. If the largest "difference ratio" is greater than the rejection threshold,
   flag the group corresponding to that ratio as having a jump.
#. If a jump is found in a given pixel, iterate the above steps with the
   jump-impacted group excluded, looking for additional lower-level jumps
   that still exceed the rejection threshold.
#. Stop iterating on a given pixel when no new jumps are found or only one
   difference remains.
#. If there are only two differences (three groups), the smallest one is compared
   to the larger one and if the larger one is above a threshold, it is flagged
   as a jump.

Snowball Detection
^^^^^^^^^^^^^^^^^^

To identify a snowball, use the algorithm below to find a contiguous block of newly
saturated pixels, then compute an enclosing ellipse for the block of pixels. Refer
to `Regan (2024) <https://www.stsci.edu/files/live/sites/www/files/home/jwst/documentation/technical-documents/_documents/JWST-STScI-008545.pdf>`_ for
more detail.


#. For each group plane in an integration, find newly saturated pixels.

#. Find contiguous saturated pixels with ``min_sat_area``, default=1.0. Then solve
   for minimum enclosing ellipses.

#. Find contiguous jump detected pixels with ``min_jump_area``, default=5.0. Then
   solve for minimum enclosing ellipses.

#. For each jump ellipses that has a newly saturated pixel at the center, add the
   jump ellipse parameters to the list of snowballs. Using ``edge_size`` for jump
   ellipses close to the edge, the saturated center requirement is removed.

#. For saturated ellipses with minor axis > ``min_sat_extend``, extend the minor
   axis for saturationg by ``sat_expand``.

#. For jump ellipses with minor axis > ``min_sat_extend``, extend the minor axis for
   saturationg by expand_factor, then expand major axis by same number of pixels.

#. Limit expansion by ``max_extended_radius``.
