.. _jump_algorithm:

Algorithm
---------
This routine detects jumps by looking for outliers
in the up-the-ramp signal for each pixel in each integration within
an input exposure. On output, the GROUPDQ array is updated with the DQ flag
"JUMP_DET" to indicate the location of each jump that was found.
In addition, any pixels that have non-positive or NaN values in the gain
reference file will have DQ flags "NO_GAIN_VALUE" and "DO_NOT_USE" set in the
output PIXELDQ array.
The SCI array of the input data is not modified.

The current implementation uses the two-point difference method described
in `Anderson & Gordon (2011) <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1237A>`_.

Two-Point Difference Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The two-point difference method is applied to each integration as follows:

#. Compute the first differences for each pixel (the difference between
   adjacent groups)
#. Compute the clipped median (dropping the largest difference) of the first differences for each pixel.
   If there are only three first difference values (four groups), no clipping is
   performed when computing the median.
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
#. If there are only two differences (three groups), the smallest one is compared to the larger
   one and if the larger one is above a threshold, it is flagged as a jump.
#. If flagging of the 4 neighbors is requested, then the 4 adjacent pixels will
   have ramp jumps flagged in the same group as the central pixel as long as it has
   a jump between the min and max requested levels for this option.
#. If flagging of groups after a ramp jump is requested, then the groups in the
   requested time since a detected ramp jump will be flagged as ramp jumps if
   the ramp jump is above the requested threshold.  Two thresholds and times are
   possible for this option.

Note that any ramp groups flagged as SATURATED in the input GROUPDQ array
are not used in any of the above calculations and hence will never be
marked as containing a jump.
