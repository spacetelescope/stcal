Algorithm
---------
This routine detects jumps in an exposure by looking for outliers
in the up-the-ramp signal for each pixel in each integration within
an input exposure. On output, the GROUPDQ array is updated with the DQ flag
"JUMP_DET" to indicate the location of each jump that was found.
In addition, any pixels that have non-positive or NaN values in the gain
reference file will have DQ flags "NO_GAIN_VALUE" and "DO_NOT_USE" set in the
output PIXELDQ array.
The SCI and ERR arrays of the input data are not modified.

The current implementation uses the two-point difference method described
in Anderson&Gordon2011_.

Two-Point Difference Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The two-point difference method is applied to each integration as follows:

* Compute the first differences for each pixel (the difference between
  adjacent groups)
* Compute the clipped (dropping the largest difference) median of the first differences for each pixel.
* Use the median to estimate the Poisson noise for each group and combine it
  with the read noise to arrive at an estimate of the total expected noise for
  each difference.
* Compute the "difference ratio" as the difference between the first differences
  of each group and the median, divided by the expected noise.
* If the largest "difference ratio" is greater than the rejection threshold,
  flag the group corresponding to that ratio as having a jump.
* If a jump is found in a given pixel, iterate the above steps with the
  jump-impacted group excluded, looking for additional lower-level jumps
  that still exceed the rejection threshold.
* Stop iterating on a given pixel when no new jumps are found or only one
  difference remains.
* If the there are only three differences (four groups), the standard median
  is used rather than the clipped median.
* If there are only two differences (three groups), the smallest one is compared to the larger
  one and if the larger one is above a threshold, it is flagged as a jump.

Note that any ramp values flagged as SATURATED in the input GROUPDQ array
are not used in any of the above calculations and hence will never be
marked as containing a jump.

.. _Anderson&Gordon2011: https://ui.adsabs.harvard.edu/abs/2011PASP..123.1237A
