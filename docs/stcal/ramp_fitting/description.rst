Description
===========

This step determines the mean count rate, in units of counts per second, for
each pixel by performing a linear fit to the data in the input file.  The fit
is done using the "ordinary least squares" method.
The fit is performed independently for each pixel.

The count rate for each pixel is determined by a linear fit to the
cosmic-ray-free and saturation-free ramp intervals for each pixel; hereafter
this interval will be referred to as a "segment." The fitting algorithm uses an
'optimal' weighting scheme, as described by Fixsen et al, PASP, 112, 1350.
Segments are determined using
the 4-D GROUPDQ array of the input data set, under the assumption that the jump
step will have already flagged CR's. Segments are terminated where
saturation flags are found. Pixels are processed simultaneously in blocks
using the array-based functionality of numpy.  The size of the block depends
on the image size and the number of groups.

.. _ramp_slopes_and_variances:

Slope and Variance Calculations
+++++++++++++++++++++++++++++++
Slopes and their variances are calculated for each segment, for each integration,
and for the entire exposure. As defined above, a segment is a set of contiguous
groups where none of the groups is saturated or cosmic ray-affected.  The
appropriate slopes and variances are output to the primary output product, the
integration-specific output product, and the optional output product. The
following is a description of these computations. The notation in the equations
is the following: the type of noise (when appropriate) will appear as the superscript
‘R’, ‘P’, or ‘C’ for readnoise, Poisson noise, or combined, respectively;
and the form of the data will appear as the subscript: ‘s’, ‘i’, ‘o’ for segment,
integration, or overall (for the entire dataset), respectively.

It is possible for an integration or pixel to have invalid data, so usable
slope data will not be available.  If a pixel has an invalid integration, the value
for that integration for that pixel will be set to NaN in the rateints product.
Further, if all integrations for a given pixel are invalid the pixel value for
the rate product will be set to NaN.  An example of invalid data would be a
fully saturated integration for a pixel.

Optimal Weighting Algorithm
---------------------------
The slope of each segment is calculated using the least-squares method with optimal
weighting, as described by
`Fixsen et al 2000 <https://ui.adsabs.harvard.edu/abs/2000PASP..112.1350F/abstract>`_
and Regan 2007, JWST-STScI-001212.
Optimal weighting determines the relative weighting of each sample
when calculating the least-squares fit to the ramp. When the data have low signal-to-noise
ratio :math:`S`, the data are read noise dominated and equal weighting of samples is the
best approach. In the high signal-to-noise regime, data are Poisson-noise dominated and
the least-squares fit is calculated with the first and last samples. In most practical
cases, the data will fall somewhere in between, where the weighting is scaled between the
two extremes.

The signal-to-noise ratio :math:`S` used for weighting selection is calculated from the
last sample as:

.. math::
    S = \frac{data \times gain} { \sqrt{(read\_noise)^2 + (data \times gain) } } \,,

The weighting for a sample :math:`i` is given as:

.. math::
    w_i = (i - i_{midpoint})^P \,,

where :math:`i_{midpoint}` is the the sample number of the midpoint of the sequence, and
:math:`P` is the exponent applied to weights, determined by the value of :math:`S`. Fixsen
et al. 2000 found that defining a small number of P values to apply to values of S was
sufficient; they are given as:

+-------------------+------------------------+----------+
| Minimum S         | Maximum S              | P        |
+===================+========================+==========+
| 0                 | 5                      | 0        |
+-------------------+------------------------+----------+
| 5                 | 10                     | 0.4      |
+-------------------+------------------------+----------+
| 10                | 20                     | 1        |
+-------------------+------------------------+----------+
| 20                | 50                     | 3        |
+-------------------+------------------------+----------+
| 50                | 100                    | 6        |
+-------------------+------------------------+----------+
| 100               |                        | 10       |
+-------------------+------------------------+----------+

Segment-specific Computations
-----------------------------
The variance of the slope of a segment due to read noise is:

.. math::
   var^R_{s} = \frac{12 \ R^2 }{ (ngroups_{s}^3 - ngroups_{s})(tgroup^2) } \,,

where :math:`R` is the noise in the difference between 2 frames,
:math:`ngroups_{s}` is the number of groups in the segment, and :math:`tgroup` is the group
time in seconds (from the keyword TGROUP).

The variance of the slope in a segment due to Poisson noise is:

.. math::
   var^P_{s} = \frac{ slope_{est} }{  tgroup \times gain\ (ngroups_{s} -1)}  \,,

where :math:`gain` is the gain for the pixel (from the GAIN reference file),
in e/DN. The :math:`slope_{est}` is an overall estimated slope of the pixel,
calculated by taking the median of the first differences of the groups that are
unaffected by saturation and cosmic rays, in all integrations. This is a more
robust estimate of the slope than the segment-specific slope, which may be noisy
for short segments.

The combined variance of the slope of a segment is the sum of the variances:

.. math::
   var^C_{s} = var^R_{s} + var^P_{s}


Integration-specific computations
---------------------------------
The variance of the slope for an integration due to read noise is:

.. math::
   var^R_{i} = \frac{1}{ \sum_{s} \frac{1}{ var^R_{s} }}  \,,

where the sum is over all segments in the integration.

The variance of the slope for an integration due to Poisson noise is:

.. math::
   var^P_{i} = \frac{1}{ \sum_{s} \frac{1}{ var^P_{s}}}

The combined variance of the slope for an integration due to both Poisson and read
noise is:

.. math::
   var^C_{i} = \frac{1}{ \sum_{s} \frac{1}{ var^R_{s} + var^P_{s}}}

The slope for an integration depends on the slope and the combined variance of each segment's slope:

.. math::
   slope_{i} = \frac{ \sum_{s}{ \frac{slope_{s}} {var^C_{s}}}} { \sum_{s}{ \frac{1} {var^C_{s}}}}

Exposure-level computations
---------------------------

The variance of the slope due to read noise depends on a sum over all integrations:

.. math::
   var^R_{o} = \frac{1}{ \sum_{i} \frac{1}{ var^R_{i}}}

The variance of the slope due to Poisson noise is:

.. math::
   var^P_{o} = \frac{1}{ \sum_{i} \frac{1}{ var^P_{i}}}

The combined variance of the slope is the sum of the variances:

.. math::
   var^C_{o} = var^R_{o} + var^P_{o}

The square-root of the combined variance is stored in the ERR array of the output product.

The overall slope depends on the slope and the combined variance of the slope of each integration's
segments, and hence is a sum over integrations and segments:

.. math::
    slope_{o} = \frac{ \sum_{i,s}{ \frac{slope_{i,s}} {var^C_{i,s}}}} { \sum_{i,s}{ \frac{1} {var^C_{i,s}}}}

