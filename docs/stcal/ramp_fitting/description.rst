Description
===========

This step determines the mean count rate, in units of counts per second, for
each pixel by performing a linear fit to the data in the input file.  The default
method uses "ordinary least squares" based on the Fixsen fitting algorithm
described by
`Fixsen et al. (2011) <https://ui.adsabs.harvard.edu/abs/2000PASP..112.1350F>`_.

The count rate for each pixel is determined by a linear fit to the
cosmic-ray-free and saturation-free ramp intervals for each pixel; hereafter
this interval will be referred to as a "segment."

Segments are determined using
the 4-D GROUPDQ array of the input data set, under the assumption that the jump
step will have already flagged CR's. Segments are terminated where
saturation flags are found. Pixels are processed simultaneously in blocks
using the array-based functionality of numpy.  The size of the block depends
on the image size and the number of groups.


There is also a new algorithm available for testing, the likelihood
algorithm, implementing an algorithm based on the group differences
of a ramp.  See :ref:`likelihood algorithm <likelihood_algo>`.

.. _ramp_output_products:

Output Products
---------------

There are two output products created by default, with a third optional
product also available:

#. The primary output file ("rate") contains slope and variance/error
   estimates for each pixel that are the result of averaging over all
   integrations in the exposure. This is a product with 2-D data arrays.
#. The secondary product ("rateints") contains slope and variance/error
   estimates for each pixel on a per-integration basis, stored as 3-D
   data cubes.
#. The third, optional, output product contains detailed
   fit information for every ramp segment for each pixel.

RATE Product
++++++++++++
After computing the slopes and variances for all segments for a given pixel, the final slope is
determined as a weighted average from all segments in all integrations, and is
written to the "rate" output product.  In this output product, the
4-D GROUPDQ from all integrations is collapsed into 2-D, merged
(using a bitwise OR) with the input 2-D PIXELDQ, and stored as a 2-D DQ array.
The 3-D VAR_POISSON and VAR_RNOISE arrays from all integrations are averaged
into corresponding 2-D output arrays.  In cases where the median rate
for a pixel is negative, the VAR_POISSON is set to zero, in order to avoid the
unphysical situation of having a negative variance.

RATEINTS Product
++++++++++++++++
The slope images for each integration are stored as a data cube in "rateints" output data
product.  Each plane of the 3-D SCI, ERR, DQ, VAR_POISSON, and VAR_RNOISE
arrays in this product corresponds to the result for a given integration.  In this output
product, the GROUPDQ data for a given integration is collapsed into 2-D and then
merged with the input 2-D PIXELDQ array to create the output DQ array for each
integration. The 3-D VAR_POISSON and VAR_RNOISE arrays are
calculated by averaging over the fit segments in the corresponding 4-D
variance arrays.

FITOPT Product
++++++++++++++
A third, optional output product is also available and is produced only when
the step parameter ``save_opt`` is True (the default is False).  This optional
product contains 4-D arrays called SLOPE, SIGSLOPE, YINT, SIGYINT, WEIGHTS,
VAR_POISSON, and VAR_RNOISE, which contain the slopes, uncertainties in the
slopes, y-intercept, uncertainty in the y-intercept, fitting weights,
variance of the slope due to poisson noise, and the variance of the slope
due to read noise for each segment of each pixel, respectively. The y-intercept refers
to the result of the fit at an effective exposure time of zero.  This product also
contains a 3-D array called PEDESTAL, which gives the signal at zero exposure
time for each pixel, and the 4-D CRMAG array, which contains the magnitude of
each group that was flagged as having a CR hit.

By default, the name of this
output file will have the product type suffix "_fitopt".
In this optional output product, the pedestal array is
calculated for each integration by extrapolating the final slope (the weighted
average of the slopes of all ramp segments in the integration) for each pixel
from its value at the first group to an exposure time of zero. Any pixel that is
saturated on the first group is given a pedestal value of 0. Before compression,
the cosmic-ray magnitude array is equivalent to the input SCI array but with the
only nonzero values being those whose pixel locations are flagged in the input
GROUPDQ as cosmic ray hits. The array is compressed, removing all groups in
which all the values are 0 for pixels having at least one group with a non-zero
magnitude. The order of the cosmic rays within the ramp is preserved.

.. _ramp_special_cases:

Special Cases
-------------
If the input dataset has only one group in each integration (NGROUPS=1), the count rate
for all unsaturated pixels in each integration will be calculated as the
value of the science data in the one group divided by the group time.  If the
input dataset has only two groups per integration (NGROUPS=2), the count rate for all
unsaturated pixels in each integration will be calculated using the differences
between the two valid groups of the science data divided by the group time.

For datasets having more than one group in each integration (NGROUPS>1), a ramp having
a segment with only one good group is processed differently depending on the
number and size of the other segments in the ramp. If a ramp has only one
segment and that segment contains a single group, the count rate will be calculated
to be the value of the science data in that group divided by the group time.  If a ramp
has a segment with only one good group, and at least one other segment having more
than one good group, only data from the segment(s) having more than one
good group will be used to calculate the count rate.

For ramps in a given integration that are saturated beginning in their second group,
the count rate for that integration will be calculated as the value of the science data
in the first group divided by the group time, but only if the step parameter
``suppress_one_group`` is set to ``False``. If set to ``True``, the computation of
slopes for pixels that have only one good group will be suppressed and the slope
for that integration will be set to zero.

.. _ramp_slopes_and_variances:

Slope and Variance Calculations
-------------------------------
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
+++++++++++++++++++++++++++
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


For segment :math:`k` of length :math:`n`, which includes groups :math:`[g_{k}, ...,
g_{k+n-1}]`, the signal-to-noise ratio :math:`S` used for weighting selection is
calculated from the last sample as:

.. math::
    S = \frac{data \times gain} { \sqrt{(read\_noise)^2 + (data \times gain) } } \,,

where :math:`data = g_{k+n-1} - g_{k}`.

The weighting for a sample :math:`i` is given as:

.. math::
    w_i = \frac{ [(i - i_{midpoint}) / i_{midpoint}]^P }{ (read\_noise)^2 } \,,

where  :math:`i_{midpoint} = \frac{n-1}{2}` and :math:`i = 0, 1, ..., n-1`.


is the the sample number of the midpoint of the sequence, and :math:`P` is the exponent
applied to weights, determined by the value of :math:`S`. Fixsen et al. 2000 found that
defining a small number of P values to apply to values of S was sufficient; they are given as:

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
+++++++++++++++++++++++++++++
The variance of the slope of a segment due to read noise is:

.. math::  
   var^R_{s} = \frac{12 \ R^2 }{ (ngroups_{s}^3 - ngroups_{s})(tgroup^2)(gain^2) } \,,

where :math:`R` is the noise in the difference between 2 frames, 
:math:`ngroups_{s}` is the number of groups in the segment, and :math:`tgroup` is the group 
time in seconds (from the keyword TGROUP).  The divide by gain converts to
:math:`DN`.  For the special case where as segment has length 1, the
:math:`ngroups_{s}` is set to :math:`2`.

The variance of the slope in a segment due to Poisson noise is:

.. math::
   var^P_{s} = \frac{ slope_{est} + darkcurrent}{  tgroup \times gain\ (ngroups_{s} -1)}  \,,

where :math:`gain` is the gain for the pixel (from the GAIN reference file),
in e/DN. The :math:`slope_{est}` is an overall estimated slope of the pixel,
calculated by taking the median of the first differences of the groups that are
unaffected by saturation and cosmic rays, in all integrations. This is a more
robust estimate of the slope than the segment-specific slope, which may be noisy
for short segments. The contributions from the dark current are added when present;
the value can be provided by the user during the `jwst.dark_current.DarkCurrentStep`,
or it can be specified in scalar or 2D array form by the dark reference file.

The combined variance of the slope of a segment is the sum of the variances:

.. math::
   var^C_{s} = var^R_{s} + var^P_{s}


Integration-specific computations
+++++++++++++++++++++++++++++++++
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
+++++++++++++++++++++++++++

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
segments, so is a sum over integration values computed from the segements:

.. math::    
    slope_{o} = \frac{ \sum_{i}{ \frac{slope_{i}} {var^C_{i}}}} { \sum_{i}{ \frac{1} {var^C_{i}}}}


.. _ramp_error_propagation:

Error Propagation
-----------------
Error propagation in the ``ramp_fitting`` step is implemented by carrying along
the individual variances in the slope due to Poisson noise and read noise at all
levels of calculations. The total error estimate at each level is computed as
the square-root of the sum of the two variance estimates.

In each type of output product generated by the step, the variance in the slope
due to Poisson noise is stored in the "VAR_POISSON" extension, the variance in
the slope due to read noise is stored in the "VAR_RNOISE" extension, and the
total error is stored in the "ERR" extension. In the optional output product,
these arrays contain information for every segment used in the fitting for each
pixel. In the "rateints" product they contain values for each integration, and
in the "rate" product they contain values for the exposure as a whole.

.. _ramp_dq_propagation:

Data Quality Propagation
------------------------
For a given pixel, if all groups in an integration are flagged as DO_NOT_USE or
SATURATED, then that pixel will be flagged as DO_NOT_USE in the corresponding
integration in the "rateints" product.  Note this does NOT mean that all groups
are flagged as SATURATED, nor that all groups are flagged as DO_NOT_USE.  For
example, slope calculations that are suppressed due to a ramp containing only
one good group will be flagged as DO_NOT_USE in the
first group, but not necessarily any other group, while only groups two and
beyond are flagged as SATURATED.  Further, only if all integrations in the "rateints"
product are flagged as DO_NOT_USE, then the pixel will be flagged as DO_NOT_USE
in the "rate" product.

For a given pixel, if all groups in an integration are flagged as SATURATED,
then that pixel will be flagged as SATURATED and DO_NOT_USE in the corresponding
integration in the "rateints" product.  This is different from the above case in
that this is only for all groups flagged as SATURATED, not for some combination
of DO_NOT_USE and SATURATED.  Further, only if all integrations in the "rateints"
product are flagged as SATURATED, then the pixel will be flagged as SATURATED
and DO_NOT_USE in the "rate" product.

For a given pixel, if any group in an integration is flagged as JUMP_DET, then
that pixel will be flagged as JUMP_DET in the corresponding integration in the
"rateints" product.  That pixel will also be flagged as JUMP_DET in the "rate"
product.

.. _likelihood_algo:

Likelihood Algorithm Details
----------------------------
As an alternative to the OLS algorithm, a likelihood algorithm can be selected
with the step argument ``--ramp_fitting.algorithm=LIKELY``.  This algorithm has
its own algorithm for jump detection that augments anything identified by
the regular jump detection step.
The LIKELY algorithm requires a minimum of four (4) NGROUPS.  If the LIKELY
algorithm is selected for data with NGROUPS less than four, the ramp fitting
algorithm is changed to OLS_C.

Each pixel is independently processed, but rather than operate on each
group/resultant directly, the likelihood algorithm is based on differences of
the groups/resultants :math:`d_i = r_i - r_{i-1}`.  The model used to determine
the slope/countrate, :math:`a`, is:

.. math::    
    \chi^2 = ({\bf d} - a \cdot {\bf 1})^T C ({\bf d} - a \cdot {\bf 1}) \,,

Differentiating, setting to zero, then solving for :math:`a` results in 

.. math::    
    a = ({\bf 1}^T C {\bf d})({\bf 1}^T C {\bf 1})^T \,,

The covariance matrix :math:`C` is a tridiagonal matrix, due to the nature of the
differences.  Because the covariance matrix is tridiagonal, the  computational
complexity reduces from :math:`O(n^3)` to :math:`O(n)`.  To see the detailed
derivation and computations implemented, refer to the links above.
The Poisson and read noise  computations are based on equations (27) and (28), in
`Brandt (2024) <https://iopscience.iop.org/article/10.1088/1538-3873/ad38d9>`__

For more details, especially for the jump detection portion in the liklihood
algorithm, see
`Brandt (2024) <https://iopscience.iop.org/article/10.1088/1538-3873/ad38da>`__.
