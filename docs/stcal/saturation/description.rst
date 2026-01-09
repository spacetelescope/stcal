Description
===========

:Alias: saturation

Saturation Checking
-------------------
The ``flag_saturated_pixels`` routine flags pixels at or below the A/D floor or above the
saturation threshold.  Pixel values are flagged as saturated if the pixel value
is larger than the defined saturation threshold.  Pixel values are flagged as
below the A/D floor if they have a value of zero DN or less.

The method loops over all integrations within an exposure, examining each one
group-by-group, comparing the pixel values in the ``data`` array with defined
saturation thresholds for each pixel. When it finds a pixel value in a given
group that is above the saturation threshold (high saturation), it sets the
"SATURATED" flag (as defined by the input ``dqflags`` dictionary)
in the corresponding location of the input ``gdq`` array.
When it finds a pixel in a given group that has a zero or
negative value (below the A/D floor), it sets the "AD_FLOOR" and "DO_NOT_USE"
flags in the corresponding location of the input ``gdq`` array.
exposure. For the saturation case, it also flags all subsequent groups for that
pixel as saturated. For example, if there are 10 groups in an integration and
group 7 is the first one to cross the saturation threshold for a given pixel,
then groups 7 through 10 will all be flagged for that pixel.

Pixels with thresholds set to NaN or flagged as "NO_SAT_CHECK" in the ``sat_dq`` array
have their thresholds set above the 16-bit A-to-D converter limit
of 65535 and hence will never be flagged as saturated.
The "NO_SAT_CHECK" flag is propagated to the pixel data quality
(``pdq``) array output to indicate which pixels fall into this category.

If the optional ``read_pattern`` input is provided, this method will use information about the
read pattern to find pixels that saturated in the middle of grouped data.  This
can be particularly important for flagging data that saturated during
the second group but did not trigger the normal saturation threshold due to the
grouped data averaging.  To trigger second group saturation in a pixel all three
of the following criteria must be met:

#. The count rate estimated from the first group is not expected to saturate by
   the third group (as estimated by the difference between the first group counts
   and the superbias if available), which may occur for bright sources.

#. The difference in counts between the first and second group is larger than the
   remaining counts needed to saturate divided by the number of frames in the 
   second group, i.e., the expected frame-averaged counts of a saturating signal 
   that occurs in the last frame of the group.

#. The third group is saturated.


Charge Migration
----------------
There is an effect in IR detectors that results in charge migrating (spilling)
from a pixel that has "hard" saturation (i.e. where the pixel no longer accumulates
charge) into neighboring pixels. This results in non-linearities in the accumulating
signal ramp in the neighboring pixels and hence the ramp data following the onset
of saturation is not usable.

The ``flag_saturated_pixels`` routine accounts for charge migration by flagging - as saturated -
all pixels neighboring a pixel that goes above the saturation threshold. This is
accomplished by first flagging all pixels that cross their saturation thresholds
and then making a second pass through the data to flag neighbors within a specified
region. The region of neighboring pixels is specified as a 2N+1 pixel wide box that
is centered on the saturating pixel and N is set by the input parameter
``n_pix_grow_sat``. The default value is 1, resulting in a 3x3 box of neighboring
pixels that will be flagged.