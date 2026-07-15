.. _jump_algorithm:

Algorithm
---------
This routine detects jumps by looking for outliers in the up-the-ramp signal
for each pixel.  On output, the GROUPDQ array is updated with the DQ flag
"JUMP_DET" to indicate the location of each jump that was found. In addition,
any pixels that have non-positive or NaN values in the gain reference file will
have DQ flags "NO_GAIN_VALUE" and "DO_NOT_USE" set in the output PIXELDQ array.
The SCI array of the input data is not modified.

Jumps can be detected in two different ways.  The primary way is the two-point
difference method described below.  The other way is by selecting ``only_use_ints``
as ``True`` and if there are enough integrations, then ``sigma_clip`` from the
``astropy.stats`` package will be used to detect jumps.  The ``sigma_clip`` method
will also be used if the total number of usable groups (number of groups per
integration multiplied by the number of integrations) is above a minimum threshold.

The current implementation uses the two-point difference method described
in `Anderson & Gordon (2011) <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1237A>`_.

Two-Point Difference Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The two-point difference method is applied as follows:

It is based on two conditions:

A. If ``only_use_ints`` is ``True``, then the number of integrations needs to be
   greater than ``minimum_sigclip_groups``, which has a default of 100.

B. If ``only_use_ints`` is ``False``, then the total number of available groupsi
   needs to be greater than ``minimum_sigclip_groups``.

If both A and B are false and there are not enough usable groups in each integration,
then the jump step is skipped.

The jump step runs using the first differences of groups in integrations runs as follows.

#. If the groups are evenly spaced in time and A or B is true, then:
    #. If A is true, then use ``astropy.stats.sigma_clip`` across integrations,
       using ``rejection_threshold`` for jump detection.
    #. If B is true, then use ``astropy.stats.sigma_clip`` across integrations
       and groups ``rejection_threshold`` for jump detection.
#. Else:
    #. If the minimum number of usable groups is greater than ``min_diffs_single_pass+1``.
       The ``+1`` is needed because first differences are used; ``min_diffs_single_pass``
       defaults to 10. Then detect jumps in each integration using 4 sigma outliers.
    #. Else, do an iterative flagging within each integration.

Note that any ramp groups flagged as SATURATED in the input GROUPDQ array
are not used in any of the above calculations and hence will never be
marked as containing a jump.
