1.2.2 (2022-12-01)
==================

General
-------

- Moved build configuration from ``setup.cfg`` to ``pyproject.toml`` to support PEP621 [#95]
- made dependency on ``opencv-python`` conditional [#126]

ramp_fitting
~~~~~~~~~~~~

- Set saturation flag only for full saturation.  The rateints product will
  have the saturation flag set for an integration only if saturation starts
  in group 0.  The rate product will have the saturation flag set only if
  each integration for a pixel is marked as fully saturated. [#125]

1.2.1 (2022-10-14)
==================

Bug Fixes
---------

jump
~~~~
- Changes to limit the expansion of MIRI shower ellipses to be the same
  number of pixels for both the major and minor axis. JP-2944 [#123]

1.2.0 (2022-10-07)
==================

Bug Fixes
---------

dark_current
~~~~~~~~~~~~

- Bug fix for computation of the total number of frames when science data
  use on-board frame averaging and/or group gaps. [#121]

jump
~~~~

- Changes to flag both NIR snowballs and MIRI showers
  for  JP-#2645. [#118]

- Early in the step, the object arrays are converted from DN to electrons
  by multiplying by the gain. The values need to be reverted back to DN
  at the end of the step. [#116]

1.1.0 (2022-08-17)
==================

General
-------

- Made style changes due to the new 5.0.3 version of flake8, which
  noted many missing white spaces after keywords. [#114]

Bug Fixes
---------

ramp_fitting
~~~~~~~~~~~~

- Updating multi-integration processing to correctly combine multiple
  integration computations for the final image information. [#108]

- Fixed crash due to two group ramps with saturated groups that used
  an intermediate array with an incorrect shape. [#109]

- Updating how NaNs and DO_NOT_USE flags are handled in the rateints
  product. [#112]

- Updating how GLS handles bad gain values.  NaNs and negative gain
  values have the DO_NOT_USE and NO_GAIN_VALUE flag set.  Any NaNs
  found in the image data are set to 0.0 and the corresponding DQ flag
  is set to DO_NOT_USE. [#115]

Changes to API
--------------

jump
~~~~

 - Added flagging after detected ramp jumps based on two DN thresholds and
   two number of groups to flag [#110]

1.0.0 (2022-06-24)
==================

Bug Fixes
---------

ramp_fitting
~~~~~~~~~~~~

- Adding special case handler for GLS to handle one group ramps. [#97]

- Updating how one group suppression and ZEROFRAME processing works with
  multiprocessing, as well as fixing the multiprocessing failure. [#99]

- Changing how ramp fitting handles fully saturated ramps. [#102]

saturation
~~~~~~~~~~

- Modified the saturation threshold applied to pixels flagged with
  NO_SAT_CHECK, so that they never get flagged as saturated. [#106]

Changes to API
--------------

ramp_fitting
~~~~~~~~~~~~

- The tuple ``integ_info`` no longer returns ``int_times`` as a part of it,
  so the tuple is one element shorter. [#99]

- For fully saturated exposures, all returned values are ``None``, instead
  of tuples. [#102]

saturation
~~~~~~~~~~~

- Changing parameter name in twopoint_difference from 'normal_rej_thresh' to rejection_thresh' for consistency. [#105]

Other
-----

general
~~~~~~~

- Update CI workflows to cache test environments and depend upon style and security checks [#96]
- Increased required ``Python`` version from ``>=3.7`` to ``>=3.8`` (to align with ``astropy``) [#98]

0.7.3 (2022-05-20)
==================

jump
~~~~

- Update ``twopoint_difference.py`` [#90]

ramp_fitting
~~~~~~~~~~~~

- Updating the one good group ramp suppression handler works. [#92]

0.7.2 (2022-05-19)
==================

ramp_fitting
~~~~~~~~~~~~

- Fix for accessing zero-frame in model to account for Roman data not using
  zero-frame. [#89]


0.7.1 (2022-05-16)
==================

jump
~~~~
- Enable multiprocessing for jump detection, which is controlled by the 'max_cores' parameter. [#87]

0.7.0 (2022-05-13)
==================

linearity
~~~~~~~~~
- Added functionality to linearly process ZEROFRAME data the same way
  as the SCI data. [#81]

ramp_fitting
~~~~~~~~~~~~
- Added functionality to use ZEROFRAME data in place of group 0 data
  for ramps that are fully saturated, but still have good ZEROFRAME
  data. [#81]

saturation
~~~~~~~~~~
- Added functionality to process ZEROFRAME data for saturation the same
  way as the SCI data. [#81]


0.6.4 (2022-05-02)
==================

saturation
~~~~~~~~~~

- Added in functionality to deal with charge spilling from saturated pixels onto neighboring pixels [#83]

0.6.3 (2022-04-27)
==================

- Pin astropy min version to 5.0.4. [#82]

- Fix for jumps in first good group after dropping groups [#84]


0.6.2 (22-03-29)
================

jump
~~~~
- Neighboring pixels with 'SATURATION' or 'DONOTUSE' flags are no longer flagged as jumps. [#79]

ramp_fitting
~~~~~~~~~~~~

- Adding feature to use ZEROFRAME for ramps that are fully saturated, but
  the ZEROFRAME data for that ramp is good. [#81]

0.6.1 (22-03-04)
================

ramp_fitting
~~~~~~~~~~~~

- Adding feature to suppress calculations for saturated ramps having only
  the 0th group be a good group.  [#76]

0.6.0 (22-01-14)
================

ramp_fitting
~~~~~~~~~~~~

- Adding GLS code back to ramp fitting. [#64]

jump
~~~~

- Fix issue in jump detection that occured when there were only 2 usable
  differences with no other groups flagged. This PR also added tests and
  fixed some of the logging statements in twopoint difference. [#74]

0.5.1 (2022-01-07)
==================

jump
~~~~

- fixes to several existing errors in the jump detection step. added additional
  tests to ensure step is no longer flagging jumps for pixels with only two
  usable groups / one usable diff. [#72]

0.5.0 (2021-12-28)
==================

dark_current
~~~~~~~~~~~~

- Moved dark current code from JWST to STCAL. [#63]

0.4.3 (2021-12-27)
==================

linearity
~~~~~~~~~
- Let software set the pixel dq flag to NO_LIN_CORR if linear term of linearity coefficient is zero. [#65]

ramp_fitting
~~~~~~~~~~~~

- Fix special handling for 2 group ramp. [#70]

- Fix issue with inappropriately including a flagged group at the beginning
  of a ramp segment. [#68]

- Changed Ramp Fitting Documentation [#61]

0.4.2 (2021-10-28)
==================

ramp_fitting
~~~~~~~~~~~~

- For slopes with negative median rates, the Poisson variance is zero. [#59]

- Changed the way the final DQ array gets computed when handling the DO_NOT_USE
  flag for multi-integration data. [#60]

0.4.1 (2021-10-14)
==================

jump_detection
~~~~~~~~~~~~~~

- Reverts "Fix issue with flagging for MIRI three and four group integrations. [#44]


0.4.0 (2021-10-13)
==================

jump_detection
~~~~~~~~~~~~~~

- Fix issue with flagging for MIRI three and four group integrations. [#44]

linearity
~~~~~~~~~

- Adds common code for linearity correction [#55]

ramp_fitting
~~~~~~~~~~~~

- Global DQ variable removed [#54]

0.3.0 (2021-09-28)
==================

saturation
~~~~~~~~~~

- Adds common code for saturation [#39]


0.2.5 (2021-08-27)
==================

jump
~~~~

- added tests for two point difference [#37]

ramp_fitting
~~~~~~~~~~~~

- Adds support for Roman ramp data. [#43] [#49]

0.2.4 (2021-08-26)
==================

Workaround for setuptools_scm issues with recent versions of pip. [#45]


0.2.3 (2021-08-06)
==================

jump
~~~~
- documentation changes + docs for jump detection [#14]

ramp_fitting
~~~~~~~~~~~~

- Fix ramp fitting multiprocessing. [#30]


0.2.2 (2021-07-19)
==================

jump
~~~~

- Move common ``jump`` code to stcal [#27]

ramp_fitting
~~~~~~~~~~~~

- Implemented multiprocessing for OLS. [#30]
- Added DQ flag parameter to `ramp_fit` [#25]
- Reduced data model dependency [#26]

0.2.1 (2021-05-20)
==================

ramp_fitting
~~~~~~~~~~~~

- Fixed bug for median ramp rate computation in report JP-1950. [#12]


0.2.0 (2021-05-18)
==================

ramp_fitting
~~~~~~~~~~~~

- Added ramp fitting code [#6]


0.1.0 (2021-03-19)
==================

- Added code to manipulate bitmasks.
