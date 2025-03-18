1.12.0 (2025-03-18)
===================

Changes to API
--------------

- Added ``resample`` submodule. (`#320
  <https://github.com/spacetelescope/stcal/issues/320>`_)
- Refactoring the jump module to improve memory consumption, readability, and
  maintenance. (`#330 <https://github.com/spacetelescope/stcal/issues/330>`_)
- Removed all references to the unused ramp error array in dark, jump, and ramp
  fitting steps. (`#334 <https://github.com/spacetelescope/stcal/issues/334>`_)
- Make sregion_to_footprint part of public API (`#345
  <https://github.com/spacetelescope/stcal/issues/345>`_)


Bug Fixes
---------

- Fix a bug in ``alignment.util.wcs_from_sregions()`` and
  ``alignment.util.wcs_from_footprints()`` functions that was causing incorrect
  WCS bounding boxes
  when the ``crpix`` argument was provided. (`#326
  <https://github.com/spacetelescope/stcal/issues/326>`_)
- Allow tweakreg to parse absolute reference catalogs with skycoord objects in
  them (`#333 <https://github.com/spacetelescope/stcal/issues/333>`_)
- Fix a bug in the jump step with counting the number of usable groups. (`#338
  <https://github.com/spacetelescope/stcal/issues/338>`_)
- Do not reset existing JUMP flags at the start of the likelihood ramp fitting
  algorithm. (`#339 <https://github.com/spacetelescope/stcal/issues/339>`_)
- Fix a bug in ``jump.twopointdifference.calc_med_first_diffs()`` where
  ``argmax`` was being used instead of ``nanargmax``. (`#344
  <https://github.com/spacetelescope/stcal/issues/344>`_)
- Changed multiprocessing from forkserver to spawn. (`#350
  <https://github.com/spacetelescope/stcal/issues/350>`_)


General
-------

- Move common parts of skymatch shared by both jwst and romancal into stcal.
  (`#310 <https://github.com/spacetelescope/stcal/issues/310>`_)
- Drop support for Python 3.10 and add support for Python 3.13. (`#327
  <https://github.com/spacetelescope/stcal/issues/327>`_)
- Performance improvements for saturation step (`#331
  <https://github.com/spacetelescope/stcal/issues/331>`_)
- Performance improvements for jump step targeting both runtime and memory
  consumption. Results are mostly identical, but there are some differences in
  the MIRI shower flagging (it now flags somewhat fewer showers). (`#337
  <https://github.com/spacetelescope/stcal/issues/337>`_)
- Refactoring twopoint_difference.py for the jump step. (`#340
  <https://github.com/spacetelescope/stcal/issues/340>`_)


1.11.0 (2024-12-20)
===================

Changes to API
--------------

- Add maximum_shower_amplitude parameter to MIRI cosmic rays showers routine
  to fix accidental flagging of bright science pixels. (`#306
  <https://github.com/spacetelescope/stcal/issues/306>`_)


Bug Fixes
---------

- Do not evaluate the inverse WCS transform within the bounding box in cases
  where a resampled WCS is computed. Do not pass table columns to the WCS
  Shared API. (`#314 <https://github.com/spacetelescope/stcal/issues/314>`_)
- For `ramp_fitting`, the `CRMAG` element was not originally implemented in
  the C-extension for ramp fitting.  It is now implemented.  A bug in the read
  noise recalculation for CHARGELOSS when using the multiprocessing option has
  been fixed.  Further, in `JWST` regression tests have been added to test for
  multiprocessing to ensure testing for anything that could affect
  multiprocessing. (`#318
  <https://github.com/spacetelescope/stcal/issues/318>`_)
- Update weight threshold calculation in outlier detection to work around numpy
  bug that introduces small numerical differences for a mean of a masked array.
  (`#319 <https://github.com/spacetelescope/stcal/issues/319>`_)
- Change flagging of 'pre-saturation' grouped data to use DO_NOT_USE instead of
  SATURATION flag to avoid confusing the snowball routine downstream. (`#321
  <https://github.com/spacetelescope/stcal/issues/321>`_)


1.10.0 (2024-11-15)
===================

Changes to API
--------------

- Add `outlier_detection` median calculators from jwst. (`#292
  <https://github.com/spacetelescope/stcal/issues/292>`_)
- Deprecate wcs_from_footprints. Use wcs_from_sregions instead. (`#307
  <https://github.com/spacetelescope/stcal/issues/307>`_)
- Add wcs_from_sregions function to compute a combined WCS from a list of
  s_regions. (`#307 <https://github.com/spacetelescope/stcal/issues/307>`_)


Bug Fixes
---------

- Fix `IntEnum` saturation flag issue with numpy 2+ for romancal. (`#305
  <https://github.com/spacetelescope/stcal/issues/305>`_)
- Fix abs_deriv handling of off-edge and nan values. (`#311
  <https://github.com/spacetelescope/stcal/issues/311>`_)


General
-------

- Added fillval option to ``gwcs_blot`` utility. (`#291
  <https://github.com/spacetelescope/stcal/issues/291>`_)
- Update downstream tests for jwst and romancal to fix pytest configurations.
  (`#297 <https://github.com/spacetelescope/stcal/issues/297>`_)
- Changed the default `ramp fitting` CI test algorithm to ``OLS_C``.  This also
  revealed
  a bug in control flow that allowed for the CHARGELOSS recalculation in error,
  which
  resulted in a crash while attempting to dereference a ``NULL`` pointer.
  Further, when
  creating the optional results product, the object creation was changed to
  `PyArray_ZEROS`
  to ensure invalid data was set to zero.  The use of `PyArray_EMPTY` does not
  initialize
  memory, so junk data could be in used array elements. (`#298
  <https://github.com/spacetelescope/stcal/issues/298>`_)
- Add infrastructure for testing memory usage (`#299
  <https://github.com/spacetelescope/stcal/issues/299>`_)
- Preparing ramp fitting for the upgrade to python 3.13.  In python 3.13, the
  C-API
  function ``PyLong_AsLong`` raises an exception if the object passed to it is
  ``NoneType``.  There are two integer attributes for the ``RampData`` class
  that
  can be ``NoneType``, so a check for ``NoneType`` for these attributes was
  added. (`#303 <https://github.com/spacetelescope/stcal/issues/303>`_)


1.9.0 (2024-09-19)
==================

Changes to API
--------------

- [ramp_fitting] Add the likelihood algorithm to ramp fitting. (`#278
  <https://github.com/spacetelescope/stcal/issues/278>`_)


Bug Fixes
---------

- [saturation] Add option for using the readout pattern information to improve
  saturation flagging in grouped data. (`#283
  <https://github.com/spacetelescope/stcal/issues/283>`_)


General
-------

- Add clip_accum parameter to alignment methods. (`#286
  <https://github.com/spacetelescope/stcal/issues/286>`_)
- Improve handling of catalog web service connectivity issues. (`#286
  <https://github.com/spacetelescope/stcal/issues/286>`_)


1.8.2 (2024-09-10)
==================

Bug Fixes
---------

- Implement byteorder swap method that is forward-compatible with numpy 2.0 in
  jwst ramp_fitting. (`#282
  <https://github.com/spacetelescope/stcal/issues/282>`_)
- [jump] Fix a logical bug in the jump step for usage of > vs >= per JP-3689.
  (`#285 <https://github.com/spacetelescope/stcal/issues/285>`_)


General
-------

- [ramp_fitting] Moving the read noise recalculation due to CHARGELOSS flagging
  from
  the JWST ramp fit step code into the STCAL ramp fit C-extension. (`#275
  <https://github.com/spacetelescope/stcal/issues/275>`_)


1.8.1 (2024-09-08)
==================

Bug Fixes
---------

- Fixed memory leak in C-extension. (`#281
  <https://github.com/spacetelescope/stcal/issues/281>`_)


General
-------

- use ``towncrier`` to handle changelog entries (`#284
  <https://github.com/spacetelescope/stcal/issues/284>`_)


1.8.0 (2024-08-14)
==================

General
-------

- Add TweakReg submodule. [#267]

ramp_fitting
~~~~~~~~~~~~

- Move the CHARGELOSS read noise variance recalculation from the JWST step
  code to the C extension to simplify the code and improve performance.[#275]

Changes to API
--------------

- Add ``outlier_detection`` submodule with ``utils`` included
  from jwst. [#270] 

1.7.3 (2024-07-05)
==================

Bug Fixes
---------

ramp_fitting
~~~~~~~~~~~~

- Fix bugs in the C algorithm Poisson variance calculation when provided with
  an average dark current. [#269]

- When OLS_C was selected as the ramp fitting algorithm with multiprocessing, the C
  extension was not called.  The old python code was called.  This bug has been fixed,
  so the C extension is properly run when selecting multiprocessing. [#268]

1.7.2 (2024-06-12)
==================

General
-------

- build with Numpy 2.0 release candidate [#260]

Bug Fixes
---------

jump
~~~~
- Flag asymmetrical snowballs that are missed by the current code (JP-3638). This was changed to
  not require that the center of the snowball jump ellipse is a saturated
  pixel. [#261]

1.7.1 (2024-05-21)
==================

Bug Fixes
---------

jump
~~~~

- Catch some additional warnings about all-NaN slices. [#258]

ramp_fitting
~~~~~~~~~~~~

- Fix a bug in Poisson variance calculation visible when providing an average
  dark current value in which the specified dark current was not converted to the
  appropriate units for pixels with negative slopes.  This resulted in
  incorrect SCI, ERR, and VAR_POISSON values. Also required revising the approach
  for catching all-zero variance cases when average dark current was not
  specified. [#255]

- Refactor ramp fitting using a C extension to improve performance. [#156]

1.7.0 (2024-03-25)
==================

Changes to API
--------------

jump
~~~~

- Switch multiprocessing method to ``fork_server``. [#249]

ramp_fitting
~~~~~~~~~~~~

- Switch multiprocessing method to ``fork_server``. [#249]

Bug Fixes
---------

jump
~~~~

- Updated the shower flagging code to mask reference pixels, require a minimum
  number of groups to trigger the detection, and use all integrations to determine
  the median value. [#248]

ramp_fitting
~~~~~~~~~~~~

- Changed the data type of three variables that are used in measuring
  the jump free segments of integrations. The variables were uint8 and
  they would yield wrong results for integrations with more than 256
  groups. [#251]

- Use ``sqrtf`` instead of ``sqrt`` in ols_cas22 ramp fitting with
  jump detection to avoid small numerical errors on different systems
  due to a cast to/from double. [#252]


Other
-----

jump
~~~~

- Enable the use of multiple integrations to find outliers. Also,
  when the number of groups is above a threshold, use single pass
  outlier flagging rather than the iterative flagging. [#242]

- Use ``sqrtf`` instead of ``sqrt`` in ols_cas22 ramp fitting with
  jump detection to avoid small numerical errors on different systems
  due to a cast to/from double. [#252]

1.6.1 (2024-02-29)
==================

Changes to API
--------------

ramp_fitting
~~~~~~~~~~~~

- Add ``average_dark_current`` to calculations of poisson variance. [#243]

1.6.0 (2024-02-15)
==================

Changes to API
--------------

jump
~~~~

- Add in the flagging of groups in the integration after a snowball
  occurs. The saturated core of the snowball gets flagged as jump
  for a number of groups passed in as a parameter [#238]

Bug Fixes
---------

jump
~~~~

- Fixed the computation of the number of rows per slice for multiprocessing, which
  was causing different results when running the step with multiprocess [#239]

- Fix the code to at least always flag the group with the shower and the requested
  groups after the primary shower. [#237]

Other
-----

jump
~~~~

- Reorganize jump docs between the jwst and stcal repos. [#240]

ramp_fitting
~~~~~~~~~~~~

- Reorganize ramp_fitting docs between the jwst and stcal repos. [#240]


1.5.2 (2023-12-13)
==================

- non-code updates to testing and development infrastructure

1.5.1 (2023-11-16)
==================

- re-release to publish source distribution

1.5.0 (2023-11-15)
==================

Other
-----

- Added ``alignment`` sub-package. [#179]

- Enable automatic linting and code style checks [#187]

ramp_fitting
~~~~~~~~~~~~

- Refactor Casertano, et.al, 2022 uneven ramp fitting and incorporate the matching
  jump detection algorithm into it. [#215]

- Fix memory issue with Cas22 uneven ramp fitting [#226]

- Fix some bugs in the jump detection algorithm within the Cas22 ramp fitting [#227]

- Moving some CI tests from JWST to STCAL. [#228, spacetelescope/jwst#6080]

- Significantly improve the performance of the Cas22 uneven ramp fitting algorithm. [#229]

Changes to API
--------------

-

Bug Fixes
---------

-

1.4.4 (2023-09-15)
==================

Other
-----

- small hotfix for Numpy 2.0 deprecations [#211]

1.4.3 (2023-09-13)
==================

Changes to API
--------------

saturation
~~~~~~~~~~

- Added read_pattern argument to flag_saturated_pixels.  When used,
  this argument adjusts the saturation group-by-group to handle
  different numbers of frames entering different groups for Roman.
  When not set, the original behavior is preserved. [#188]

Bug Fixes
---------

- Fixed failures with Numpy 2.0. [#210, #211]

Other
-----

jump
~~~~

- enable the detection of snowballs that occur in frames that are
  within a group. [#207]

- Added more allowable selections for the number of cores to use for
  multiprocessing [#183]

- Fixed the computation of the number of rows per slice for multiprocessing,
  which caused different results when running the step with multiprocess [#239]

ramp_fitting
~~~~~~~~~~~~

- Added more allowable selections for the number of cores to use for
  multiprocessing [#183]

- Updating variance computation for invalid integrations, as well as
  updating the median rate computation by excluding groups marked as
  DO_NOT_USE. [#208]

- Implement the Casertano, et.al, 2022 uneven ramp fitting [#175]

1.4.2 (2023-07-11)
==================

Bug Fixes
---------

jump
~~~~

- Added setting of number_extended_events for non-multiprocessing
  mode. This is the value that is put into the header keyword EXTNCRS. [#178]

1.4.1 (2023-06-29)
==================

Bug Fixes
---------

jump
~~~~

- Added setting of number_extended_events for non-multiprocessing
  mode. This is the value that is put into the header keyword EXTNCRS. [#178]

1.4.1 (2023-06-29)

Bug Fixes
---------

jump
~~~~

- Added statement to prevent the number of cores used in multiprocessing from
  being larger than the number of rows. This was causing some CI tests to fail. [#176]

1.4.0 (2023-06-27)
==================

Bug Fixes
---------

jump
~~~~

- Updated the jump detection to switch to using the numpy sigmaclip routine to
  find the actual rms across integrations when there are at least 101 integrations
  in the exposure. This still allows cosmic rays and snowballs/showers to be flagged
  without being affected by slope variations due to either brigher-fatter/charge-spilling
  or errors in the nonlinearity correction.
  Also added the counting of the number of cosmic rays and snowballs/showers that
  is then placed in the FITS header in the JWST routines. [#174]

ramp_fitting
~~~~~~~~~~~~

- Changing where time division occurs during ramp fitting in order to
  properly handle special cases where the time is not group time, such
  as when ZEROFRAME data is used, so the time is frame time. [#173]

- Added another line of code to be included in the section where warnings are turned
  off. The large number of warnings can cause a hang in the Jupyter notebook when
  running with multiprocessing. [#174]

Changes to API
--------------

-

Other
-----

-

1.3.8 (2023-05-31)
==================

Bug Fixes
---------

dark_current
~~~~~~~~~~~~

- Fixed handling of MIRI segmented data files so that the correct dark
  integrations get subtracted from the correct science integrations. [#165]

ramp_fitting
~~~~~~~~~~~~

- Correct the "averaging" of the final image slope by properly excluding
  variances as a part of the denominator from integrations with invalid slopes.
  [#167]
- Removing the usage of ``numpy.where`` where possible for performance
  reasons. [#169]

1.3.7 (2023-04-26)
==================

Bug Fixes
---------

ramp_fitting
~~~~~~~~~~~~

- Correctly compute the number of groups in a segment to properly compute the
  optimal weights for the OLS ramp fitting algorithm.  Originally, this
  computation had the potential to include groups not in the segment being
  computed. [#163]

Changes to API
--------------

- Drop support for Python 3.8 [#162]

1.3.6 (2023-04-19)
==================

Bug Fixes
---------

ramp_fitting
~~~~~~~~~~~~

- The ``meta`` tag was missing when checking for ``drop_frame1``.  It has been
  added to the check. [#161]


Changes to API
--------------

-

Other
-----

- Remove use of deprecated ``pytest-openfiles`` ``pytest`` plugin. This has been replaced by
  catching ``ResourceWarning``. [#159]


1.3.5 (2023-03-30)
==================

Bug Fixes
---------

jump
~~~~

- Updated the code for both NIR Snowballs and MIRI Showers. The snowball
  flagging will now extend the saturated core of snowballs. Also,
  circles are no longer used for snowballs preventing the huge circles
  of flagged pixels from a glancing CR.
  Shower code is completely new and is now able to find extended
  emission far below the single pixel SNR. It also allows detected
  showers to flag groups after the detection. [#144]

ramp_fitting
~~~~~~~~~~~~

- During multiprocessing, if the number of processors requested are greater
  than the number of rows in the image, then ramp fitting errors out.  To
  prevent this error, during multiprocessing, the number of processors actually
  used will be no greater than the number of rows in the image. [#154]

Other
~~~~~

- Remove the ``dqflags``, ``dynamicdq``, and ``basic_utils`` modules and replace
  them with thin imports from ``stdatamodels`` where the code as been moved. [#146]

- update minimum version of ``numpy`` to ``1.20`` and add minimum dependency testing to CI [#153]

- restore ``opencv-python`` to a hard dependency [#155]

1.3.4 (2023-02-13)
==================

Bug Fixes
---------

ramp_fitting
~~~~~~~~~~~~

- Changed computations for ramps that have only one good group in the 0th
  group.  Ramps that have a non-zero groupgap should not use group_time, but
  (NFrames+1)*TFrame/2, instead. [#142]

1.3.3 (2023-01-26)
==================

Bug Fixes
---------

ramp_fitting
~~~~~~~~~~~~

- Fixed zeros that should be NaNs in rate and rateints product and suppressed
  a cast warning due to attempts to cast NaN to an integer. [#141]

Changes to API
--------------

dark
----

- Modified dark class to support quantities in Roman.[#140]

1.3.2 (2023-01-10)
==================

Bug Fixes
---------

ramp_fitting
~~~~~~~~~~~~

- Changed a cast due to numpy deprecation that now throws a warning.  The
  negation of a DQ flag then cast to a np.uint32 caused an over flow.  The
  flag is now cast to a np.uint32 before negation. [#139]


1.3.1 (2023-01-03)
==================

Bug Fixes
---------

- improve exception handling when attempting to use ellipses without ``opencv-python`` installed [#136]

1.3.0 (2022-12-15)
==================

General
-------

- use ``tox`` environments [#130]

Changes to API
--------------

- Added support for Quantities in models required for the RomanCAL
  pipeline. [#124]

ramp_fitting
~~~~~~~~~~~~

- Set values in the rate and rateints product to NaN when no usable data is
  available to compute slopes. [#131]


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

Bug Fixes
---------

jump
~~~~

- Update ``twopoint_difference.py`` [#90]

ramp_fitting
~~~~~~~~~~~~

- Updating the one good group ramp suppression handler works. [#92]

0.7.2 (2022-05-19)
==================

Bug Fixes
---------

ramp_fitting
~~~~~~~~~~~~

- Fix for accessing zero-frame in model to account for Roman data not using
  zero-frame. [#89]


0.7.1 (2022-05-16)
==================

Bug Fixes
---------

jump
~~~~
- Enable multiprocessing for jump detection, which is controlled by the 'max_cores' parameter. [#87]

0.7.0 (2022-05-13)
==================

Bug Fixes
---------

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

Bug Fixes
---------

saturation
~~~~~~~~~~

- Added in functionality to deal with charge spilling from saturated pixels onto neighboring pixels [#83]

0.6.3 (2022-04-27)
==================

Bug Fixes
---------

- Pin astropy min version to 5.0.4. [#82]

- Fix for jumps in first good group after dropping groups [#84]


0.6.2 (22-03-29)
================

Bug Fixes
---------

jump
~~~~
- Neighboring pixels with 'SATURATION' or 'DONOTUSE' flags are no longer flagged as jumps. [#79]

ramp_fitting
~~~~~~~~~~~~

- Adding feature to use ZEROFRAME for ramps that are fully saturated, but
  the ZEROFRAME data for that ramp is good. [#81]

0.6.1 (22-03-04)
================

Bug Fixes
---------

ramp_fitting
~~~~~~~~~~~~

- Adding feature to suppress calculations for saturated ramps having only
  the 0th group be a good group.  [#76]

0.6.0 (22-01-14)
================

Bug Fixes
---------

ramp_fitting
~~~~~~~~~~~~

- Adding GLS code back to ramp fitting. [#64]

jump
~~~~

- Fix issue in jump detection that occurred when there were only 2 usable
  differences with no other groups flagged. This PR also added tests and
  fixed some of the logging statements in twopoint difference. [#74]

0.5.1 (2022-01-07)
==================

Bug Fixes
---------

jump
~~~~

- fixes to several existing errors in the jump detection step. added additional
  tests to ensure step is no longer flagging jumps for pixels with only two
  usable groups / one usable diff. [#72]

0.5.0 (2021-12-28)
==================

Bug Fixes
---------

dark_current
~~~~~~~~~~~~

- Moved dark current code from JWST to STCAL. [#63]

0.4.3 (2021-12-27)
==================

Bug Fixes
---------

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

Bug Fixes
---------

ramp_fitting
~~~~~~~~~~~~

- For slopes with negative median rates, the Poisson variance is zero. [#59]

- Changed the way the final DQ array gets computed when handling the DO_NOT_USE
  flag for multi-integration data. [#60]

0.4.1 (2021-10-14)
==================

Bug Fixes
---------

jump_detection
~~~~~~~~~~~~~~

- Reverts "Fix issue with flagging for MIRI three and four group integrations. [#44]


0.4.0 (2021-10-13)
==================

Bug Fixes
---------

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

Bug Fixes
---------

saturation
~~~~~~~~~~

- Adds common code for saturation [#39]


0.2.5 (2021-08-27)
==================

Bug Fixes
---------

jump
~~~~

- added tests for two point difference [#37]

ramp_fitting
~~~~~~~~~~~~

- Adds support for Roman ramp data. [#43] [#49]

0.2.4 (2021-08-26)
==================

Bug Fixes
---------

Workaround for setuptools_scm issues with recent versions of pip. [#45]


0.2.3 (2021-08-06)
==================

Bug Fixes
---------

jump
~~~~
- documentation changes + docs for jump detection [#14]

ramp_fitting
~~~~~~~~~~~~

- Fix ramp fitting multiprocessing. [#30]


0.2.2 (2021-07-19)
==================

Bug Fixes
---------

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

Bug Fixes
---------

ramp_fitting
~~~~~~~~~~~~

- Fixed bug for median ramp rate computation in report JP-1950. [#12]


0.2.0 (2021-05-18)
==================

Bug Fixes
---------

ramp_fitting
~~~~~~~~~~~~

- Added ramp fitting code [#6]


0.1.0 (2021-03-19)
==================

- Added code to manipulate bitmasks.
