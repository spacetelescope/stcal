0.7.2 (2022-05-19)
==================

ramp_fitting
------------

- Fix for accessing zero-frame in model to account for Roman data not using
  zero-frame. [#89]


0.7.1 (2022-05-16)
==================

jump
---- 
- Enable multiprocessing for jump detection, which is controlled by the 'max_cores' parameter. [#87]

0.7.0 (2022-05-13)
==================

linearity
---------
- Added functionality to linearly process ZEROFRAME data the same way
  as the SCI data. [#81]

ramp_fitting
------------
- Added functionality to use ZEROFRAME data in place of group 0 data
  for ramps that are fully saturated, but still have good ZEROFRAME
  data. [#81]

saturation
---------- 
- Added functionality to process ZEROFRAME data for saturation the same
  way as the SCI data. [#81]


0.6.4 (2022-05-02)
==================

saturation
---------- 

- Added in functionality to deal with charge spilling from saturated pixels onto neighboring pixels [#83]

0.6.3 (2022-04-27)
==================

- Pin astropy min version to 5.0.4. [#82]

- Fix for jumps in first good group after dropping groups [#84]
  

0.6.2 (22-03-29)
================

jump
----
- Neighboring pixels with 'SATURATION' or 'DONOTUSE' flags are no longer flagged as jumps. [#79]

ramp_fitting
------------

- Adding feature to use ZEROFRAME for ramps that are fully saturated, but
  the ZEROFRAME data for that ramp is good. [#81]

0.6.1 (22-03-04)
================

ramp_fitting
------------

- Adding feature to suppress calculations for saturated ramps having only
  the 0th group be a good group.  [#76]

0.6.0 (22-01-14)
================

ramp_fitting
------------

- Adding GLS code back to ramp fitting. [#64]

jump
----

- Fix issue in jump detection that occured when there were only 2 usable
  differences with no other groups flagged. This PR also added tests and
  fixed some of the logging statements in twopoint difference. [#74]

0.5.1 (2022-01-07)
==================

jump
----

- fixes to several existing errors in the jump detection step. added additional
  tests to ensure step is no longer flagging jumps for pixels with only two
  usable groups / one usable diff. [#72]

0.5.0 (2021-12-28)
==================

dark_current
------------

- Moved dark current code from JWST to STCAL. [#63]

0.4.3 (2021-12-27)
==================

linearity
---------
- Let software set the pixel dq flag to NO_LIN_CORR if linear term of linearity coefficient is zero. [#65]

ramp_fitting
------------

- Fix special handling for 2 group ramp. [#70]

- Fix issue with inappropriately including a flagged group at the beginning
  of a ramp segment. [#68]

0.4.2 (2021-10-28)
==================

ramp_fitting
------------

- For slopes with negative median rates, the Poisson variance is zero. [#59]

- Changed the way the final DQ array gets computed when handling the DO_NOT_USE
  flag for multi-integration data. [#60]

0.4.1 (2021-10-14)
==================

jump_detection
--------------

- Reverts "Fix issue with flagging for MIRI three and four group integrations. [#44]


0.4.0 (2021-10-13)
==================

jump_detection
--------------

- Fix issue with flagging for MIRI three and four group integrations. [#44]

linearity
---------

- Adds common code for linearity correction [#55]


0.3.0 (2021-09-28)
==================

saturation
----------

- Adds common code for saturation [#39]


0.2.5 (2021-08-27)
==================

ramp_fitting
------------

- Adds support for Roman ramp data. [#49]


0.2.4 (2021-08-26)
==================

Workaround for setuptools_scm issues with recent versions of pip. [#45]


0.2.3 (2021-08-06)
==================

ramp_fitting
------------

- Fix ramp fitting multiprocessing. (#30)


0.2.2 (2021-07-19)
==================

ramp_fitting
------------

- Implemented multiprocessing for OLS. [#30]
- Added DQ flag parameter to `ramp_fit` [#25]

- Move common ``jump`` code to stcal [#27]


0.2.1 (2021-05-20)
==================

ramp_fitting
------------

- Fixed bug for median ramp rate computation in report JP-1950. [#12]


0.2.0 (2021-05-18)
==================

ramp_fitting
------------

- Added ramp fitting code [#6]


0.1.0 (2021-03-19)
==================

- Added code to manipulate bitmasks.
