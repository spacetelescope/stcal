Writing news fragments for the change log
#########################################

This ``changes/`` directory contains "news fragments": small ReStructured Text files describing a change in a few sentences.
When making a release, run ``towncrier build --version <VERSION>`` to consume existing fragments in ``changes/`` and insert them as a full change log entry at the top of ``CHANGES.rst`` for the released version.

News fragment filenames consist of the pull request number and the change log category (see below). A single change can have more than one news fragment, if it spans multiple categories:

.. code-block::

  488.alignment.rst
  488.breaking.rst
  488.outlier_detection.rst
  491.ramp_fitting.rst
  492.docs.rst

Change log categories
*********************
- ``<PR#>.breaking.rst``: Also add this fragment if your change breaks existing functionality

Step Changes
============

- ``<PR#>.alignment.rst``
- ``<PR#>.dark_current.rst``
- ``<PR#>.jump.rst``
- ``<PR#>.linearity.rst``
- ``<PR#>.outlier_detection.rst``
- ``<PR#>.ramp_fitting.rst``
- ``<PR#>.resample.rst``
- ``<PR#>.saturation.rst``
- ``<PR#>.skymatch.rst``
- ``<PR#>.tweakreg.rst``

Other Changes
=============

- ``<PR#>.docs.rst``: Documentation change
- ``<PR#>.other.rst``: Infrastructure or miscellaneous changes

.. note:: This README was adapted from the Astropy changelog readme under the terms of BSD license, which in turn adapted from the Numpy changelog readme under the terms of the MIT licence.
