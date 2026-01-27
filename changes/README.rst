Changelog
=========

This directory contains "news fragments" which are short files that contain a
small **ReST**-formatted text that will be added to the full changelog.

Make sure to use full sentences with correct case and punctuation.

Consuming news fragments in `changes/` into a new change log entry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running `towncrier build` will read all existing fragment files in `changes/`
and create a new entry at the top of `CHANGES.rst` with the specified version number.

```shell
pip install towncrier
towncrier build --version <VERSION>
```

News fragment change types
--------------------------
- ``<PR#>.breaking.rst``: Also add this fragment if your change breaks existing functionality

Step Changes
""""""""""""

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
"""""""""""""

- ``<PR#>.docs.rst``: Documentation change
- ``<PR#>.other.rst``: Infrastructure or miscellaneous changes

Note
----

This README was adapted from the Astropy changelog readme under the terms
of BSD license, which in turn adapted from the Numpy changelog readme under
the terms of the MIT licence.
