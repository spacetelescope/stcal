# STCAL

[![Documentation Status](https://readthedocs.org/projects/stcal/badge/?version=latest)](http://stcal.readthedocs.io/en/latest/?badge=latest)

[![CI](https://github.com/spacetelescope/stcal/actions/workflows/ci.yml/badge.svg)](https://github.com/spacetelescope/stcal/actions/workflows/ci.yml)

[![codecov](https://codecov.io/gh/spacetelescope/stcal/branch/main/graph/badge.svg?token=C1LO00W9CZ)](https://codecov.io/gh/spacetelescope/stcal)

STScI Calibration algorithms and tools.

![STScI Logo](docs/_static/stsci_logo.png)

**STCAL requires Python 3.9 or above and a C compiler for dependencies.**

**Linux and MacOS platforms are tested and supported. Windows is not currently supported.**

**If installing on MacOS Mojave 10.14, you must install
into an environment with python 3.9. Installation will fail on python 3.10 due
to lack of a stable build for dependency `opencv-python`.**

`STCAL` is intended to be used as a support package for calibration pipeline
software, such as the `JWST` and `Roman` calibration pipelines. `STCAL` is a
separate package because it is also intended to be software that can be reused
by multiple calibration pipelines. Even though it is intended to be a support
package for calibration pipelines, it can be installed and used as a stand alone
package. This could make usage unwieldy as it is easier to use `STCAL` through
calibration software. The main use case for stand alone installation is for
development purposes, such as bug fixes and feature additions. When installing
calibration pipelines that depend on `STCAL` this package automatically gets
installed as a dependency.

## Installation

The easiest way to install the latest `stcal` release into a fresh virtualenv or conda environment is

    pip install stcal

### Detailed Installation

The `stcal` package can be installed into a virtualenv or conda environment via `pip`.
We recommend that for each installation you start by creating a fresh
environment that only has Python installed and then install the `stcal` package and
its dependencies into that bare environment.
If using conda environments, first make sure you have a recent version of Anaconda
or Miniconda installed.
If desired, you can create multiple environments to allow for switching between different
versions of the `stcal` package (e.g. a released version versus the current development version).

In all cases, the installation is generally a 3-step process:

- Create a conda environment
- Activate that environment
- Install the desired version of the `stcal` package into that environment

Details are given below on how to do this for different types of installations,
including tagged releases and development versions.
Remember that all conda operations must be done from within a bash/zsh shell.

### Installing latest releases

You can install the latest released version via `pip`. From a bash/zsh shell:

    conda create -n <env_name> python
    conda activate <env_name>
    pip install stcal

You can also install a specific version, for example `stcal 1.3.2`:

    conda create -n <env_name> python
    conda activate <env_name>
    pip install stcal==1.3.2

### Installing the development version from Github

You can install the latest development version (not as well tested) from the
Github master branch:

    conda create -n <env_name> python
    conda activate <env_name>
    pip install git+https://github.com/spacetelescope/stcal

### Installing for Developers

If you want to be able to work on and test the source code with the `stcal` package,
the high-level procedure to do this is to first create a conda environment using
the same procedures outlined above, but then install your personal copy of the
code overtop of the original code in that environment. Again, this should be done
in a separate conda environment from any existing environments that you may have
already installed with released versions of the `stcal` package.

As usual, the first two steps are to create and activate an environment:

    conda create -n <env_name> python
    conda activate <env_name>

To install your own copy of the code into that environment, you first need to
fork and clone the `stcal` repo:

    cd <where you want to put the repo>
    git clone https://github.com/spacetelescope/stcal
    cd stcal

_Note: `python setup.py install` and `python setup.py develop` commands do not work._

Install from your local checked-out copy as an "editable" install:

    pip install -e .

If you want to run the unit or regression tests and/or build the docs, you can make
sure those dependencies are installed too:

    pip install -e ".[test]"
    pip install -e ".[docs]"
    pip install -e ".[test,docs]"

Need other useful packages in your development environment?

    pip install ipython jupyter matplotlib pylint ipdb

## Contributions and Feedback

We welcome contributions and feedback on the project. Please follow the
[contributing guidelines](CONTRIBUTING.md) to submit an issue or a pull request.

We strive to provide a welcoming community to all of our users by abiding with
the [Code of Conduct](CODE_OF_CONDUCT.md).

If you have questions or concerns regarding the software, please open an issue
at https://github.com/spacetelescope/stcal/issues.

## Unit Tests

Unit tests can be run via `pytest`. Within the top level of your local `stcal` repo checkout:

    pip install -e ".[test]"
    pytest

Need to parallelize your test runs over all available cores?

    pip install pytest-xdist
    pytest -n auto
