import numpy as np
import os
import sys
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

Options.docstrings = True
Options.annotate = False

# package_data values are glob patterns relative to each specific subpackage.
package_data = {
    "stcal.ramp_fitting.src": ["*.c"],
}

# Setup C module include directories
include_dirs = [np.get_include()]

# Setup C module macros
define_macros = [
    ("NUMPY", "1"),
]

# Setup C libraries
libraries = []

if sys.platform.startswith("win"):
    define_macros.append(("PSAPI_VERSION", "1"))
    libraries.append("psapi")

debug_logdir= os.environ.get("DEBUG_LOGDIR")
if debug_logdir:
    define_macros.append(("DEBUG_LOGDIR", f"\"{debug_logdir}\""))

# importing these extension modules is tested in `.github/workflows/build.yml`;
# when adding new modules here, make sure to add them to the `test_command` entry there
extensions = [
    Extension(
        "stcal.ramp_fitting.ols_cas22._ramp",
        ["src/stcal/ramp_fitting/ols_cas22/_ramp.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        "stcal.ramp_fitting.ols_cas22._jump",
        ["src/stcal/ramp_fitting/ols_cas22/_jump.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        "stcal.ramp_fitting.ols_cas22._fit",
        ["src/stcal/ramp_fitting/ols_cas22/_fit.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        "stcal.ramp_fitting.slope_fitter",
        ["src/stcal/ramp_fitting/src/slope_fitter.c"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        libraries=libraries,
    ),
]

setup(ext_modules=cythonize(extensions))
