import os
import sys
import sysconfig

import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup


FREE_THREADED_PYTHON = sysconfig.get_config_var("Py_GIL_DISABLED") == 1


Options.docstrings = True
Options.annotate = False

# package_data values are glob patterns relative to each specific subpackage.
package_data = {
    "stcal.ramp_fitting.src": ["*.c"],
}


cfg = {
    "include_dirs": [np.get_include()],
    "libraries": [],
    "define_macros": [("NUMPY", "1"), ("NDEBUG", None)],
}

if not FREE_THREADED_PYTHON:
    cfg["py_limited_api"] = True
    cfg["define_macros"].append(("Py_LIMITED_API", 0x030B0000))  # PY_VERSION_HEX for 3.11

if sys.platform == "win32":
    cfg["define_macros"].extend(
        [
            ("WIN32", None),
            ("__STDC__", 1),
            ("_CRT_SECURE_NO_WARNINGS", None),
        ]
    )
else:
    cfg["libraries"].append("m")
    cfg["extra_compile_args"] = [
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wpedantic",
        "-Wno-unused-parameter",
        "-Wincompatible-pointer-types",
    ]


# importing these extension modules is tested in `.github/workflows/build.yml`;
# when adding new modules here, make sure to add them to the `test_command` entry there
extensions = [
    Extension(
        "stcal.ramp_fitting.ols_cas22._ramp",
        ["src/stcal/ramp_fitting/ols_cas22/_ramp.pyx"],
        language="c++",
        **cfg,
    ),
    Extension(
        "stcal.ramp_fitting.ols_cas22._jump",
        ["src/stcal/ramp_fitting/ols_cas22/_jump.pyx"],
        language="c++",
        **cfg,
    ),
    Extension(
        "stcal.ramp_fitting.ols_cas22._fit",
        ["src/stcal/ramp_fitting/ols_cas22/_fit.pyx"],
        language="c++",
        **cfg,
    ),
    Extension(
        "stcal.ramp_fitting.slope_fitter",
        ["src/stcal/ramp_fitting/src/slope_fitter.c"],
        **cfg,
    ),
]

SETUPTOOLS_OPTIONS = {}
if not FREE_THREADED_PYTHON:
    SETUPTOOLS_OPTIONS["bdist_wheel"] = {"py_limited_api": "cp311"}

setup(
    ext_modules=cythonize(extensions),
    options=SETUPTOOLS_OPTIONS,
)
