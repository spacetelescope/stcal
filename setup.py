import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

Options.docstrings = True
Options.annotate = False

extensions = [
    Extension(
        "stcal.ramp_fitting._ols_cas22._fit",
        ["src/stcal/ramp_fitting/_ols_cas22/_fit.py"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
]

setup(ext_modules=cythonize(extensions))
