from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np

Options.docstrings = True
Options.annotate = False

extensions = [
    Extension(
        'stcal.ramp_fitting.ols_cas22._core',
        ['src/stcal/ramp_fitting/ols_cas22/_core.pyx'],
        include_dirs=[np.get_include()],
        language='c++'
    ),
    Extension(
        'stcal.ramp_fitting.ols_cas22._fixed',
        ['src/stcal/ramp_fitting/ols_cas22/_fixed.pyx'],
        include_dirs=[np.get_include()],
        language='c++'
    ),
    Extension(
        'stcal.ramp_fitting.ols_cas22._pixel',
        ['src/stcal/ramp_fitting/ols_cas22/_pixel.pyx'],
        include_dirs=[np.get_include()],
        language='c++'
    ),
    Extension(
        'stcal.ramp_fitting.ols_cas22._ramp',
        ['src/stcal/ramp_fitting/ols_cas22/_ramp.pyx'],
        include_dirs=[np.get_include()],
        language='c++'
    ),
    Extension(
        'stcal.ramp_fitting.ols_cas22._fit_ramps',
        ['src/stcal/ramp_fitting/ols_cas22/_fit_ramps.pyx'],
        include_dirs=[np.get_include()],
        language='c++'
    ),
]

setup(ext_modules=cythonize(extensions))
