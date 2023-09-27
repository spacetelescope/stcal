from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np

Options.docstrings = True
Options.annotate = False

extensions = [Extension('stcal.ramp_fitting.ols_cas22',
                        ['src/stcal/ramp_fitting/ols_cas22.pyx'],
                        include_dirs=[np.get_include()],
                        extra_compile_args=['-std=c99'])]

setup(ext_modules=cythonize(extensions))
