from pathlib import Path

from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np

Options.docstrings = True
Options.annotate = False

extensions = [Extension('stcal.ramp_fitting.matable_fit_cas2022',
                        ['src/stcal/ramp_fitting/matable_fit_cas2022.pyx'],
                        include_dirs=[np.get_include()])]

setup(ext_modules=cythonize(extensions))
