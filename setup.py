from setuptools import setup
from extension_helpers import get_extensions

ext_modules = get_extensions()

# Specify the minimum version for the Numpy C-API
for ext in ext_modules:
    if ext.include_dirs and "numpy" in ext.include_dirs[0]:
        ext.define_macros.append(("NPY_TARGET_VERSION", "NPY_1_21_API_VERSION"))
        ext.extra_compile_args.append("-std=c99")

setup(ext_modules=ext_modules)
