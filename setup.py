from setuptools import setup, Extension
from Cython.Build import cythonize

exts = Extension(name = "_c3rlda", 
                 sources = [".c3rlda/_c3rlda.pyx", ".c3rlda/gamma.c"])

setup(
    name = "sampling_tools",
    ext_modules = cythonize(exts, gdb_debug=True),
    zip_safe=False
)
