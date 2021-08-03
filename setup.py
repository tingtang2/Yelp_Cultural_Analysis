from setuptools import setup, Extension
from Cython.Build import cythonize

exts = Extension(name = "_lda", 
                 sources = [".c3rlda/_lda.pyx", ".c3rlda/gamma.c"])

setup(
    name = "sampling_tools",
    ext_modules = cythonize(exts, gdb_debug=True),
    zip_safe=False
)
