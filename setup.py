from setuptools import setup, Extension
from Cython.Build import cythonize

exts = Extension(name = "_lda", 
                 sources = ["./lda/_lda.pyx", "./lda/gamma.c"])

setup(
    ext_modules = cythonize(exts, gdb_debug=True)
)
