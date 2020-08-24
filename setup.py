from setuptools import setup, Extension
from Cython.Build import cythonize

exts = Extension(name = "_lda", 
                 sources = ["./lda/_lda.pyx", "./lda/gamma.c"])

setup(
    name = "sampling_tools",
    ext_modules = cythonize(exts, gdb_debug=True),
    zip_safe=False
)
