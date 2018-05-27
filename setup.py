from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy
extensions = [
    Extension("ceer_logn", ["ceer_logn.pyx"],
              define_macros=[
                  # ('CYTHON_TRACE', '1')
    ])
]

setup(
    ext_modules = cythonize(extensions)
)