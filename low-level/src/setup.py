from distutils.command.build_ext import build_ext
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("./facedetection/cython_code/get_faces.pyx"),
    include_dirs=[numpy.get_include()],
)
