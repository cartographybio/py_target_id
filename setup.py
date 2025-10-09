from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

# HDF5 configuration
include_dirs = ['/usr/include/hdf5/serial']
library_dirs = ['/usr/lib/x86_64-linux-gnu']
libraries = ['hdf5_serial', 'hdf5_serial_cpp']

ext_modules = [
    Pybind11Extension(
        "py_target_id.hdf5_sparse_reader",
        ["py_target_id/cpp/hdf5_sparse_reader.cpp"],  # Correct path based on your structure
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=['-std=c++14', '-fopenmp', '-O3', '-march=native'],
        extra_link_args=['-fopenmp'],
        language='c++'
    ),
]

setup(
    name="py_target_id",
    version="1.0.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'h5py',
        'pybind11',
        'scanpy',
        'anndata',
        'torch',
        'tqdm',
        'plotnine'
    ],
    python_requires='>=3.7',
)