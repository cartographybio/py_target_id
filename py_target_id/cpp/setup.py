from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

include_dirs = ['/usr/include/hdf5/serial']
library_dirs = ['/usr/lib/x86_64-linux-gnu']

# Link BOTH the C library AND the C++ library
libraries = ['hdf5_serial', 'hdf5_serial_cpp']

print(f"Using include_dirs: {include_dirs}")
print(f"Using library_dirs: {library_dirs}")
print(f"Using libraries: {libraries}")

ext_modules = [
    Pybind11Extension(
        "hdf5_sparse_reader",
        ["hdf5_sparse_reader.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=['-std=c++14', '-fopenmp', '-O3', '-march=native'],
        extra_link_args=['-fopenmp'],
        language='c++'
    ),
]

setup(
    name="hdf5_reader",
    version="1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)