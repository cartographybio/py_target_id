from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import subprocess

# Flexible HDF5 detection
def get_hdf5_config():
    """Detect HDF5 installation and return include/library dirs"""
    
    # Try conda environment first
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_include = os.path.join(conda_prefix, 'include')
        conda_lib = os.path.join(conda_prefix, 'lib')
        if os.path.exists(os.path.join(conda_lib, 'libhdf5.so')) or \
           os.path.exists(os.path.join(conda_lib, 'libhdf5.a')):
            return {
                'include_dirs': [conda_include],
                'library_dirs': [conda_lib],
                'libraries': ['hdf5', 'hdf5_cpp']
            }
    
    # Try h5py's HDF5 location
    try:
        import h5py
        h5py_config = h5py.get_config()
        if hasattr(h5py_config, 'hdf5_includedir'):
            return {
                'include_dirs': [h5py_config.hdf5_includedir],
                'library_dirs': [h5py_config.hdf5_libdir],
                'libraries': ['hdf5', 'hdf5_cpp']
            }
    except:
        pass
    
    # Try system paths (Debian/Ubuntu)
    system_paths = [
        {
            'include_dirs': ['/usr/include/hdf5/serial'],
            'library_dirs': ['/usr/lib/x86_64-linux-gnu'],
            'libraries': ['hdf5_serial', 'hdf5_serial_cpp']
        },
        {
            'include_dirs': ['/usr/include'],
            'library_dirs': ['/usr/lib/x86_64-linux-gnu'],
            'libraries': ['hdf5', 'hdf5_cpp']
        }
    ]
    
    for config in system_paths:
        lib_dir = config['library_dirs'][0]
        # Check for either serial or standard variants
        for lib_name in config['libraries']:
            if os.path.exists(os.path.join(lib_dir, f'lib{lib_name}.so')) or \
               os.path.exists(os.path.join(lib_dir, f'lib{lib_name}.a')):
                return config
    
    # Default fallback
    return {
        'include_dirs': ['/usr/include'],
        'library_dirs': ['/usr/lib/x86_64-linux-gnu'],
        'libraries': ['hdf5', 'hdf5_cpp']
    }

hdf5_config = get_hdf5_config()
print(f"Using HDF5 config: {hdf5_config}")

ext_modules = [
    Pybind11Extension(
        "py_target_id.hdf5_sparse_reader",
        ["py_target_id/cpp/hdf5_sparse_reader.cpp"],
        include_dirs=hdf5_config['include_dirs'],
        library_dirs=hdf5_config['library_dirs'],
        libraries=hdf5_config['libraries'],
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
    package_data={
        'py_target_id': [
            'data/**/*',  # Include everything under data/ recursively
        ],
    },
    include_package_data=True,
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