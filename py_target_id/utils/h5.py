"""
Gene pair generation utilities.
"""
__all__ = ['h5ls']
import h5py

def h5ls(filename):
    """Python equivalent of R's h5ls() with more details"""
    def print_attrs(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name:50s} Dataset {str(obj.shape):20s} {obj.dtype}")
        else:
            print(f"{name:50s} Group")
    
    with h5py.File(filename, 'r') as f:
        f.visititems(print_attrs)
