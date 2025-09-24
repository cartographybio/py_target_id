"""
Target ID package for genomics data analysis.

This package provides tools for loading and processing manifest files
from Google Cloud Storage for genomics workflows.
"""

__version__ = "0.1.0"
__author__ = "Jeffrey Granja"

# Import main functions to make them available at package level
from .manifest import load_manifest, set_google_copy_version
from .utils import google_copy

# Define what gets imported with "from target_id import *"
__all__ = [
    'load_manifest',
    'set_google_copy_version', 
    'google_copy',
]