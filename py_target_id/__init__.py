"""
Target ID package for genomics data analysis.

This package provides tools for loading and processing manifest files
from Google Cloud Storage for genomics workflows.
"""

__version__ = "0.1.0"
__author__ = "Jeffrey Granja"

from .google import *
from .load_manifest import *
from .download_manifest import *