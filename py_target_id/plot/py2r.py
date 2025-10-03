"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['pd2r']

"""
Helper function to add repelled labels to plotnine plots using adjustText
"""
from rpy2.robjects import r, globalenv
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Check and install packages once
_packages_loaded = False

def ensure_r_packages():
    global _packages_loaded
    if not _packages_loaded:
        r('''
        if (!require("jgplot2", quietly = TRUE)) {
            if (!require("devtools", quietly = TRUE)) {
                install.packages("devtools")
            }
            devtools::install_github("jeffmgranja/jgplot2")
        }
        suppressPackageStartupMessages(library(jgplot2))
        ''')
        _packages_loaded = True

def pd2r(name, df):
    ensure_r_packages()  # Auto-load packages on first use
    with localconverter(pandas2ri.converter):
        r_df = pandas2ri.py2rpy(df)
        globalenv[name] = r_df