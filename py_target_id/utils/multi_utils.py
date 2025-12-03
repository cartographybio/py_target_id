"""
Gene pair generation utilities.
"""
__all__ = ['split_multis', 'multis_to_singles']

import numpy as np
import pandas as pd

def split_multis(multis):

    # Pre-allocate arrays
    gene1 = np.empty(len(multis), dtype=object)
    gene2 = np.empty(len(multis), dtype=object)

    for i, g in enumerate(multis):
        parts = g.split("_")
        gene1[i] = parts[0]
        gene2[i] = parts[1]

    return gene1, gene2

def multis_to_singles(multis):
    genes = np.concatenate([[pair.split('_')[0], pair.split('_')[1]] for pair in multis])
    return np.unique(genes).tolist()