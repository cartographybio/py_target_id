"""
Gene pair generation utilities.
"""
__all__ = ['split_multis', 'get_unique_genes']

import numpy as np
import pandas as pd

def split_multis(multis):
    df = pd.DataFrame([pair.split('_') for pair in multis], columns=['Gene1', 'Gene2'])
    return df

def get_unique_genes(multis):
    genes = np.concatenate([[pair.split('_')[0], pair.split('_')[1]] for pair in multis])
    return np.unique(genes).tolist()