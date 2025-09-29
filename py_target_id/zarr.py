"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['h5map_to_zarr']


import zarr
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import os
from typing import List, Union, Tuple, Optional, Any
from abc import ABC, abstractmethod

# Utility
def h5_to_zarr(h5map_path: str, zarr_path: str, h5_path: str = 'RNA', genes_per_chunk: int = 500):
    """Convert h5map directly to zarr with optimal chunking
    
    Args:
        h5map_path: Path to input h5 file
        zarr_path: Path to output zarr store
        h5_path: Path within h5 file to data (default: 'RNA')
        genes_per_chunk: Number of genes per chunk (default: 5)
    """
    
    with h5py.File(h5map_path, 'r') as h5f:
        rna_group = h5f[h5_path]
        
        # Load sparse matrix components
        data = rna_group['data'][:]
        indices = rna_group['indices'][:]
        indptr = rna_group['indptr'][:]
        shape = rna_group['shape'][:]
        
        # Load metadata
        genes = [x.decode() if isinstance(x, bytes) else x for x in rna_group['genes'][:]]
        barcodes = [x.decode() if isinstance(x, bytes) else x for x in rna_group['barcodes'][:]]
        
        # Reconstruct sparse matrix
        sparse_matrix = sp.csc_matrix((data, indices, indptr), shape=shape)
    
        # Compute colSums efficiently from sparse matrix (before transposing)
        row_sums = np.array(sparse_matrix.sum(axis=1)).flatten()
        # Convert to dense (zarr works better with dense for now)
        dense_matrix = sparse_matrix.toarray()
    
    # Get dimensions
    n_cells = len(barcodes)
    n_genes = len(genes)
    
    # Create zarr store
    zarr_store = zarr.open(zarr_path, mode='w')
    
    # Store matrix with optimal chunking (transposed to cells x genes)
    # Chunk format: [all_cells, genes_per_chunk] - similar to the example pattern
    chunk_size = (n_cells, genes_per_chunk)
    
    zarr_store.create_dataset(
        'X', 
        data=dense_matrix.T,  # Transpose to cells x genes
        chunks=chunk_size,
        compressor=zarr.Blosc(cname='zstd', clevel=3),
        dtype=np.float32
    )
    
    # Store metadata
    zarr_store.create_dataset('obs_names', data=barcodes)
    zarr_store.create_dataset('var_names', data=genes)
    # Store rowSums (sum per cell across all genes)
    zarr_store.create_dataset('row_sums', data=row_sums, dtype=np.float32)
    
    zarr_store.attrs['n_obs'] = n_cells
    zarr_store.attrs['n_vars'] = n_genes
    
    #print(f"Converted {h5map_path} to {zarr_path}")
    #print(f"Shape: {dense_matrix.T.shape}, Chunks: {chunk_size}")
    
    return zarr_store
