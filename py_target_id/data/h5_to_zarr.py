"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['h5_to_zarr']


import numpy as np
import scipy.sparse as sp
import zarr
import h5py

def h5_to_zarr(
    h5map_path: str,
    zarr_path: str,
    h5_path: str = 'RNA', 
    cells_per_chunk: int = 100,
    genes_per_chunk: int = 1000,
    remove_duplicates: bool = False
):
    """
    Convert h5map directly to Zarr using a ZipStore (single file).
    
    Args:
        h5map_path: Path to input h5 file
        zarr_path: Path to output zarr zip store (e.g. 'mydata.zarr.zip')
        h5_path: Path within h5 file to data (default: 'RNA')
        cells_per_chunk: Number of cells per chunk (default: 100)
        genes_per_chunk: Number of genes per chunk (default: 1000)
        remove_duplicates: Whether to remove duplicate genes (default: False)
    """
    
    # Open the HDF5 file
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
        
        # Reconstruct sparse matrix (genes × cells)
        sparse_matrix = sp.csc_matrix((data, indices, indptr), shape=shape)
        
        # Optionally remove duplicate genes
        if remove_duplicates:
            _, unique_indices = np.unique(genes, return_index=True)
            unique_indices_sorted = sorted(unique_indices)
            genes = [genes[i] for i in unique_indices_sorted]
            sparse_matrix = sparse_matrix[unique_indices_sorted, :]
        
        # Convert to dense and transpose to cells × genes
        dense_matrix = sparse_matrix.toarray().T
                
        # Compute row_sums (sum per cell)
        row_sums = dense_matrix.sum(axis=1).astype(np.float32)
    
    # Dimensions
    n_cells = len(barcodes)
    n_genes = len(genes)
    
    # Adjust chunk sizes if dataset is smaller
    actual_cells_per_chunk = min(cells_per_chunk, n_cells)
    actual_genes_per_chunk = min(genes_per_chunk, n_genes)
    chunk_size = (actual_cells_per_chunk, actual_genes_per_chunk)
    
    print(f"Creating Zarr ZipStore with chunk size: {chunk_size} for data shape: {dense_matrix.shape}")
    
    # Use a ZipStore
    store = zarr.ZipStore(zarr_path, mode='w')
    zarr_store = zarr.group(store=store)
    
    # Create main dataset
    zarr_store.create_dataset(
        'X',
        data=dense_matrix,
        chunks=chunk_size,
        compressor=zarr.Blosc(cname='zstd', clevel=3),
        dtype=np.float32
    )
    
    # Store metadata
    zarr_store.create_dataset('obs_names', data=barcodes)
    zarr_store.create_dataset('var_names', data=genes)
    zarr_store.create_dataset('row_sums', data=row_sums, dtype=np.float32)
    
    zarr_store.attrs['n_obs'] = n_cells
    zarr_store.attrs['n_vars'] = n_genes
    
    # Close the ZipStore
    store.close()
    
    return zarr_store
