"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['DaskAnnData']

import zarr
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import dask.array as da
import os
from typing import List, Union, Tuple, Optional, Any
from abc import ABC, abstractmethod

class DaskAnnData:
    """AnnData-like interface with lazy dask arrays and proper name tracking"""
    
    def __init__(self, X, obs_names, var_names, obs=None, var=None):
        """
        Parameters
        ----------
        X : dask.array
            Expression matrix (cells x genes)
        obs_names : pd.Index
            Cell/observation names
        var_names : pd.Index
            Gene/variable names
        obs : pd.DataFrame, optional
            Cell metadata
        var : pd.DataFrame, optional
            Gene metadata
        """
        self._X = X
        self._obs_names = pd.Index(obs_names)
        self._var_names = pd.Index(var_names)
        
        # Create metadata DataFrames if not provided
        self._obs = obs if obs is not None else pd.DataFrame(index=self._obs_names)
        self._var = var if var is not None else pd.DataFrame(index=self._var_names)
        
        # Validate shapes
        if self._X.shape[0] != len(self._obs_names):
            raise ValueError(f"X has {self._X.shape[0]} rows but obs_names has {len(self._obs_names)} entries")
        if self._X.shape[1] != len(self._var_names):
            raise ValueError(f"X has {self._X.shape[1]} cols but var_names has {len(self._var_names)} entries")
    
    @classmethod
    def concat_zarr(cls, zarr_paths: List[str], axis: int = 0):
        """
        Create DaskAnnData by concatenating multiple zarr stores
        
        Parameters
        ----------
        zarr_paths : list of str
            Paths to zarr stores to concatenate
        axis : int
            0 for rbind (concatenate cells), 1 for cbind (concatenate genes)
        """
        # Open all zarr stores
        stores = [zarr.open(path, mode='r') for path in zarr_paths]
        
        # Create dask arrays
        dask_arrays = [
            da.from_array(store['X'], chunks=store['X'].chunks) 
            for store in stores
        ]
        
        # Concatenate
        X_concat = da.concatenate(dask_arrays, axis=axis)
        
        # Combine metadata
        if axis == 0:  # rbind - concatenate cells
            obs_names_list = [store['obs_names'][:].astype(str) for store in stores]
            obs_names = pd.Index(np.concatenate(obs_names_list))
            var_names = pd.Index(stores[0]['var_names'][:].astype(str))
            
            # Combine obs metadata
            obs_list = []
            for i, store in enumerate(stores):
                obs_df = pd.DataFrame(index=obs_names_list[i])
                
                # Add row_sums if present and valid
                if 'row_sums' in store:
                    row_sums = store['row_sums'][:]
                    if len(row_sums) == len(obs_names_list[i]):
                        obs_df['row_sums'] = row_sums
                
                obs_df['batch'] = f'batch_{i}'
                obs_df['source_file'] = os.path.basename(zarr_paths[i])
                obs_list.append(obs_df)
            
            obs = pd.concat(obs_list)
            var = pd.DataFrame(index=var_names)
            
        else:  # cbind - concatenate genes
            obs_names = pd.Index(stores[0]['obs_names'][:].astype(str))
            var_names_list = [store['var_names'][:].astype(str) for store in stores]
            var_names = pd.Index(np.concatenate(var_names_list))
            
            obs = pd.DataFrame(index=obs_names)
            
            # Add row_sums from first store if present
            if 'row_sums' in stores[0]:
                row_sums = stores[0]['row_sums'][:]
                if len(row_sums) == len(obs_names):
                    obs['row_sums'] = row_sums
            
            # Combine var metadata
            var_list = []
            for i, store in enumerate(stores):
                var_df = pd.DataFrame(index=var_names_list[i])
                var_df['source'] = f'source_{i}'
                var_list.append(var_df)
            var = pd.concat(var_list)
        
        print(f"Created DaskAnnData: {X_concat.shape[0]} obs x {X_concat.shape[1]} vars")
        print(f"Chunks: {X_concat.chunks}")
        
        return cls(X_concat, obs_names, var_names, obs, var)
    
    @classmethod
    def from_zarr(cls, zarr_path: str):
        """Load a single zarr store"""
        store = zarr.open(zarr_path, mode='r')
        
        X = da.from_array(store['X'], chunks=store['X'].chunks)
        obs_names = pd.Index(store['obs_names'][:].astype(str))
        var_names = pd.Index(store['var_names'][:].astype(str))
        
        obs = pd.DataFrame(index=obs_names)
        if 'row_sums' in store:
            row_sums = store['row_sums'][:]
            if len(row_sums) == len(obs_names):
                obs['row_sums'] = row_sums
        
        var = pd.DataFrame(index=var_names)
        
        return cls(X, obs_names, var_names, obs, var)
    
    # Properties
    @property
    def X(self):
        """Expression matrix (dask array)"""
        return self._X
    
    @property
    def obs_names(self):
        """Observation (cell) names"""
        return self._obs_names
    
    @property
    def var_names(self):
        """Variable (gene) names"""
        return self._var_names
    
    @property
    def obs(self):
        """Observation metadata"""
        return self._obs
    
    @property
    def var(self):
        """Variable metadata"""
        return self._var
    
    @property
    def shape(self):
        return self._X.shape
    
    @property
    def n_obs(self):
        return self.shape[0]
    
    @property
    def n_vars(self):
        return self.shape[1]
    
    # Subsetting
    def __getitem__(self, key):
        """
        Subset the data (returns new DaskAnnData with lazy dask array)
        
        Supports:
        - Integer indexing: ddata[0, 5]
        - Slice indexing: ddata[:100, :50]
        - List indexing: ddata[[0,1,2], [5,10,15]]
        - Boolean indexing: ddata[mask, :]
        """
        # Normalize key to (obs_key, var_key)
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Only 2D indexing supported")
            obs_key, var_key = key
        else:
            obs_key, var_key = key, slice(None)
        
        # Subset dask array (still lazy!)
        X_subset = self._X[obs_key, var_key]
        
        # Subset names
        obs_names_subset = self._subset_index(self._obs_names, obs_key)
        var_names_subset = self._subset_index(self._var_names, var_key)
        
        # Subset metadata
        obs_subset = self._obs.loc[obs_names_subset]
        var_subset = self._var.loc[var_names_subset]
        
        return DaskAnnData(X_subset, obs_names_subset, var_names_subset, obs_subset, var_subset)
    
    def _subset_index(self, index, key):
        """Helper to subset a pandas Index"""
        if isinstance(key, slice):
            return index[key]
        elif isinstance(key, (list, np.ndarray)):
            if len(key) > 0 and isinstance(key[0], bool):
                return index[key]
            else:
                return index[key]
        elif isinstance(key, int):
            return pd.Index([index[key]])
        elif isinstance(key, pd.Series) and key.dtype == bool:
            return index[key.values]
        else:
            return index[key]
    
    def obs_subset(self, names: Union[str, List[str]]):
        """Subset by observation names"""
        if isinstance(names, str):
            names = [names]
        indices = [self._obs_names.get_loc(name) for name in names]
        return self[indices, :]
    
    def var_subset(self, names: Union[str, List[str]]):
        """Subset by variable names"""
        if isinstance(names, str):
            names = [names]
        indices = [self._var_names.get_loc(name) for name in names]
        return self[:, indices]
    
    def obs_mask(self, mask: pd.Series):
        """Subset by boolean mask on observations"""
        return self[mask.values, :]
    
    def var_mask(self, mask: pd.Series):
        """Subset by boolean mask on variables"""
        return self[:, mask.values]
    
    # Computation
    def compute(self):
        """Compute the dask array and return numpy array"""
        return self._X.compute()
    
    def to_memory(self):
        """Load into memory and return new DaskAnnData with numpy array"""
        X_computed = self._X.compute()
        X_dask = da.from_array(X_computed, chunks=X_computed.shape)
        return DaskAnnData(X_dask, self._obs_names, self._var_names, self._obs.copy(), self._var.copy())
    
    def to_adata(self):
        """Convert to AnnData (computes the dask array)"""
        import anndata as ad
        
        X_computed = self._X.compute()
        return ad.AnnData(
            X=X_computed,
            obs=self._obs.copy(),
            var=self._var.copy()
        )
    
    def to_pandas(self):
        """Convert to pandas DataFrame (computes the dask array)"""
        data = self._X.compute()
        return pd.DataFrame(
            data,
            index=self._obs_names,
            columns=self._var_names
        )
    
    # Convenient accessors
    def head(self, n: int = 5):
        """Get first n observations"""
        return self[:n, :]
    
    def tail(self, n: int = 5):
        """Get last n observations"""
        return self[-n:, :]
    
    # GPU support
    def to_gpu(self):
        """Convert to GPU-backed version (still lazy!)"""
        try:
            import cupy as cp
            X_gpu = self._X.map_blocks(cp.asarray, dtype=self._X.dtype)
            return DaskAnnData(X_gpu, self._obs_names, self._var_names, self._obs, self._var)
        except ImportError:
            print("CuPy not installed. Install with: pip install cupy-cuda12x")
            return self
    
    def normalize_gpu(self, target_sum=10000):
        """GPU-accelerated normalization (lazy)"""
        try:
            import cupy as cp
            X_gpu = self._X.map_blocks(cp.asarray)
            counts = X_gpu.sum(axis=1, keepdims=True)
            X_norm = (X_gpu / counts) * target_sum
            return DaskAnnData(X_norm, self._obs_names, self._var_names, self._obs, self._var)
        except ImportError:
            print("CuPy not installed. Falling back to CPU")
            counts = self._X.sum(axis=1, keepdims=True)
            X_norm = (self._X / counts) * target_sum
            return DaskAnnData(X_norm, self._obs_names, self._var_names, self._obs, self._var)
    
    def log1p_gpu(self):
        """GPU-accelerated log1p (lazy)"""
        try:
            import cupy as cp
            X_gpu = self._X.map_blocks(cp.asarray)
            X_log = X_gpu.map_blocks(lambda x: cp.log1p(x))
            return DaskAnnData(X_log, self._obs_names, self._var_names, self._obs, self._var)
        except ImportError:
            print("CuPy not installed. Falling back to CPU")
            X_log = self._X.map_blocks(lambda x: np.log1p(x))
            return DaskAnnData(X_log, self._obs_names, self._var_names, self._obs, self._var)
    
    # Save
    def to_zarr(self, path: str, chunks: Optional[Tuple[int, int]] = None):
        """
        Save to zarr format
        
        Parameters
        ----------
        path : str
            Path to save zarr store
        chunks : tuple, optional
            Chunk size for output. If None, uses input chunks
        """
        if chunks is not None:
            X_to_save = self._X.rechunk(chunks)
        else:
            X_to_save = self._X
        
        # Save using dask
        da.to_zarr(X_to_save, path, component='X', overwrite=True)
        
        # Add metadata
        store = zarr.open(path, mode='a')
        store.create_dataset('obs_names', data=self._obs_names.values, overwrite=True)
        store.create_dataset('var_names', data=self._var_names.values, overwrite=True)
        
        if 'row_sums' in self._obs.columns:
            store.create_dataset('row_sums', data=self._obs['row_sums'].values, overwrite=True)
        
        store.attrs['n_obs'] = self.n_obs
        store.attrs['n_vars'] = self.n_vars
        
        print(f"Saved to {path}")
    
    def __repr__(self):
        obs_cols = ', '.join(self._obs.columns[:5].tolist())
        if len(self._obs.columns) > 5:
            obs_cols += ", ..."
        
        var_cols = ', '.join(self._var.columns[:5].tolist()) if len(self._var.columns) > 0 else ""
        if len(self._var.columns) > 5:
            var_cols += ", ..."
        
        return (f"DaskAnnData object with n_obs × n_vars = {self.n_obs} × {self.n_vars}\n"
                f"    obs: {obs_cols}\n"
                f"    var: {var_cols}\n"
                f"    X: dask.array with chunks={self._X.chunks}")


