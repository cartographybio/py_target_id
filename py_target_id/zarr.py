"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['h5map_to_zarr', 'DelayedZarrConcat']


import zarr
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import os
from typing import List, Union, Tuple, Optional, Any
from abc import ABC, abstractmethod


class DelayedOperation(ABC):
    """Base class for delayed operations"""
    
    @abstractmethod
    def compute(self):
        """Execute the delayed operation"""
        pass
    
    @abstractmethod  
    def shape(self):
        """Return the shape this operation would produce"""
        pass


class DelayedZarrArray(DelayedOperation):
    """Delayed zarr array that doesn't load until compute()"""
    
    def __init__(self, zarr_path: str):
        self.zarr_path = zarr_path
        self._zarr_array = None  # Not loaded yet
        self._obs_names = None
        self._var_names = None
        
    def _ensure_metadata_loaded(self):
        """Load only metadata, not the actual matrix data"""
        if self._zarr_array is None:
            store = zarr.open(self.zarr_path, mode='r')
            self._zarr_array = store['X']  # This is still lazy!
            self._obs_names = pd.Index(store['obs_names'][:].astype(str))
            self._var_names = pd.Index(store['var_names'][:].astype(str))
    
    @property
    def shape(self) -> Tuple[int, int]:
        self._ensure_metadata_loaded()
        return self._zarr_array.shape
    
    @property
    def obs_names(self) -> pd.Index:
        self._ensure_metadata_loaded()
        return self._obs_names
    
    @property
    def var_names(self) -> pd.Index:
        self._ensure_metadata_loaded() 
        return self._var_names
    
    def compute(self) -> np.ndarray:
        """Load the entire zarr array into memory"""
        self._ensure_metadata_loaded()
        return self._zarr_array[:]  # This actually loads data
    
    def __getitem__(self, key) -> 'DelayedSlice':
        """Return a delayed slice operation"""
        return DelayedSlice(self, key)


class DelayedConcat(DelayedOperation):
    """Delayed concatenation operation"""
    
    def __init__(self, sources: List[DelayedOperation], axis: int = 0, parent_zarr_concat=None):
        self.sources = sources
        self.axis = axis
        self.parent_zarr_concat = parent_zarr_concat  # Store reference to parent
        self._validate_sources()
    
    def _validate_sources(self):
        """Validate that concatenation is possible"""
        if not self.sources:
            raise ValueError("No sources provided for concatenation")
        
        # Check that non-concat dimensions match
        shapes = [source.shape for source in self.sources]
        if self.axis == 0:  # Concatenating along rows
            var_dims = [shape[1] for shape in shapes]
            if not all(dim == var_dims[0] for dim in var_dims):
                raise ValueError("Variable dimensions don't match for row concatenation")
        else:  # Concatenating along columns
            obs_dims = [shape[0] for shape in shapes]
            if not all(dim == obs_dims[0] for dim in obs_dims):
                raise ValueError("Observation dimensions don't match for column concatenation")
    
    @property
    def shape(self) -> Tuple[int, int]:
        shapes = [source.shape for source in self.sources]
        if self.axis == 0:  # Row concatenation
            total_obs = sum(shape[0] for shape in shapes)
            n_vars = shapes[0][1]
            return (total_obs, n_vars)
        else:  # Column concatenation
            n_obs = shapes[0][0]
            total_vars = sum(shape[1] for shape in shapes)
            return (n_obs, total_vars)
    
    def compute(self) -> np.ndarray:
        """Execute concatenation by computing all sources"""
        computed_sources = [source.compute() for source in self.sources]
        return np.concatenate(computed_sources, axis=self.axis)
    
    def __getitem__(self, key) -> 'DelayedSlice':
        """Return delayed slice of concatenated result"""
        return DelayedSlice(self, key)


class DelayedSlice(DelayedOperation):
    """Delayed slicing operation with proper slice composition and name tracking"""
    
    def __init__(self, source: DelayedOperation, key: Any):
        self.source = source
        self.key = key
        # Compute the composed key for chained slices
        self._composed_key = self._compose_keys()
    
    def _compose_keys(self):
        """Compose this slice with any previous slices in the chain"""
        if not isinstance(self.source, DelayedSlice):
            # Base case - no previous slice to compose with
            return self.key
        
        # Get the previous slice key
        prev_key = self.source._composed_key
        current_key = self.key
        
        # Normalize keys to tuples
        if isinstance(prev_key, tuple) and len(prev_key) == 2:
            prev_obs, prev_var = prev_key
        else:
            prev_obs, prev_var = prev_key, slice(None)
            
        if isinstance(current_key, tuple) and len(current_key) == 2:
            curr_obs, curr_var = current_key
        else:
            curr_obs, curr_var = current_key, slice(None)
        
        # Compose the observation slices
        composed_obs = self._compose_slice_keys(prev_obs, curr_obs, axis=0)
        
        # Compose the variable slices  
        composed_var = self._compose_slice_keys(prev_var, curr_var, axis=1)
        
        return (composed_obs, composed_var)
    
    def _compose_slice_keys(self, first_key, second_key, axis=0):
        """Compose two slice keys (handles slice, list, int, etc.)"""
        
        # Get the appropriate dimension size
        base_source = self._get_base_source()
        max_size = base_source.shape[axis]
        
        # Handle slice composition
        if isinstance(first_key, slice) and isinstance(second_key, slice):
            # Get the actual indices from the first slice
            first_start, first_stop, first_step = first_key.indices(max_size)
            
            # Apply second slice to the result of first slice
            second_start, second_stop, second_step = second_key.indices(first_stop - first_start)
            
            # Compose the slices
            composed_start = first_start + second_start * (first_step or 1)
            composed_stop = first_start + min(second_stop * (first_step or 1), first_stop - first_start)
            composed_step = (first_step or 1) * (second_step or 1)
            
            return slice(composed_start, composed_stop, composed_step)
        
        # Handle slice + list/array
        elif isinstance(first_key, slice) and isinstance(second_key, (list, np.ndarray)):
            # Convert first slice to indices, then index with second
            first_start, first_stop, first_step = first_key.indices(max_size)
            first_indices = list(range(first_start, first_stop, first_step))
            return [first_indices[i] for i in second_key if i < len(first_indices)]
        
        # Handle list + slice
        elif isinstance(first_key, (list, np.ndarray)) and isinstance(second_key, slice):
            second_start, second_stop, second_step = second_key.indices(len(first_key))
            return first_key[second_start:second_stop:second_step]
        
        # Handle list + list
        elif isinstance(first_key, (list, np.ndarray)) and isinstance(second_key, (list, np.ndarray)):
            return [first_key[i] for i in second_key if i < len(first_key)]
        
        # Handle int indexing
        elif isinstance(first_key, slice) and isinstance(second_key, int):
            first_start, first_stop, first_step = first_key.indices(max_size)
            first_indices = list(range(first_start, first_stop, first_step))
            return first_indices[second_key] if second_key < len(first_indices) else 0
        
        # Handle boolean indexing
        elif isinstance(second_key, (pd.Series, np.ndarray)) and second_key.dtype == bool:
            if isinstance(first_key, slice):
                first_start, first_stop, first_step = first_key.indices(max_size)
                first_indices = np.array(range(first_start, first_stop, first_step))
                # Apply boolean mask to the sliced indices
                return first_indices[second_key[:len(first_indices)]]
            else:
                # Apply boolean mask to list/array
                return np.array(first_key)[second_key[:len(first_key)]]
        
        # Default case
        else:
            return second_key
    
    def _get_base_source(self):
        """Get the base source (not a DelayedSlice)"""
        base = self.source
        while isinstance(base, DelayedSlice):
            base = base.source
        return base
    
    def _find_zarr_concat(self):
        """Find the DelayedZarrConcat object in the source chain"""
        current = self.source
        while current is not None:
            # Check if this is a DelayedZarrConcat
            if hasattr(current, '__class__') and current.__class__.__name__ == 'DelayedZarrConcat':
                return current
            # Check if this is a DelayedConcat with a parent reference
            elif hasattr(current, 'parent_zarr_concat') and current.parent_zarr_concat is not None:
                return current.parent_zarr_concat
            # Keep looking up the chain
            elif isinstance(current, DelayedSlice):
                current = current.source
            else:
                current = None
        return None
    
    @property
    def obs_names(self) -> pd.Index:
        """Get observation names for this slice"""
        # Find the DelayedZarrConcat (not just any base source)
        zarr_concat = self._find_zarr_concat()
        
        if zarr_concat:
            full_obs_names = zarr_concat.obs_names
        else:
            # Fallback to generating default names
            base_source = self._get_base_source()
            full_obs_names = pd.Index([f'obs_{i}' for i in range(base_source.shape[0])])
        
        # Apply the composed observation key
        obs_key, _ = self._get_composed_keys()
        
        if isinstance(obs_key, slice):
            return full_obs_names[obs_key]
        elif isinstance(obs_key, (list, np.ndarray)):
            return full_obs_names[obs_key]
        elif isinstance(obs_key, pd.Series) and obs_key.dtype == bool:
            return full_obs_names[obs_key]
        elif isinstance(obs_key, int):
            return pd.Index([full_obs_names[obs_key]])
        else:
            return full_obs_names
    
    @property
    def var_names(self) -> pd.Index:
        """Get variable names for this slice"""
        # Find the DelayedZarrConcat (not just any base source)
        zarr_concat = self._find_zarr_concat()
        
        if zarr_concat:
            full_var_names = zarr_concat.var_names
        else:
            # Fallback to generating default names
            base_source = self._get_base_source()
            full_var_names = pd.Index([f'var_{i}' for i in range(base_source.shape[1])])
        
        # Apply the composed variable key
        _, var_key = self._get_composed_keys()
        
        if isinstance(var_key, slice):
            return full_var_names[var_key]
        elif isinstance(var_key, (list, np.ndarray)):
            return full_var_names[var_key]
        elif isinstance(var_key, pd.Series) and var_key.dtype == bool:
            return full_var_names[var_key]
        elif isinstance(var_key, int):
            return pd.Index([full_var_names[var_key]])
        else:
            return full_var_names
    
    def _get_composed_keys(self):
        """Get the composed observation and variable keys"""
        if isinstance(self._composed_key, tuple) and len(self._composed_key) == 2:
            return self._composed_key
        else:
            return self._composed_key, slice(None)
    
    @property 
    def shape(self) -> Tuple[int, int]:
        """Compute what the shape would be after slicing"""
        obs_key, var_key = self._get_composed_keys()
        
        # Calculate dimensions using the actual names to handle boolean indexing correctly
        obs_names = self.obs_names
        var_names = self.var_names
        
        return (len(obs_names), len(var_names))
    
    def compute(self) -> np.ndarray:
        """Execute the slice operation using the composed key"""
        # Find the base source (not a slice)
        base_source = self._get_base_source()
        
        if isinstance(base_source, DelayedZarrArray):
            # Use composed key directly on zarr array
            base_source._ensure_metadata_loaded()
            return base_source._zarr_array[self._composed_key]
        elif isinstance(base_source, DelayedConcat):
            # Apply composed key to concatenated result
            return base_source.compute()[self._composed_key]
        else:
            # Fallback - compute source and apply key
            source_data = base_source.compute()
            return source_data[self._composed_key]
            
    def __getitem__(self, key) -> 'DelayedSlice':
        """Allow further slicing - creates another DelayedSlice"""
        return DelayedSlice(self, key)
    
    def load_data(self) -> np.ndarray:
        """Execute the delayed computation and return the result"""
        return self.compute()
    
    def to_pandas(self) -> pd.DataFrame:
        """Execute computation and return as pandas DataFrame with proper names"""
        data = self.compute()
        return pd.DataFrame(
            data,
            index=self.obs_names,
            columns=self.var_names
        )
    
    def to_adata(self):
        """Execute computation and convert to AnnData"""
        import anndata as ad
        
        matrix = self.compute()
        return ad.AnnData(
            X=matrix,
            obs=pd.DataFrame(index=self.obs_names),
            var=pd.DataFrame(index=self.var_names)
        )
    
    def __repr__(self) -> str:
        obs_key, var_key = self._get_composed_keys()
        return (f"DelayedSlice with shape {self.shape}\n"
                f"  obs_key: {obs_key}\n"
                f"  var_key: {var_key}\n"
                f"  source: {type(self.source).__name__}")


class DelayedZarrConcat:
    """Delayed concatenation of multiple zarr arrays - no computation until requested"""
    
    def __init__(self, zarr_paths: List[str], axis: int = 0):
        # Create delayed zarr arrays but don't load anything
        self.delayed_arrays = [DelayedZarrArray(path) for path in zarr_paths]
        self.axis = axis
        
        # Create the delayed concatenation operation with reference to parent
        self._delayed_concat = DelayedConcat(self.delayed_arrays, axis=axis, parent_zarr_concat=self)
        
        print(f"Created delayed concatenation of {len(zarr_paths)} zarr arrays")
        print("No data loaded - operations will be computed on demand")
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of concatenated result (computed from metadata only)"""
        return self._delayed_concat.shape
    
    @property
    def n_obs(self) -> int:
        return self.shape[0]
    
    @property
    def n_vars(self) -> int:
        return self.shape[1]
    
    @property
    def obs_names(self) -> pd.Index:
        """Combined observation names (loads metadata only)"""
        if self.axis == 0:  # Row concatenation
            all_names = []
            for delayed_array in self.delayed_arrays:
                all_names.extend(delayed_array.obs_names.tolist())
            return pd.Index(all_names)
        else:
            return self.delayed_arrays[0].obs_names
    
    @property
    def var_names(self) -> pd.Index:
        """Variable names (loads metadata only)"""
        if self.axis == 0:  # Row concatenation
            return self.delayed_arrays[0].var_names
        else:  # Column concatenation
            all_names = []
            for delayed_array in self.delayed_arrays:
                all_names.extend(delayed_array.var_names.tolist())
            return pd.Index(all_names)
    
    def __getitem__(self, key) -> DelayedSlice:
        """Create delayed slice - no computation performed"""
        print(f"Created delayed slice operation: {key}")
        return self._delayed_concat[key]
    
    def compute(self, key=None) -> np.ndarray:
        """Execute the delayed computation chain"""
        if key is not None:
            delayed_op = self._delayed_concat[key]
            print(f"Computing delayed slice: {key}")
            return delayed_op.compute()
        else:
            print("Computing full delayed concatenation")
            return self._delayed_concat.compute()
    
    def load_data(self) -> np.ndarray:
        """Alias for compute() to match your preferred API"""
        return self.compute()
    
    def to_pandas(self) -> pd.DataFrame:
        """Execute computation and return as pandas DataFrame with proper names"""
        data = self.compute()
        return pd.DataFrame(
            data,
            index=self.obs_names,
            columns=self.var_names
        )
    
    def to_adata(self, key=None):
        """Compute subset and convert to AnnData"""
        import anndata as ad
        
        if key is not None:
            matrix = self.compute(key)
            
            # Get subset names
            if isinstance(key, tuple) and len(key) == 2:
                obs_key, var_key = key
            else:
                obs_key, var_key = key, slice(None)
            
            obs_subset = self.obs_names[obs_key] if obs_key != slice(None) else self.obs_names
            var_subset = self.var_names[var_key] if var_key != slice(None) else self.var_names
            
        else:
            matrix = self.compute()
            obs_subset = self.obs_names
            var_subset = self.var_names
        
        return ad.AnnData(
            X=matrix,
            obs=pd.DataFrame(index=obs_subset),
            var=pd.DataFrame(index=var_subset)
        )
    
    def __repr__(self) -> str:
        return (f"DelayedZarrConcat with shape {self.shape}\n"
                f"    {len(self.delayed_arrays)} zarr arrays concatenated along axis {self.axis}\n"
                f"    No data loaded - call .compute() to execute operations")


# Utility functions

def h5map_to_zarr(h5map_path: str, zarr_path: str, chunk_size=(500, 5000)):
    """Convert h5map directly to zarr with optimal chunking"""
    
    with h5py.File(h5map_path, 'r') as h5f:
        rna_group = h5f['assays/RNA.counts']
        
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
        
        # Convert to dense (zarr works better with dense for now)
        dense_matrix = sparse_matrix.toarray()
    
    # Create zarr store
    zarr_store = zarr.open(zarr_path, mode='w')
    
    # Store matrix with optimal chunking (transposed to cells x genes)
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
    zarr_store.attrs['n_obs'] = len(barcodes)
    zarr_store.attrs['n_vars'] = len(genes)
    
    print(f"Converted {h5map_path} to {zarr_path}")
    print(f"Shape: {dense_matrix.T.shape}, Chunks: {chunk_size}")
    
    return zarr_store
