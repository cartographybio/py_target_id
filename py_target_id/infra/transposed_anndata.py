# transposed_anndata.py
__all__ = [
    'load_transposed_h5ad',
    'TransposedAnnData',
    'TransposedMatrix',
    'TransposedLayers'
]

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Optional

class TransposedAnnData:
    """
    Virtual transposed AnnData: cells × genes interface for a genes × cells h5ad.
    Supports backed mode, lazy slicing, and materialization to memory.
    
    State model:
    - _subset_cell_idx and _subset_gene_idx are ALWAYS materialized as integer arrays
    - obs and var are ALWAYS subsetted to match these indices
    - This ensures consistency across chained operations
    """

    def __init__(self, adata_genes_x_cells: ad.AnnData, source_path: Optional[str] = None):
        self._adata = adata_genes_x_cells
        self._source_path = source_path
        
        # Always store as materialized integer arrays, never slices
        self._subset_cell_idx = np.arange(len(adata_genes_x_cells.var))
        self._subset_gene_idx = np.arange(len(adata_genes_x_cells.obs))

        # In-memory obs/var: always rebuilt from backing indices to ensure consistency
        # Never copy metadata directly—always derive from backing file
        self._obs = pd.DataFrame(index=self._adata.var.index[self._subset_cell_idx])
        self._var = pd.DataFrame(index=self._adata.obs.index[self._subset_gene_idx])
        self._X_cache = None

    # ---- Properties ----
    @property
    def obs_names(self):
        """Cell names (from backing file's var index)."""
        return self._adata.var.index[self._subset_cell_idx]

    @property
    def var_names(self):
        """Gene names (from backing file's obs index)."""
        return self._adata.obs.index[self._subset_gene_idx]

    @property
    def obs(self):
        """Cell metadata, already subsetted."""
        return self._obs

    @obs.setter
    def obs(self, value: pd.DataFrame):
        if not value.index.equals(self.obs_names):
            raise ValueError("Index of obs must match obs_names")
        self._obs = value

    @property
    def var(self):
        """Gene metadata, already subsetted."""
        return self._var

    @var.setter
    def var(self, value: pd.DataFrame):
        if not value.index.equals(self.var_names):
            raise ValueError("Index of var must match var_names")
        self._var = value

    @property
    def n_obs(self):
        return len(self._subset_cell_idx)

    @property
    def n_vars(self):
        return len(self._subset_gene_idx)

    @property
    def shape(self):
        return (self.n_obs, self.n_vars)

    @property
    def X(self):
        """Lazy transposed matrix wrapper for current subset."""
        if self._X_cache is None:
            self._X_cache = TransposedMatrix(
                self._adata.X, 
                self._subset_gene_idx, 
                self._subset_cell_idx
            )
        return self._X_cache

    @property
    def isbacked(self):
        return getattr(self._adata, "isbacked", False)

    @property
    def layers(self):
        return TransposedLayers(
            getattr(self._adata, "layers", {}),
            self._subset_gene_idx,
            self._subset_cell_idx
        )

    @property
    def uns(self):
        return getattr(self._adata, "uns", {})

    # ---- Lazy subsetting ----
    def __getitem__(self, index):
        """
        Subset cells and/or genes with lazy evaluation.
        Supports: slices, integer arrays, string lists, boolean arrays.
        
        Examples:
            obj[10:100, :]           # cells 10-100, all genes
            obj[:, ["EGFR", "TP53"]] # all cells, specific genes
            obj[cell_mask, gene_mask] # boolean indexing
        """
        if not isinstance(index, tuple):
            index = (index, slice(None))
        cell_idx, gene_idx = index

        # Convert to integer indices relative to current state
        new_cell_idx = self._resolve_index(cell_idx, self.n_obs, self.obs_names)
        new_gene_idx = self._resolve_index(gene_idx, self.n_vars, self.var_names)
        
        # Map back to global indices
        new_cell_idx = self._subset_cell_idx[new_cell_idx]
        new_gene_idx = self._subset_gene_idx[new_gene_idx]

        # Create new object with materialized indices
        new_obj = TransposedAnnData(self._adata, self._source_path)
        new_obj._subset_cell_idx = new_cell_idx
        new_obj._subset_gene_idx = new_gene_idx
        
        # Rebuild metadata from scratch using backing file indices
        # This ensures obs/var are always in sync and have unique indices
        new_obj._obs = pd.DataFrame(index=self._adata.var.index[new_cell_idx])
        new_obj._var = pd.DataFrame(index=self._adata.obs.index[new_gene_idx])
        
        return new_obj

    @staticmethod
    def _resolve_index(idx, length, names):
        """
        Resolve any index type to an integer array relative to current dimension.
        Mimics AnnData's indexing: supports slices, ints, arrays, strings, booleans, Series.
        """
        # Single integer
        if isinstance(idx, (int, np.integer)):
            return np.array([int(idx)])
        
        # Slice
        if isinstance(idx, slice):
            return np.arange(length)[idx]
        
        # Convert pandas Series to numpy array first
        if isinstance(idx, pd.Series):
            idx = idx.values
        
        # Handle VirtualAnnData or TransposedAnnData boolean masks
        if type(idx).__name__ in ('VirtualAnnData', 'TransposedAnnData'):
            # If it's boolean mask-like, extract the boolean array
            if hasattr(idx, '_subset_cell_idx') or hasattr(idx, '_subset_gene_idx'):
                raise TypeError("Cannot index with another VirtualAnnData/TransposedAnnData object. Use boolean array instead.")
            idx = idx.values if isinstance(idx, pd.Series) else idx
        
        # Now convert everything else to array
        idx_arr = np.asarray(idx)
        
        # Boolean array
        if idx_arr.dtype == bool:
            if len(idx_arr) != length:
                raise ValueError(f"Boolean index length {len(idx_arr)} != dimension {length}")
            return np.where(idx_arr)[0]
        
        # Integer array
        if idx_arr.dtype.kind in ('i', 'u'):
            return idx_arr
        
        # String array (names)
        if idx_arr.dtype.kind in ('U', 'O', 'S'):
            name_to_pos = {name: i for i, name in enumerate(names)}
            try:
                return np.array([name_to_pos[str(name)] for name in idx_arr])
            except KeyError as e:
                raise KeyError(f"Name {e} not found in dimension")
        
        raise TypeError(f"Unsupported index type: {type(idx)}")

    # ---- Copy ----
    def copy(self):
        """Shallow copy: new object, same backing file."""
        new_obj = TransposedAnnData(self._adata, self._source_path)
        new_obj._subset_cell_idx = self._subset_cell_idx.copy()
        new_obj._subset_gene_idx = self._subset_gene_idx.copy()
        new_obj._obs = self._obs.copy()
        new_obj._var = self._var.copy()
        return new_obj

    # ---- Materialize to memory ----
    def to_memory(self, dense=True, chunk_size=5000, dtype=None, show_progress=True):
        """
        Load selected subset to memory as regular AnnData.
        
        Returns
        -------
        ad.AnnData
            cells × genes AnnData object with subset data
        """
        if self._source_path is None and self.isbacked:
            raise ValueError(
                "Cannot load backed data: source_path not available. "
                "Use load_transposed_h5ad() or provide source_path."
            )
        
        # Load full backing if needed
        adata_full = self._adata if not self.isbacked else ad.read_h5ad(
            self._source_path, backed=None
        )

        # Extract subset: backing is genes × cells, transpose to cells × genes
        X = adata_full.X[self._subset_gene_idx, :][:, self._subset_cell_idx].T
        
        if dense and sparse.issparse(X):
            X = X.toarray()

        # Create obs/var with correct indices
        obs = pd.DataFrame(index=self._adata.var.index[self._subset_cell_idx])
        var = pd.DataFrame(index=self._adata.obs.index[self._subset_gene_idx])
        
        # Debug: check for duplicates before creating AnnData
        if not var.index.is_unique:
            print(f"WARNING: var.index has {var.index.duplicated().sum()} duplicates")
            print(f"self._subset_gene_idx length: {len(self._subset_gene_idx)}")
            print(f"var.index length: {len(var.index)}")
            # This shouldn't happen, but if it does, make unique
            var.index = var.index.make_unique()

        # Create new AnnData with correct metadata
        adata = ad.AnnData(X=X, obs=obs, var=var)
        
        return adata

    # ---- Pretty-print ----
    def __repr__(self):
        backed_str = " (backed)" if self.isbacked else ""
        lines = [
            f"TransposedAnnData{backed_str} object with n_obs × n_vars = {self.n_obs} × {self.n_vars}"
        ]
        if len(self.obs.columns) > 0:
            cols = list(self.obs.columns[:5])
            lines.append(f"obs columns: {cols}")
        if len(self.var.columns) > 0:
            cols = list(self.var.columns[:5])
            lines.append(f"var columns: {cols}")
        lines.append(f"layers: {list(self.layers.keys())}")
        return "\n".join(lines)


class TransposedMatrix:
    """Lazy transposed matrix wrapper with subset tracking."""
    
    def __init__(self, X, gene_idx=None, cell_idx=None):
        self._X = X
        # If not provided, use full dimensions
        self._gene_idx = gene_idx if gene_idx is not None else np.arange(X.shape[0])
        self._cell_idx = cell_idx if cell_idx is not None else np.arange(X.shape[1])

    @property
    def T(self):
        return self._X

    def __getitem__(self, idx):
        """Access subset of transposed matrix (cells × genes)."""
        if not isinstance(idx, tuple):
            idx = (idx, slice(None))
        cell_slice, gene_slice = idx
        
        # Convert slices to indices
        if isinstance(cell_slice, slice):
            cell_positions = np.arange(len(self._cell_idx))[cell_slice]
        else:
            cell_positions = np.asarray(cell_slice)
            
        if isinstance(gene_slice, slice):
            gene_positions = np.arange(len(self._gene_idx))[gene_slice]
        else:
            gene_positions = np.asarray(gene_slice)
        
        # Map to original matrix indices
        orig_genes = self._gene_idx[gene_positions]
        orig_cells = self._cell_idx[cell_positions]
        
        # Access backing (genes × cells), then transpose
        result = self._X[orig_genes, :][:, orig_cells].T
        return result

    @property
    def shape(self):
        return (len(self._cell_idx), len(self._gene_idx))


class TransposedLayers:
    """Lazy transposed layers wrapper with subset tracking."""
    
    def __init__(self, layers, gene_idx=None, cell_idx=None):
        self._layers = layers or {}
        self._gene_idx = gene_idx
        self._cell_idx = cell_idx

    def __getitem__(self, key):
        return TransposedMatrix(
            self._layers[key],
            self._gene_idx,
            self._cell_idx
        )

    def keys(self):
        return self._layers.keys()

    def __repr__(self):
        return f"TransposedLayers with keys: {list(self._layers.keys())}"


# ---- Convenience loader ----
def load_transposed_h5ad(filepath: str, backed: Optional[str] = 'r') -> TransposedAnnData:
    """Load a genes × cells h5ad as a cells × genes TransposedAnnData."""
    adata_gxc = ad.read_h5ad(filepath, backed=backed)
    return TransposedAnnData(adata_gxc, source_path=filepath)

    