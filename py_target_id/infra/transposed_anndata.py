# transposed_anndata.py
# Define what gets exported
__all__ = [
    'load_t_h5ad',
    'TransposedAnnData',
    'TransposedMatrix',
    'TransposedLayers'
]

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Union, List, Optional


class TransposedAnnData:
    """
    Virtual transposed AnnData wrapper providing a cells × genes interface
    while maintaining fast gene access from underlying genes × cells storage.

    Example
    -------
    >>> adata_gxc = ad.read_h5ad("genes_x_cells.h5ad", backed='r')
    >>> adata = TransposedAnnData(adata_gxc)
    >>> gene_expr = adata[:, "CD19"].X  # Fast gene access
    >>> cell_expr = adata["cell_001", :].X  # Cell access also works
    """

    def __init__(self, adata_genes_x_cells: ad.AnnData):
        """Initialize with a genes × cells AnnData object."""
        self._adata = adata_genes_x_cells
        # Swap obs and var to present cells × genes interface
        self._obs = adata_genes_x_cells.var
        self._var = adata_genes_x_cells.obs

    # ---- Basic properties ----
    @property
    def X(self):
        """Return lazily transposed data matrix."""
        if not hasattr(self, '_X_cache'):
            self._X_cache = TransposedMatrix(self._adata.X)
        return self._X_cache

    @property
    def obs(self):
        """Cell metadata (originally var)."""
        return self._obs

    @property
    def var(self):
        """Gene metadata (originally obs)."""
        return self._var

    @property
    def obs_names(self):
        """Cell names."""
        return self._obs.index

    @property
    def var_names(self):
        """Gene names."""
        return self._var.index

    @property
    def n_obs(self):
        """Number of cells."""
        return self._adata.n_vars

    @property
    def n_vars(self):
        """Number of genes."""
        return self._adata.n_obs

    @property
    def shape(self):
        """Shape as (n_cells, n_genes)."""
        return (self.n_obs, self.n_vars)

    # ---- Metadata containers ----
    @property
    def obsm(self):
        """Cell embeddings (originally varm)."""
        return getattr(self._adata, 'varm', {})

    @property
    def varm(self):
        """Gene embeddings (originally obsm)."""
        return getattr(self._adata, 'obsm', {})

    @property
    def obsp(self):
        """Cell-cell graphs (originally varp)."""
        return getattr(self._adata, 'varp', {})

    @property
    def varp(self):
        """Gene-gene graphs (originally obsp)."""
        return getattr(self._adata, 'obsp', {})

    @property
    def layers(self):
        """Transposed layers."""
        return TransposedLayers(getattr(self._adata, 'layers', {}))

    @property
    def uns(self):
        """Unstructured metadata."""
        return getattr(self._adata, 'uns', {})

    # ---- Indexing and subsetting ----
    def __getitem__(self, index):
        """
        Subset the transposed AnnData.
        Converts cells × genes indexing to genes × cells for underlying data.
        """
        if not isinstance(index, tuple):
            index = (index, slice(None))

        cell_idx, gene_idx = index
        subset = self._adata[gene_idx, cell_idx]
        return TransposedAnnData(subset)

    # ---- Utilities ----
    def __repr__(self):
        return (
            f"TransposedAnnData object with n_obs × n_vars = "
            f"{self.n_obs} × {self.n_vars}"
        )

    def copy(self):
        """Return a deep copy of the transposed AnnData."""
        return TransposedAnnData(self._adata.copy())


class TransposedMatrix:
    """Wrapper for transposed matrix access."""

    def __init__(self, matrix):
        self._matrix = matrix

    @property
    def T(self):
        """Return the original (non-transposed) matrix."""
        return self._matrix

    def __getitem__(self, index):
        """Get transposed slice."""
        if not isinstance(index, tuple):
            index = (index, slice(None))

        row_idx, col_idx = index
        # Swap indices for the underlying matrix
        result = self._matrix[col_idx, row_idx]

        # Transpose the result if it's an array-like
        if hasattr(result, 'T'):
            return result.T
        return result

    def toarray(self):
        """Convert to dense array (transposed)."""
        if hasattr(self._matrix, 'toarray'):
            return self._matrix.toarray().T
        return np.asarray(self._matrix).T

    def todense(self):
        """Convert to dense matrix (transposed)."""
        if hasattr(self._matrix, 'todense'):
            return self._matrix.todense().T
        return np.asmatrix(self._matrix).T

    @property
    def shape(self):
        """Transposed shape."""
        orig_shape = self._matrix.shape
        return (orig_shape[1], orig_shape[0])


class TransposedLayers:
    """Wrapper for transposed layers."""

    def __init__(self, layers):
        self._layers = layers or {}

    def __getitem__(self, key):
        """Get transposed layer."""
        return TransposedMatrix(self._layers[key])

    def keys(self):
        """Layer names."""
        return self._layers.keys()

    def __repr__(self):
        return f"TransposedLayers with keys: {list(self._layers.keys())}"


# ---- Convenience function ----
def load_t_h5ad(filepath: str, backed: Optional[str] = 'r') -> TransposedAnnData:
    """
    Load a genes × cells .h5ad file as a virtual cells × genes AnnData.

    Parameters
    ----------
    filepath : str
        Path to genes × cells h5ad file.
    backed : str or None, optional (default: 'r')
        Whether to use backed mode.

    Returns
    -------
    TransposedAnnData
        Transposed wrapper providing a cells × genes interface.
    """
    adata_gxc = ad.read_h5ad(filepath, backed=backed)
    return TransposedAnnData(adata_gxc)

