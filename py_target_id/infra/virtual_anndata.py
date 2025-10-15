# Define what gets exported
__all__ = [
    'read_h5',
    'VirtualAnnData',
    'VirtualMatrix'
]

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import h5py
from py_target_id import hdf5_sparse_reader

class VirtualMatrix:
    """Fast lazy/virtual matrix using C++ backend with parallel HDF5 reading for sparse matrices."""

    def __init__(self, h5_files, dataset_path, num_threads=None):
        """
        h5_files: list of HDF5 file paths
        dataset_path: path to sparse matrix group within HDF5 files (e.g., "/assays/RNA.counts")
        num_threads: number of OpenMP threads (default: use all available)
        """
        self.h5_files = h5_files
        self.dataset_path = dataset_path
        self.num_threads = num_threads if num_threads is not None else 0

        # Get dimensions and load row/column names
        with h5py.File(h5_files[0], 'r') as f:
            shape = f[dataset_path]['shape'][:]
            self.nrow = int(shape[0])
            self.ncol = int(shape[1])

            # Load row names (genes)
            if 'genes' in f[dataset_path]:
                self.rownames = np.array([g.decode() if isinstance(g, bytes) else g
                                         for g in f[dataset_path]['genes'][:]])
            else:
                self.rownames = np.array([f"Row_{i}" for i in range(self.nrow)])

            # Load column names (barcodes) from first file
            if 'barcodes' in f[dataset_path]:
                self.colnames = [b.decode() if isinstance(b, bytes) else b
                                for b in f[dataset_path]['barcodes'][:]]
            else:
                self.colnames = [f"Col_{i}" for i in range(self.ncol)]

        # Add columns from remaining files
        for h5_file in h5_files[1:]:
            with h5py.File(h5_file, 'r') as f:
                shape = f[dataset_path]['shape'][:]
                self.ncol += int(shape[1])

                if 'barcodes' in f[dataset_path]:
                    barcodes = [b.decode() if isinstance(b, bytes) else b
                               for b in f[dataset_path]['barcodes'][:]]
                    self.colnames.extend(barcodes)
                else:
                    start_idx = len(self.colnames)
                    self.colnames.extend([f"Col_{i}" for i in range(start_idx, start_idx + int(shape[1]))])

        self.colnames = np.array(self.colnames)
        self.shape = (self.nrow, self.ncol)
        self.row_indices = None
        self.col_indices = None

    def __getitem__(self, key):
        """Lazy subset - returns new VirtualMatrix with subset info."""
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("Must provide (row, col) indexing")

        row_idx, col_idx = key

        # Convert to numpy arrays
        if isinstance(row_idx, slice):
            row_idx = np.arange(*row_idx.indices(self.nrow))
        elif isinstance(row_idx, int):
            row_idx = np.array([row_idx])
        else:
            row_idx = np.asarray(row_idx)

        if isinstance(col_idx, slice):
            col_idx = np.arange(*col_idx.indices(self.ncol))
        elif isinstance(col_idx, int):
            col_idx = np.array([col_idx])
        else:
            col_idx = np.asarray(col_idx)

        # Create new VirtualMatrix with subset
        new_vm = VirtualMatrix.__new__(VirtualMatrix)
        new_vm.h5_files = self.h5_files
        new_vm.dataset_path = self.dataset_path
        new_vm.num_threads = self.num_threads
        new_vm.rownames = self.rownames
        new_vm.colnames = self.colnames

        # Chain subsetting
        if self.row_indices is not None:
            new_vm.row_indices = self.row_indices[row_idx]
        else:
            new_vm.row_indices = row_idx

        if self.col_indices is not None:
            new_vm.col_indices = self.col_indices[col_idx]
        else:
            new_vm.col_indices = col_idx

        new_vm.nrow = len(new_vm.row_indices)
        new_vm.ncol = len(new_vm.col_indices)
        new_vm.shape = (new_vm.nrow, new_vm.ncol)

        return new_vm

    def realize(self):
        """Load data into memory as sparse CSR matrix using fast C++ backend."""
        row_idx = self.row_indices if self.row_indices is not None else np.arange(self.nrow)
        col_idx = self.col_indices if self.col_indices is not None else np.arange(self.ncol)

        # Call C++ extension
        sparse_data = hdf5_sparse_reader.read_sparse_hdf5_subset(
            self.h5_files,
            self.dataset_path,
            row_idx.astype(np.int64).tolist(),
            col_idx.astype(np.int64).tolist(),
            self.num_threads
        )

        # Convert to scipy sparse matrix
        return csr_matrix(
            (sparse_data.data, sparse_data.indices, sparse_data.indptr),
            shape=(sparse_data.nrows, sparse_data.ncols)
        )


class VirtualAnnData:
    """AnnData-compatible virtual matrix container."""

    def __init__(self, h5_files, dataset_path="/assays/RNA.counts", num_threads=None):
        """
        Create AnnData-like object backed by HDF5 files.

        Parameters
        ----------
        h5_files : list of str
            List of HDF5 file paths
        dataset_path : str
            Path to dataset within HDF5 files
        num_threads : int, optional
            Number of threads for parallel reading
        """
        self._X = VirtualMatrix(h5_files, dataset_path, num_threads)

        # Create obs (cell metadata)
        self._obs = pd.DataFrame(index=self._X.colnames)
        self._obs.index.name = 'cell_id'

        # Create var (gene metadata)
        self._var = pd.DataFrame(index=self._X.rownames)
        self._var.index.name = 'gene_id'

        # Placeholder for other slots
        self._obsm = {}
        self._varm = {}
        self._uns = {}
        self._obsp = {}
        self._varp = {}
        self._layers = {}

    @property
    def X(self):
        """Access to the data matrix (returns VirtualMatrix or can be realized)."""
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def obs(self):
        """Cell metadata (observations)."""
        return self._obs

    @obs.setter
    def obs(self, value):
        self._obs = value

    @property
    def var(self):
        """Gene metadata (variables)."""
        return self._var

    @var.setter
    def var(self, value):
        self._var = value

    @property
    def obsm(self):
        """Multi-dimensional cell annotations."""
        return self._obsm

    @property
    def varm(self):
        """Multi-dimensional gene annotations."""
        return self._varm

    @property
    def uns(self):
        """Unstructured annotations."""
        return self._uns

    @property
    def obsp(self):
        """Pairwise cell annotations."""
        return self._obsp

    @property
    def varp(self):
        """Pairwise gene annotations."""
        return self._varp

    @property
    def layers(self):
        """Additional data layers."""
        return self._layers

    @property
    def n_obs(self):
        """Number of observations (cells)."""
        return self._X.shape[1]

    @property
    def n_vars(self):
        """Number of variables (genes)."""
        return self._X.shape[0]

    @property
    def obs_names(self):
        """Cell names."""
        return self._obs.index

    @property
    def var_names(self):
        """Gene names."""
        return self._var.index

    @property
    def shape(self):
        """Shape of the data matrix (n_obs, n_vars)."""
        return (self.n_obs, self.n_vars)

    def __getitem__(self, key):
        """
        Subset the AnnData object.

        Examples
        --------
        adata[0:10, 0:20]  # First 10 cells, first 20 genes
        adata[:, ['GENE1', 'GENE2']]  # All cells, specific genes
        adata[adata.obs['group'] == 'A', :]  # Cells from group A
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("Must provide (obs, var) indexing like adata[obs_idx, var_idx]")

        obs_idx, var_idx = key

        # Convert obs indexing
        if isinstance(obs_idx, str):
            obs_idx = [obs_idx]
        if isinstance(obs_idx, list) and isinstance(obs_idx[0], str):
            # String names to indices
            obs_idx = [self.obs.index.get_loc(name) for name in obs_idx]
        elif isinstance(obs_idx, pd.Series):
            # Boolean mask from pandas Series
            obs_idx = np.where(obs_idx.values)[0]
        elif isinstance(obs_idx, np.ndarray) and obs_idx.dtype == bool:
            # Boolean mask from numpy array
            obs_idx = np.where(obs_idx)[0]

        # Convert var indexing
        if isinstance(var_idx, str):
            var_idx = [var_idx]
        if isinstance(var_idx, list) and isinstance(var_idx[0], str):
            # String names to indices
            var_idx = [self.var.index.get_loc(name) for name in var_idx]
        elif isinstance(var_idx, pd.Series):
            # Boolean mask from pandas Series
            var_idx = np.where(var_idx.values)[0]
        elif isinstance(var_idx, np.ndarray) and var_idx.dtype == bool:
            # Boolean mask from numpy array
            var_idx = np.where(var_idx)[0]

        # Note: X is stored transposed (genes x cells), so swap indices
        new_adata = VirtualAnnData.__new__(VirtualAnnData)
        new_adata._X = self._X[var_idx, obs_idx]
        new_adata._obs = self._obs.iloc[obs_idx].copy()
        new_adata._var = self._var.iloc[var_idx].copy()
        new_adata._obsm = {}
        new_adata._varm = {}
        new_adata._uns = self._uns.copy()
        new_adata._obsp = {}
        new_adata._varp = {}
        new_adata._layers = {}

        return new_adata

    def to_memory(self, chunk_size: int = 5000, show_progress: bool = True):
        """
        Realize the virtual matrix in chunks to minimize peak memory usage.

        Parameters
        ----------
        chunk_size : int, optional (default: 5000)
            Number of cells (observations) to load per chunk.
        show_progress : bool, optional (default: True)
            Whether to display a tqdm progress bar.

        Returns
        -------
        anndata.AnnData
            Standard in-memory AnnData object (sparse by default).
        """
        try:
            import anndata
            from tqdm import tqdm
            import scipy.sparse as sp
        except ImportError:
            raise ImportError("Required packages missing. Install with: pip install anndata tqdm scipy")

        print("Realizing to memory...")

        n_cells = self.n_obs
        n_genes = self.n_vars

        # Prepare obs and var copies
        obs = self._obs.copy()
        var = self._var.copy()

        # Storage for data chunks
        X_chunks = []
        obs_chunks = []

        iterator = range(0, n_cells, chunk_size)
        if show_progress:
            iterator = tqdm(iterator, total=(n_cells + chunk_size - 1) // chunk_size, desc="Realizing chunks")

        for start in iterator:
            end = min(start + chunk_size, n_cells)

            # Subset chunk (lazy)
            sub = self[start:end, :]

            # Realize to memory (sparse)
            X_chunk = sub._X.realize().T  # shape: cells × genes
            X_chunks.append(X_chunk)
            obs_chunks.append(obs.iloc[start:end])

            # Explicitly free temporary VirtualAnnData to lower memory peak
            del sub, X_chunk

        # Stack chunks vertically
        X_full = sp.vstack(X_chunks, format='csr')

        # Build final AnnData
        adata = anndata.AnnData(
            X=X_full,
            obs=pd.concat(obs_chunks),
            var=var,
            obsm=self._obsm.copy(),
            varm=self._varm.copy(),
            uns=self._uns.copy(),
            obsp=self._obsp.copy(),
            varp=self._varp.copy(),
            layers=self._layers.copy()
        )

        return adata

    def __repr__(self):
        """AnnData-style representation."""
        lines = []
        lines.append(f"VirtualAnnData object with n_obs × n_vars = {self.n_obs} × {self.n_vars}")

        if len(self._obs.columns) > 0:
            lines.append(f"    obs: {', '.join(repr(c) for c in self._obs.columns[:5])}")

        if len(self._var.columns) > 0:
            lines.append(f"    var: {', '.join(repr(c) for c in self._var.columns[:5])}")

        if self._uns:
            lines.append(f"    uns: {', '.join(repr(k) for k in list(self._uns.keys())[:5])}")

        if self._obsm:
            lines.append(f"    obsm: {', '.join(repr(k) for k in self._obsm.keys())}")

        if self._varm:
            lines.append(f"    varm: {', '.join(repr(k) for k in self._varm.keys())}")

        if self._layers:
            lines.append(f"    layers: {', '.join(repr(k) for k in self._layers.keys())}")

        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


# Convenience function
def read_h5(h5_files, dataset_path="/assays/RNA.counts", num_threads=None):
    """
    Read HDF5 files into VirtualAnnData object.

    Parameters
    ----------
    h5_files : str or list of str
        Path(s) to HDF5 file(s)
    dataset_path : str
        Path to dataset within HDF5 files
    num_threads : int, optional
        Number of threads for parallel reading

    Returns
    -------
    VirtualAnnData
        AnnData-compatible virtual matrix
    """
    if isinstance(h5_files, str):
        h5_files = [h5_files]

    return VirtualAnnData(h5_files, dataset_path, num_threads)
