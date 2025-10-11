"""
Reference data loading functions.
"""

__all__ = ['summarize_matrix']

import numpy as np
import pandas as pd
from scipy import sparse
from typing import Literal, Union
import time

def summarize_matrix(
    mat: Union[np.ndarray, sparse.spmatrix],
    groups: np.ndarray,
    metric: Literal["mean", "median", "max", "quantile", "percent", "custom_ha"] = "median",
    na_rm: bool = True,
    prob: float = 0.9,
    axis: int = 1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Summarize matrix by groups - optimized for sparse matrices.
    
    Parameters
    ----------
    mat : array-like
        Matrix to summarize
    groups : array-like
        Group labels for each element along the specified axis
    metric : str
        Summarization metric
    na_rm : bool
        Remove NA values
    prob : float
        Quantile probability
    axis : int
        Axis to summarize over:
        - 1 (default): summarize over columns (expects genes × cells, groups labels cells)
        - 0: summarize over rows (expects cells × genes, groups labels cells)
    verbose : bool
        Print progress messages
        
    Returns
    -------
    pd.DataFrame
        Summarized matrix
    """
    start = time.time()
    
    # Handle axis parameter by transposing if needed
    if axis == 0:
        mat = mat.T
        transpose_output = True
    else:
        transpose_output = False
    
    # Now mat is always (genes × cells) format
    assert mat.shape[1] == len(groups), f"ncol(mat)={mat.shape[1]} must equal len(groups)={len(groups)}"
    
    # Get unique groups
    uniq_groups = np.unique(groups)
    uniq_groups = sorted(uniq_groups)
    
    n_genes = mat.shape[0]
    group_mat = np.zeros((n_genes, len(uniq_groups)))
    
    if verbose:
        print(f"Summarizing ({len(uniq_groups)}): ", end='', flush=True)
    
    is_sparse = sparse.issparse(mat)
    
    # Convert to CSC for efficient column slicing if sparse
    if is_sparse and not sparse.isspmatrix_csc(mat):
        mat = sparse.csc_matrix(mat)
    
    for i, group in enumerate(uniq_groups):
        if verbose:
            print(f"{i+1} ", end='', flush=True)
        
        mask = groups == group
        mat_i = mat[:, mask]
        n_cells = mat_i.shape[1]
        
        if n_cells == 1:
            if metric == "percent":
                group_mat[:, i] = (mat_i.toarray().ravel() > 0).astype(float) if is_sparse else (mat_i.ravel() > 0).astype(float)
            else:
                group_mat[:, i] = mat_i.toarray().ravel() if is_sparse else mat_i.ravel()
            continue
        
        if metric == "custom_ha":
            group_mat[:, i] = _custom_ha_summarize(mat_i, prob, is_sparse, n_cells)
            continue
        
        if n_cells <= 2 and metric != "percent":
            if metric == "max":
                group_mat[:, i] = _row_max(mat_i, is_sparse)
            else:
                if verbose:
                    print(f"\nGroup: {group} has {n_cells} cells, using mean!\n")
                group_mat[:, i] = _row_mean(mat_i, is_sparse)
            continue
        
        # Main summarization
        if metric == "mean":
            group_mat[:, i] = _row_mean(mat_i, is_sparse)
        elif metric == "median":
            group_mat[:, i] = _row_median_fast(mat_i, is_sparse)
        elif metric == "max":
            group_mat[:, i] = _row_max(mat_i, is_sparse)
        elif metric == "quantile":
            group_mat[:, i] = _row_quantile_fast(mat_i, prob, is_sparse)
        elif metric == "percent":
            group_mat[:, i] = _row_percent_positive(mat_i, is_sparse)
        else:
            raise ValueError(f"metric {metric} unrecognized!")
    
    if verbose:
        print(f"\nRunTime (s): {time.time() - start:.2f}")
    
    result = pd.DataFrame(group_mat, columns=uniq_groups)
    
    # Transpose back if we transposed the input
    if transpose_output:
        result = result.T
    
    return result


def _custom_ha_summarize(mat_i, prob, is_sparse, n_cells):
    """Custom healthy atlas summarization."""
    if n_cells <= 2:
        return _row_mean(mat_i, is_sparse)
    elif 2 < n_cells < 10:
        val_mean = _row_mean(mat_i, is_sparse)
        val_top2 = _row_n_maxs(mat_i, n=2, is_sparse=is_sparse)
        return np.where(val_top2 >= val_mean, val_top2, (val_top2 + val_mean) / 2)
    else:
        val_q = _row_quantile_fast(mat_i, prob, is_sparse)
        val_mean = _row_mean(mat_i, is_sparse)
        return np.where(val_q >= val_mean, val_q, (val_q + val_mean) / 2)


def _row_mean(mat, is_sparse):
    """Compute row means - fast for sparse."""
    if is_sparse:
        return np.array(mat.mean(axis=1)).ravel()
    return np.mean(mat, axis=1)


def _row_max(mat, is_sparse):
    """Compute row maxs - fast for sparse."""
    if is_sparse:
        return np.array(mat.max(axis=1).toarray()).ravel()
    return np.max(mat, axis=1)


def _row_median_fast(mat, is_sparse):
    """
    Compute row medians - vectorized approach.
    
    For sparse matrices with many zeros, we can optimize by:
    1. If >50% zeros, adjust median calculation
    2. Otherwise convert to dense (if manageable)
    """
    if is_sparse:
        n_genes, n_cells = mat.shape
        
        # Check sparsity
        nnz_per_row = np.diff(mat.tocsr().indptr)
        sparsity = 1 - (nnz_per_row / n_cells)
        
        # If mostly sparse, we need special handling
        # For now, batch convert to dense for efficiency
        if n_cells <= 1000:
            # Small enough to convert to dense
            return np.median(mat.toarray(), axis=1)
        else:
            # Process in chunks to manage memory
            chunk_size = 1000
            medians = np.zeros(n_genes)
            mat_csr = sparse.csr_matrix(mat)
            
            for start_idx in range(0, n_genes, chunk_size):
                end_idx = min(start_idx + chunk_size, n_genes)
                chunk = mat_csr[start_idx:end_idx, :].toarray()
                medians[start_idx:end_idx] = np.median(chunk, axis=1)
            
            return medians
    else:
        return np.median(mat, axis=1)


def _row_quantile_fast(mat, prob, is_sparse):
    """Compute row quantiles - vectorized approach."""
    if is_sparse:
        n_genes, n_cells = mat.shape
        
        if n_cells <= 1000:
            return np.quantile(mat.toarray(), prob, axis=1)
        else:
            # Process in chunks
            chunk_size = 1000
            quantiles = np.zeros(n_genes)
            mat_csr = sparse.csr_matrix(mat)
            
            for start_idx in range(0, n_genes, chunk_size):
                end_idx = min(start_idx + chunk_size, n_genes)
                chunk = mat_csr[start_idx:end_idx, :].toarray()
                quantiles[start_idx:end_idx] = np.quantile(chunk, prob, axis=1)
            
            return quantiles
    else:
        return np.quantile(mat, prob, axis=1)


def _row_n_maxs(mat, n=2, is_sparse=False):
    """Get nth largest value in each row."""
    ncol = mat.shape[1]
    if ncol <= n:
        return _row_max(mat, is_sparse)
    prob = (ncol - n) / (ncol - 1)
    return _row_quantile_fast(mat, prob, is_sparse)


def _row_percent_positive(mat, is_sparse):
    """Compute percent positive - very fast for sparse."""
    if is_sparse:
        # Count non-zeros per row (very efficient for sparse)
        mat_csr = sparse.csr_matrix(mat)
        nnz_per_row = np.diff(mat_csr.indptr)
        return nnz_per_row / mat.shape[1]
    else:
        return np.sum(mat > 0, axis=1) / mat.shape[1]
