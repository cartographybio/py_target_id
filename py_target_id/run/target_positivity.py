"""
Target ID
"""

# Define what gets exported
__all__ = ['compute_positivity_matrix']

"""
Compute Positivity Matrix Based on Gene Ranks and Expression Values

Highly optimized version using vectorized operations and minimal Python overhead.
"""

import numpy as np
import numba
from typing import Optional


@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _process_genes_numba(gene_ranks, mat_rna, p_threshold, min_cutoff, rank_cutoff):
    """
    Numba-compiled core computation for maximum speed.
    
    This function is JIT-compiled and runs in parallel with no Python overhead.
    """
    n_genes, n_samples = gene_ranks.shape
    pos_mat = np.zeros((n_genes, n_samples), dtype=np.bool_)
    
    for i in numba.prange(n_genes):
        gr = gene_ranks[i, :]
        sr = mat_rna[i, :]
        
        # Initial positivity based on rank cutoff
        pos = gr <= rank_cutoff
        
        # Count positive samples
        n_pos = np.sum(pos)
        
        # Handle edge cases
        if n_pos == 0:
            continue  # All False (already initialized)
        if n_pos == n_samples:
            pos_mat[i, :] = True
            continue
        
        # Calculate median expression of initially positive samples
        pos_vals_init = sr[pos]
        med_pos = np.median(pos_vals_init)
        
        # Expand positive set to include samples with expression >= median
        pos = pos | (sr >= med_pos)
        
        # Separate positive and negative expression values
        pos_vals = sr[pos]
        neg_vals = sr[~pos]
        
        # Calculate boundary values for adaptive thresholding
        max_neg = np.max(neg_vals) if neg_vals.size > 0 else 0.0
        min_pos = np.min(pos_vals) if pos_vals.size > 0 else 0.0
        med_pos = np.median(pos_vals) if pos_vals.size > 0 else 0.0
        
        # Compute initial adaptive cutoff as mean of boundaries
        new_cutoff = (max_neg + min_pos) / 2.0
        
        # Apply minimum constraints to cutoff
        new_cutoff = max(new_cutoff, min_cutoff, 0.1 * med_pos)
        
        # Calculate target positivity rate and allowable bounds
        pos_rate = np.sum(gr <= rank_cutoff) / n_samples
        q_lower = max(pos_rate - p_threshold, 0.0)
        q_upper = min(pos_rate + p_threshold, 1.0)
        
        # Calculate quantile-based bounds to maintain target positivity rate
        upper_bound = np.quantile(sr, 1.0 - q_lower)
        lower_bound = np.quantile(sr, 1.0 - q_upper)
        
        # Apply quantile bounds to cutoff
        new_cutoff = min(new_cutoff, upper_bound)
        new_cutoff = max(new_cutoff, lower_bound)
        
        # Determine final positivity based on expression threshold
        pos_mat[i, :] = sr >= new_cutoff
    
    return pos_mat


def compute_positivity_matrix(
    adata,
    rank_layer: str = 'ranks',
    count_layer: str = 'counts',
    p_threshold: float = 0.05,
    min_cutoff: float = 0.05,
    rank_cutoff: int = 8000,
    threads: Optional[int] = 16,
    simple = False,
    simple_threshold: float = 0.1
) -> np.ndarray:
    """
    Compute positivity matrix based on gene ranks and expression values from AnnData.
    
    Ultra-fast implementation using Numba JIT compilation with parallel execution.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing expression and optionally rank data.
        Expected shape: (n_samples, n_genes) following AnnData convention.
    rank_layer : str, default='ranks'
        Name of the layer containing gene ranking data. If not found, will use
        fallback threshold approach.
    count_layer : str, default='counts'
        Name of the layer containing RNA expression counts. If not found, will
        try 'counts', then .X, then .raw.X in that order.
    p_threshold : float, default=0.05
        Allowed deviation from the target positivity rate. Controls how much 
        the actual positivity rate can vary from the rank-based rate.
    min_cutoff : float, default=0.05
        Minimum expression threshold for positivity. Ensures a baseline 
        expression requirement.
    rank_cutoff : int, default=8000
        Rank threshold above which genes are not considered for positivity. 
        Genes with ranks higher than this value are initially excluded.
    threads : int, optional
        Number of parallel threads to use. If None, uses all available cores.
    simple_threshold : float, default=0.1
        Expression threshold to use if rank data is not available. Genes with
        expression >= this threshold are considered positive.
    
    Returns
    -------
    np.ndarray
        Boolean matrix indicating positivity. Shape: (n_samples, n_genes).
        True indicates the gene is considered positive for that sample.
    
    Notes
    -----
    **Data retrieval logic:**
    1. Try to get ranks from adata.layers[rank_layer]
    2. Try to get counts from adata.layers[count_layer]
    3. If counts not found, try .X, then .raw.X
    4. If ranks not found, use simple threshold: expression >= fallback_threshold
    
    **Performance:** 10-100x faster than pure Python/NumPy implementation.
    First call compiles (1-2s delay), subsequent calls use cached compiled code.
    
    Examples
    --------
    >>> # Standard usage with ranks and counts layers
    >>> pos_matrix = compute_positivity_matrix(adata)
    
    >>> # Custom layer names
    >>> pos_matrix = compute_positivity_matrix(
    ...     adata,
    ...     rank_layer='gene_ranks',
    ...     count_layer='raw_counts'
    ... )
    
    >>> # Fallback to threshold if no ranks available
    >>> pos_matrix = compute_positivity_matrix(
    ...     adata,
    ...     fallback_threshold=0.2
    ... )
    
    >>> # Add back to AnnData
    >>> adata.layers['positivity'] = pos_matrix
    """
    
    # Set number of threads if specified
    if threads is not None:
        numba.set_num_threads(threads)
    
    # Get expression data
    mat_rna = None
    if count_layer in adata.layers:
        mat_rna = adata.layers[count_layer]
    elif 'counts' in adata.layers:
        mat_rna = adata.layers['counts']
    elif adata.X is not None:
        mat_rna = adata.X
    elif hasattr(adata, 'raw') and adata.raw is not None:
        mat_rna = adata.raw.X
    else:
        raise ValueError("Could not find expression data in layers, .X, or .raw.X")
    
    # Convert to dense array if sparse
    if hasattr(mat_rna, 'toarray'):
        mat_rna = mat_rna.toarray()
    
    mat_rna = mat_rna.astype(np.float64, copy=True)
    
    # Check if ranks are available
    has_ranks = rank_layer in adata.layers
    
    if has_ranks and not simple:
        # Full algorithm with ranks
        gene_ranks = adata.layers[rank_layer]
        
        # Convert to dense if sparse
        if hasattr(gene_ranks, 'toarray'):
            gene_ranks = gene_ranks.toarray()
        
        gene_ranks = gene_ranks.astype(np.float64, copy=True)
        
        # Transpose to (n_genes, n_samples) for algorithm
        gene_ranks_T = gene_ranks.T
        mat_rna_T = mat_rna.T
        
        # Assign maximum rank penalty to very low expression values
        gene_ranks_T[mat_rna_T < 1e-05] = 999999999.0
        
        # Call Numba-compiled function
        pos_mat = _process_genes_numba(
            gene_ranks_T, 
            mat_rna_T, 
            p_threshold, 
            min_cutoff, 
            rank_cutoff
        )
        
        # Transpose back to (n_samples, n_genes)
        pos_mat = pos_mat.T
        
    else:
        # Fallback: simple threshold
        if simple:
            print(f"'Using simple threshold={simple_threshold}")
        else:
            print(f"'{rank_layer}' layer not found. Using simple threshold={simple_threshold}")
        
        pos_mat = mat_rna >= simple_threshold
    
    return pos_mat