"""
Target ID
"""

# Define what gets exported
__all__ = ['compute_positivity_matrix']

"""
GPU-accelerated positivity matrix computation using PyTorch.
No compilation overhead, instant execution, works on CPU or GPU.
"""

import numpy as np
import torch
from typing import Optional

def compute_positivity_matrix(
    adata,
    rank_layer: str = 'ranks',
    count_layer: str = 'counts',
    p_threshold: float = 0.05,
    min_cutoff: float = 0.05,
    rank_cutoff: int = 8000,
    fallback_threshold: float = 0.1,
    simple: bool = False,
    device: Optional[str] = None,
    return_adata: bool = True,
    layer_name: str = 'positivity'
):
    """
    Compute positivity matrix using PyTorch (CPU or GPU accelerated).
    
    Much faster than Numba with no compilation overhead. Automatically uses
    GPU if available, falls back to CPU otherwise.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing expression and optionally rank data.
        Expected shape: (n_samples, n_genes).
    rank_layer : str, default='ranks'
        Name of layer containing gene ranking data.
    count_layer : str, default='counts'
        Name of layer containing RNA expression counts.
    p_threshold : float, default=0.05
        Allowed deviation from target positivity rate.
    min_cutoff : float, default=0.05
        Minimum expression threshold for positivity.
    rank_cutoff : int, default=8000
        Rank threshold above which genes are not considered.
    fallback_threshold : float, default=0.1
        Expression threshold if rank data unavailable or simple=True.
    simple : bool, default=False
        If True, use simple threshold approach.
    device : str, optional
        Device to use ('cuda', 'cpu', or None for auto-detect).
        If None, automatically uses GPU if available.
    return_adata : bool, default=True
        If True, returns AnnData with positivity added as a layer.
        If False, returns just the boolean numpy array.
    layer_name : str, default='positivity'
        Name of layer to store positivity matrix (only used if return_adata=True).
    
    Returns
    -------
    AnnData or np.ndarray
        If return_adata=True: Returns modified AnnData object with positivity 
        matrix added as adata.layers[layer_name].
        If return_adata=False: Returns boolean numpy array (n_samples, n_genes).
    
    Examples
    --------
    >>> # Returns AnnData with positivity layer
    >>> adata = compute_positivity_matrix(adata)
    >>> print(adata.layers['positivity'])
    
    >>> # Return just the matrix
    >>> pos_mat = compute_positivity_matrix(adata, return_adata=False)
    
    >>> # Custom layer name
    >>> adata = compute_positivity_matrix(adata, layer_name='pos_mat')
    
    >>> # Force CPU
    >>> adata = compute_positivity_matrix(adata, device='cpu')
    """
    
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
        raise ValueError("Could not find expression data")
    
    # Convert to dense if sparse
    if hasattr(mat_rna, 'toarray'):
        mat_rna = mat_rna.toarray()
    
    # Convert to torch tensors and move to device
    mat_rna_torch = torch.from_numpy(mat_rna.astype(np.float32)).to(device)
    
    # Check if using rank-based approach
    has_ranks = rank_layer in adata.layers
    
    if has_ranks and not simple:
        # Full algorithm with ranks
        gene_ranks = adata.layers[rank_layer]
        if hasattr(gene_ranks, 'toarray'):
            gene_ranks = gene_ranks.toarray()
        
        gene_ranks_torch = torch.from_numpy(gene_ranks.astype(np.float32)).to(device)
        
        # Transpose to (n_genes, n_samples)
        gene_ranks_T = gene_ranks_torch.T
        mat_rna_T = mat_rna_torch.T
        
        # Assign max rank penalty to very low expression
        gene_ranks_T = torch.where(mat_rna_T < 1e-05, 999999999.0, gene_ranks_T)
        
        # Process genes in parallel on GPU
        pos_mat = _process_genes_torch(
            gene_ranks_T,
            mat_rna_T,
            p_threshold,
            min_cutoff,
            rank_cutoff,
            device
        )
        
        # Transpose back and convert to numpy
        pos_mat = pos_mat.T.cpu().numpy()
        
    else:
        # Simple threshold approach
        if simple:
            print(f"Using simple threshold approach: expression >= {fallback_threshold}")
        else:
            print(f"Warning: '{rank_layer}' layer not found. Using fallback threshold={fallback_threshold}")
        
        pos_mat = (mat_rna_torch >= fallback_threshold).cpu().numpy()
    
    # Return based on preference
    if return_adata:
        adata.layers[layer_name] = pos_mat
        return adata
    else:
        return pos_mat


def _process_genes_torch(gene_ranks, mat_rna, p_threshold, min_cutoff, rank_cutoff, device):
    """
    Process all genes in parallel using PyTorch vectorized operations.
    
    Fully vectorized - no Python loops for maximum GPU performance.
    """
    n_genes, n_samples = gene_ranks.shape
    
    # Initial positivity based on rank cutoff (n_genes, n_samples)
    pos = gene_ranks <= rank_cutoff
    
    # For genes with no positive samples, handle separately
    n_pos_per_gene = pos.sum(dim=1)
    all_neg_mask = n_pos_per_gene == 0
    all_pos_mask = n_pos_per_gene == n_samples
    
    # Calculate median expression of initially positive samples
    # Set negative samples to inf so they don't affect median
    mat_for_median = torch.where(pos, mat_rna, torch.tensor(float('inf'), device=device))
    med_pos = torch.median(mat_for_median, dim=1).values
    
    # Handle edge cases
    med_pos = torch.where(all_neg_mask, torch.tensor(0.0, device=device), med_pos)
    
    # Expand positive set: include samples >= median
    pos = pos | (mat_rna >= med_pos.unsqueeze(1))
    
    # Separate positive and negative values for each gene
    # Use masked operations to avoid loops
    pos_vals_for_max = torch.where(~pos, mat_rna, torch.tensor(float('-inf'), device=device))
    pos_vals_for_min = torch.where(pos, mat_rna, torch.tensor(float('inf'), device=device))
    
    max_neg = pos_vals_for_max.max(dim=1).values  # max of negative samples
    min_pos = pos_vals_for_min.min(dim=1).values  # min of positive samples
    
    # Recalculate median of expanded positive set
    mat_for_median2 = torch.where(pos, mat_rna, torch.tensor(float('inf'), device=device))
    med_pos = torch.median(mat_for_median2, dim=1).values
    
    # Handle edge cases
    max_neg = torch.where(torch.isinf(max_neg), torch.tensor(0.0, device=device), max_neg)
    min_pos = torch.where(torch.isinf(min_pos), torch.tensor(0.0, device=device), min_pos)
    med_pos = torch.where(torch.isinf(med_pos), torch.tensor(0.0, device=device), med_pos)
    
    # Initial adaptive cutoff (vectorized)
    new_cutoff = (max_neg + min_pos) / 2.0
    
    # Apply minimum constraints (vectorized)
    new_cutoff = torch.maximum(new_cutoff, torch.tensor(min_cutoff, device=device))
    new_cutoff = torch.maximum(new_cutoff, 0.1 * med_pos)
    
    # Calculate quantile bounds (vectorized)
    pos_rate = (gene_ranks <= rank_cutoff).float().mean(dim=1)
    q_lower = torch.clamp(pos_rate - p_threshold, min=0.0)
    q_upper = torch.clamp(pos_rate + p_threshold, max=1.0)
    
    # Compute quantiles per gene using linear interpolation (match R/NumPy default)
    # Sort each gene's expression values
    sorted_rna, _ = torch.sort(mat_rna, dim=1)
    
    # Convert quantile to continuous index (with interpolation)
    # R default: quantile type 7 (linear interpolation)
    indices_upper_f = (1.0 - q_lower) * (n_samples - 1)
    indices_lower_f = (1.0 - q_upper) * (n_samples - 1)
    
    # Get integer parts and fractional parts for interpolation
    idx_upper_low = indices_upper_f.long()
    idx_upper_high = torch.clamp(idx_upper_low + 1, max=n_samples - 1)
    frac_upper = indices_upper_f - idx_upper_low.float()
    
    idx_lower_low = indices_lower_f.long()
    idx_lower_high = torch.clamp(idx_lower_low + 1, max=n_samples - 1)
    frac_lower = indices_lower_f - idx_lower_low.float()
    
    # Linear interpolation
    arange_idx = torch.arange(n_genes, device=device)
    upper_bound = (1 - frac_upper) * sorted_rna[arange_idx, idx_upper_low] + \
                  frac_upper * sorted_rna[arange_idx, idx_upper_high]
    lower_bound = (1 - frac_lower) * sorted_rna[arange_idx, idx_lower_low] + \
                  frac_lower * sorted_rna[arange_idx, idx_lower_high]
    
    # Apply quantile bounds (vectorized)
    new_cutoff = torch.minimum(new_cutoff, upper_bound)
    new_cutoff = torch.maximum(new_cutoff, lower_bound)
    
    # Final positivity determination
    pos_mat = mat_rna >= new_cutoff.unsqueeze(1)
    
    # Handle edge cases
    pos_mat = torch.where(all_neg_mask.unsqueeze(1), torch.tensor(False, device=device), pos_mat)
    pos_mat = torch.where(all_pos_mask.unsqueeze(1), torch.tensor(True, device=device), pos_mat)
    
    return pos_mat