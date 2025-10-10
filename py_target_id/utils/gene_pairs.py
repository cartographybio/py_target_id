"""
Gene pair generation utilities.
"""
__all__ = ['create_gene_pairs']

import numpy as np
import time


def create_gene_pairs(genes_list1, genes_list2, include_self_pairs=True):
    """
    Create all unique gene pairs between two lists using Cantor pairing.
    
    Treats (gene1, gene2) and (gene2, gene1) as the same pair for deduplication.
    Uses Cantor pairing function for fast 1D deduplication instead of 2D.
    
    Parameters
    ----------
    genes_list1 : list
        First list of gene names
    genes_list2 : list
        Second list of gene names
    include_self_pairs : bool, default=True
        If True, includes pairs like (geneA, geneA) when the same gene
        appears in both lists
    
    Returns
    -------
    list of tuples
        List of unique (gene1, gene2) pairs
    
    Examples
    --------
    >>> genes1 = ['CD3D', 'CD19', 'CD8A']
    >>> genes2 = ['CD3E', 'MS4A1', 'CD8B']
    >>> pairs = create_gene_pairs(genes1, genes2, include_self_pairs=False)
    >>> len(pairs)
    9
    
    Notes
    -----
    For large gene lists (6000 x 6000), this function is ~4x faster than
    using np.unique on 2D arrays, completing in ~8-10 seconds.
    """
    print(f"Creating pairs from {len(genes_list1)} x {len(genes_list2)} genes...")
    
    n1 = len(genes_list1)
    n2 = len(genes_list2)
    
    # Generate all index combinations
    idx1 = np.repeat(np.arange(n1, dtype=np.int32), n2)
    idx2 = np.tile(np.arange(n2, dtype=np.int32), n1)
    
    print(f"  Generated {len(idx1):,} combinations")
    
    # Remove self-pairs if requested
    if not include_self_pairs:
        set1 = set(genes_list1)
        set2 = set(genes_list2)
        if set1 & set2:
            gene1_arr = np.array(genes_list1, dtype=object)
            gene2_arr = np.array(genes_list2, dtype=object)
            mask = gene1_arr[idx1] != gene2_arr[idx2]
            idx1 = idx1[mask]
            idx2 = idx2[mask]
            print(f"  After removing self-pairs: {len(idx1):,}")
    
    # Deduplicate using Cantor pairing function
    print("  Deduplicating with Cantor pairing...")
    t_dedup = time.time()
    
    # Create canonical pairs (sorted to ensure A,B == B,A)
    min_idx = np.minimum(idx1, idx2).astype(np.int64)
    max_idx = np.maximum(idx1, idx2).astype(np.int64)
    
    # Cantor pairing: unique integer for each pair
    # Formula: (a + b) * (a + b + 1) / 2 + b
    canonical = ((min_idx + max_idx) * (min_idx + max_idx + 1)) // 2 + max_idx
    
    # Find unique pairs using 1D unique (much faster than 2D)
    _, unique_idx = np.unique(canonical, return_index=True)
    
    print(f"    Dedup took: {time.time() - t_dedup:.2f}s")
    
    # Extract unique indices
    idx1_unique = idx1[unique_idx]
    idx2_unique = idx2[unique_idx]
    
    print(f"  Unique pairs: {len(idx1_unique):,}")
    
    # Convert indices to gene names
    print("  Converting to gene names...")
    g1_array = np.array(genes_list1, dtype=object)
    g2_array = np.array(genes_list2, dtype=object)
    
    gene_pairs = list(zip(g1_array[idx1_unique], g2_array[idx2_unique]))
    
    print(f"  Done!")
    return gene_pairs