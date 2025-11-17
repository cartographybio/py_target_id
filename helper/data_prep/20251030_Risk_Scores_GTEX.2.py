#########################################################################################################
#########################################################################################################
import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
from importlib.resources import files
from py_target_id import utils


gtex = tid.utils.get_gtex_adata()
risk = tid.utils.get_multi_risk_scores()
gene_pairs = [tuple(gene.split("_")) for gene in risk["gene_name"].tolist()]

def compute_gtex_risk_scores_single(gtex):
    """
    Compute composite hazard-weighted GTEx score for genes.
    
    Parameters:
    -----------
    gtex : AnnData
        GTEx AnnData object
    
    Returns:
    --------
    pd.DataFrame with columns: gene_name, Hazard_GTEX_v1
    """
    import numpy as np
    import pandas as pd
    from py_target_id import run
    from py_target_id import utils
    
    # Compute median expression by tissue
    gtex_med = utils.summarize_matrix(mat=gtex.X, groups=gtex.obs["GTEX"].values, metric="median", axis=0)
    
    # Build tissue to hazard mapping
    tissue_to_hazard = {}
    for tier, config in run.hazard_map.items():
        for tissue in config['gtex_tissues']:
            tissue_to_hazard[tissue] = config['hazard_score']
    
    hazard_score = gtex_med.index.map(tissue_to_hazard)
    
    # Threshold expression
    gtex_med2 = np.where(gtex_med >= 25, 10, 
               np.where(gtex_med >= 10, 5, 
               np.where(gtex_med >= 5, 1, 
               np.where(gtex_med > 1, 0.25, 0))))
    
    hazard_score = np.array(hazard_score)
    
    # Composite score: tissue-weighted expression + critical tissue penalty
    tissue_weighted = np.sum(gtex_med2.T * hazard_score, axis=1)
    critical_penalty = np.sqrt(np.sum(gtex_med2[hazard_score==4, :] >= 5, axis=0)) * 10
    
    return pd.DataFrame({
        "gene_name": gtex.var_names,
        "Hazard_GTEX_v1": tissue_weighted + critical_penalty
    })

risk = tid.utils.get_multi_risk_scores()
gene_pairs = [tuple(gene.split("_")) for gene in risk["gene_name"].tolist()]

def compute_gtex_risk_scores_multi(gtex, gene_pairs, batch_size=5000):

    import numpy as np
    import pandas as pd
    from py_target_id import run
    from py_target_id import utils
    
    # Validate gene_pairs
    if gene_pairs is None or len(gene_pairs) == 0:
        raise ValueError("gene_pairs is required and must contain at least one gene pair")
    
    # Extract all unique genes from pairs
    all_genes_in_pairs = set()
    for gene1, gene2 in gene_pairs:
        all_genes_in_pairs.add(gene1)
        all_genes_in_pairs.add(gene2)

    # Check which genes are present in the matrices
    available_genes = set(gtex.var_names)
    missing_genes = all_genes_in_pairs - available_genes
    
    if missing_genes:
        raise ValueError(f"The following genes are not found in the data: {sorted(missing_genes)}")

    genes_to_keep = sorted(all_genes_in_pairs)
    
    print(f"Subsetting GTEX to {len(genes_to_keep)} genes...")
    gtex_subset = gtex[:, genes_to_keep].copy()

    # Build tissue to hazard mapping
    tissue_to_hazard = {}
    for tier, config in run.hazard_map.items():
        for tissue in config['gtex_tissues']:
            tissue_to_hazard[tissue] = config['hazard_score']
    
    hazard_score = gtex_med.index.map(tissue_to_hazard)
    
    # Convert gene pairs to indices
    print(f"Converting {len(gene_pairs)} gene pairs to indices...")
    gene_to_idx = pd.Series(range(len(genes_to_keep)), index=genes_to_keep)
    if isinstance(gene_pairs, list):
        gene_pairs_df = pd.DataFrame(gene_pairs, columns=['gene1', 'gene2'])
    else:
        gene_pairs_df = gene_pairs

    gx_all = gene_to_idx[gene_pairs_df['gene1'].values].values
    gy_all = gene_to_idx[gene_pairs_df['gene2'].values].values
    n_batches = int(np.ceil(len(gx_all) / batch_size))

    for batch_idx in range(n_batches):

        iter_start = time.time()
        start = batch_idx * batch_size
        end = min(start + batch_size, len(gx_all))

        gx_t = gx_all[start:end]
        gy_t = gy_all[start:end]

        print(f"  {batch_idx + 1}/{n_batches} | ", end='', flush=True)

        np.minimum(gtex.X[:,gx_t], gtex.X[:,gy_t])

        gtex_med = utils.summarize_matrix(mat=np.minimum(gtex.X[:,gx_t], gtex.X[:,gy_t]), groups=gtex.obs["GTEX"].values, metric="median", axis=0)

        # Composite score: tissue-weighted expression + critical tissue penalty
        tissue_weighted = np.sum(gtex_med2.T * hazard_score, axis=1)
        critical_penalty = np.sqrt(np.sum(gtex_med2[hazard_score==4, :] >= 5, axis=0)) * 10
        
        names = [f"{genes[i]}_{genes[j]}" for i, j in zip(gx_all[start:end], gy_all[start:end])]

        return_df = pd.DataFrame({
            "gene_name": names,
            "Hazard_GTEX_v1": tissue_weighted + critical_penalty
        })

    df_all = pd.concat(return_df)


    import numpy as np
    import pandas as pd
    import torch
    import time
    import gc
    from scipy import sparse
    from py_target_id import run, utils
    
    # Validate gene_pairs
    if gene_pairs is None or len(gene_pairs) == 0:
        raise ValueError("gene_pairs is required and must contain at least one gene pair")
    
    print(f"Validating {len(gene_pairs)} gene pairs...")
    
    # Extract all unique genes from pairs
    all_genes_in_pairs = set()
    for gene1, gene2 in gene_pairs:
        all_genes_in_pairs.add(gene1)
        all_genes_in_pairs.add(gene2)
    
    # Check which genes are present
    available_genes = set(gtex.var_names)
    missing_genes = all_genes_in_pairs - available_genes
    
    if missing_genes:
        raise ValueError(f"The following genes are not found in the data: {sorted(missing_genes)}")
    
    genes_to_keep = sorted(all_genes_in_pairs)
    print(f"Subsetting GTEX to {len(genes_to_keep)} genes...")
    gtex_subset = gtex[:, genes_to_keep].copy()
    
    # Load to memory if virtual
    if type(gtex_subset).__name__ in ['VirtualAnnData', 'TransposedAnnData']:
        print("  Loading data to memory...")
        gtex_subset = gtex_subset.to_memory(dense=True, chunk_size=5000, show_progress=True)
    
    gtex_subset.X = gtex_subset.X.toarray()

    # Build tissue to hazard mapping
    tissue_to_hazard = {}
    for tier, config in run.hazard_map.items():
        for tissue in config['gtex_tissues']:
            tissue_to_hazard[tissue] = config['hazard_score']
    
    hazard_score = np.array([tissue_to_hazard.get(t, 0) for t in gtex_subset.obs["GTEX"].values])
    
    # Convert gene pairs to indices
    print(f"Converting {len(gene_pairs)} gene pairs to indices...")
    gene_to_idx = pd.Series(range(len(genes_to_keep)), index=genes_to_keep)
    
    if isinstance(gene_pairs, list):
        gene_pairs_df = pd.DataFrame(gene_pairs, columns=['gene1', 'gene2'])
    else:
        gene_pairs_df = gene_pairs
    
    gx_all = gene_to_idx[gene_pairs_df['gene1'].values].values
    gy_all = gene_to_idx[gene_pairs_df['gene2'].values].values
        
    # Load to GPU once
    device_obj = torch.device(device)
    print(f"Loading expression matrix to {device}...")
    X_gpu = torch.tensor(np.array(gtex_subset.X.T, dtype=np.float32), device=device_obj)
    hazard_gpu = torch.tensor(hazard_score.astype(np.float32), device=device_obj)
    
    n_batches = int(np.ceil(len(gx_all) / batch_size))
    results = []
    overall_start = time.time()
    
    print(f"Processing {len(gx_all)} pairs in {n_batches} batches...\n")
   import numpy as np
    import pandas as pd
    import torch
    import time
    import gc
    from scipy import sparse
    from py_target_id import run, utils
    
    # Validate gene_pairs
    if gene_pairs is None or len(gene_pairs) == 0:
        raise ValueError("gene_pairs is required and must contain at least one gene pair")
    
    print(f"Validating {len(gene_pairs)} gene pairs...")
    
    # Extract all unique genes from pairs
    all_genes_in_pairs = set()
    for gene1, gene2 in gene_pairs:
        all_genes_in_pairs.add(gene1)
        all_genes_in_pairs.add(gene2)
    
    # Check which genes are present
    available_genes = set(gtex.var_names)
    missing_genes = all_genes_in_pairs - available_genes
    
    if missing_genes:
        raise ValueError(f"The following genes are not found in the data: {sorted(missing_genes)}")
    
    genes_to_keep = sorted(all_genes_in_pairs)
    print(f"Subsetting GTEX to {len(genes_to_keep)} genes...")
    gtex_subset = gtex[:, genes_to_keep].copy()
    
    # Load to memory if virtual
    if type(gtex_subset).__name__ in ['VirtualAnnData', 'TransposedAnnData']:
        print("  Loading data to memory...")
        gtex_subset = gtex_subset.to_memory(dense=True, chunk_size=5000, show_progress=True)
    
    gtex_subset.X = gtex_subset.X.toarray()

    # Build tissue to hazard mapping
    tissue_to_hazard = {}
    for tier, config in run.hazard_map.items():
        for tissue in config['gtex_tissues']:
            tissue_to_hazard[tissue] = config['hazard_score']
    
    # hazard_score should match gtex_subset observations (tissues), not all original tissues
    hazard_score = np.array([tissue_to_hazard.get(t, 0) for t in gtex_subset.obs["GTEX"].values])
    
    # Convert gene pairs to indices
    print(f"Converting {len(gene_pairs)} gene pairs to indices...")
    gene_to_idx = pd.Series(range(len(genes_to_keep)), index=genes_to_keep)
    
    if isinstance(gene_pairs, list):
        gene_pairs_df = pd.DataFrame(gene_pairs, columns=['gene1', 'gene2'])
    else:
        gene_pairs_df = gene_pairs    for batch_idx in range(n_batches):
        iter_start = time.time()
        start = batch_idx * batch_size
        end = min(start + batch_size, len(gx_all))
        
        gx_t = torch.tensor(gx_all[start:end], device=device_obj, dtype=torch.long)
        gy_t = torch.tensor(gy_all[start:end], device=device_obj, dtype=torch.long)
        
        print(f"  {batch_idx + 1}/{n_batches} | ", end='', flush=True)
        
        # GPU minimum (tissues x pairs)
        min_expr = torch.minimum(X_gpu[:, gx_t], X_gpu[:, gy_t])
        
        # Compute Median
        gtex_med = utils.summarize_matrix(mat=min_expr, groups=gtex.obs["GTEX"].values, metric="median", axis=0)

        # Threshold expression on GPU
        gtex_med2 = torch.where(gtex_med >= 25, 10.0,
                   torch.where(gtex_med >= 10, 5.0,
                   torch.where(gtex_med >= 5, 1.0,
                   torch.where(gtex_med > 1, 0.25, 0.0))))
        
        # Composite score (vectorized on GPU)
        tissue_weighted = torch.sum(gtex_med2 * hazard_gpu.unsqueeze(1), dim=0)
        
        # Critical tissue penalty (hazard_score == 4 for tier-1)
        tier1_mask = (hazard_gpu == 4.0)
        if tier1_mask.any():
            critical_penalty = torch.sqrt(torch.sum((gtex_med2[tier1_mask] >= 5).float(), dim=0)) * 10
        else:
            critical_penalty = torch.zeros(gtex_med2.shape[1], device=device_obj)
        
        # Create gene pair names
        names = [f"{genes_to_keep[i]}_{genes_to_keep[j]}" for i, j in zip(gx_all[start:end], gy_all[start:end])]
        
        batch_df = pd.DataFrame({
            "gene_name": names,
            "Hazard_GTEX_v1": (tissue_weighted + critical_penalty).cpu().numpy()
        })
        
        results.append(batch_df)
        
        # Cleanup
        del min_expr, gtex_med2, tissue_weighted, critical_penalty, gx_t, gy_t
        torch.cuda.empty_cache()
        gc.collect()
        
        iter_time = time.time() - iter_start
        total_time = time.time() - overall_start
        print(f"{iter_time:.2f}s | Total: {total_time/60:.1f}m")
    
    # Cleanup GPU memory
    del X_gpu, hazard_gpu
    torch.cuda.empty_cache()
    gc.collect()
    
    df_all = pd.concat(results, ignore_index=True)


def compute_hazard_gtex_pairs_gpu(gtex, gene_pairs, batch_size=10000, device='cuda'):
    """
    GPU-optimized hazard-weighted GTEx scores for gene pairs.
    
    Parameters:
    -----------
    gtex : AnnData
        GTEx AnnData object
    gene_pairs : list of tuples
        List of (gene1, gene2) tuples
    batch_size : int
        Number of pairs per batch
    device : str
        'cuda' or 'cpu'
    
    Returns:
    --------
    pd.DataFrame with columns: gene_name, Hazard_GTEX_v1
    """
    import numpy as np
    import pandas as pd
    import torch
    import time
    import gc
    from scipy import sparse
    from py_target_id import run, utils
    
    # Validate gene_pairs
    if gene_pairs is None or len(gene_pairs) == 0:
        raise ValueError("gene_pairs is required and must contain at least one gene pair")
    
    print(f"Validating {len(gene_pairs)} gene pairs...")
    
    # Extract all unique genes from pairs
    all_genes_in_pairs = set()
    for gene1, gene2 in gene_pairs:
        all_genes_in_pairs.add(gene1)
        all_genes_in_pairs.add(gene2)
    
    # Check which genes are present
    available_genes = set(gtex.var_names)
    missing_genes = all_genes_in_pairs - available_genes
    
    if missing_genes:
        raise ValueError(f"The following genes are not found in the data: {sorted(missing_genes)}")
    
    genes_to_keep = sorted(all_genes_in_pairs)
    print(f"Subsetting GTEX to {len(genes_to_keep)} genes...")
    gtex_subset = gtex[:, genes_to_keep].copy()
    
    # Load to memory if virtual
    if type(gtex_subset).__name__ in ['VirtualAnnData', 'TransposedAnnData']:
        print("  Loading data to memory...")
        gtex_subset = gtex_subset.to_memory(dense=True, chunk_size=5000, show_progress=True)
    
    gtex_subset.X = gtex_subset.X.toarray()

    # Build tissue to hazard mapping
    tissue_to_hazard = {}
    for tier, config in run.hazard_map.items():
        for tissue in config['gtex_tissues']:
            tissue_to_hazard[tissue] = config['hazard_score']
    
    # hazard_score should match gtex_subset observations (tissues), not all original tissues
    hazard_score = np.array([tissue_to_hazard.get(t, 0) for t in gtex_subset.obs["GTEX"].values])
    
    # Convert gene pairs to indices
    print(f"Converting {len(gene_pairs)} gene pairs to indices...")
    gene_to_idx = pd.Series(range(len(genes_to_keep)), index=genes_to_keep)
    
    if isinstance(gene_pairs, list):
        gene_pairs_df = pd.DataFrame(gene_pairs, columns=['gene1', 'gene2'])
    else:
        gene_pairs_df = gene_pairs
    
    gx_all = gene_to_idx[gene_pairs_df['gene1'].values].values
    gy_all = gene_to_idx[gene_pairs_df['gene2'].values].values
        
    # Load to GPU once before batching
    device_obj = torch.device(device)
    print(f"Loading expression matrix to {device}...")
    X_gpu = torch.tensor(gtex_subset.X.astype(np.float32), device=device_obj)
    
    n_batches = int(np.ceil(len(gx_all) / batch_size))
    results = []
    overall_start = time.time()
    
    print(f"Processing {len(gx_all)} pairs in {n_batches} batches...\n")
    
    for batch_idx in range(n_batches):
        iter_start = time.time()
        start = batch_idx * batch_size
        end = min(start + batch_size, len(gx_all))
        
        gx_t = gx_all[start:end]
        gy_t = gy_all[start:end]
        
        print(f"  {batch_idx + 1}/{n_batches} | ", end='', flush=True)
        
        # GPU minimum (cells, pairs)
        min_expr = torch.minimum(X_gpu[gx_t, :], X_gpu[gy_t, :])
        
        # Compute median by tissue
        gtex_med = utils.summarize_matrix(mat=min_expr.cpu().numpy(), groups=gtex_subset.obs["GTEX"].values, metric="median", axis=0)
        
        # Build tissue to hazard mapping (only once per batch, could be moved out)
        tissue_to_hazard = {}
        for tier, config in run.hazard_map.items():
            for tissue in config['gtex_tissues']:
                tissue_to_hazard[tissue] = config['hazard_score']
        
        hazard_score = gtex_med.index.map(tissue_to_hazard).values.astype(np.float32)
        
        # Threshold expression
        gtex_med2 = np.where(gtex_med >= 25, 10, 
                   np.where(gtex_med >= 10, 5, 
                   np.where(gtex_med >= 5, 1, 
                   np.where(gtex_med > 1, 0.25, 0))))
        
        # Convert to GPU tensors
        gtex_med2_gpu = torch.tensor(gtex_med2.astype(np.float32), device=device_obj)
        hazard_gpu_batch = torch.tensor(hazard_score, device=device_obj)
        
        # Composite score (tissues, pairs) * (tissues, 1)
        tissue_weighted = torch.sum(gtex_med2_gpu * hazard_gpu_batch.unsqueeze(1), dim=0)
        
        # Critical tissue penalty
        tier1_mask = (hazard_gpu_batch == 4.0)
        if tier1_mask.any():
            critical_penalty = torch.sqrt(torch.sum((gtex_med2_gpu[tier1_mask] >= 5).float(), dim=0)) * 10
        else:
            critical_penalty = torch.zeros(gtex_med2_gpu.shape[1], device=device_obj)
        
        # Create gene pair names
        names = [f"{genes_to_keep[i]}_{genes_to_keep[j]}" for i, j in zip(gx_all[start:end], gy_all[start:end])]
        
        batch_df = pd.DataFrame({
            "gene_name": names,
            "Hazard_GTEX_v1": (tissue_weighted + critical_penalty).cpu().numpy()
        })
        
        results.append(batch_df)
        
        # Cleanup
        del min_expr, gtex_med2_gpu, tissue_weighted, critical_penalty
        torch.cuda.empty_cache()
        gc.collect()
        
        iter_time = time.time() - iter_start
        total_time = time.time() - overall_start
        print(f"{iter_time:.2f}s | Total: {total_time/60:.1f}m")
    
    # Cleanup GPU memory
    del X_gpu
    torch.cuda.empty_cache()
    gc.collect()
    
    df_all = pd.concat(results, ignore_index=True)
    return df_all
def compute_hazard_gtex_pairs_gpu(gtex, gene_pairs, batch_size=10000, device='cuda'):
    """
    GPU-optimized hazard-weighted GTEx scores for gene pairs.
    
    Parameters:
    -----------
    gtex : AnnData
        GTEx AnnData object
    gene_pairs : list of tuples
        List of (gene1, gene2) tuples
    batch_size : int
        Number of pairs per batch
    device : str
        'cuda' or 'cpu'
    
    Returns:
    --------
    pd.DataFrame with columns: gene_name, Hazard_GTEX_v1
    """
    import numpy as np
    import pandas as pd
    import torch
    import time
    import gc
    from scipy import sparse
    from py_target_id import run, utils
    
    # Validate gene_pairs
    if gene_pairs is None or len(gene_pairs) == 0:
        raise ValueError("gene_pairs is required and must contain at least one gene pair")
    
    print(f"Validating {len(gene_pairs)} gene pairs...")
    
    # Extract all unique genes from pairs
    all_genes_in_pairs = set()
    for gene1, gene2 in gene_pairs:
        all_genes_in_pairs.add(gene1)
        all_genes_in_pairs.add(gene2)
    
    # Check which genes are present
    available_genes = set(gtex.var_names)
    missing_genes = all_genes_in_pairs - available_genes
    
    if missing_genes:
        raise ValueError(f"The following genes are not found in the data: {sorted(missing_genes)}")
    
    genes_to_keep = sorted(all_genes_in_pairs)
    print(f"Subsetting GTEX to {len(genes_to_keep)} genes...")
    gtex_subset = gtex[:, genes_to_keep].copy()
    
    # Load to memory if virtual
    if type(gtex_subset).__name__ in ['VirtualAnnData', 'TransposedAnnData']:
        print("  Loading data to memory...")
        gtex_subset = gtex_subset.to_memory(dense=True, chunk_size=5000, show_progress=True)
    
    gtex_subset.X = gtex_subset.X.toarray()

    # Build tissue to hazard mapping
    tissue_to_hazard = {}
    for tier, config in run.hazard_map.items():
        for tissue in config['gtex_tissues']:
            tissue_to_hazard[tissue] = config['hazard_score']
    
    # hazard_score should match gtex_subset observations (tissues), not all original tissues
    hazard_score = np.array([tissue_to_hazard.get(t, 0) for t in gtex_subset.obs["GTEX"].values])
    
    # Convert gene pairs to indices
    print(f"Converting {len(gene_pairs)} gene pairs to indices...")
    gene_to_idx = pd.Series(range(len(genes_to_keep)), index=genes_to_keep)
    
    if isinstance(gene_pairs, list):
        gene_pairs_df = pd.DataFrame(gene_pairs, columns=['gene1', 'gene2'])
    else:
        gene_pairs_df = gene_pairs
    
    gx_all = gene_to_idx[gene_pairs_df['gene1'].values].values
    gy_all = gene_to_idx[gene_pairs_df['gene2'].values].values
        
    # Load to GPU once before batching
    device_obj = torch.device(device)
    print(f"Loading expression matrix to {device}...")
    X_gpu = torch.tensor(gtex_subset.X.astype(np.float32), device=device_obj)
    
    n_batches = int(np.ceil(len(gx_all) / batch_size))
    results = []
    overall_start = time.time()
    
    print(f"Processing {len(gx_all)} pairs in {n_batches} batches...\n")
    
    for batch_idx in range(n_batches):
        iter_start = time.time()
        start = batch_idx * batch_size
        end = min(start + batch_size, len(gx_all))
        
        gx_t = gx_all[start:end]
        gy_t = gy_all[start:end]
        
        print(f"  {batch_idx + 1}/{n_batches} | ", end='', flush=True)
        
        # GPU minimum (pairs, cells) - element-wise min for each pair across cells
        min_expr = torch.minimum(X_gpu[gx_t, :], X_gpu[gy_t, :])
        
        # Compute median by tissue (pairs, tissues)
        gtex_med = utils.summarize_matrix(mat=min_expr.cpu().numpy(), groups=gtex_subset.obs["GTEX"].values, metric="median", axis=1)
        
        # Build tissue to hazard mapping
        tissue_to_hazard = {}
        for tier, config in run.hazard_map.items():
            for tissue in config['gtex_tissues']:
                tissue_to_hazard[tissue] = config['hazard_score']
        
        hazard_score = gtex_med.columns.map(tissue_to_hazard).values.astype(np.float32)
        
        # Threshold expression
        gtex_med2 = np.where(gtex_med >= 25, 10, 
                   np.where(gtex_med >= 10, 5, 
                   np.where(gtex_med >= 5, 1, 
                   np.where(gtex_med > 1, 0.25, 0))))
        
        # Composite score (pairs, tissues) * (tissues,) = (pairs,)
        tissue_weighted = np.sum(gtex_med2 * hazard_score, axis=1)
        
        # Critical tissue penalty
        tier1_mask = (hazard_score == 4)
        if tier1_mask.any():
            critical_penalty = np.sqrt(np.sum(gtex_med2[:, tier1_mask] >= 5, axis=1)) * 10
        else:
            critical_penalty = np.zeros(len(tissue_weighted))
        
        # Create gene pair names
        names = [f"{genes_to_keep[i]}_{genes_to_keep[j]}" for i, j in zip(gx_all[start:end], gy_all[start:end])]
        
        batch_df = pd.DataFrame({
            "gene_name": names,
            "Hazard_GTEX_v1": tissue_weighted + critical_penalty
        })
        
        results.append(batch_df)
        
        # Cleanup
        del min_expr
        torch.cuda.empty_cache()
        gc.collect()
        
        iter_time = time.time() - iter_start
        total_time = time.time() - overall_start
        print(f"{iter_time:.2f}s | Total: {total_time/60:.1f}m")
    
    # Cleanup GPU memory
    del X_gpu
    torch.cuda.empty_cache()
    gc.collect()
    
    df_all = pd.concat(results, ignore_index=True)
    return df_all


# Usage:
# gene_pairs = [tuple(gene.split("_")) for gene in risk["gene_name"].tolist()]
# df = compute_hazard_gtex_pairs_gpu(gtex, gene_pairs, batch_size=10000, device='cuda')

# Usage:
# gene_pairs = [tuple(gene.split("_")) for gene in risk["gene_name"].tolist()]
# df = compute_hazard_gtex_pairs_gpu(gtex, gene_pairs, batch_size=10000, device='cuda')





def compute_hazard_gtex_pairs_gpu(gtex, gene_pairs, batch_size=10000, device='cuda'):
    """
    GPU-optimized hazard-weighted GTEx scores for gene pairs.
    
    Parameters:
    -----------
    gtex : AnnData
        GTEx AnnData object
    gene_pairs : list of tuples
        List of (gene1, gene2) tuples
    batch_size : int
        Number of pairs per batch
    device : str
        'cuda' or 'cpu'
    
    Returns:
    --------
    pd.DataFrame with columns: gene_name, Hazard_GTEX_v1
    """
    import numpy as np
    import pandas as pd
    import torch
    import time
    import gc
    from scipy import sparse
    from py_target_id import run, utils
    
    # Validate gene_pairs
    if gene_pairs is None or len(gene_pairs) == 0:
        raise ValueError("gene_pairs is required and must contain at least one gene pair")
    
    print(f"Validating {len(gene_pairs)} gene pairs...")
    
    # Extract all unique genes from pairs
    all_genes_in_pairs = set()
    for gene1, gene2 in gene_pairs:
        all_genes_in_pairs.add(gene1)
        all_genes_in_pairs.add(gene2)
    
    # Check which genes are present
    available_genes = set(gtex.var_names)
    missing_genes = all_genes_in_pairs - available_genes
    
    if missing_genes:
        raise ValueError(f"The following genes are not found in the data: {sorted(missing_genes)}")
    
    genes_to_keep = sorted(all_genes_in_pairs)
    print(f"Subsetting GTEX to {len(genes_to_keep)} genes...")
    gtex_subset = gtex[:, genes_to_keep].copy()
    
    # Load to memory if virtual
    if type(gtex_subset).__name__ in ['VirtualAnnData', 'TransposedAnnData']:
        print("  Loading data to memory...")
        gtex_subset = gtex_subset.to_memory(dense=True, chunk_size=5000, show_progress=True)
    
    gtex_subset.X = gtex_subset.X.toarray()

    # Build tissue to hazard mapping
    tissue_to_hazard = {}
    for tier, config in run.hazard_map.items():
        for tissue in config['gtex_tissues']:
            tissue_to_hazard[tissue] = config['hazard_score']
    
    # hazard_score should match gtex_subset observations (tissues), not all original tissues
    hazard_score = np.array([tissue_to_hazard.get(t, 0) for t in gtex_subset.obs["GTEX"].values])
    
    # Convert gene pairs to indices
    print(f"Converting {len(gene_pairs)} gene pairs to indices...")
    gene_to_idx = pd.Series(range(len(genes_to_keep)), index=genes_to_keep)
    
    if isinstance(gene_pairs, list):
        gene_pairs_df = pd.DataFrame(gene_pairs, columns=['gene1', 'gene2'])
    else:
        gene_pairs_df = gene_pairs
    
    gx_all = gene_to_idx[gene_pairs_df['gene1'].values].values
    gy_all = gene_to_idx[gene_pairs_df['gene2'].values].values
        
    # Load to GPU once before batching
    device_obj = torch.device(device)
    print(f"Loading expression matrix to {device}...")
    X_gpu = torch.tensor(gtex_subset.X.T.astype(np.float32), device=device_obj)
    hazard_gpu = torch.tensor(hazard_score.astype(np.float32), device=device_obj)
    
    n_batches = int(np.ceil(len(gx_all) / batch_size))
    results = []
    overall_start = time.time()
    
    print(f"Processing {len(gx_all)} pairs in {n_batches} batches...\n")
    
    for batch_idx in range(n_batches):
        iter_start = time.time()
        start = batch_idx * batch_size
        end = min(start + batch_size, len(gx_all))
        
        gx_t = gx_all[start:end]
        gy_t = gy_all[start:end]
        
        print(f"  {batch_idx + 1}/{n_batches} | ", end='', flush=True)
        
        # GPU minimum (tissues x pairs)
        min_expr = torch.minimum(X_gpu[gx_t, :], X_gpu[gy_t, :])
        
        # Threshold expression on GPU
        gtex_med2 = torch.where(min_expr >= 25, 10.0,
                   torch.where(min_expr >= 10, 5.0,
                   torch.where(min_expr >= 5, 1.0,
                   torch.where(min_expr > 1, 0.25, 0.0))))
        
        # gtex_med2 is (tissues, pairs), hazard_gpu is (tissues,)
        # Composite score (vectorized on GPU)
        tissue_weighted = torch.sum(gtex_med2 * hazard_gpu.unsqueeze(1), dim=0)
        
        # Critical tissue penalty (hazard_score == 4 for tier-1)
        tier1_mask = (hazard_gpu == 4.0)
        if tier1_mask.any():
            critical_penalty = torch.sqrt(torch.sum((gtex_med2[tier1_mask] >= 5).float(), dim=0)) * 10
        else:
            critical_penalty = torch.zeros(gtex_med2.shape[1], device=device_obj)
        
        # Create gene pair names
        names = [f"{genes_to_keep[i]}_{genes_to_keep[j]}" for i, j in zip(gx_all[start:end], gy_all[start:end])]
        
        batch_df = pd.DataFrame({
            "gene_name": names,
            "Hazard_GTEX_v1": (tissue_weighted + critical_penalty).cpu().numpy()
        })
        
        results.append(batch_df)
        
        # Cleanup
        del min_expr, gtex_med2, tissue_weighted, critical_penalty
        torch.cuda.empty_cache()
        gc.collect()
        
        iter_time = time.time() - iter_start
        total_time = time.time() - overall_start
        print(f"{iter_time:.2f}s | Total: {total_time/60:.1f}m")
    
    # Cleanup GPU memory
    del X_gpu, hazard_gpu
    torch.cuda.empty_cache()
    gc.collect()
    
    df_all = pd.concat(results, ignore_index=True)
    return df_all


# Usage:
# gene_pairs = [tuple(gene.split("_")) for gene in risk["gene_name"].tolist()]
# df = compute_hazard_gtex_pairs_gpu(gtex, gene_pairs, batch_size=10000, device='cuda')

































    import numpy as np
    import pandas as pd
    import time
    from py_target_id import run, utils
    
    # Validate gene_pairs
    if gene_pairs is None or len(gene_pairs) == 0:
        raise ValueError("gene_pairs is required and must contain at least one gene pair")
    
    # Extract all unique genes from pairs
    all_genes_in_pairs = set()
    for gene1, gene2 in gene_pairs:
        all_genes_in_pairs.add(gene1)
        all_genes_in_pairs.add(gene2)
    
    # Check which genes are present in the matrices
    available_genes = set(gtex.var_names)
    missing_genes = all_genes_in_pairs - available_genes
    
    if missing_genes:
        raise ValueError(f"The following genes are not found in the data: {sorted(missing_genes)}")
    
    genes_to_keep = sorted(all_genes_in_pairs)
    print(f"Subsetting GTEX to {len(genes_to_keep)} genes...")
    gtex_subset = gtex[:, genes_to_keep].copy()
    
    # Build tissue to hazard mapping
    tissue_to_hazard = {}
    for tier, config in run.hazard_map.items():
        for tissue in config['gtex_tissues']:
            tissue_to_hazard[tissue] = config['hazard_score']
    
    hazard_score = np.array([tissue_to_hazard.get(t, 0) for t in gtex_subset.obs["GTEX"].values])
    
    # Convert gene pairs to indices
    print(f"Converting {len(gene_pairs)} gene pairs to indices...")
    gene_to_idx = pd.Series(range(len(genes_to_keep)), index=genes_to_keep)
    
    if isinstance(gene_pairs, list):
        gene_pairs_df = pd.DataFrame(gene_pairs, columns=['gene1', 'gene2'])
    else:
        gene_pairs_df = gene_pairs
    
    gx_all = gene_to_idx[gene_pairs_df['gene1'].values].values
    gy_all = gene_to_idx[gene_pairs_df['gene2'].values].values
    
    n_batches = int(np.ceil(len(gx_all) / batch_size))
    results = []
    
    for batch_idx in range(n_batches):
        iter_start = time.time()
        start = batch_idx * batch_size
        end = min(start + batch_size, len(gx_all))
        gx_t = gx_all[start:end]
        gy_t = gy_all[start:end]
        
        print(f"  {batch_idx + 1}/{n_batches} | ", end='', flush=True)
        
        # Take minimum expression across gene pairs
        X1 = gtex_subset.X[:, gx_t]
        X2 = gtex_subset.X[:, gy_t]

        # Use scipy for sparse minimum
        min_expr = sp.csr_matrix(np.minimum(X1.toarray(), X2.toarray()))
        
        # Summarize by tissue
        gtex_med = utils.summarize_matrix(mat=min_expr, groups=gtex_subset.obs["GTEX"].values, metric="median", axis=0)
         
        # Threshold expression
        gtex_med2 = np.where(gtex_med >= 25, 10, 
                   np.where(gtex_med >= 10, 5, 
                   np.where(gtex_med >= 5, 1, 
                   np.where(gtex_med > 1, 0.25, 0))))
        
        # Composite score: tissue-weighted expression + critical tissue penalty
        tissue_weighted = np.sum(gtex_med2.T * hazard_score, axis=1)
        critical_penalty = np.sqrt(np.sum(gtex_med2[hazard_score==4, :] >= 5, axis=0)) * 10
        
        # Create gene pair names
        names = [f"{genes_to_keep[i]}_{genes_to_keep[j]}" for i, j in zip(gx_t, gy_t)]
        
        batch_df = pd.DataFrame({
            "gene_name": names,
            "Hazard_GTEX_v1": tissue_weighted + critical_penalty
        })
        
        results.append(batch_df)
        iter_time = time.time() - iter_start
        print(f"{iter_time:.2f}s")
    
    df_all = pd.concat(results, ignore_index=True)

    return df_all










tissue_to_hazard = {}
for tier, config in tid.run.hazard_map.items():
    for tissue in config['gtex_tissues']:
        tissue_to_hazard[tissue] = config['hazard_score']

hazard_score = gtex_med.index.map(tissue_to_hazard)

gtex = gtex.to_memory()
gtex_med = tid.utils.summarize_matrix(mat = gtex.X, groups= gtex.obs["GTEX"].values, metric = "median", axis = 0)

gtex_med2 = np.where(gtex_med >= 25, 10, 
           np.where(gtex_med >= 10, 5, 
           np.where(gtex_med >= 5, 1, 
           np.where(gtex_med > 1, 0.25, 0))))

hazard_score = np.array(hazard_score)

df = pd.DataFrame({
    "gene_name": gtex.var_names,
    "Hazard_GTEX_v1": np.sum(gtex_med2.T * hazard_score, axis=1) + np.sqrt(np.sum(gtex_med2[hazard_score==4, :] >= 5, axis = 0)) * 10
})



np.min(df.loc[np.sum(gtex_med2[hazard_score==4, :] >= 5, axis = 0) >  0, "Hazard_GTEX_v1"])











gtex_med2 = np.minimum(gtex_med, 16).copy()
gtex_med2 = np.sqrt(gtex_med2) / 4
risk_score = np.sqrt(np.sum(gtex_med2.T * hazard_score ** 2, axis = 1)).values

df = pd.DataFrame({
    "gene_name" : gtex.var_names,
    "Hazard_GTEX_v1" : np.sqrt(np.sum(gtex_med2.T * hazard_score ** 2, axis = 1)).values
})

df[df["gene_name"] == "TACSTD2"]

gtex_med2 = np.where(gtex_med >= 25, 10, 
           np.where(gtex_med >= 10, 5, 
           np.where(gtex_med >= 5, 1, 
           np.where(gtex_med > 1, 0.25, 0))))

hazard_score = np.array(hazard_score)
np.sqrt(np.sum(gtex_med2[hazard_score==4, :] >= 5, axis = 0) * 10)


df = pd.DataFrame({
    "gene_name": gtex.var_names,
    "Hazard_GTEX_v1": np.sum(gtex_med2.T * hazard_score, axis=1) + np.sqrt(np.sum(gtex_med2[hazard_score==4, :] >= 5, axis = 0)) * 10
})

df[df["gene_name"] == "TACSTD2"]
df[df["gene_name"] == "LY6G6D"]
df[df["gene_name"] == "PRSS21"]
df[df["gene_name"] == "ROS1"]
df[df["gene_name"] == "SLC22A31"]
df[df["gene_name"] == "CA9"]
df[df["gene_name"] == "LY6K"]
df[df["gene_name"] == "SLC17A3"]


df["Hazard_GTEX_v1"]





df[df["gene_name"] == "TACSTD2"]
df[df["gene_name"] == "LY6G6D"]
df[df["gene_name"] == "PRSS21"]
df[df["gene_name"] == "ROS1"]
df[df["gene_name"] == "SLC22A31"]
df[df["gene_name"] == "CA9"]
df[df["gene_name"] == "LY6K"]
df[df["gene_name"] == "KISS1R"]



gtex_med.loc[:,gtex.var_names=="ROS1"] 

np.sum(gtex_med2[:,gtex.var_names=="TACSTD2"].T * hazard_score)