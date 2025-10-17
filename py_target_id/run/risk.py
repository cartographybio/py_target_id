"""
Target ID
"""

# Define what gets exported
__all__ = ['compute_ref_risk_scores']
import torch
import numpy as np
import pandas as pd
import time
import gc
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

hazard_map = {
    'tier_1': {
        'hazard_score': 2,
        'sc_tissues': [
            'Brain_Diencephalon', 'Brain_Forebrain', 'Brain_Hindbrain',
            'Brain_Hippocampus', 'Brain_Medial_Temporal_Gyrus', 'Brain_Midbrain',
            'Brain_Pre_Frontal_Cortex', 'Brain_Substantia_Nigra',
            'Brain_Superior_Frontal_Cortex', 'Brain_Telencephalon',
            'Spine', 'Heart', 'Peripheral_Nerve', 'Lung', 'Trachea',
        ],
        'gtex_tissues': [
            'Brain.ACC_BA24', 'Brain.Amygdala', 'Brain.Basal_G_Caud',
            'Brain.Basal_G_NAcc', 'Brain.Cerebellar', 'Brain.Cerebellum',
            'Brain.Cortex', 'Brain.Frontal_BA9', 'Brain.Hippo',
            'Brain.Hypothal', 'Brain.Putamen', 'Brain.Spinal_C1',
            'Brain.SubNigra', 'Heart.Atrium', 'Heart.Ventr', 'Lung',
            'Nerve.Tibial', 'Vessel.Aorta', 'Vessel.Coronary', 'Vessel.Tibial',
        ],
        'ffpe_tissues': [
            'Brain', 'Heart', 'Lung', 'Peripheral_Nerve',
        ],
    },
    'tier_2': {
        'hazard_score': 1,
        'sc_tissues': [
            'Liver', 'Kidney', 'Pancreas',
            'Adrenal_Gland', 'Thyroid_Gland',
            'Bone_Marrow', 'Blood', 'Thymus', 'Lymph_Node', 'Spleen',
            'Small_Intestine', 'Large_Intestine', 'Stomach', 'Esophagus',
            'Bile_Duct', 'Gallbladder',
        ],
        'gtex_tissues': [
            'Adrenal', 'Blood', 'Colon.Sigmoid', 'Colon.Transverse',
            'Esophagus.GE_Jxn', 'Esophagus.Mucosa', 'Esophagus.Muscle',
            'Kidney.Cortex', 'Kidney.Medulla', 'Liver', 'Pancreas',
            'Pituitary', 'Small_Int.Ileum', 'Spleen', 'Stomach', 'Thyroid',
        ],
        'ffpe_tissues': [
            'Bone_Marrow', 'Liver', 'Kidney', 'Pancreas', 'Adrenal_Gland',
            'Thyroid_Gland', 'Lymph_Node', 'Spleen', 'Thymus',
            'Small_Intestine', 'Large_Intestine', 'Stomach', 'Esophagus',
            'Bile_Duct', 'Gallbladder',
        ],
    },
    'tier_3': {
        'hazard_score': 0.25,
        'sc_tissues': [
            'Testes', 'Ovary', 'Prostate', 'Uterus', 'Cervix', 'Fallopian_Tube',
            'Breast', 'Appendix', 'Skin', 'Adipose', 'Skeletal_Muscle',
            'Eye', 'Bladder', 'Salivary_Gland', 'Lacrimal_Gland', 'Tongue',
        ],
        'gtex_tissues': [
            'Adipose.Subcut', 'Adipose.Visc', 'Bladder', 'Breast.Mammary',
            'Cervix.Ecto', 'Cervix.Endo', 'Fallopian', 'Muscle.Skeletal',
            'Ovary', 'Prostate', 'Salivary.Minor', 'Skin.Exposed',
            'Skin.Unexposed', 'Testis', 'Uterus', 'Vagina',
        ],
        'ffpe_tissues': [
            'Adipose', 'Breast', 'Bladder', 'Ovary', 'Uterus', 'Salivary_Gland',
            'Skeletal_Muscle', 'Skin',
        ],
    },
}


def clear_gpu_memory(verbose=True):
    """Clear GPU memory"""
    if verbose:
        print(f"Before: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if verbose:
        print(f"After: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")


def get_gpu_memory_info():
    """Get GPU memory stats"""
    allocated = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    usage_pct = (allocated / total) * 100
    return allocated, total, usage_pct


def compute_group_median_cached_gpu(data: torch.Tensor, group_indices: list) -> torch.Tensor:
    """Compute group medians on GPU using quantile"""
    n_pairs = data.shape[0]
    n_groups = len(group_indices)
    result = torch.zeros((n_pairs, n_groups), dtype=torch.float32, device=data.device)
    
    for g in range(n_groups):
        indices = group_indices[g]
        if len(indices) > 0:
            result[:, g] = torch.quantile(data[:, indices].float(), q=0.5, dim=1)
    
    return result

def compute_dedup_multipliers_gpu(
    ref_med: torch.Tensor,
    tissues_per_group: np.ndarray,
    lv1_per_group: np.ndarray,
    lv2_per_group: np.ndarray,
    lv3_per_group: np.ndarray,
    lv4_per_group: np.ndarray,
    combo_lv4_per_group: np.ndarray,
    device='cuda'
) -> torch.Tensor:
    """
    Compute tissue deduplication multipliers for batch of pairs.
    Uses stable sort (NumPy) to match CPU behavior exactly.
    
    Parameters
    ----------
    ref_med : torch.Tensor
        Shape (n_pairs, n_groups) - median expressions per group
    tissues_per_group, lv1_per_group, ..., combo_lv4_per_group : np.ndarray
        Shape (n_groups,) - metadata for each group
    device : str
        'cuda' or 'cpu'
    
    Returns
    -------
    multipliers : torch.Tensor
        Shape (n_pairs, n_groups) - multipliers per group per pair
    """
    n_pairs, n_groups = ref_med.shape
    multipliers = torch.ones((n_pairs, n_groups), dtype=torch.float32, device=device)
    
    # For each pair, sort groups by median expression (descending)
    for pair_idx in range(n_pairs):
        pair_vals = ref_med[pair_idx, :]
        
        # Sort on CPU with quicksort to match CPU code exactly
        # CPU code uses: np.argsort(col, kind='quicksort')[::-1]
        pair_vals_np = pair_vals.cpu().numpy()
        sorted_group_indices_np = np.argsort(-pair_vals_np, kind='quicksort')
        sorted_group_indices = torch.tensor(sorted_group_indices_np, dtype=torch.long, device=device)
        
        # Track seen values at each hierarchy level
        seen_tissue = set()
        seen_lv1 = set()
        seen_lv2 = set()
        seen_lv3 = set()
        seen_lv4 = set()
        seen_combo_lv4 = set()
        
        for position, group_idx in enumerate(sorted_group_indices):
            group_idx_int = group_idx.item()
            
            if position == 0:
                # First group always gets 1.0
                multipliers[pair_idx, group_idx_int] = 1.0
            else:
                # Get metadata for this group
                tissue = tissues_per_group[group_idx_int]
                lv1 = lv1_per_group[group_idx_int]
                lv2 = lv2_per_group[group_idx_int]
                lv3 = lv3_per_group[group_idx_int]
                lv4 = lv4_per_group[group_idx_int]
                combo_lv4 = combo_lv4_per_group[group_idx_int]
                
                mult = 1.0
                
                # Apply 0.5x penalty for each repeated hierarchy level
                if tissue in seen_tissue:
                    mult *= 0.5
                if lv1 in seen_lv1:
                    mult *= 0.5
                if lv2 in seen_lv2:
                    mult *= 0.5
                if lv3 in seen_lv3:
                    mult *= 0.5
                if lv4 in seen_lv4:
                    mult *= 0.5
                
                # Exact duplicate = skip entirely
                if combo_lv4 in seen_combo_lv4:
                    mult = 0.0
                
                multipliers[pair_idx, group_idx_int] = mult
            
            # Mark as seen
            seen_tissue.add(tissues_per_group[group_idx_int])
            seen_lv1.add(lv1_per_group[group_idx_int])
            seen_lv2.add(lv2_per_group[group_idx_int])
            seen_lv3.add(lv3_per_group[group_idx_int])
            seen_lv4.add(lv4_per_group[group_idx_int])
            seen_combo_lv4.add(combo_lv4_per_group[group_idx_int])
    
    return multipliers

def compute_dedup_multipliers_gpu_fast(
    ref_med: torch.Tensor,
    tissues_per_group: np.ndarray,
    lv1_per_group: np.ndarray,
    lv2_per_group: np.ndarray,
    lv3_per_group: np.ndarray,
    lv4_per_group: np.ndarray,
    combo_lv4_per_group: np.ndarray,
    device='cuda'
) -> torch.Tensor:
    """
    Fast GPU dedup multipliers using NumPy for heavy lifting, minimal GPU work.
    """
    n_pairs, n_groups = ref_med.shape
    multipliers = torch.ones((n_pairs, n_groups), dtype=torch.float32, device=device)
    
    # Batch process on CPU using NumPy (faster than GPU loops for this)
    ref_med_np = ref_med.cpu().numpy()
    
    for pair_idx in range(n_pairs):
        pair_vals = ref_med_np[pair_idx, :]
        
        # Sort
        sorted_indices = np.argsort(-pair_vals, kind='quicksort')
        
        # Reorder metadata
        tissues_i = tissues_per_group[sorted_indices]
        lv1_i = lv1_per_group[sorted_indices]
        lv2_i = lv2_per_group[sorted_indices]
        lv3_i = lv3_per_group[sorted_indices]
        lv4_i = lv4_per_group[sorted_indices]
        combo_lv4_i = combo_lv4_per_group[sorted_indices]
        v = pair_vals[sorted_indices]
        
        # Compute multipliers on CPU (vectorized with NumPy sets)
        seen_tissue = set()
        seen_lv1 = set()
        seen_lv2 = set()
        seen_lv3 = set()
        seen_lv4 = set()
        seen_combo_lv4 = set()
        
        mults = np.ones(n_groups, dtype=np.float32)
        
        for j in range(n_groups):
            if j > 0:
                mj = 1.0
                if tissues_i[j] in seen_tissue:
                    mj *= 0.5
                if lv1_i[j] in seen_lv1:
                    mj *= 0.5
                if lv2_i[j] in seen_lv2:
                    mj *= 0.5
                if lv3_i[j] in seen_lv3:
                    mj *= 0.5
                if lv4_i[j] in seen_lv4:
                    mj *= 0.5
                if combo_lv4_i[j] in seen_combo_lv4:
                    mj = 0.0
                mults[j] = mj
            
            seen_tissue.add(tissues_i[j])
            seen_lv1.add(lv1_i[j])
            seen_lv2.add(lv2_i[j])
            seen_lv3.add(lv3_i[j])
            seen_lv4.add(lv4_i[j])
            seen_combo_lv4.add(combo_lv4_i[j])
        
        # Map back to original order and store on GPU
        mults_original_order = np.zeros(n_groups, dtype=np.float32)
        mults_original_order[sorted_indices] = mults
        multipliers[pair_idx, :] = torch.tensor(mults_original_order, dtype=torch.float32, device=device)
    
    return multipliers

def compute_ref_risk_scores(
    ref_adata,
    gene_pairs,
    hazard_map=hazard_map,  # Pass the module-level hazard_map
    device='cuda',
    batch_size=5000,
    type = "SC",
    use_fp16=True
):
    """
    GPU-optimized hazard-weighted risk scoring for multi-gene combinations.
    
    Parameters
    ----------
    ref_adata : AnnData
        Reference atlas data (will be preprocessed)
    gene_pairs : list of tuples
        List of (gene1, gene2) pairs to analyze
    hazard_map : dict
        Tissue hazard tier mapping with tier_1, tier_2, tier_3
    device : str
        'cuda' or 'cpu'
    batch_size : int
        Number of pairs per batch
    use_fp16 : bool
        Use FP16 mixed precision
    
    Returns
    -------
    pd.DataFrame
        Results with hazard-weighted risk scores
    """
    
    print("Preprocessing reference data...")
    
    # 1. Remove immune system
    ref_adata = ref_adata[ref_adata.obs['Combo_Lv1'].str.split(':', n=1, expand=True)[1] != "Immune", :].copy()
    
    # 2. Use existing Combo_Lv4_Norm if available
    if 'Combo_Lv4_Norm' not in ref_adata.obs.columns:
        ref_adata.obs['Combo_Lv4_Norm'] = ref_adata.obs['Combo_Lv4'].str.replace(r'_\d+$', '', regex=True)
    
    # 3. Extract hierarchy
    tissues = ref_adata.obs['Combo_Lv1'].str.split(':', n=1, expand=True)[0]
    
    if type == "SC":
        haz_type = "sc_tissues"
    elif type == "FFPE":
        haz_type = "ffpe_tissues"
    else:
        raise ValueError("'type' must be SC or FFPE")

    # 4. Compute hazard weights
    v_hazard = np.zeros(len(tissues))
    v_hazard[tissues.isin(hazard_map["tier_1"][haz_type])] = hazard_map["tier_1"]["hazard_score"]
    v_hazard[tissues.isin(hazard_map["tier_2"][haz_type])] = hazard_map["tier_2"]["hazard_score"]
    v_hazard[tissues.isin(hazard_map["tier_3"][haz_type])] = hazard_map["tier_3"]["hazard_score"]
    
    print(f"Validating {len(gene_pairs)} gene pairs...")
    
    # Extract unique genes
    all_genes = set()
    for g1, g2 in gene_pairs:
        all_genes.add(g1)
        all_genes.add(g2)
    
    genes_to_keep = sorted(all_genes)
    print(f"Found {len(genes_to_keep)} unique genes")
    
    # Subset reference data
    ref_subset = ref_adata[:, genes_to_keep].copy().to_memory(dense=True, chunk_size=5000, dtype=np.float16, show_progress=True)
    genes = ref_subset.var_names
 
    # Convert sparse to dense
    from scipy.sparse import issparse
    if issparse(ref_subset.X):
        print("Converting reference sparse matrix to dense...")
        ref_subset.X = ref_subset.X.toarray()
        
    # 5. Weight matrix by hazard
    m = np.minimum(ref_subset.X.copy(), 2)
    m = (m.T * v_hazard).T
    ref_subset.X = m
    
    print("Matrix subsetting complete.\n")
    
    # Convert gene pairs to indices
    gene_to_idx = pd.Series(range(len(genes)), index=genes)
    gene_pairs_df = pd.DataFrame(gene_pairs, columns=['gene1', 'gene2'])
    gx_all = gene_to_idx[gene_pairs_df['gene1'].values].values
    gy_all = gene_to_idx[gene_pairs_df['gene2'].values].values
    n_batches = int(np.ceil(len(gx_all) / batch_size))
    
    clear_gpu_memory(verbose=False)
    
    overall_start = time.time()
    device = torch.device(device)
    
    if use_fp16 and device.type == 'cuda':
        if torch.cuda.get_device_capability(device)[0] < 7:
            use_fp16 = False
        else:
            print(f"Using FP16 mixed precision\n")
    
    try:
        # Load reference data to GPU
        print(f"------  Loading Data - | Total: {time.time()-overall_start:.1f}s")
        dtype = torch.float16 if use_fp16 else torch.float32
        ref_X = torch.tensor(np.array(ref_subset.X.T), dtype=dtype, device=device)
        
        # Encode groups for reference
        ref_ids = (ref_subset.obs_names.str.extract(r'^([^:]+:[^:]+)', expand=False)
                  .str.replace(r'[ -]', '_', regex=True).values)
        ref_unique = np.unique(ref_ids)
        ref_id_to_idx = {id_val: idx for idx, id_val in enumerate(ref_unique)}
        ref_ids_encoded = torch.tensor([ref_id_to_idx[x] for x in ref_ids], dtype=torch.long, device=device)
        
        n_ref_groups = len(ref_unique)
        ref_group_indices = [(ref_ids_encoded == g).nonzero(as_tuple=True)[0] for g in range(n_ref_groups)]

        # Extract tissue from Combo_Lv1 (before colon), and cell-type parts (after colon)

        # Get metadata arrays from ref_subset
        ref_obs_lv1_full = ref_subset.obs['Combo_Lv1'].values
        ref_obs_lv2_full = ref_subset.obs['Combo_Lv2'].values
        ref_obs_lv3_full = ref_subset.obs['Combo_Lv3'].values
        ref_obs_lv4_full = ref_subset.obs['Combo_Lv4'].values
        ref_obs_combo_lv4 = ref_subset.obs['Combo_Lv4_Norm'].values

        # Extract tissue names (before colon) and cell-type parts (after colon)
        # Using pandas Series split for efficiency
        lv1_series = pd.Series(ref_obs_lv1_full).str.split(':', n=1, expand=True)
        lv2_series = pd.Series(ref_obs_lv2_full).str.split(':', n=1, expand=True)
        lv3_series = pd.Series(ref_obs_lv3_full).str.split(':', n=1, expand=True)
        lv4_series = pd.Series(ref_obs_lv4_full).str.split(':', n=1, expand=True)

        tissues = lv1_series[0].values  # Before colon
        lv1 = lv1_series[1].values      # After colon
        lv2 = lv2_series[1].values      # After colon
        lv3 = lv3_series[1].values      # After colon
        lv4 = lv4_series[1].values      # After colon

        # Build per-group metadata arrays (for each of 734 groups)
        tissues_per_group = np.zeros(n_ref_groups, dtype=object)
        lv1_per_group = np.zeros(n_ref_groups, dtype=object)
        lv2_per_group = np.zeros(n_ref_groups, dtype=object)
        lv3_per_group = np.zeros(n_ref_groups, dtype=object)
        lv4_per_group = np.zeros(n_ref_groups, dtype=object)
        combo_lv4_per_group = np.zeros(n_ref_groups, dtype=object)

        # For each group, grab metadata from the first cell in that group
        for g in range(n_ref_groups):
            cell_mask = (ref_ids_encoded == g).nonzero(as_tuple=True)[0]
            if len(cell_mask) > 0:
                first_cell_idx = cell_mask[0].item()
                
                tissues_per_group[g] = tissues[first_cell_idx]
                lv1_per_group[g] = lv1[first_cell_idx]
                lv2_per_group[g] = lv2[first_cell_idx]
                lv3_per_group[g] = lv3[first_cell_idx]
                lv4_per_group[g] = lv4[first_cell_idx]
                combo_lv4_per_group[g] = ref_obs_combo_lv4[first_cell_idx]

        print(f"Built per-group metadata for {n_ref_groups} groups")
        
        print(f"------  Running Hazard-Weighted Risk Scoring ({len(gx_all)} pairs, {n_batches} batches)\n")
        
        all_results = []
        
        for batch_idx in range(n_batches):
            iter_start = time.time()
            start = batch_idx * batch_size
            end = min(start + batch_size, len(gx_all))
            
            gx_t = torch.tensor(gx_all[start:end], dtype=torch.long, device=device)
            gy_t = torch.tensor(gy_all[start:end], dtype=torch.long, device=device)
            
            print(f"  {batch_idx + 1}/{n_batches} | ", end='', flush=True)
            
            # Compute minimum expression across pair genes
            t_expr = time.time()
            ref_min = torch.minimum(ref_X[gx_t], ref_X[gy_t])
            expr_time = time.time() - t_expr
            print(f"MinExpr:{expr_time:.1f}s | ", end='', flush=True)
            
            # Compute group medians
            t_median = time.time()
            ref_med = compute_group_median_cached_gpu(ref_min, ref_group_indices)
            median_time = time.time() - t_median
            print(f"Median:{median_time:.1f}s | ", end='', flush=True)
            
            # Sum risk (hazard already applied to matrix)
            t_risk = time.time()
            
            # Compute deduplication multipliers
            multipliers = compute_dedup_multipliers_gpu_fast(
                ref_med,
                tissues_per_group,
                lv1_per_group,
                lv2_per_group,
                lv3_per_group,
                lv4_per_group,
                combo_lv4_per_group,
                device=device
            )
            
            # Apply multipliers and sum
            weighted_med = ref_med * multipliers
            risk_scores = weighted_med.sum(dim=1).cpu().numpy()
            
            risk_time = time.time() - t_risk
            print(f"Risk(dedup):{risk_time:.1f}s | ", end='', flush=True)

            # Store results
            pair_names = [f"{genes[i]}_{genes[j]}" for i, j in zip(gx_all[start:end], gy_all[start:end])]
            df_batch = pd.DataFrame({
                'pair_name': pair_names,
                'hazard_weighted_risk': risk_scores
            })
            all_results.append(df_batch)
            
            # Cleanup
            del gx_t, gy_t, ref_min, ref_med
            torch.cuda.empty_cache()
            
            allocated, total, usage_pct = get_gpu_memory_info()
            iter_total = time.time() - iter_start
            print(f"IterTotal:{iter_total:.1f}s | GPU:{usage_pct:.0f}% | ", end='', flush=True)
            
            total_time = time.time() - overall_start
            print(f"Total:{total_time/60:.1f}m")
        
        # Combine results
        print(f"\n------  Combining Results - | Total: {time.time()-overall_start:.1f}s")
        df_all = pd.concat(all_results, ignore_index=True)
        df_all = df_all.sort_values('hazard_weighted_risk', ascending=True)
        
        print(f"------  Complete - | Total: {time.time()-overall_start:.1f}s\n")
        
        return df_all
        
    finally:
        try:
            del ref_X, ref_ids_encoded
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()

