"""
Target ID Multi v1 - GPU Optimized
"""

__all__ = ['run_target_id_multi_v1']

import torch
import numpy as np
import pandas as pd
import time
import gc
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

# ============================================================================
# Helper functions
# ============================================================================

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

def compute_group_median_cached(data: torch.Tensor, group_indices: list) -> torch.Tensor:
    """Compute group medians using quantile"""
    n_pairs = data.shape[0]
    n_groups = len(group_indices)
    result = torch.zeros((n_pairs, n_groups), dtype=torch.float32, device=data.device)
    
    for g in range(n_groups):
        indices = group_indices[g]
        if len(indices) > 0:
            result[:, g] = torch.quantile(data[:, indices].float(), q=0.5, dim=1)
    
    return result

def compute_score_matrix_gpu(
    target_val: torch.Tensor,
    healthy_matrix_ordered_T: torch.Tensor,
    healthy_matrix_ordered: torch.Tensor,
    test_n_off_targets: int,
    device: torch.device,
    batch_size: int = 2000,
    min_lfc: float = 0.25,
    max_off_val: float = 1.0,
    offset: float = 1e-8
) -> torch.Tensor:
    """Compute specificity scores with filtering"""
    n_genes = target_val.shape[0]
    n_healthy = healthy_matrix_ordered_T.shape[1]
    test_n = min(test_n_off_targets, n_healthy)
    
    score_matrix_raw = torch.zeros((n_genes, test_n), dtype=torch.float32, device=device)
    eps = 1e-10
    
    # Compute raw JS divergence scores
    for batch_start in range(0, n_genes, batch_size):
        batch_end = min(batch_start + batch_size, n_genes)
        target_batch = target_val[batch_start:batch_end]
        healthy_batch = healthy_matrix_ordered_T[batch_start:batch_end]
        
        for iter_idx in range(test_n):
            n_cols = n_healthy - iter_idx
            if n_cols <= 0:
                continue
            
            combined = torch.cat([
                target_batch.unsqueeze(1),
                healthy_batch[:, iter_idx:iter_idx + n_cols]
            ], dim=1)
            
            combined_sum = combined.sum(dim=1, keepdim=True).clamp(min=eps)
            combined_norm = combined / combined_sum
            
            spec_vector = torch.zeros_like(combined)
            spec_vector[:, 0] = 1.0
            mid = (combined_norm + spec_vector) * 0.5
            
            log_norm = torch.where(combined_norm > eps, torch.log2(combined_norm.clamp(min=eps)), torch.zeros_like(combined_norm))
            log_mid = torch.where(mid > eps, torch.log2(mid.clamp(min=eps)), torch.zeros_like(mid))
            
            entropy_a = (combined_norm * log_norm).sum(dim=1)
            entropy_mid = (mid * log_mid).sum(dim=1)
            
            jsdist = torch.clamp(-entropy_mid + 0.5 * entropy_a, min=0.0, max=1.0)
            scores = 1.0 - torch.sqrt(jsdist)
            scores = torch.where(target_batch > 0, scores, torch.zeros_like(scores))
            
            score_matrix_raw[batch_start:batch_end, iter_idx] = scores
    
    # Apply filters
    score_matrix = score_matrix_raw.clone()
    target_val_nontransformed = torch.sqrt(target_val)
    
    lfc = torch.log2(target_val_nontransformed.unsqueeze(1) + offset) - torch.log2(healthy_matrix_ordered[:, :test_n] + offset)
    score_matrix[lfc < min_lfc] = 0
    score_matrix[healthy_matrix_ordered[:, :test_n] > max_off_val] = 0
    
    # Handle all-zero rows
    all_zero = score_matrix.sum(dim=1) == 0
    if all_zero.any():
        score_matrix[all_zero, -1] = score_matrix_raw[all_zero, -1]
    
    return score_matrix

def compute_positive_patients_inline(
    target_val: torch.Tensor,
    malig_med: torch.Tensor,
    ha_med: torch.Tensor,
    ha_ordered_idx: torch.Tensor,
    score_matrix: torch.Tensor,
    passing_mask: torch.Tensor,
    n_malig_groups: int,
    device: torch.device
) -> tuple:
    """
    Compute positive patients inline during main loop
    """
    n_genes_batch = target_val.shape[0]
    n_ha_groups = ha_med.shape[1]
    
    target_val_pos = torch.full((n_genes_batch,), float('nan'), device=device)
    n_pos = torch.zeros(n_genes_batch, dtype=torch.long, device=device)
    p_pos = torch.zeros(n_genes_batch, dtype=torch.float32, device=device)
    
    if not passing_mask.any():
        return target_val_pos, n_pos, p_pos
    
    # Build off-target mask
    off_target_mask = torch.zeros((n_genes_batch, n_ha_groups), dtype=torch.bool, device=device)
    for i in range(n_genes_batch):
        if passing_mask[i]:
            failing_positions = (score_matrix[i] < 0.35).nonzero(as_tuple=True)[0]
            if len(failing_positions) > 0:
                original_failing_indices = ha_ordered_idx[i, failing_positions]
                off_target_mask[i, original_failing_indices] = True
    
    # Zero out off-targets
    ha_med_masked = ha_med.clone()
    ha_med_masked[off_target_mask] = 0
    
    # Get passing genes
    passing_indices = passing_mask.nonzero(as_tuple=True)[0]
    
    # Process in sub-batches to avoid memory issues
    sub_batch_size = 100
    for sub_start in range(0, len(passing_indices), sub_batch_size):
        sub_end = min(sub_start + sub_batch_size, len(passing_indices))
        sub_indices = passing_indices[sub_start:sub_end]
        actual_sub_batch = len(sub_indices)
        
        sub_targets = target_val[sub_indices]
        sub_malig = malig_med[sub_indices]
        sub_healthy = ha_med_masked[sub_indices]
        
        # Scaling factors
        scaling_factors = torch.tensor(
            [1e-6, 1e-5, 1e-4] + list(np.arange(0.001, 1.001, 0.005)),
            dtype=torch.float32, device=device
        )
        n_tests = len(scaling_factors)
        
        # Build thresholds
        thresholds = sub_targets.unsqueeze(1) * scaling_factors.unsqueeze(0)
        thresholds_T = thresholds ** 2
        sub_healthy_T = sub_healthy ** 2
        
        # Vectorized JS computation
        combined = torch.zeros((actual_sub_batch, n_tests, n_ha_groups + 1), dtype=torch.float32, device=device)
        combined[:, :, 0] = thresholds_T
        combined[:, :, 1:] = sub_healthy_T.unsqueeze(1).expand(-1, n_tests, -1)
        
        eps = 1e-10
        combined_sum = combined.sum(dim=2, keepdim=True).clamp(min=eps)
        combined_norm = combined / combined_sum
        
        spec_vector = torch.zeros_like(combined)
        spec_vector[:, :, 0] = 1.0
        mid = (combined_norm + spec_vector) * 0.5
        
        log_norm = torch.where(combined_norm > eps, torch.log2(combined_norm.clamp(min=eps)), torch.zeros_like(combined_norm))
        log_mid = torch.where(mid > eps, torch.log2(mid.clamp(min=eps)), torch.zeros_like(mid))
        
        entropy_a = (combined_norm * log_norm).sum(dim=2)
        entropy_mid = (mid * log_mid).sum(dim=2)
        jsdist = torch.clamp(-entropy_mid + 0.5 * entropy_a, min=0.0, max=1.0)
        specificities = 1.0 - torch.sqrt(jsdist)
        
        valid_thresholds = specificities >= 0.35
        
        # Find first valid threshold for each gene
        for local_i, gene_idx in enumerate(sub_indices):
            if valid_thresholds[local_i].any():
                first_valid = torch.argmax(valid_thresholds[local_i].float()).item()
                final_threshold = thresholds[local_i, first_valid]
                n_pos_count = (sub_malig[local_i] >= final_threshold).sum()
                
                if n_pos_count > 0:
                    target_val_pos[gene_idx] = final_threshold
                    n_pos[gene_idx] = n_pos_count
                    p_pos[gene_idx] = n_pos_count.float() / n_malig_groups
        
        del combined, combined_norm, spec_vector, mid, log_norm, log_mid
        del entropy_a, entropy_mid, jsdist, specificities
    
    return target_val_pos, n_pos, p_pos


def compute_positive_patients_inline2(
    target_val: torch.Tensor,
    malig_med: torch.Tensor,
    ha_med: torch.Tensor,
    ha_ordered_idx: torch.Tensor,
    score_matrix: torch.Tensor,
    passing_mask: torch.Tensor,
    n_malig_groups: int,
    device: torch.device
) -> tuple:
    """
    Compute positive patients inline during main loop - fully vectorized
    """
    n_genes_batch = target_val.shape[0]
    n_ha_groups = ha_med.shape[1]
    
    target_val_pos = torch.full((n_genes_batch,), float('nan'), device=device)
    n_pos = torch.zeros(n_genes_batch, dtype=torch.long, device=device)
    p_pos = torch.zeros(n_genes_batch, dtype=torch.float32, device=device)
    
    if not passing_mask.any():
        return target_val_pos, n_pos, p_pos
    
    # Build off-target mask (vectorized)
    off_target_mask = torch.zeros((n_genes_batch, n_ha_groups), dtype=torch.bool, device=device)
    passing_indices = passing_mask.nonzero(as_tuple=True)[0]
    
    for i in passing_indices:
        failing_positions = (score_matrix[i] < 0.35).nonzero(as_tuple=True)[0]
        if len(failing_positions) > 0:
            original_failing_indices = ha_ordered_idx[i, failing_positions]
            off_target_mask[i, original_failing_indices] = True
    
    # Zero out off-targets
    ha_med_masked = ha_med.clone()
    ha_med_masked[off_target_mask] = 0
    
    # Process in sub-batches for memory efficiency
    sub_batch_size = 1000
    for sub_start in range(0, len(passing_indices), sub_batch_size):
        sub_end = min(sub_start + sub_batch_size, len(passing_indices))
        sub_indices = passing_indices[sub_start:sub_end]
        actual_sub_batch = len(sub_indices)
        
        sub_targets = target_val[sub_indices]
        sub_malig = malig_med[sub_indices]
        sub_healthy = ha_med_masked[sub_indices]
        
        # Scaling factors
        scaling_factors = torch.tensor(
            [1e-6, 1e-5, 1e-4] + list(np.arange(0.001, 1.001, 0.005)),
            dtype=torch.float32, device=device
        )
        n_tests = len(scaling_factors)
        
        # Build thresholds
        thresholds = sub_targets.unsqueeze(1) * scaling_factors.unsqueeze(0)
        thresholds_T = thresholds ** 2
        sub_healthy_T = sub_healthy ** 2
        
        # Vectorized JS computation
        combined = torch.zeros((actual_sub_batch, n_tests, n_ha_groups + 1), dtype=torch.float32, device=device)
        combined[:, :, 0] = thresholds_T
        combined[:, :, 1:] = sub_healthy_T.unsqueeze(1).expand(-1, n_tests, -1)
        
        eps = 1e-10
        combined_sum = combined.sum(dim=2, keepdim=True).clamp(min=eps)
        combined_norm = combined / combined_sum
        
        spec_vector = torch.zeros_like(combined)
        spec_vector[:, :, 0] = 1.0
        mid = (combined_norm + spec_vector) * 0.5
        
        log_norm = torch.where(combined_norm > eps, torch.log2(combined_norm.clamp(min=eps)), torch.zeros_like(combined_norm))
        log_mid = torch.where(mid > eps, torch.log2(mid.clamp(min=eps)), torch.zeros_like(mid))
        
        entropy_a = (combined_norm * log_norm).sum(dim=2)
        entropy_mid = (mid * log_mid).sum(dim=2)
        jsdist = torch.clamp(-entropy_mid + 0.5 * entropy_a, min=0.0, max=1.0)
        specificities = 1.0 - torch.sqrt(jsdist)
        
        # VECTORIZED: Find first valid threshold for ALL genes at once
        valid_thresholds = specificities >= 0.35
        has_valid = valid_thresholds.any(dim=1)
        
        if has_valid.any():
            # Get first valid index for each gene
            first_valid_idx = torch.argmax(valid_thresholds.float(), dim=1)
            
            # Gather the actual thresholds
            final_thresholds = torch.gather(thresholds, 1, first_valid_idx.unsqueeze(1)).squeeze(1)
            
            # Count positive patients (vectorized)
            n_pos_counts = (sub_malig >= final_thresholds.unsqueeze(1)).sum(dim=1)
            
            # Update only genes with valid thresholds and positive patients
            valid_with_pos = has_valid & (n_pos_counts > 0)
            
            if valid_with_pos.any():
                valid_gene_indices = sub_indices[valid_with_pos]
                target_val_pos[valid_gene_indices] = final_thresholds[valid_with_pos]
                n_pos[valid_gene_indices] = n_pos_counts[valid_with_pos]
                p_pos[valid_gene_indices] = n_pos_counts[valid_with_pos].float() / n_malig_groups
        
        del combined, combined_norm, spec_vector, mid, log_norm, log_mid
        del entropy_a, entropy_mid, jsdist, specificities
    
    return target_val_pos, n_pos, p_pos

def compute_target_quality_score(df: pd.DataFrame, surface_evidence_path: str) -> pd.DataFrame:
    """Compute quality scores"""
    try:
        surface_evidence = pd.read_csv(surface_evidence_path)
        surface_dict = dict(zip(surface_evidence['gene_name'], surface_evidence['surface_evidence']))
        gene_splits = df['gene_name'].str.split('.', n=1, expand=True)
        gene1 = gene_splits[0].map(surface_dict).fillna(1.0)
        gene2 = gene_splits[1].map(surface_dict).fillna(1.0)
        df['Surface_Prob'] = np.minimum(gene1, gene2)
    except FileNotFoundError:
        df['Surface_Prob'] = 1.0

    df['Score_1'] = 10
    df.loc[df['N_Off_Targets'] <= 3, 'Score_1'] = df.loc[df['N_Off_Targets'] <= 3, 'N_Off_Targets']
    
    df['Score_2'] = 10
    df.loc[df['N_Off_Targets_0.5'] <= 3, 'Score_2'] = df.loc[df['N_Off_Targets_0.5'] <= 3, 'N_Off_Targets_0.5']
    
    df['Score_3'] = 10
    df.loc[df['Corrected_Specificity'] >= 0.75, 'Score_3'] = 0
    df.loc[(df['Corrected_Specificity'] >= 0.5) & (df['Corrected_Specificity'] < 0.75), 'Score_3'] = 1
    df.loc[(df['Corrected_Specificity'] >= 0.35) & (df['Corrected_Specificity'] < 0.5), 'Score_3'] = 3
    
    df['Score_4'] = 10
    df.loc[df['P_Pos_Per'] > 0.25, 'Score_4'] = 0
    df.loc[(df['P_Pos_Per'] > 0.15) & (df['P_Pos_Per'] <= 0.25), 'Score_4'] = 1
    df.loc[(df['P_Pos_Per'] > 0.025) & (df['P_Pos_Per'] <= 0.15), 'Score_4'] = 3
    df.loc[df['N_Pos_Val'] == 1, 'Score_4'] = 10
    
    df['Score_5'] = 10
    df.loc[df['P_Pos'] > 0.25, 'Score_5'] = 0
    df.loc[(df['P_Pos'] > 0.15) & (df['P_Pos'] <= 0.25), 'Score_5'] = 1
    df.loc[(df['P_Pos'] > 0.025) & (df['P_Pos'] <= 0.15), 'Score_5'] = 3
    df.loc[df['N_Pos'] == 1, 'Score_5'] = 10
    
    df['Score_6'] = 10
    df.loc[df['SC_2nd_Target_Val'] > 2, 'Score_6'] = 0
    df.loc[(df['SC_2nd_Target_Val'] > 1) & (df['SC_2nd_Target_Val'] <= 2), 'Score_6'] = 1
    df.loc[(df['SC_2nd_Target_Val'] > 0.5) & (df['SC_2nd_Target_Val'] <= 1), 'Score_6'] = 3
    df.loc[(df['SC_2nd_Target_Val'] > 0.1) & (df['SC_2nd_Target_Val'] <= 0.5), 'Score_6'] = 5
    
    df['Score_7'] = 10
    df.loc[df['Surface_Prob'] >= 0.5, 'Score_7'] = 0
    df.loc[(df['Surface_Prob'] >= 0.1875) & (df['Surface_Prob'] < 0.5), 'Score_7'] = 3
    df.loc[(df['Surface_Prob'] >= 0.125) & (df['Surface_Prob'] < 0.1875), 'Score_7'] = 7
    
    score_columns = ['Score_1', 'Score_2', 'Score_3', 'Score_5', 'Score_6', 'Score_7']
    penalty_columns = ['Score_1', 'Score_2', 'Score_3']
    
    raw_scores = df[score_columns].sum(axis=1)
    penalty_count = (df[penalty_columns] == 10).sum(axis=1)
    penalized_scores = raw_scores / 60 + 0.25 * penalty_count
    df['TargetQ_Final_v1'] = (100 / 1.75) * (1.75 - penalized_scores)
    
    return df

# ============================================================================
# Main pipeline
# ============================================================================

def target_id_multi_v1(
    malig_adata,
    ha_adata,
    device: str = 'cuda',
    batch_size: int = 25000,
    surface_evidence_path: str = "surface_evidence.v1.20240715.csv",
    use_fp16: bool = True
) -> pd.DataFrame:
    """GPU-optimized target ID pipeline"""
    
    genes = malig_adata.var_names
    malig_adata = malig_adata[:, genes]
    ha_adata = ha_adata[:, genes]
    clear_gpu_memory(verbose=False)

    overall_start = time.time()
    device = torch.device(device)
    
    if use_fp16 and device.type == 'cuda':
        if torch.cuda.get_device_capability(device)[0] < 7:
            use_fp16 = False
        else:
            print(f"Using FP16 mixed precision\n")
    
    try:
        # Load data
        print(f"------  Loading Data - | Total: {time.time()-overall_start:.1f}s")
        dtype = torch.float16 if use_fp16 else torch.float32
        malig_X = torch.tensor(np.array(malig_adata.X.T), dtype=dtype, device=device)
        ha_X = torch.tensor(np.array(ha_adata.X.T), dtype=dtype, device=device)
        
        # Encode groups
        m_ids = malig_adata.obs_names.str.split("._.", regex=False).str[1].values
        m_unique = np.unique(m_ids)
        m_id_to_idx = {id_val: idx for idx, id_val in enumerate(m_unique)}
        m_ids_encoded = torch.tensor([m_id_to_idx[x] for x in m_ids], dtype=torch.long, device=device)
        
        ha_ids = (ha_adata.obs_names.str.extract(r'^([^:]+:[^:]+)', expand=False)
                  .str.replace(r'[ -]', '_', regex=True).values)
        ha_unique = np.unique(ha_ids)
        ha_id_to_idx = {id_val: idx for idx, id_val in enumerate(ha_unique)}
        ha_ids_encoded = torch.tensor([ha_id_to_idx[x] for x in ha_ids], dtype=torch.long, device=device)
        
        n_malig_groups = len(m_unique)
        n_ha_groups = len(ha_unique)
        
        # Generate pairs
        n = len(genes)
        indices = np.triu_indices(n, k=0)
        gx_all = indices[0]
        gy_all = indices[1]
        n_batches = int(np.ceil(len(gx_all) / batch_size))
        
        print(f"------  Running Target ID ({len(gx_all)} pairs, {n_batches} batches)")

        m_group_indices = [(m_ids_encoded == g).nonzero(as_tuple=True)[0] for g in range(n_malig_groups)]
        ha_group_indices = [(ha_ids_encoded == g).nonzero(as_tuple=True)[0] for g in range(n_ha_groups)]

        all_results = []
        
        for batch_idx in range(n_batches):
            iter_start = time.time()
            start = batch_idx * batch_size
            end = min(start + batch_size, len(gx_all))
            
            gx_t = torch.tensor(gx_all[start:end], dtype=torch.long, device=device)
            gy_t = torch.tensor(gy_all[start:end], dtype=torch.long, device=device)
            
            print(f"  {batch_idx + 1}/{n_batches} | ", end='', flush=True)
            
            # Matrix computation
            t_matrix = time.time()
            malig_xy = torch.minimum(malig_X[gx_t], malig_X[gy_t])
            ha_xy = torch.minimum(ha_X[gx_t], ha_X[gy_t])
            malig_med = compute_group_median_cached(malig_xy, m_group_indices)
            ha_med = compute_group_median_cached(ha_xy, ha_group_indices)
            matrix_time = time.time() - t_matrix
            print(f"Matrix:{matrix_time:.1f}s | ", end='', flush=True)
            
            # Target ID scoring
            t_target = time.time()
            target_val = torch.max(malig_med, dim=1).values
            target_val_T = target_val ** 2
            ha_T = ha_med ** 2
            
            ha_ordered_idx = torch.argsort(ha_T, dim=1, descending=True)
            ha_ordered_T = torch.gather(ha_T, 1, ha_ordered_idx)
            ha_ordered = torch.gather(ha_med, 1, ha_ordered_idx)
            
            score_matrix = compute_score_matrix_gpu(
                target_val=target_val_T.float(),
                healthy_matrix_ordered_T=ha_ordered_T.float(),
                healthy_matrix_ordered=ha_ordered.float(),
                test_n_off_targets=10,
                device=device,
                batch_size=2000,
                min_lfc=0.25,
                max_off_val=1.0,
                offset=1e-8
            )
            target_time = time.time() - t_target
            print(f"Target:{target_time:.1f}s | ", end='', flush=True)
            
            # Metrics computation
            t_metrics = time.time()
            n_genes_batch = len(gx_t)
            corrected_idx_gpu = torch.argmax((score_matrix >= 0.35).int(), dim=1)
            n_pos_val = (malig_med >= 0.5).sum(dim=1)
            p_pos_per = n_pos_val.float() / n_malig_groups

            off_targets_gpu = {}
            for thresh in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
                off_targets_gpu[f'N_Off_Targets_{thresh}'] = (ha_med >= thresh).sum(dim=1)

            if malig_med.shape[1] > 1:
                sc_2nd_target_val = torch.kthvalue(malig_med, k=malig_med.shape[1]-1, dim=1).values
            else:
                sc_2nd_target_val = malig_med[:, 0]

            offset = 1e-8
            corrected_off_val = torch.gather(ha_ordered, 1, corrected_idx_gpu.unsqueeze(1)).squeeze(1)
            ha_ordered_first = ha_ordered[:, 0]

            log2_fc = torch.log2(target_val + offset) - torch.log2(ha_ordered_first + offset)
            corrected_log2_fc = torch.log2(target_val + offset) - torch.log2(corrected_off_val + offset)
            sc_2nd_lfc = torch.log2(sc_2nd_target_val + offset) - torch.log2(corrected_off_val + offset)

            log2_fc = torch.nan_to_num(log2_fc, nan=0.0, posinf=0.0, neginf=0.0).clamp(-5, 5)
            corrected_log2_fc = torch.nan_to_num(corrected_log2_fc, nan=0.0, posinf=0.0, neginf=0.0).clamp(-5, 5)
            sc_2nd_lfc = torch.nan_to_num(sc_2nd_lfc, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 10)

            n_off_targets = (score_matrix < 0.35).sum(dim=1)
            corrected_spec = score_matrix[torch.arange(n_genes_batch, device=device), corrected_idx_gpu]
            metrics_time = time.time() - t_metrics
            print(f"Metrics:{metrics_time:.1f}s | ", end='', flush=True)
            
            # Positive patients - computed inline
            t_pos = time.time()
            passing_mask = corrected_spec >= 0.35
            target_val_pos, n_pos, p_pos = compute_positive_patients_inline2(
                target_val=target_val,
                malig_med=malig_med,
                ha_med=ha_med,
                ha_ordered_idx=ha_ordered_idx,
                score_matrix=score_matrix,
                passing_mask=passing_mask,
                n_malig_groups=n_malig_groups,
                device=device
            )
            pos_time = time.time() - t_pos
            print(f"PosPat:{pos_time:.1f}s | ", end='', flush=True)
            
            # Storage
            t_storage = time.time()
            names = [f"{genes[i]}.{genes[j]}" for i, j in zip(gx_all[start:end], gy_all[start:end])]
            
            df = pd.DataFrame({
                'gene_name': names,
                'Target_Val': target_val.cpu().numpy(),
                'Specificity': score_matrix[:, 0].cpu().numpy(),
                'Corrected_Specificity': corrected_spec.cpu().numpy(),
                'Corrected_Top_Off_Target_Val': corrected_off_val.cpu().numpy(),
                'Top_Off_Target_Val': ha_ordered_first.cpu().numpy(),
                'Log2_Fold_Change': log2_fc.cpu().numpy(),
                'Corrected_Log2_Fold_Change': corrected_log2_fc.cpu().numpy(),
                'N_Off_Targets': n_off_targets.cpu().numpy(),
                'N_Pos_Val': n_pos_val.cpu().numpy(),
                'P_Pos_Per': p_pos_per.cpu().numpy(),
                'SC_2nd_Target_Val': sc_2nd_target_val.cpu().numpy(),
                'SC_2nd_Target_LFC': sc_2nd_lfc.cpu().numpy(),
                'N': n_malig_groups,
                'Target_Val_Pos': target_val_pos.cpu().numpy(),
                'N_Pos': n_pos.cpu().numpy(),
                'P_Pos': p_pos.cpu().numpy(),
                **{k: v.cpu().numpy() for k, v in off_targets_gpu.items()}
            })
            
            all_results.append(df)
            storage_time = time.time() - t_storage
            print(f"Storage:{storage_time:.1f}s | ", end='', flush=True)

            allocated, total, usage_pct = get_gpu_memory_info()
            iter_total = time.time() - iter_start
            print(f"IterTotal:{iter_total:.1f}s | GPU:{usage_pct:.0f}% | ", end='', flush=True)

            total_total = time.time() - overall_start
            total_minutes = total_total / 60
            print(f"Total:{total_minutes:.1f}m")

            # Cleanup
            del gx_t, gy_t, malig_xy, ha_xy, malig_med, ha_med
            del target_val, target_val_T, ha_T, ha_ordered, ha_ordered_T, ha_ordered_idx
            del score_matrix, target_val_pos, n_pos, p_pos
            torch.cuda.empty_cache()

        # Combine results
        print(f"------  Combining Results - | Total: {time.time()-overall_start:.1f}s")
        df_all = pd.concat(all_results, ignore_index=True)
        
        print(f"------  Computing Quality Scores - | Total: {time.time()-overall_start:.1f}s")
        df_all = compute_target_quality_score(df_all, surface_evidence_path)
        df_all = df_all.sort_values('TargetQ_Final_v1', ascending=False)
        
        print(f"------  Complete - | Total: {time.time()-overall_start:.1f}s")
        return df_all
        
    finally:
        try:
            del malig_X, ha_X, m_ids_encoded, ha_ids_encoded
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()


