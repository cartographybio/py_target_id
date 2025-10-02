"""
Target ID
"""

# Define what gets exported
__all__ = ['run_target_id_multi_v1']

import torch
import numpy as np
import pandas as pd
import time
import gc
import IPython
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

# ============================================================================
# Helper functions
# ============================================================================

def clear_gpu_memory(verbose=True):
    """
    Aggressively clear all PyTorch tensors and GPU memory
    
    Parameters
    ----------
    verbose : bool
        Print memory stats before and after cleanup
    """
    if verbose:
        print(f"Before cleanup:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    # Clear Jupyter output cache
    try:
        ipython = IPython.get_ipython()
        if ipython is not None:
            ipython.cache_size = 0
            # Clear output history variables
            if hasattr(ipython, 'history_manager'):
                ipython.history_manager.reset()
    except:
        pass
    
    # Clear underscore variables (Jupyter output cache)
    try:
        del _, __, ___
    except:
        pass
    
    # Find and delete all CUDA tensors
    deleted_count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    del obj
                    deleted_count += 1
        except:
            pass
    
    # Python garbage collection
    gc.collect()
    
    # Clear PyTorch cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    if verbose:
        print(f"\nDeleted {deleted_count} CUDA tensors")
        print(f"\nAfter cleanup:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
        
        # Show remaining large tensors if any
        remaining = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    size_gb = obj.element_size() * obj.nelement() / 1e9
                    if size_gb > 0.01:
                        remaining.append((obj.shape, size_gb))
            except:
                pass
        
        if remaining:
            print(f"\nWarning: {len(remaining)} large tensors still on GPU:")
            for shape, size in remaining[:5]:
                print(f"  Shape {shape}: {size:.3f} GB")
        else:
            print("\nAll large tensors cleared!")
    
    return torch.cuda.memory_allocated() / 1e9

def get_gpu_memory_info():
    """Get current GPU memory usage statistics"""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    usage_pct = (allocated / total) * 100
    return allocated, reserved, total, usage_pct

def compute_group_median_vectorized(data: torch.Tensor, group_ids: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Vectorized group median - processes ALL pairs in parallel"""
    n_pairs = data.shape[0]
    device = data.device
    result = torch.zeros((n_pairs, n_groups), dtype=data.dtype, device=device)
    
    for g in range(n_groups):
        mask = (group_ids == g)
        if mask.any():
            group_data = data[:, mask]
            result[:, g] = torch.median(group_data, dim=1).values
    
    return result

# Replace compute_group_median_vectorized with this:
def compute_group_median_cached(data: torch.Tensor, group_indices: list) -> torch.Tensor:
    n_pairs = data.shape[0]
    n_groups = len(group_indices)
    result = torch.zeros((n_pairs, n_groups), dtype=data.dtype, device=data.device)
    
    for g in range(n_groups):
        indices = group_indices[g]
        if len(indices) > 0:
            result[:, g] = torch.median(data[:, indices], dim=1).values
    
    return result

def compute_group_median_scatter(data: torch.Tensor, group_ids: torch.Tensor, n_groups: int) -> torch.Tensor:
    """
    Compute group medians across rows in parallel using segment_reduce.
    data: (n_pairs, n_samples)
    group_ids: (n_samples,)
    returns: (n_pairs, n_groups)
    """
    # Expand group ids for each pair so shape matches
    # (n_pairs, n_samples)
    expanded_group_ids = group_ids.unsqueeze(0).expand(data.size(0), -1)

    # Use segment_reduce directly (no sorting needed)
    medians = torch.segment_reduce(
        data,
        expanded_group_ids,
        reduce="median",
        unsafe=True  # skips bounds checking → faster
    )
    return medians

def compute_score_matrix_gpu(
    target_val: torch.Tensor,
    healthy_matrix_ordered_T: torch.Tensor,
    test_n_off_targets: int,
    device: torch.device,
    batch_size: int = 2000
) -> torch.Tensor:
    """GPU-optimized score matrix computation"""
    n_genes = target_val.shape[0]
    n_healthy = healthy_matrix_ordered_T.shape[1]
    test_n = min(test_n_off_targets, n_healthy)
    
    score_matrix = torch.zeros((n_genes, test_n), dtype=torch.float32, device=device)
    eps = 1e-10
    
    for batch_start in range(0, n_genes, batch_size):
        batch_end = min(batch_start + batch_size, n_genes)
        
        target_batch = target_val[batch_start:batch_end]
        healthy_batch = healthy_matrix_ordered_T[batch_start:batch_end]
        
        for iter_idx in range(test_n):
            n_cols_iter = n_healthy - iter_idx
            if n_cols_iter <= 0:
                continue
            
            combined = torch.cat([
                target_batch.unsqueeze(1),
                healthy_batch[:, iter_idx:iter_idx + n_cols_iter]
            ], dim=1)
            
            combined_sum = combined.sum(dim=1, keepdim=True).clamp(min=eps)
            combined_norm = combined / combined_sum
            
            spec_vector = torch.zeros_like(combined)
            spec_vector[:, 0] = 1.0
            
            mid = (combined_norm + spec_vector) * 0.5
            
            log_norm = torch.where(
                combined_norm > eps,
                torch.log2(combined_norm.clamp(min=eps)),
                torch.zeros_like(combined_norm)
            )
            log_mid = torch.where(
                mid > eps,
                torch.log2(mid.clamp(min=eps)),
                torch.zeros_like(mid)
            )
            
            entropy_a = (combined_norm * log_norm).sum(dim=1)
            entropy_mid = (mid * log_mid).sum(dim=1)
            
            jsdist = torch.clamp(-entropy_mid + 0.5 * entropy_a, min=0.0, max=1.0)
            scores = 1.0 - torch.sqrt(jsdist)
            
            scores = torch.where(target_batch > 0, scores, torch.zeros_like(scores))
            score_matrix[batch_start:batch_end, iter_idx] = scores
            
            del combined, combined_sum, combined_norm, spec_vector, mid
            del log_norm, log_mid, entropy_a, entropy_mid, jsdist, scores
    
    return score_matrix


def determine_positive_patients_gpu(
    df_summary: pd.DataFrame,
    malig_xy: torch.Tensor,
    thresh: float = 0.35,
    device: torch.device = None
) -> pd.DataFrame:
    """Determine positive patients for gene pairs"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    scaling_factors = torch.tensor(
        [1e-6, 1e-5, 1e-4] + list(np.arange(0.001, 1.001, 0.005)),
        dtype=torch.float32, device=device
    )
    
    n_genes = len(df_summary)
    n_malig = malig_xy.shape[1]
    
    valid_mask = (df_summary['Corrected_Specificity'].values >= thresh)
    valid_indices = np.where(valid_mask)[0]
    
    target_vals = torch.tensor(df_summary['Target_Val'].values, dtype=torch.float32, device=device)
    
    results = np.full((n_genes, 3), np.nan)
    results[:, 1] = 0
    results[:, 2] = 0
    
    if len(valid_indices) == 0:
        df_summary['Target_Val_Pos'] = results[:, 0]
        df_summary['N_Pos'] = results[:, 1].astype(int)
        df_summary['P_Pos'] = results[:, 2]
        return df_summary
    
    batch_size = 250
    
    for batch_start in range(0, len(valid_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_indices))
        batch_indices = valid_indices[batch_start:batch_end]
        
        batch_targets = target_vals[batch_indices]
        batch_malig = malig_xy[batch_indices]
        
        for local_idx, gene_idx in enumerate(batch_indices):
            target_val = batch_targets[local_idx]
            
            for scale in scaling_factors:
                test_threshold = (target_val * scale).item()
                n_pos = (batch_malig[local_idx] >= test_threshold).sum().item()
                
                if n_pos > 0:
                    results[gene_idx, 0] = test_threshold
                    results[gene_idx, 1] = n_pos
                    results[gene_idx, 2] = n_pos / n_malig
                    break
    
    df_summary['Target_Val_Pos'] = results[:, 0]
    df_summary['N_Pos'] = results[:, 1].astype(int)
    df_summary['P_Pos'] = results[:, 2]
    
    del target_vals, scaling_factors
    
    return df_summary


def compute_target_quality_score(
    df: pd.DataFrame,
    surface_evidence_path: str = "surface_evidence.v1.20240715.csv"
) -> pd.DataFrame:
    """Compute target quality scores"""
    
    try:
        surface_evidence = pd.read_csv(surface_evidence_path)
        surface_dict = dict(zip(surface_evidence['gene_name'], surface_evidence['surface_evidence']))
        df['Surface_Prob'] = df['gene_name'].map(surface_dict).fillna(1.0)
    except FileNotFoundError:
        print(f"Warning: {surface_evidence_path} not found")
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


def clear_gpu_memory(verbose=True):
    """Aggressively clear all PyTorch tensors and GPU memory"""
    if verbose:
        print(f"Before cleanup:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    try:
        import IPython
        ipython = IPython.get_ipython()
        if ipython is not None:
            ipython.cache_size = 0
            if hasattr(ipython, 'history_manager'):
                ipython.history_manager.reset()
    except:
        pass
    
    try:
        del _, __, ___
    except:
        pass
    
    deleted_count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    del obj
                    deleted_count += 1
        except:
            pass
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    if verbose:
        print(f"\nDeleted {deleted_count} CUDA tensors")
        print(f"\nAfter cleanup:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    return torch.cuda.memory_allocated() / 1e9


# ============================================================================
# Main pipeline with FP16
# ============================================================================

def run_target_id_multi_v1(
    malig_adata,
    ha_adata,
    surface: np.ndarray,
    device: str = 'cuda',
    batch_size: int = 20000,
    surface_evidence_path: str = "surface_evidence.v1.20240715.csv",
    use_fp16: bool = True
) -> pd.DataFrame:
    """
    GPU-optimized pipeline with FP16 mixed precision
    """
    
    clear_gpu_memory(verbose = False)

    overall_start = time.time()
    device = torch.device(device)
    
    # Check mixed precision support
    if use_fp16 and device.type == 'cuda':
        capability = torch.cuda.get_device_capability(device)
        supports_amp = capability[0] >= 7
        if not supports_amp:
            print(f"Warning: GPU doesn't support FP16, using FP32")
            use_fp16 = False
        else:
            print(f"Using FP16 mixed precision for ~2x speedup\n")
    
    try:
        elapsed = time.time() - overall_start
        print(f"------  Realizing Malig Matrix - | Total: {elapsed/60:.1f}min")
        
        # Load data in FP16 from the start
        dtype = torch.float16 if use_fp16 else torch.float32
        malig_X = torch.tensor(np.array(malig_adata.X.T), dtype=dtype, device=device)
        
        elapsed = time.time() - overall_start
        print(f"------  Realizing Healthy Matrix - | Total: {elapsed/60:.1f}min")
        
        ha_X = torch.tensor(np.array(ha_adata.X.T), dtype=dtype, device=device)
        
        # Encode group IDs
        m_ids = malig_adata.obs_names.str.split("._.", regex=False).str[1].values
        m_unique = np.unique(m_ids)
        m_id_to_idx = {id_val: idx for idx, id_val in enumerate(m_unique)}
        m_ids_encoded = torch.tensor([m_id_to_idx[x] for x in m_ids], dtype=torch.long, device=device)
        
        ha_ids = (ha_adata.obs_names
                  .str.extract(r'^([^:]+:[^:]+)', expand=False)
                  .str.replace(r'[ -]', '_', regex=True).values)
        ha_unique = np.unique(ha_ids)
        ha_id_to_idx = {id_val: idx for idx, id_val in enumerate(ha_unique)}
        ha_ids_encoded = torch.tensor([ha_id_to_idx[x] for x in ha_ids], dtype=torch.long, device=device)
        
        n_malig_groups = len(m_unique)
        n_ha_groups = len(ha_unique)
        n_malig_samples = malig_X.shape[1]
        
        # Generate pairs
        n = len(surface)
        indices = np.triu_indices(n, k=1)
        gx_all = indices[0]
        gy_all = indices[1]
        
        n_batches = int(np.ceil(len(gx_all) / batch_size))
        
        elapsed = time.time() - overall_start
        print(f"------  Running Target ID - | Total: {elapsed/60:.1f}min")

        m_group_indices = [(m_ids_encoded == g).nonzero(as_tuple=True)[0] for g in range(n_malig_groups)]
        ha_group_indices = [(ha_ids_encoded == g).nonzero(as_tuple=True)[0] for g in range(n_ha_groups)]

        all_results = []
        
        for batch_idx in range(n_batches):
            iter_start = time.time()
            
            start = batch_idx * batch_size
            end = min(start + batch_size, len(gx_all))
            
            gx_t = torch.tensor(gx_all[start:end], dtype=torch.long, device=device)
            gy_t = torch.tensor(gy_all[start:end], dtype=torch.long, device=device)
            
            print(f"  {batch_idx + 1}/{n_batches} | ", end='')
            
            # Matrix computation - ALL IN FP16
            print("Matrix:", end='')
            matrix_start = time.time()
            
            # Operations stay in FP16
            malig_xy = torch.minimum(malig_X[gx_t], malig_X[gy_t])
            ha_xy = torch.minimum(ha_X[gx_t], ha_X[gy_t])
            
            #malig_med = group_median_quantile(malig_xy, m_ids_encoded, n_malig_groups)
            #ha_med = group_median_quantile(ha_xy, ha_ids_encoded, n_ha_groups)
            malig_med = compute_group_median_cached(malig_xy, m_group_indices)
            ha_med = compute_group_median_cached(ha_xy, ha_group_indices)

            matrix_time = time.time() - matrix_start
            print(f"{matrix_time:.1f}s | ", end='')
            
            # Target ID
            print("Target:", end='')
            target_start = time.time()
            
            target_val = torch.max(malig_med, dim=1).values
            target_val_T = target_val ** 2
            ha_T = ha_med ** 2
            
            ha_ordered_idx = torch.argsort(ha_T, dim=1, descending=True)
            ha_ordered_T = torch.gather(ha_T, 1, ha_ordered_idx)
            ha_ordered = torch.gather(ha_med, 1, ha_ordered_idx)
            
            # Convert to FP32 for score matrix (needs more precision for log operations)
            score_matrix = compute_score_matrix_gpu(
                target_val=target_val_T.float(),
                healthy_matrix_ordered_T=ha_ordered_T.float(),
                test_n_off_targets=10,
                device=device,
                batch_size=2000
            )
            
            del target_val_T, ha_T, ha_ordered_idx, ha_ordered_T
            
            target_time = time.time() - target_start
            print(f"{target_time:.1f}s | ", end='')
                                    
            # Metrics - PROFILED VERSION
            print("Metrics:", end='')
            metrics_start = time.time()

            # Synchronize GPU to get accurate timings
            torch.cuda.synchronize()

            # Section 1: Basic computations
            t1 = time.time()
            n_genes_batch = len(gx_t)
            corrected_idx_gpu = torch.argmax((score_matrix >= 0.35).int(), dim=1)
            n_pos_val = (malig_xy >= 0.5).sum(dim=1)
            p_pos_per = n_pos_val.half() / n_malig_samples
            torch.cuda.synchronize()
            section1_time = time.time() - t1

            # Section 2: Off-targets - ON MEDIANS (like R does)
            t2 = time.time()

            thresholds = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
            off_targets_gpu = {}

            for thresh in thresholds:
                off_targets_gpu[f'N_Off_Targets_{thresh}'] = (ha_med >= thresh).sum(dim=1)

            torch.cuda.synchronize()
            section2_time = time.time() - t2

            # Section 3: Second target value
            t3 = time.time()
            if malig_med.shape[1] > 1:
                sc_2nd_target_val = torch.kthvalue(malig_med, k=malig_med.shape[1]-1, dim=1).values
            else:
                sc_2nd_target_val = malig_med[:, 0]
            torch.cuda.synchronize()
            section3_time = time.time() - t3

            # Section 4: Log computations
            t4 = time.time()
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
            torch.cuda.synchronize()
            section4_time = time.time() - t4

            # Section 5: CPU transfer
            t5 = time.time()
            names = [f"{surface[i]}.{surface[j]}" for i, j in zip(gx_all[start:end], gy_all[start:end])]

            # Time individual transfers
            transfer_times = {}
            t_temp = time.time()
            target_val_cpu = target_val.cpu().numpy()
            transfer_times['target_val'] = time.time() - t_temp

            t_temp = time.time()
            spec_cpu = score_matrix[:, 0].cpu().numpy()
            transfer_times['specificity'] = time.time() - t_temp

            t_temp = time.time()
            corrected_spec_cpu = score_matrix[torch.arange(n_genes_batch, device=device), corrected_idx_gpu].cpu().numpy()
            transfer_times['corrected_spec'] = time.time() - t_temp

            t_temp = time.time()
            off_targets_cpu = {k: v.cpu().numpy() for k, v in off_targets_gpu.items()}
            transfer_times['off_targets'] = time.time() - t_temp

            # Remaining transfers
            corrected_off_val_cpu = corrected_off_val.cpu().numpy()
            ha_ordered_first_cpu = ha_ordered_first.cpu().numpy()
            log2_fc_cpu = log2_fc.cpu().numpy()
            corrected_log2_fc_cpu = corrected_log2_fc.cpu().numpy()
            n_off_targets_cpu = n_off_targets.cpu().numpy()
            n_pos_val_cpu = n_pos_val.cpu().numpy()
            p_pos_per_cpu = p_pos_per.cpu().numpy()
            sc_2nd_target_val_cpu = sc_2nd_target_val.cpu().numpy()
            sc_2nd_lfc_cpu = sc_2nd_lfc.cpu().numpy()

            section5_time = time.time() - t5

            # Section 6: DataFrame creation
            t6 = time.time()
            df = pd.DataFrame({
                'gene_name': names,
                'batch_start_idx': start,
                'local_idx': np.arange(len(names)),
                'Target_Val': target_val_cpu,
                'Specificity': spec_cpu,
                'Corrected_Specificity': corrected_spec_cpu,
                'Corrected_Top_Off_Target_Val': corrected_off_val_cpu,
                'Top_Off_Target_Val': ha_ordered_first_cpu,
                'Log2_Fold_Change': log2_fc_cpu,
                'Corrected_Log2_Fold_Change': corrected_log2_fc_cpu,
                'N_Off_Targets': n_off_targets_cpu,
                'N_Pos_Val': n_pos_val_cpu,
                'P_Pos_Per': p_pos_per_cpu,
                'SC_2nd_Target_Val': sc_2nd_target_val_cpu,
                'SC_2nd_Target_LFC': sc_2nd_lfc_cpu,
                'N': n_malig_samples,
                **off_targets_cpu
            })
            all_results.append(df)
            section6_time = time.time() - t6

            metrics_time = time.time() - metrics_start

            # Print detailed breakdown
            print(f"{metrics_time:.1f}s ", end='')
            print(f"[GPU:{section1_time:.2f}+{section2_time:.2f}+{section3_time:.2f}+{section4_time:.2f}={section1_time+section2_time+section3_time+section4_time:.2f}s, ", end='')
            print(f"Xfer:{section5_time:.2f}s, DF:{section6_time:.2f}s] | ", end='')

            # Get GPU stats
            allocated, reserved, total, usage_pct = get_gpu_memory_info()

            # CLEANUP
            del gx_t, gy_t, malig_xy, ha_xy, malig_med, ha_med
            del target_val, ha_ordered, score_matrix, ha_ordered_first
            del corrected_idx_gpu, n_pos_val, p_pos_per, off_targets_gpu
            del sc_2nd_target_val, corrected_off_val, n_off_targets
            del log2_fc, corrected_log2_fc, sc_2nd_lfc
            torch.cuda.empty_cache()

            iter_time = time.time() - iter_start
            overall_elapsed = time.time() - overall_start

            print(f"Done:{iter_time:.1f}s | Total ({overall_elapsed/60:.1f}min) | GPU:{usage_pct:.1f}% ({allocated:.1f}/{total:.1f}GB)")
        
        df_all = pd.concat(all_results, ignore_index=True)
        
        # PASS 2: Positive patients
        passing_mask = df_all['Corrected_Specificity'] >= 0.35
        n_passing = passing_mask.sum()
        
        df_all['Target_Val_Pos'] = np.nan
        df_all['N_Pos'] = 0
        df_all['P_Pos'] = 0.0
        
        if n_passing > 0:
            passing_indices = df_all[passing_mask].index.values
            
            for batch_idx in range(0, len(passing_indices), batch_size):
                batch_passing = passing_indices[batch_idx:min(batch_idx + batch_size, len(passing_indices))]
                
                batch_df = df_all.loc[batch_passing]
                gx_batch = gx_all[batch_df['batch_start_idx'].values + batch_df['local_idx'].values]
                gy_batch = gy_all[batch_df['batch_start_idx'].values + batch_df['local_idx'].values]
                
                gx_t = torch.tensor(gx_batch, dtype=torch.long, device=device)
                gy_t = torch.tensor(gy_batch, dtype=torch.long, device=device)
                
                # Use FP16 for memory efficiency
                malig_xy = torch.minimum(malig_X[gx_t], malig_X[gy_t])
                
                # Convert to FP32 for positive patient computation
                batch_results = determine_positive_patients_gpu(
                    df_summary=batch_df.copy(),
                    malig_xy=malig_xy.float(),
                    thresh=0.35,
                    device=device
                )
                
                df_all.loc[batch_passing, 'Target_Val_Pos'] = batch_results['Target_Val_Pos'].values
                df_all.loc[batch_passing, 'N_Pos'] = batch_results['N_Pos'].values
                df_all.loc[batch_passing, 'P_Pos'] = batch_results['P_Pos'].values
                
                del gx_t, gy_t, malig_xy
                torch.cuda.empty_cache()
        
        df_all = df_all.drop(columns=['batch_start_idx', 'local_idx'])
        
        elapsed = time.time() - overall_start
        print(f"------  Running Single Stats - | Total: {elapsed/60:.1f}min")
        
        df_all = compute_target_quality_score(df_all, surface_evidence_path)
        df_all = df_all.sort_values('TargetQ_Final_v1', ascending=False)
        
        return df_all
        
    finally:
        try:
            del malig_X, ha_X, m_ids_encoded, ha_ids_encoded
        except:
            pass
        
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

