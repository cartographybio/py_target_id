"""
Target ID
"""

# Define what gets exported
__all__ = ['target_id_v1']

import numpy as np
import pandas as pd
import torch
from typing import Optional, Tuple
from tqdm import tqdm
import warnings

# ============================================================================
# CORE COMPUTATION FUNCTIONS
# ============================================================================

def ultra_fast_js_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized Jensen-Shannon divergence"""
    a_sum = a.sum(dim=-1, keepdim=True)
    a_sum = torch.where(a_sum > 0, a_sum, torch.ones_like(a_sum))
    a_norm = a / a_sum
    
    mid = (a_norm + b) * 0.5
    eps = 1e-10
    
    entropy_a = torch.where(
        a_norm > eps,
        a_norm * torch.log2(a_norm + eps),
        torch.zeros_like(a_norm)
    ).sum(dim=-1)
    
    entropy_b = torch.where(
        b > eps,
        b * torch.log2(b + eps),
        torch.zeros_like(b)
    ).sum(dim=-1)
    
    entropy_mid = torch.where(
        mid > eps,
        mid * torch.log2(mid + eps),
        torch.zeros_like(mid)
    ).sum(dim=-1)
    
    jsdist = -entropy_mid + 0.5 * (entropy_a + entropy_b)
    jsdist = torch.clamp(jsdist, min=0.0, max=1.0)
    
    return torch.sqrt(jsdist)


def compute_score_matrix(
    target_val: torch.Tensor,
    healthy_matrix_ordered_T: torch.Tensor,
    test_n_off_targets: int,
    device: torch.device,
    batch_size: int = 1000
) -> torch.Tensor:
    """Compute specificity score matrix"""
    n_genes = target_val.shape[0]
    n_healthy = healthy_matrix_ordered_T.shape[1]
    test_n = min(test_n_off_targets, n_healthy)
    
    score_matrix = torch.zeros((n_genes, test_n), dtype=torch.float32, device=device)
    
    for batch_start in tqdm(range(0, n_genes, batch_size), desc="Computing scores"):
        batch_end = min(batch_start + batch_size, n_genes)
        batch_size_actual = batch_end - batch_start
        
        target_batch = target_val[batch_start:batch_end]
        healthy_batch = healthy_matrix_ordered_T[batch_start:batch_end]
        
        for iter_idx in range(test_n):
            n_cols_iter = n_healthy - iter_idx
            
            if n_cols_iter <= 0:
                continue
            
            combined = torch.zeros((batch_size_actual, n_cols_iter + 1),
                                  dtype=torch.float32, device=device)
            combined[:, 0] = target_batch
            combined[:, 1:] = healthy_batch[:, iter_idx:iter_idx + n_cols_iter]
            
            spec_vector = torch.zeros_like(combined)
            spec_vector[:, 0] = 1.0
            
            js_scores = ultra_fast_js_score(combined, spec_vector)
            scores = 1.0 - js_scores
            
            scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
            scores = torch.where(target_batch > 0, scores, torch.zeros_like(scores))
            
            score_matrix[batch_start:batch_end, iter_idx] = scores
    
    return score_matrix


def specificity_summary(
    target_val: np.ndarray,
    healthy_matrix: np.ndarray,
    threshold: float = 0.35,
    test_n_off_targets: int = 10,
    min_lfc: float = 0.25,
    max_lfc: float = 5.0,
    offset: float = 1e-8,
    max_off_val: float = 1.0,
    transform: str = "X^2",
    device: Optional[str] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Compute specificity summary and return off-target masks"""
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Processing {len(target_val)} genes on {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    target_val_np = target_val.copy()
    
    # Apply transform
    if transform == "X^2" or transform == "X * X":
        target_val_T = torch.tensor(target_val ** 2, dtype=torch.float32, device=device)
        healthy_matrix_T = torch.tensor(healthy_matrix ** 2, dtype=torch.float32, device=device)
    else:
        target_val_T = torch.tensor(target_val, dtype=torch.float32, device=device)
        healthy_matrix_T = torch.tensor(healthy_matrix, dtype=torch.float32, device=device)
    
    n_genes = len(target_val)
    n_healthy = healthy_matrix.shape[1]
    test_n_off_targets = min(test_n_off_targets, n_healthy - 1)
    
    # Order matrices
    print("Ordering matrices...")
    healthy_ordered_idx = torch.argsort(
        torch.tensor(healthy_matrix, dtype=torch.float32, device=device),
        dim=1, descending=True
    )
    healthy_ordered = torch.gather(
        torch.tensor(healthy_matrix, dtype=torch.float32, device=device),
        1, healthy_ordered_idx
    )
    
    healthy_ordered_idx_T = torch.argsort(healthy_matrix_T, dim=1, descending=True)
    healthy_ordered_T = torch.gather(healthy_matrix_T, 1, healthy_ordered_idx_T)
    
    # Compute scores
    print("Computing score matrix...")
    scores = compute_score_matrix(target_val_T, healthy_ordered_T, test_n_off_targets, device)
    
    scores_cpu = scores.cpu().numpy()
    healthy_ordered_cpu = healthy_ordered.cpu().numpy()
    healthy_ordered_idx_cpu = healthy_ordered_idx.cpu().numpy()
    
    score_matrix = scores_cpu.copy()
    
    # Apply LFC filter
    with np.errstate(divide='ignore', invalid='ignore'):
        lfc_check = (np.log2(target_val_np[:, None] + offset) - 
                    np.log2(healthy_ordered_cpu[:, :test_n_off_targets] + offset)) < min_lfc
    score_matrix[lfc_check] = 0
    
    # Apply max value filter
    max_val_check = healthy_ordered_cpu[:, :test_n_off_targets] > max_off_val
    score_matrix[max_val_check] = 0
    
    # Handle all-zero rows
    all_zero = np.sum(score_matrix, axis=1) == 0
    score_matrix[all_zero, -1] = scores_cpu[all_zero, -1]
    
    # Create summary
    print("Creating summary...")
    df = pd.DataFrame()
    
    corrected_idx = np.argmax(score_matrix >= threshold, axis=1)
    corrected_specificity = score_matrix[np.arange(n_genes), corrected_idx]
    corrected_off_val = healthy_ordered_cpu[np.arange(n_genes), corrected_idx]
    
    df['Target_Val'] = target_val_np
    df['Specificity'] = score_matrix[:, 0]
    df['Corrected_Specificity'] = corrected_specificity
    df['Corrected_Top_Off_Target_Val'] = corrected_off_val
    df['Top_Off_Target_Val'] = healthy_ordered_cpu[:, 0]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        df['Log2_Fold_Change'] = np.log2(target_val_np + offset) - np.log2(healthy_ordered_cpu[:, 0] + offset)
        df['Corrected_Log2_Fold_Change'] = np.log2(target_val_np + offset) - np.log2(corrected_off_val + offset)
    
    df['Log2_Fold_Change'] = np.where(np.isfinite(df['Log2_Fold_Change']), 
                                      df['Log2_Fold_Change'], 
                                      np.sign(df['Log2_Fold_Change']) * max_lfc)
    df['Corrected_Log2_Fold_Change'] = np.where(np.isfinite(df['Corrected_Log2_Fold_Change']),
                                                 df['Corrected_Log2_Fold_Change'],
                                                 np.sign(df['Corrected_Log2_Fold_Change']) * max_lfc)
    
    df['Log2_Fold_Change'] = df['Log2_Fold_Change'].clip(-max_lfc, max_lfc)
    df['Corrected_Log2_Fold_Change'] = df['Corrected_Log2_Fold_Change'].clip(-max_lfc, max_lfc)
    df['N_Off_Targets'] = np.sum(score_matrix < threshold, axis=1)
    
    # Build off-target masks
    print("Building off-target masks...")
    off_target_masks = np.zeros((n_genes, n_healthy), dtype=bool)
    
    for gene_idx in range(n_genes):
        failing_positions = np.where(score_matrix[gene_idx, :] < threshold)[0]
        failing_original_indices = healthy_ordered_idx_cpu[gene_idx, failing_positions]
        off_target_masks[gene_idx, failing_original_indices] = True
    
    return df, off_target_masks


def determine_positive_patients(
    df_summary: pd.DataFrame,
    malig_matrix: np.ndarray,
    healthy_matrix: np.ndarray,
    off_target_masks: np.ndarray,
    thresh: float = 0.35,
    transform: str = "X^2",
    device: Optional[str] = None,
    gene_batch_size: int = 200
) -> pd.DataFrame:
    """Fully parallelized positive patient determination"""
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    scaling_factors = np.array([1e-6, 1e-5, 1e-4] + list(np.arange(0.001, 1.001, 0.005)))
    scaling_factors = np.sort(scaling_factors)
    apply_transform = transform in ["X^2", "X * X"]
    
    n_genes = len(df_summary)
    n_malig = malig_matrix.shape[1]
    n_healthy = healthy_matrix.shape[1]
    
    valid_mask = (df_summary['Corrected_Specificity'].values >= thresh) & \
                 (~df_summary['Corrected_Specificity'].isna())
    valid_indices = np.where(valid_mask)[0]
    
    print(f"Processing {len(valid_indices)} genes in parallel batches on {device}")
    
    # Move to GPU
    target_vals = torch.tensor(df_summary['Target_Val'].values, dtype=torch.float32, device=device)
    malig_t = torch.tensor(malig_matrix, dtype=torch.float32, device=device)
    healthy_t = torch.tensor(healthy_matrix, dtype=torch.float32, device=device)
    scales_t = torch.tensor(scaling_factors, dtype=torch.float32, device=device)
    
    # Apply transform and zero out off-targets
    if apply_transform:
        healthy_T = healthy_t ** 2
    else:
        healthy_T = healthy_t.clone()
    
    off_target_masks_t = torch.tensor(off_target_masks, dtype=torch.bool, device=device)
    healthy_T[off_target_masks_t] = 0
    
    results = np.full((n_genes, 3), np.nan)
    results[:, 1] = 0
    results[:, 2] = 0
    
    n_tests = len(scaling_factors)
    eps = 1e-10
    
    # Process in batches
    for batch_start in tqdm(range(0, len(valid_indices), gene_batch_size), desc="Parallel batches"):
        batch_end = min(batch_start + gene_batch_size, len(valid_indices))
        batch_genes = valid_indices[batch_start:batch_end]
        batch_size = len(batch_genes)
        
        batch_targets = target_vals[batch_genes]
        batch_healthy = healthy_T[batch_genes]
        
        # Build test thresholds
        thresholds = batch_targets.unsqueeze(1) * scales_t.unsqueeze(0)
        if apply_transform:
            thresholds = thresholds ** 2
        
        # Build combined arrays
        combined = torch.zeros((batch_size, n_tests, n_healthy + 1), 
                              dtype=torch.float32, device=device)
        combined[:, :, 0] = thresholds
        combined[:, :, 1:] = batch_healthy.unsqueeze(1).expand(-1, n_tests, -1)
        
        # Normalize
        combined_sum = combined.sum(dim=2, keepdim=True).clamp(min=eps)
        combined_norm = combined / combined_sum
        
        # Specificity vector
        spec_vector = torch.zeros_like(combined)
        spec_vector[:, :, 0] = 1.0
        
        # Midpoint
        mid = (combined_norm + spec_vector) * 0.5
        
        # Vectorized entropy
        log_norm = torch.where(combined_norm > eps, 
                              torch.log2(combined_norm.clamp(min=eps)), 
                              torch.zeros_like(combined_norm))
        log_mid = torch.where(mid > eps, 
                             torch.log2(mid.clamp(min=eps)), 
                             torch.zeros_like(mid))
        
        entropy_a = (combined_norm * log_norm).sum(dim=2)
        entropy_mid = (mid * log_mid).sum(dim=2)
        
        # JS divergence
        jsdist = torch.clamp(-entropy_mid + 0.5 * entropy_a, min=0.0, max=1.0)
        specificities = 1.0 - torch.sqrt(jsdist)
        
        # Find first valid threshold
        valid_thresholds = specificities >= thresh
        
        for local_idx, gene_idx in enumerate(batch_genes):
            if valid_thresholds[local_idx].any():
                first_valid_idx = torch.argmax(valid_thresholds[local_idx].float()).item()
                final_threshold = (scaling_factors[first_valid_idx] * 
                                 batch_targets[local_idx]).cpu().item()
                
                results[gene_idx, 0] = final_threshold
                results[gene_idx, 1] = (malig_t[gene_idx] >= final_threshold).sum().cpu().item()
                results[gene_idx, 2] = results[gene_idx, 1] / n_malig
    
    df_summary['Target_Val_Pos'] = results[:, 0]
    df_summary['N_Pos'] = results[:, 1].astype(int)
    df_summary['P_Pos'] = results[:, 2]
    
    return df_summary


def compute_target_quality_score(
    df: pd.DataFrame,
    surface_evidence_path: str = "surface_evidence.v1.20240715.csv"
) -> pd.DataFrame:
    """Compute target quality scores with surface protein evidence"""
    
    # Load surface evidence if path provided
    try:
        surface_evidence = pd.read_csv(surface_evidence_path)
        surface_dict = dict(zip(surface_evidence['gene_name'], surface_evidence['surface_evidence']))
        df['Surface_Prob'] = df['gene_name'].map(surface_dict).fillna(1.0)
    except FileNotFoundError:
        print(f"Warning: {surface_evidence_path} not found, skipping surface evidence")
        df['Surface_Prob'] = 1.0
    
    # Score components
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
    
    # Compute final TargetQ score
    score_columns = ['Score_1', 'Score_2', 'Score_3', 'Score_5', 'Score_6', 'Score_7']
    penalty_columns = ['Score_1', 'Score_2', 'Score_3']
    
    raw_scores = df[score_columns].sum(axis=1)
    penalty_count = (df[penalty_columns] == 10).sum(axis=1)
    penalized_scores = raw_scores / 60 + 0.25 * penalty_count
    df['TargetQ_Final_v1'] = (100 / 1.75) * (1.75 - penalized_scores)
    
    return df


# ============================================================================
# MAIN RUN FUNCTION
# ============================================================================

def target_id_v1(
    malig_adata,
    healthy_adata,
    device: Optional[str] = None,
    surface_evidence_path: Optional[str] = "surface_evidence.v1.20240715.csv",
    version: str = "1.02"
) -> pd.DataFrame:
    """
    Run complete Target ID analysis
    
    Parameters
    ----------
    malig_adata : AnnData
        Malignant patient data (patients x genes)
    healthy_adata : AnnData
        Healthy atlas data (samples x genes)
    device : str, optional
        'cuda', 'cpu', or None (auto-detect)
    surface_evidence_path : str, optional
        Path to surface evidence CSV file
    version : str
        Version of algorithm
        
    Returns
    -------
    pd.DataFrame
        Target identification results with TargetQ scores
    """
    
    from scipy.sparse import issparse
    from py_target_id import run

    print(f"Starting Target ID v{version}...")
    
    # Find common genes
    genes = np.intersect1d(malig_adata.var_names, healthy_adata.var_names)
    print(f"Found {len(genes)} common genes")
    
    # Subset data
    malig_subset = malig_adata[:, genes].copy()
    healthy_subset = healthy_adata[:, genes].copy()

    # Compute Positivity Quickly
    if "positivity" not in malig_adata.layers:
        malig_adata = run.compute_positivity_matrix(malig_adata)
    
    # Extract matrices
    mat_malig = malig_subset.X.T if hasattr(malig_subset.X, 'T') else malig_subset.X.T
    mat_healthy = healthy_subset.X.T if hasattr(healthy_subset.X, 'T') else healthy_subset.X.T

    # Convert sparse to dense
    if issparse(mat_malig):
        print("Converting malignant sparse matrix to dense...")
        mat_malig = mat_malig.toarray()
    if issparse(mat_healthy):
        print("Converting healthy sparse matrix to dense...")
        mat_healthy = mat_healthy.toarray()
    
    if not isinstance(mat_malig, np.ndarray):
        mat_malig = np.array(mat_malig)
    if not isinstance(mat_healthy, np.ndarray):
        mat_healthy = np.array(mat_healthy)
    
    print(f"Malignant matrix: {mat_malig.shape}")
    print(f"Healthy matrix: {mat_healthy.shape}")
    
    # Compute target values
    target_val = np.max(mat_malig, axis=1)
    
    # Compute specificity
    print("\nComputing specificity...")
    df, off_target_masks = specificity_summary(
        target_val=target_val,
        healthy_matrix=mat_healthy,
        threshold=0.35,
        test_n_off_targets=10,
        min_lfc=0.25,
        offset=1e-8,
        max_off_val=1.0,
        transform="X^2",
        device=device
    )
    
    df.index = genes
    df['gene_name'] = genes
    
    # Determine positive patients
    df = determine_positive_patients(
        df_summary=df,
        malig_matrix=mat_malig,
        healthy_matrix=mat_healthy,
        off_target_masks=off_target_masks,
        thresh=0.35,
        transform="X^2",
        device=device,
        gene_batch_size=250
    )
    
    # Compute additional metrics
    print("\nComputing additional metrics...")
    df['N_Pos_Val'] = np.sum(mat_malig >= 0.5, axis=1)
    df['P_Pos_Per'] = df['N_Pos_Val'] / mat_malig.shape[1]
    
    for thresh in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
        df[f'N_Off_Targets_{thresh}'] = np.sum(mat_healthy >= thresh, axis=1)
    
    if mat_malig.shape[1] > 1:
        df['SC_2nd_Target_Val'] = np.partition(mat_malig, -2, axis=1)[:, -2]
    else:
        df['SC_2nd_Target_Val'] = mat_malig[:, 0]
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        with np.errstate(divide='ignore', invalid='ignore'):
            df['SC_2nd_Target_LFC'] = np.log2(df['SC_2nd_Target_Val'].values / 
                                              df['Corrected_Top_Off_Target_Val'].values)
    
    df['SC_2nd_Target_LFC'] = np.where(
        np.isfinite(df['SC_2nd_Target_LFC']),
        df['SC_2nd_Target_LFC'],
        0.0
    )
    df['SC_2nd_Target_LFC'] = df['SC_2nd_Target_LFC'].clip(0, 10)
    df['N'] = mat_malig.shape[1]
    
    # Compute target quality scores
    print("\nComputing target quality scores...")
    if surface_evidence_path:
        df = compute_target_quality_score(df, surface_evidence_path)

    df["Postivie_Final_0.1"] = (malig_adata[:, df['gene_name']].X >= 0.1).mean(axis=0) * 100    
    df["Postivie_Final_v2"] = malig_adata[:, df['gene_name']].layers['positivity'].mean(axis=0) * 100    
    df = df.sort_values('TargetQ_Final_v1', ascending=False)

    print("Target ID complete!")

    return df