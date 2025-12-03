"""
compute_target_quality_v2
"""

import numpy as np
import pandas as pd
from typing import List, Optional

__all__ = ['tcga_multis_tpm_threshold']

def tcga_multis_tpm_threshold(
    multis: list,
    tcga_adata = None,
    tpm_threshold: float = 10,
    pct_threshold: float = 0.20,
    main_indication: str = None
):
    """
    Analyze TCGA indications for gene combinations with co-expression above TPM threshold.
    
    Parameters
    ----------
    multis : list
        Gene combinations in format ["GENE1_GENE2", ...]
    tcga_adata : AnnData
        TCGA expression data (samples × genes)
    tpm_threshold : float
        TPM threshold for "positive" (default: 10)
    pct_threshold : float
        Percentage threshold for indication inclusion (default: 0.20 = 20%)
    main_indication : str
        Indication to report percentage for (e.g., "LUAD"). If None, skipped.
    
    Returns
    -------
    DataFrame with columns:
        - 'gene_pair': gene pair name
        - 'n_indications_above_threshold': number of indications with >= pct_threshold patients
        - 'n_patients_above_threshold': total patients meeting criteria across qualifying indications
        - '{main_indication}_pct': percentage for the specified indication (if provided)
    """
    
    from py_target_id import utils
    
    # Lazy-load TCGA data only if needed
    if tcga_adata is None:
        tcga_adata = utils.get_tcga_adata()
    
    # Parse gene combinations - convert to list if pandas Series
    if hasattr(multis, 'tolist'):
        multis = multis.tolist()
    
    multis_split = np.array([m.split("_") for m in multis])
    genes = np.unique(multis_split.ravel()).tolist()
    
    print(f"Analyzing {len(multis)} gene combinations with {len(genes)} unique genes")
    
    # Load TCGA data
    print("Reading in TCGA...")
    if hasattr(tcga_adata, 'to_memory'):
        print("Materializing TCGA VirtualAnnData...")
        tcga_adata = tcga_adata.to_memory()
    
    # Get all samples first, then extract indication info before subsetting genes
    tcga_samples = tcga_adata.obs_names.values
    tcga_id = np.array([s.split('#')[0] for s in tcga_samples])
    
    print(f"Found {len(np.unique(tcga_id))} indications: {sorted(np.unique(tcga_id))}")
    
    # Now subset to genes
    tcga_subset = tcga_adata[:, genes]
    tcga_mat = tcga_subset.X.toarray() if hasattr(tcga_subset.X, 'toarray') else tcga_subset.X
    
    # Process each gene combination
    results_list = []
    
    for idx, (gx, gy) in enumerate(multis_split):
        # Get gene indices
        idx_gx = genes.index(gx)
        idx_gy = genes.index(gy)
        
        # TCGA data
        df_tcga = pd.DataFrame({
            'x1': tcga_mat[:, idx_gx],
            'x2': tcga_mat[:, idx_gy],
            'facet': tcga_id
        })
        
        # Minimum of the pair
        df_tcga['xm'] = np.minimum(df_tcga['x1'], df_tcga['x2'])
        df_tcga['TPM_threshold'] = df_tcga['xm'] > tpm_threshold
        
        # Calculate percentage positive by cancer type
        pct_results = df_tcga.groupby('facet')['TPM_threshold'].agg(['mean', 'size']).reset_index()
        pct_results.columns = ['cancer_type', 'percentage', 'n']
        pct_results['percentage'] = pct_results['percentage'] * 100
        
        # Count indications meeting threshold
        pct_results['meets_threshold'] = pct_results['percentage'] >= (pct_threshold * 100)
        n_indications = pct_results['meets_threshold'].sum()
        n_patients = pct_results[pct_results['meets_threshold']]['n'].sum()
        
        # Get percentage for main indication if specified
        main_pct = None
        if main_indication:
            main_row = pct_results[pct_results['cancer_type'] == main_indication]
            if len(main_row) > 0:
                main_pct = main_row.iloc[0]['percentage']
        
        result_dict = {
            'gene_pair': multis[idx],
            'n_indications_above_threshold': int(n_indications),
            'n_patients_above_threshold': int(n_patients)
        }
        
        if main_indication:
            result_dict[f'{main_indication}_pct'] = main_pct
        
        results_list.append(result_dict)
        
        # Debug: show which indications pass
        if idx < 3:  # Show first 3 pairs
            print(f"\n{multis[idx]}:")
            print(pct_results[['cancer_type', 'percentage', 'n', 'meets_threshold']].head(10))
    
    # Create DataFrame in same order as input
    result_df = pd.DataFrame(results_list)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete: {len(result_df)} gene pairs")
    print(f"Threshold: ≥{pct_threshold*100:.1f}% patients per indication with min(gene1, gene2) >{tpm_threshold} TPM")
    print(f"{'='*60}\n")
    
    return result_df
