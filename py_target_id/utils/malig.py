"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['malig_patient_median']

def malig_patient_median(malig_data):
    import time
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from dask.diagnostics import ProgressBar
    import pandas as pd
    import anndata as ad
    
    # Get patient info
    malig_data.obs['patient_id'] = malig_data.obs_names.str.split("._.", regex=False).str[1]
    unique_patients = malig_data.obs['patient_id'].unique()
    patient_ids = malig_data.obs['patient_id'].values
    
    start = time.time()

    full_data = malig_data.X.rechunk('auto')  # 1000 cells per chunk
    
    with ProgressBar():
        full_data = full_data.compute()
    
    def compute_patient_median(patient):
        mask = patient_ids == patient
        return np.median(full_data[mask], axis=0)
    
    # Parallel median computation
    with ThreadPoolExecutor(max_workers=8) as executor:
        patient_medians_list = list(executor.map(compute_patient_median, unique_patients))
    
    # Create AnnData with correct orientation
    adata = ad.AnnData(
        X=np.array(patient_medians_list),
        obs=pd.DataFrame(index=unique_patients),      # patients as rows (obs)
        var=pd.DataFrame(index=malig_data.var_names)  # genes as columns (var)
    )
    
    elapsed = time.time() - start
    print(f"Took {elapsed:.2f} seconds")
    
    return adata