"""
Reference data loading functions.
"""

__all__ = ['get_malig_med_adata', 'get_malig_ar_adata']

def get_malig_med_adata(manifest, positivity=True):
    import scanpy as sc
    import anndata as ad
    from tqdm import tqdm
    from py_target_id import run
    import numpy as np
    import h5py
    import pandas as pd
    
    def read_h5ad_robust(file_path):
        """Try to read h5ad with fallback for version compatibility issues"""
        try:
            # Try standard read first
            adata = sc.read_h5ad(file_path)
            return adata
        except TypeError as e:
            if "obs_names" in str(e):
                # Old AnnData format - reconstruct manually
                #@print(f"  Using legacy format reader for {file_path}")
                with h5py.File(file_path, 'r') as f:
                    X = f['X'][:]
                    obs_names = f['obs_names'][:].astype(str)
                    var_names = f['var_names'][:].astype(str)
                    
                    # Read obs and var metadata
                    obs = pd.DataFrame(index=obs_names)
                    var = pd.DataFrame(index=var_names)
                    
                    # Try to read obs/var dataframes if they're stored as groups
                    if 'obs' in f and isinstance(f['obs'], h5py.Group):
                        for col in f['obs'].keys():
                            obs[col] = f['obs'][col][:]
                    
                    if 'var' in f and isinstance(f['var'], h5py.Group):
                        for col in f['var'].keys():
                            var[col] = f['var'][col][:]
                    
                    # Create AnnData object
                    adata = ad.AnnData(X=X, obs=obs, var=var)
                    
                    # Add layers if they exist
                    if 'layers' in f:
                        for layer_name in f['layers'].keys():
                            adata.layers[layer_name] = f['layers'][layer_name][:]
                    
                    return adata
            else:
                raise
    
    # Load all files and preserve obs metadata
    adata_list = []
    for file_path in tqdm(manifest['Local_AD_Malig'], desc="Loading files"):
        try:
            adata = read_h5ad_robust(file_path)
            
            # Ensure obs metadata is preserved (especially nMalig)
            with h5py.File(file_path, 'r') as f:
                if 'obs' in f:
                    if isinstance(f['obs'], h5py.Dataset):
                        # Old format: structured array
                        obs_data = f['obs'][:]
                        for col_name in obs_data.dtype.names:
                            adata.obs[col_name] = obs_data[col_name]
                    elif isinstance(f['obs'], h5py.Group):
                        # New format: group with columns
                        for col_name in f['obs'].keys():
                            if col_name != '_index':
                                adata.obs[col_name] = f['obs'][col_name][:]
            
            adata_list.append(adata)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    # Concatenate
    adata_all = ad.concat(adata_list, axis=0)
    
    # Positivity
    if positivity:
        adata_all = run.compute_positivity_matrix(adata_all)
    
    adata_all.obs["Patient"] = np.array([name.split('._.')[1] for name in adata_all.obs_names])
    
    return adata_all

def get_malig_ar_adata(manifest):
    from py_target_id import infra
    import numpy as np
    malig_adata = infra.read_h5(manifest.Local_Archr_Malig, "RNA")
    malig_meta = ~malig_adata.obs_names.str.contains("nonmalig")
    malig_adata = malig_adata[malig_meta, :]
    malig_adata.obs["Patient"] = np.array([name.split('._.')[1] for name in malig_adata.obs_names]) 
    return malig_adata


