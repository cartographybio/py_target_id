"""
Reference data loading functions.
"""

__all__ = ['get_malig_adata', 'get_malig_archr_adata']

def get_malig_adata(manifest, positivity = True):
    import scanpy as sc
    import anndata as ad
    from tqdm import tqdm
    from py_target_id import run

    # Load all files
    adata_list = []
    for file_path in tqdm(manifest['Local_AD_Malig'], desc="Loading files"):
        adata = sc.read_h5ad(file_path)
        adata_list.append(adata)

    # Concatenate
    adata_all = ad.concat(adata_list, axis=0)

    #Positivity
    if positivity:
        adata_all = run.compute_positivity_matrix(adata_all)

    adata_all.obs["Patient"] = np.array([name.split('._.')[1] for name in adata_all.obs_names]) 
    return adata_all

def get_malig_archr_adata(manifest):
    from py_target_id import infra
    malig_adata = infra.read_h5(manifest.Local_Archr_Malig, "RNA")
    malig_meta = ~malig_adata.obs_names.str.contains("nonmalig")
    malig_adata = malig_adata[malig_meta, :]
    malig_adata.obs["Patient"] = np.array([name.split('._.')[1] for name in malig_adata.obs_names]) 
    return malig_adata


