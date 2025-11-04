"""
Reference data loading functions.
"""

import os
import numpy as np
import scanpy as sc
import pandas as pd
from py_target_id import utils

__all__ = [
    'add_ref_weights',
    'get_ref_ffpe_off_target', 'get_ref_lv4_ffpe_med_adata', 'get_ref_lv4_ffpe_ar_adata',
    'get_ref_sc_off_target', 'get_ref_lv4_sc_med_adata', 'get_ref_lv4_sc_ar_adata',
    'ref_med_gene_sorted'
]

################################################################################################################################################
# Other
################################################################################################################################################

def add_ref_weights(ref_obj, df_off, col="Off_Target.V0"):
    
    # Determine Type of Ref Object
    if df_off.shape[0] == ref_obj.shape[0]:
        CT = ref_obj.obs_names.tolist()  # Convert to list for consistency
    else:
        CT = ref_obj.obs["CellType"].values  # Expect Tissue:CellType Combo Lv4
    
    # Check for mismatches
    mismatch_count = len(set(df_off['Combo_Lv4']) ^ set(CT))
    if mismatch_count > 0:
        raise ValueError(f"Mismatch: {mismatch_count} elements differ between df_off['Combo_Lv4'] and CT")
    
    # Match indices
    idx = [df_off['Combo_Lv4'].tolist().index(x) for x in CT]
    
    # Get off-target values (use the col parameter)
    off_vals = df_off.iloc[idx][col].values
    off_vals_scaled = off_vals / np.max(off_vals)
    
    # Add to obs
    ref_obj.obs["Weights"] = off_vals_scaled
    
    return ref_obj

################################################################################################################################################
# FFPE
################################################################################################################################################

def get_ref_ffpe_off_target(
    overwrite: bool = False, 
    version: str = "latest", 
    gcs_base_path: str = "gs://cartography_target_id_package/Healthy_Atlas/FFPE/",
    local_base_path: str = "temp/Healthy_Atlas/"
):

    local_file = os.path.join(local_base_path, "FFPE.Off_Target.csv")
    
    # Check if file exists and we don't want to overwrite
    if os.path.exists(local_file) and not overwrite:
        print(f"✓ Loading existing file: {local_file}")
    else:
        # List and select version
        versions = utils.list_gcs_versions(gcs_base_path)
        print(f"Available versions: {versions}")
        selected_version = utils.select_version(versions, version)
        
        # Define paths
        gcs_file = f"{gcs_base_path}{selected_version}/Off_Target.csv"
        
        # Download
        utils.download_gcs_file(gcs_file, local_file, overwrite)
    
    # Load and return
    df = pd.read_csv(local_file)
    df["Combo_Lv4"] = df["Combo_Lv4"].str.replace('α', 'a').str.replace('β', 'B')

    return df


def get_ref_lv4_ffpe_med_adata(
    overwrite: bool = False, 
    version: str = "latest", 
    gcs_base_path: str = "gs://cartography_target_id_package/Healthy_Atlas/FFPE/",
    local_base_path: str = "temp/Healthy_Atlas/"
):
    """
    Download and load reference Healthy Atlas Level 4 FFPE data.
    
    Args:
        overwrite: If True, re-download even if file exists
        version: Version to download ("latest" or specific date like "20250225")
        gcs_base_path: Base GCS path containing versioned data
        local_base_path: Local directory to save downloaded file
    
    Returns:
        AnnData object with the reference data
    
    Examples:
        >>> import py_target_id as tid
        >>> adata = tid.data.get_ref_lv4_ffpe_med_adata()
        >>> adata = tid.data.get_ref_lv4_ffpe_med_adata(version="20250225")
    """
    local_file = os.path.join(local_base_path, "Healthy_Atlas.Lv4.FFPE.h5ad")
    
    # Check if file exists and we don't want to overwrite
    if os.path.exists(local_file) and not overwrite:
        print(f"✓ Loading existing file: {local_file}")
    else:
        # List and select version
        versions = utils.list_gcs_versions(gcs_base_path)
        print(f"Available versions: {versions}")
        selected_version = utils.select_version(versions, version)
        
        # Define paths
        gcs_file = f"{gcs_base_path}{selected_version}/Healthy_Atlas.Lv4.h5ad"
        
        # Download
        utils.download_gcs_file(gcs_file, local_file, overwrite)
    
    # Load and return
    print(f"Loading data from: {local_file}")
    ad = sc.read_h5ad(local_file)
    ad.obs["Combo_Lv4"] = ad.obs["Combo_Lv4"].str.replace('α', 'a').str.replace('β', 'B')
    ad.obs_names = ad.obs_names.str.replace('α', 'a').str.replace('β', 'B')
    ad.obs['Combo_Lv4_Norm'] = ad.obs['Combo_Lv4'].str.replace(r'_\d+$', '', regex=True).values

    return ad

def get_ref_lv4_ffpe_ar_adata(
    overwrite: bool = False, 
    version: str = "latest", 
    gcs_base_path: str = "gs://cartography_target_id_package/Healthy_Atlas/FFPE/",
    local_base_path: str = "temp/Healthy_Atlas/"
):
    """
    Download and load reference Healthy Atlas Level 4 FFPE data.
    
    Args:
        overwrite: If True, re-download even if file exists
        version: Version to download ("latest" or specific date like "20250225")
        gcs_base_path: Base GCS path containing versioned data
        local_base_path: Local directory to save downloaded file
    
    Returns:
        AnnData object with the reference data
    
    Examples:
        >>> import py_target_id as tid
        >>> adata = tid.data.get_ref_lv4_ffpe_med_adata()
        >>> adata = tid.data.get_ref_lv4_ffpe_med_adata(version="20250225")
    """
    from py_target_id import infra

    local_file = os.path.join(local_base_path, "Healthy_Atlas.Gene_Matrix.ArchRCells.FFPE.h5ad")
    
    # Check if file exists and we don't want to overwrite
    if os.path.exists(local_file) and not overwrite:
        print(f"✓ Loading existing file: {local_file}")
    else:
        # List and select version
        versions = utils.list_gcs_versions(gcs_base_path)
        print(f"Available versions: {versions}")
        selected_version = utils.select_version(versions, version)
        
        # Define paths
        gcs_file = f"{gcs_base_path}{selected_version}/Healthy_Atlas.Gene_Matrix.ArchRCells.h5ad"
        
        # Download
        utils.download_gcs_file(gcs_file, local_file, overwrite)

    #Load Virtual AnnData
    ref_adata = infra.load_transposed_h5ad(local_file)

    #Cell Type
    ref_adata.obs['CellType'] = ref_adata.obs_names.str.extract(r'^([^:]+:[^:]+)', expand=False).str.replace(r'[ -]', '_', regex=True)
    ref_adata.obs['CellType'] = ref_adata.obs['CellType'].str.replace('α', 'a').str.replace('β', 'B')

    #Additional Info
    ref_adata_med = get_ref_lv4_ffpe_med_adata()
    ref_adata_med.obs = ref_adata_med.obs.drop("CellType", axis=1)

    ref_df = ref_adata.obs.merge(ref_adata_med.obs, left_on="CellType", right_on="Combo_Lv4", how="left")
    ref_df.index = ref_adata.obs.index
    ref_adata.obs = ref_df

    return ref_adata

################################################################################################################################################
# Single Cell
################################################################################################################################################

def get_ref_sc_off_target(
    overwrite: bool = False, 
    version: str = "latest", 
    gcs_base_path: str = "gs://cartography_target_id_package/Healthy_Atlas/SingleCell/",
    local_base_path: str = "temp/Healthy_Atlas/"
):

    local_file = os.path.join(local_base_path, "SC.Off_Target.csv")
    
    # Check if file exists and we don't want to overwrite
    if os.path.exists(local_file) and not overwrite:
        print(f"✓ Loading existing file: {local_file}")
    else:
        # List and select version
        versions = utils.list_gcs_versions(gcs_base_path)
        print(f"Available versions: {versions}")
        selected_version = utils.select_version(versions, version)
        
        # Define paths
        gcs_file = f"{gcs_base_path}{selected_version}/Off_Target.csv"
        
        # Download
        utils.download_gcs_file(gcs_file, local_file, overwrite)
    
    # Load and return
    df = pd.read_csv(local_file)
    df["Combo_Lv4"] = df["Combo_Lv4"].str.replace('α', 'a').str.replace('β', 'B')

    return df


def get_ref_lv4_sc_med_adata(
    overwrite: bool = False, 
    version: str = "latest", 
    gcs_base_path: str = "gs://cartography_target_id_package/Healthy_Atlas/SingleCell/",
    local_base_path: str = "temp/Healthy_Atlas/"
):
    """
    Download and load reference Healthy Atlas Level 4 SingleCell data.
    
    Args:
        overwrite: If True, re-download even if file exists
        version: Version to download ("latest" or specific date like "20250225")
        gcs_base_path: Base GCS path containing versioned data
        local_base_path: Local directory to save downloaded file
    
    Returns:
        AnnData object with the reference data
    
    Examples:
        >>> import py_target_id as tid
        >>> adata = tid.data.get_ref_lv4_ffpe_med_adata()
        >>> adata = tid.data.get_ref_lv4_ffpe_med_adata(version="20250225")
    """
    local_file = os.path.join(local_base_path, "Healthy_Atlas.Lv4.SC.h5ad")
    
    # Check if file exists and we don't want to overwrite
    if os.path.exists(local_file) and not overwrite:
        print(f"✓ Loading existing file: {local_file}")
    else:
        # List and select version
        versions = utils.list_gcs_versions(gcs_base_path)
        print(f"Available versions: {versions}")
        selected_version = utils.select_version(versions, version)
        
        # Define paths
        gcs_file = f"{gcs_base_path}{selected_version}/Healthy_Atlas.Lv4.h5ad"
        
        # Download
        utils.download_gcs_file(gcs_file, local_file, overwrite)
    
    # Load and return
    print(f"Loading data from: {local_file}")
    ad = sc.read_h5ad(local_file)
    ad.obs["Combo_Lv4"] = ad.obs["Combo_Lv4"].str.replace('α', 'a').str.replace('β', 'B')
    ad.obs_names = ad.obs_names.str.replace('α', 'a').str.replace('β', 'B')
    ad.obs['Combo_Lv4_Norm'] = ad.obs['Combo_Lv4'].str.replace(r'_\d+$', '', regex=True).values

    return ad

def get_ref_lv4_sc_ar_adata(
    overwrite: bool = False, 
    version: str = "latest", 
    gcs_base_path: str = "gs://cartography_target_id_package/Healthy_Atlas/SingleCell/",
    local_base_path: str = "temp/Healthy_Atlas/"
):
    """
    Download and load reference Healthy Atlas Level 4 FFPE data.
    
    Args:
        overwrite: If True, re-download even if file exists
        version: Version to download ("latest" or specific date like "20250225")
        gcs_base_path: Base GCS path containing versioned data
        local_base_path: Local directory to save downloaded file
    
    Returns:
        AnnData object with the reference data
    
    Examples:
        >>> import py_target_id as tid
        >>> adata = tid.data.get_ref_lv4_ffpe_med_adata()
        >>> adata = tid.data.get_ref_lv4_ffpe_med_adata(version="20250225")
    """
    from py_target_id import infra

    local_file = os.path.join(local_base_path, "Healthy_Atlas.Gene_Matrix.ArchRCells.SC.h5ad")
    
    # Check if file exists and we don't want to overwrite
    if os.path.exists(local_file) and not overwrite:
        print(f"✓ Loading existing file: {local_file}")
    else:
        # List and select version
        versions = utils.list_gcs_versions(gcs_base_path)
        print(f"Available versions: {versions}")
        selected_version = utils.select_version(versions, version)
        
        # Define paths
        gcs_file = f"{gcs_base_path}{selected_version}/Healthy_Atlas.Gene_Matrix.ArchRCells.h5ad"
        
        # Download
        utils.download_gcs_file(gcs_file, local_file, overwrite)

    #Load Virtual AnnData
    ref_adata = infra.load_transposed_h5ad(local_file)

    #Cell Type
    ref_adata.obs['CellType'] = ref_adata.obs_names.str.extract(r'^([^:]+:[^:]+)', expand=False).str.replace(r'[ -]', '_', regex=True)
    ref_adata.obs['CellType'] = ref_adata.obs['CellType'].str.replace('α', 'a').str.replace('β', 'B')

    #Additional Info
    ref_adata_med = get_ref_lv4_sc_med_adata()
    ref_adata_med.obs = ref_adata_med.obs.drop("CellType", axis=1)

    ref_df = ref_adata.obs.merge(ref_adata_med.obs, left_on="CellType", right_on="Combo_Lv4", how="left")
    ref_df.index = ref_adata.obs.index
    ref_adata.obs = ref_df

    return ref_adata

def ref_med_gene_sorted(ref_med_adata, gene):
    obs = ref_med_adata.obs.copy()
    obs[gene] = ref_med_adata[:, gene].X.copy().flatten()
    obs = obs.sort_values(gene, ascending=False)
    return obs
