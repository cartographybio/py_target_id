"""
Reference data loading functions.
"""

import os
import scanpy as sc
from py_target_id.utils import list_gcs_versions, select_version, download_gcs_file

__all__ = ['get_ref_lv4_ffpe_med_adata']


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
        versions = list_gcs_versions(gcs_base_path)
        print(f"Available versions: {versions}")
        selected_version = select_version(versions, version)
        
        # Define paths
        gcs_file = f"{gcs_base_path}{selected_version}/Healthy_Atlas.Lv4.h5ad"
        
        # Download
        download_gcs_file(gcs_file, local_file, overwrite)
    
    # Load and return
    print(f"Loading data from: {local_file}")
    return sc.read_h5ad(local_file)

def get_ref_lv4_ffpe_h5_adata(
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

    local_file = os.path.join(local_base_path, "Healthy_Atlas.Gene_Matrix.ArchRCells.FFPE.h5")
    
    # Check if file exists and we don't want to overwrite
    if os.path.exists(local_file) and not overwrite:
        print(f"✓ Loading existing file: {local_file}")
    else:
        # List and select version
        versions = utils.list_gcs_versions(gcs_base_path)
        print(f"Available versions: {versions}")
        selected_version = utils.select_version(versions, version)
        
        # Define paths
        gcs_file = f"{gcs_base_path}{selected_version}/Healthy_Atlas.Gene_Matrix.ArchRCells.h5"
        
        # Download
        utils.download_gcs_file(gcs_file, local_file, overwrite)

    #Load Virtual AnnData
    ref_adata = infra.read_h5(local_file, "RNA_Norm_Counts")

    #Cell Type
    ref_adata.obs['CellType'] = ref_adata.obs_names.str.extract(r'^([^:]+:[^:]+)', expand=False).str.replace(r'[ -]', '_', regex=True)
    ref_adata.obs['CellType'] = ref_adata.obs['CellType'].str.replace('α', 'a').str.replace('β', 'B')

    return ref_adata


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
        versions = list_gcs_versions(gcs_base_path)
        print(f"Available versions: {versions}")
        selected_version = select_version(versions, version)
        
        # Define paths
        gcs_file = f"{gcs_base_path}{selected_version}/Healthy_Atlas.Lv4.h5ad"
        
        # Download
        download_gcs_file(gcs_file, local_file, overwrite)
    
    # Load and return
    print(f"Loading data from: {local_file}")
    return sc.read_h5ad(local_file)

def get_ref_lv4_sc_h5_adata(
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

    local_file = os.path.join(local_base_path, "Healthy_Atlas.Gene_Matrix.ArchRCells.SC.h5")
    
    # Check if file exists and we don't want to overwrite
    if os.path.exists(local_file) and not overwrite:
        print(f"✓ Loading existing file: {local_file}")
    else:
        # List and select version
        versions = utils.list_gcs_versions(gcs_base_path)
        print(f"Available versions: {versions}")
        selected_version = utils.select_version(versions, version)
        
        # Define paths
        gcs_file = f"{gcs_base_path}{selected_version}/Healthy_Atlas.Gene_Matrix.ArchRCells.h5"
        
        # Download
        utils.download_gcs_file(gcs_file, local_file, overwrite)

    #Load Virtual AnnData
    ref_adata = infra.read_h5(local_file, "RNA_Norm_Counts")

    #Cell Type
    ref_adata.obs['CellType'] = ref_adata.obs_names.str.extract(r'^([^:]+:[^:]+)', expand=False).str.replace(r'[ -]', '_', regex=True)
    ref_adata.obs['CellType'] = ref_adata.obs['CellType'].str.replace('α', 'a').str.replace('β', 'B')

    return ref_adata
