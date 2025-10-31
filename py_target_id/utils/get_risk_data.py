"""
Reference data loading functions.
"""

import os
import numpy as np
import scanpy as sc
import pandas as pd
from py_target_id import utils

__all__ = [
    'get_single_risk_scores', 'get_multi_risk_scores'
]

def get_single_risk_scores(
    genes : list = None,
    overwrite: bool = False, 
    version: str = "latest", 
    gcs_base_path: str = "gs://cartography_target_id_package/Other_Input/Risk",
    local_base_path: str = "temp/Risk/"
):
    local_file = os.path.join(local_base_path, "Single_Risk_Scores.parquet")
    
    # Check if file exists and we don't want to overwrite
    if os.path.exists(local_file) and not overwrite:
        print(f"✓ Loading existing file: {local_file}")
    else:
        # Define paths
        gcs_file = f"{gcs_base_path}/Single_Risk_Scores.20251030.parquet"
        
        # Download
        utils.download_gcs_file(gcs_file, local_file, overwrite)
    
    df = pd.read_parquet(local_file)
    
    # Load and return
    if genes is not None:

        # Convert string to list
        if isinstance(genes, str):
            genes = [genes]

        df_indexed = df.set_index("gene_name")

        # Reindex to match gene order exactly, NaN for missing
        return df_indexed.reindex(genes).reset_index()

    else:

        return df

def get_multi_risk_scores(
    genes : list = None,
    overwrite: bool = False, 
    version: str = "surface", 
    gcs_base_path: str = "gs://cartography_target_id_package/Other_Input/Risk",
    local_base_path: str = "temp/Risk/"
):
    local_file = os.path.join(local_base_path, "Multi_Risk_Scores.parquet")
    
    # Check if file exists and we don't want to overwrite
    if os.path.exists(local_file) and not overwrite:
        print(f"✓ Loading existing file: {local_file}")
    else:

        if version.lower() == "surface":
            gcs_file = f"{gcs_base_path}/Multi_Risk_Scores.20251030.parquet"
        
        # Download
        utils.download_gcs_file(gcs_file, local_file, overwrite)
    
    df = pd.read_parquet(local_file)
    print(f"✓ Loaded database")

    # Load and return
    if genes is not None:

        # Convert string to list
        if isinstance(genes, str):
            genes = [genes]

        df_indexed = df.set_index("gene_name")

        # Reindex to match gene order exactly, NaN for missing
        return df_indexed.reindex(genes).reset_index()

    else:

        return df







