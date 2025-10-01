"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['se_rds_to_anndata']

def se_rds_to_anndata(rds_path: str, assay_name: str = "median", output_path: str = None):
    """
    Convert R SummarizedExperiment RDS file to AnnData h5ad format
    
    Parameters
    ----------
    rds_path : str
        Path to the RDS file containing a SummarizedExperiment object
    assay_name : str
        Name of the assay to extract (default: "median")
    output_path : str, optional
        Path to save the h5ad file. If None, doesn't save to disk.
        
    Returns
    -------
    anndata.AnnData
        AnnData object with data transposed to cells × genes format
    """
        
    import rpy2.robjects as ro
    import numpy as np
    import pandas as pd
    import anndata as ad
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import pandas2ri

    # Load required R library
    ro.r('suppressPackageStartupMessages(library(SummarizedExperiment))')
    
    # Read the RDS file
    ro.r(f'se <- readRDS("{rds_path}")')
    
    # Extract matrix (fast conversion with asarray)
    print("Extracting matrix...")
    if assay_name:
        matrix = np.asarray(ro.r(f'assay(se, "{assay_name}")'))
    else:
        matrix = np.asarray(ro.r('assay(se)'))
    
    gene_names = list(ro.r('rownames(se)'))
    obs_names = list(ro.r('colnames(se)'))
    
    print(f"Loaded: {matrix.shape[0]} genes × {matrix.shape[1]} observations")
    
    # Extract colData as obs metadata
    try:
        with localconverter(ro.default_converter + pandas2ri.converter):
            obs_df = ro.r('as.data.frame(colData(se))')
        obs_df.index = obs_names
    except Exception as e:
        print(f"Warning: Could not extract colData: {e}")
        obs_df = pd.DataFrame(index=obs_names)
    
    # Create AnnData (transpose to obs × vars format)
    adata = ad.AnnData(
        X=matrix.T,
        obs=obs_df,
        var=pd.DataFrame(index=gene_names)
    )
    
    # Save if output path provided
    if output_path:
        adata.write_h5ad(output_path)
        print(f"Saved to: {output_path}")
    
    print(f"AnnData shape: {adata.shape[0]} obs × {adata.shape[1]} vars")
    
    return adata