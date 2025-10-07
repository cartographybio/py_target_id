"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['se_rds_to_anndata']

# def se_rds_to_anndata(rds_path: str, assay_name: str = "median", output_path: str = None):
#     """
#     Convert R SummarizedExperiment RDS file to AnnData h5ad format
    
#     Parameters
#     ----------
#     rds_path : str
#         Path to the RDS file containing a SummarizedExperiment object
#     assay_name : str
#         Name of the assay to extract (default: "median")
#     output_path : str, optional
#         Path to save the h5ad file. If None, doesn't save to disk.
        
#     Returns
#     -------
#     anndata.AnnData
#         AnnData object with data transposed to cells × genes format
#     """
        
#     import rpy2.robjects as ro
#     import numpy as np
#     import pandas as pd
#     import anndata as ad
#     from rpy2.robjects.conversion import localconverter
#     from rpy2.robjects import pandas2ri

#     # Load required R library
#     ro.r('suppressPackageStartupMessages(library(SummarizedExperiment))')
    
#     # Read the RDS file
#     ro.r(f'se <- readRDS("{rds_path}")')
    
#     # Extract matrix (fast conversion with asarray)
#     print("Extracting matrix...")
#     if assay_name:
#         matrix = np.asarray(ro.r(f'assay(se, "{assay_name}")'))
#     else:
#         matrix = np.asarray(ro.r('assay(se)'))
    
#     gene_names = list(ro.r('rownames(se)'))
#     obs_names = list(ro.r('colnames(se)'))
    
#     print(f"Loaded: {matrix.shape[0]} genes × {matrix.shape[1]} observations")
    
#     # Extract colData as obs metadata
#     try:
#         with localconverter(ro.default_converter + pandas2ri.converter):
#             obs_df = ro.r('as.data.frame(colData(se))')
#         obs_df.index = obs_names
#     except Exception as e:
#         print(f"Warning: Could not extract colData: {e}")
#         obs_df = pd.DataFrame(index=obs_names)
    
#     # Create AnnData (transpose to obs × vars format)
#     adata = ad.AnnData(
#         X=matrix.T,
#         obs=obs_df,
#         var=pd.DataFrame(index=gene_names)
#     )
    
#     # Save if output path provided
#     if output_path:
#         adata.write_h5ad(output_path)
#         print(f"Saved to: {output_path}")
    
#     print(f"AnnData shape: {adata.shape[0]} obs × {adata.shape[1]} vars")
    
#     return adata


from pathlib import Path
import logging
import numpy as np
import pandas as pd
import anndata as ad

def se_rds_to_anndata(rds_path: str, default_assay: str = None, output_path: str = None) -> ad.AnnData:
    """
    Convert R SummarizedExperiment RDS file to AnnData, including all assays as layers.
    
    Parameters
    ----------
    rds_path : str
        Path to the RDS file containing a SummarizedExperiment object.
    default_assay : str, optional
        Assay to use for main X matrix. If None, uses the first assay.
    output_path : str, optional
        Path to save the h5ad file. If None, doesn't save.
    
    Returns
    -------
    anndata.AnnData
        AnnData object with assays as layers and data transposed to obs × vars format.
    """
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import pandas2ri

    rds_path = Path(rds_path).resolve()
    if output_path:
        output_path = Path(output_path).resolve()

    # Load R library
    ro.r('suppressPackageStartupMessages(library(SummarizedExperiment))')
    ro.r(f'se <- readRDS("{rds_path}")')

    # Get all assay names
    assay_names = list(ro.r('names(assays(se))'))
    if not assay_names:
        raise ValueError("No assays found in SummarizedExperiment.")

    if default_assay and default_assay not in assay_names:
        raise ValueError(f"Default assay '{default_assay}' not found. Available: {assay_names}")

    main_assay = default_assay or assay_names[0]

    # Extract main assay
    X = np.asarray(ro.r(f'assay(se, "{main_assay}")'))

    gene_names = list(ro.r('rownames(se)'))
    obs_names = list(ro.r('colnames(se)'))

    # Extract colData
    try:
        with localconverter(ro.default_converter + pandas2ri.converter):
            obs_df = ro.r('as.data.frame(colData(se))')
        obs_df.index = obs_names
    except Exception as e:
        logging.warning(f"Could not extract colData: {e}")
        obs_df = pd.DataFrame(index=obs_names)

    # Create AnnData
    adata = ad.AnnData(X=X.T, obs=obs_df, var=pd.DataFrame(index=gene_names))

    # Add all assays as layers
    for assay in assay_names:
        matrix = np.asarray(ro.r(f'assay(se, "{assay}")'))
        adata.layers[assay] = matrix.T

    if output_path:
        adata.write_h5ad(output_path)
        logging.info(f"Saved AnnData to: {output_path}")

    logging.info(f"AnnData shape: {adata.shape[0]} obs × {adata.shape[1]} vars, layers: {list(adata.layers.keys())}")
    return adata
