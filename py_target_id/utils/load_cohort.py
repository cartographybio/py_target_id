"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['load_cohort']

def load_cohort(IND=None):
    """
    Load tumor cohort data and reference datasets for target identification analysis.
    
    Args:
        IND: Indication/cancer type to load. Options:
            - "TNBC.Magellan": Triple-negative breast cancer (FFPE samples)
            - "LUAD.Magellan": Lung adenocarcinoma (single-cell)
            - "CRC": Colorectal cancer (single-cell)
            - "KIRC": Kidney renal clear cell carcinoma (single-cell)
            - "AML": Acute myeloid leukemia (single-cell)
            - "ESCA": Esophageal carcinoma (single-cell)
            - "OVCA": Ovarian cancer (single-cell)
            - "PDAC_FFPE": Pancreatic adenocarcinoma (FFPE samples)
    
    Returns:
        tuple: (manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata)
            - manifest: Sample metadata and file paths
            - malig_adata: Malignant cell activity data (raw counts/log-normalized)
            - malig_med_adata: Malignant cell median expression
            - ref_adata: Reference tissue activity data (FFPE or single-cell)
            - ref_med_adata: Reference tissue median expression
    """
    
    import pandas as pd
    from importlib.resources import files
    from py_target_id import utils
    
    # ===== LOAD BASE MANIFEST =====
    # Manifest contains sample metadata, file paths, and quality info
    manifest = utils.load_manifest()
    
    # ===== FILTER MANIFEST BY INDICATION & DATA TYPE =====
    # Different indications may have FFPE vs single-cell samples, different formats
    
    if IND == "TNBC.Magellan":
        # Triple-negative breast cancer with FFPE (formalin-fixed paraffin-embedded) samples
        df = pd.read_csv(files('py_target_id').joinpath('data/cohorts/TNBC.FFPE.Magellan.csv'))
        manifest = manifest[manifest["Indication"] == "TNBC"]
        manifest = manifest[manifest["Sample_ID"].str.contains("FFPE", na=False)]
        manifest = manifest[manifest["Sample_ID"].isin("Breast_" + df["CBP"].astype(str) + "_FFPE")]
        manifest = manifest.reset_index(drop=True)
        ref = "FFPE"
        
    elif IND == "LUAD.Magellan":
        # Lung adenocarcinoma with single-cell (SC) samples
        df = pd.read_csv(files('py_target_id').joinpath('data/cohorts/LUAD.Magellan.Stats.csv'))
        df["CBP"] = df["ID"].str.replace(".250618", "")  # Clean sample IDs
        manifest = manifest[manifest["Indication"] == "LUAD"]
        manifest = manifest[manifest["Sample_ID"].isin(df["CBP"])]
        manifest = manifest.reset_index(drop=True)
        ref = "SC"
        
    elif IND == "CRC":
        # Colorectal cancer (single-cell, includes MSS/MSI status)
        df = pd.read_csv(files('py_target_id').joinpath('data/cohorts/CRC_MSS_MSI_Status.csv'))
        manifest = manifest[manifest["Indication"] == "COAD"]
        manifest = manifest[manifest["Sample_ID"].isin(df["id"])]
        manifest = manifest.reset_index(drop=True)
        ref = "SC"
        
    elif IND == "KIRC":
        # Kidney renal clear cell carcinoma (single-cell)
        # Exclude problematic sample
        manifest = manifest[manifest["Indication"] == "KIRC"]
        manifest = manifest[manifest["Sample_ID"] != "Kidney_TC_DTC_0121"]
        manifest = manifest.reset_index(drop=True)
        ref = "SC"
        
    elif IND == "AML":
        # Acute myeloid leukemia (single-cell)
        manifest = manifest[manifest["Indication"] == "AML"]
        manifest = manifest.reset_index(drop=True)
        ref = "SC"
        
    elif IND == "ESCA":
        # Esophageal carcinoma (single-cell)
        manifest = manifest[manifest["Indication"] == "ESCA"]
        manifest = manifest.reset_index(drop=True)
        ref = "SC"
        
    elif IND == "OVCA":
        # Ovarian cancer (single-cell)
        # Exclude replicates and low-quality samples
        manifest = manifest[manifest["Indication"] == "OVCA"]
        manifest = manifest[manifest["Sample_ID"] != "C005_Aud_Ovary_CBP2838_Tumor_GEX_CB02"]
        manifest = manifest[~manifest["Sample_ID"].str.contains("ovary1|ovary2|r1|r2", na=False)]
        manifest = manifest.reset_index(drop=True)
        ref = "SC"
        
    elif IND == "PDAC_FFPE":
        # Pancreatic adenocarcinoma (FFPE samples)
        manifest = manifest[manifest["Indication"] == "PDAC_FFPE"]
        manifest = manifest.reset_index(drop=True)
        ref = "FFPE"
        
    else:
        raise ValueError("IND must be a valid indication (TNBC.Magellan, LUAD.Magellan, CRC, KIRC, AML, ESCA, OVCA, PDAC_FFPE)")
    
    # ===== DOWNLOAD/VERIFY MANIFEST FILES =====
    # Download missing files from cloud storage, verify checksums
    manifest = utils.download_manifest(manifest=manifest, overwrite=False)
    
    # ===== LOAD MALIGNANT CELL DATA =====
    # Extract malignant cells from tumor samples
    # med_adata: median expression per cell type per sample
    # ar_adata: activity/raw counts
    malig_med_adata = utils.get_malig_med_adata(manifest)
    malig_adata = utils.get_malig_ar_adata(manifest)
    
    # ===== LOAD REFERENCE DATA =====
    # Reference datasets for off-target specificity scoring
    # Use either FFPE reference atlas or single-cell reference atlas
    if ref == "FFPE":
        df_off = utils.get_ref_ffpe_off_target()
        ref_med_adata = utils.get_ref_lv4_ffpe_med_adata()
        ref_adata = utils.get_ref_lv4_ffpe_ar_adata()
    elif ref == "SC":
        df_off = utils.get_ref_sc_off_target()
        ref_med_adata = utils.get_ref_lv4_sc_med_adata()
        ref_adata = utils.get_ref_lv4_sc_ar_adata()
    
    # ===== ADD OFF-TARGET WEIGHTS TO REFERENCE DATA =====
    # Weight tissues by criticality (e.g., heart tissue more critical than skin)
    # "Off_Target.V0" = off-target penalty scoring version
    ref_med_adata = utils.add_ref_weights(ref_med_adata, df_off, "Off_Target.V0")
    ref_adata = utils.add_ref_weights(ref_adata, df_off, "Off_Target.V0")
    
    return manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata
