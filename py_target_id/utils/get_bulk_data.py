"""
Reference data loading functions.
"""

__all__ = ['get_gtex_adata', 'get_tcga_adata']

import os
import subprocess
import h5py
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from typing import Optional
from py_target_id import utils

def get_gtex_adata(
    version: str = "20251010",
    overwrite: bool = False,
    pretty_cols: bool = True
) -> ad.AnnData:
    """
    Load GTEx H5 matrix data as AnnData.
    
    Returns:
        AnnData: GTEx expression data with genes as vars and samples as obs
    """
    if version == "20251010":
        path = "gs://cartography_target_id_package/Other_Input/GTEX/gtex.bulk_rna.20251010.h5ad"
        local = "temp/Recount3/gtex.bulk_rna.20251010.h5ad"
        
        # Create directory if needed
        os.makedirs(os.path.dirname(local), exist_ok=True)
        
        # Copy from GCS if needed
        if not os.path.exists(local) or overwrite:
            subprocess.run(
                f"{utils.google_copy()} cp {path} {local}",
                shell=True,
                check=True
            )

    adata = sc.read_h5ad(local, backed='r')

    # Parse the obs_names to extract tissue info
    def parse_gtex_id(obs_name):
        parts = obs_name.split('#')
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        return obs_name

    # Create the mapping
    name_map = {
        "ADIPOSE_TISSUE.Adipose_Subcutaneous": "Adipose.Subcut",
        "ADIPOSE_TISSUE.Adipose_Visceral_Omentum": "Adipose.Visc",
        "MUSCLE.Muscle_Skeletal": "Muscle.Skeletal",
        "BLOOD_VESSEL.Artery_Tibial": "Vessel.Tibial",
        "BLOOD_VESSEL.Artery_Aorta": "Vessel.Aorta",
        "BLOOD_VESSEL.Artery_Coronary": "Vessel.Coronary",
        "HEART.Heart_Atrial_Appendage": "Heart.Atrium",
        "HEART.Heart_Left_Ventricle": "Heart.Ventr",
        "OVARY.Ovary": "Ovary",
        "UTERUS.Uterus": "Uterus",
        "VAGINA.Vagina": "Vagina",
        "BREAST.Breast_Mammary_Tissue": "Breast.Mammary",
        "SKIN.Skin_Sun_Exposed_Lower_leg": "Skin.Exposed",
        "SKIN.Skin_Not_Sun_Exposed_Suprapubic": "Skin.Unexposed",
        "SALIVARY_GLAND.Minor_Salivary_Gland": "Salivary.Minor",
        "BRAIN.Brain_Hippocampus": "Brain.Hippo",
        "BRAIN.Brain_Cortex": "Brain.Cortex",
        "BRAIN.Brain_Putamen_basal_ganglia": "Brain.Putamen",
        "BRAIN.Brain_Anterior_cingulate_cortex_BA24": "Brain.ACC_BA24",
        "BRAIN.Brain_Cerebellar_Hemisphere": "Brain.Cerebellar",
        "BRAIN.Brain_Frontal_Cortex_BA9": "Brain.Frontal_BA9",
        "BRAIN.Brain_Spinal_cord_cervical_c_1": "Brain.Spinal_C1",
        "BRAIN.Brain_Substantia_nigra": "Brain.SubNigra",
        "BRAIN.Brain_Nucleus_accumbens_basal_ganglia": "Brain.Basal_G_NAcc",
        "BRAIN.Brain_Hypothalamus": "Brain.Hypothal",
        "BRAIN.Brain_Cerebellum": "Brain.Cerebellum",
        "BRAIN.Brain_Caudate_basal_ganglia": "Brain.Basal_G_Caud",
        "BRAIN.Brain_Amygdala": "Brain.Amygdala",
        "ADRENAL_GLAND.Adrenal_Gland": "Adrenal",
        "THYROID.Thyroid": "Thyroid",
        "LUNG.Lung": "Lung",
        "SPLEEN.Spleen": "Spleen",
        "PANCREAS.Pancreas": "Pancreas",
        "ESOPHAGUS.Esophagus_Muscularis": "Esophagus.Muscle",
        "ESOPHAGUS.Esophagus_Mucosa": "Esophagus.Mucosa",
        "ESOPHAGUS.Esophagus_Gastroesophageal_Junction": "Esophagus.GE_Jxn",
        "STOMACH.Stomach": "Stomach",
        "COLON.Colon_Transverse": "Colon.Transverse",
        "COLON.Colon_Sigmoid": "Colon.Sigmoid",
        "SMALL_INTESTINE.Small_Intestine_Terminal_Ileum": "Small_Int.Ileum",
        "PROSTATE.Prostate": "Prostate",
        "TESTIS.Testis": "Testis",
        "NERVE.Nerve_Tibial": "Nerve.Tibial",
        "PITUITARY.Pituitary": "Pituitary",
        "BLOOD.Whole_Blood": "Blood",
        "LIVER.Liver": "Liver",
        "KIDNEY.Kidney_Cortex": "Kidney.Cortex",
        "KIDNEY.Kidney_Medulla": "Kidney.Medulla",
        "CERVIX_UTERI.Cervix_Endocervix": "Cervix.Endo",
        "CERVIX_UTERI.Cervix_Ectocervix": "Cervix.Ecto",
        "FALLOPIAN_TUBE.Fallopian_Tube": "Fallopian",
        "BLADDER.Bladder": "Bladder"
    }

    # Parse and map
    og_ids = [parse_gtex_id(obs_name) for obs_name in adata.obs_names]
    adata.obs['GTEX_Old'] = og_ids
    adata.obs['GTEX'] = [name_map.get(og_id, og_id) for og_id in og_ids]

    return adata


def get_tcga_adata(
    version: str = "20251010",
    overwrite: bool = False,
    pretty_cols: bool = True
) -> ad.AnnData:
    """
    Load TCGA H5 matrix data as AnnData.
    
    Returns:
        AnnData: TCGA expression data with genes as vars and samples as obs
    """
    if version == "20251010":
        path = "gs://cartography_target_id_package/Other_Input/TCGA/tcga.bulk_rna.20251010.h5ad"
        local = "temp/Recount3/tcga.bulk_rna.20251010.h5ad"
        
        # Create directory if needed
        os.makedirs(os.path.dirname(local), exist_ok=True)
        
        # Copy from GCS if needed
        if not os.path.exists(local) or overwrite:
            subprocess.run(
                f"{utils.google_copy()} cp {path} {local}",
                shell=True,
                check=True
            )

    adata = sc.read_h5ad(local, backed='r')
    adata.obs["TCGA"] = adata.obs_names.str.split('#').str[0].tolist()
    
    return adata

