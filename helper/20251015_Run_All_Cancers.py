import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
from importlib.resources import files
from py_target_id import utils

IND = os.path.basename(os.getcwd())

#Manifest
manifest = tid.utils.load_manifest()

#Get Manifest
if IND  == "TNBC.Magellan":
	df = pd.read_csv(files('py_target_id').joinpath(f'data/cohorts/TNBC.FFPE.Magellan.csv'))
	manifest = manifest[manifest["Indication"] == "TNBC"]
	manifest = manifest[manifest["Sample_ID"].str.contains("FFPE", na=False)]
	manifest = manifest[manifest["Sample_ID"].isin("Breast_" + df["CBP"].astype(str) + "_FFPE")]
	manifest = manifest.reset_index(drop=True)
	ref = "FFPE"

elif IND == "LUAD.Magellan":
	df = pd.read_csv(files('py_target_id').joinpath(f'data/cohorts/LUAD.Magellan.Stats.csv'))
	df["CBP"] = df["ID"].str.replace(".250618", "")
	manifest = manifest[manifest["Indication"] == "LUAD"]
	manifest = manifest[manifest["Sample_ID"].isin(df["CBP"])]
	manifest = manifest.reset_index(drop=True)
	ref = "SC"

elif IND == "CRC":
	df = pd.read_csv(files('py_target_id').joinpath(f'data/cohorts/CRC_MSS_MSI_Status.csv'))
	manifest = manifest[manifest["Indication"] == "COAD"]
	manifest = manifest[manifest["Sample_ID"].isin(df["id"])]
	manifest = manifest.reset_index(drop=True)
	ref = "SC"

elif IND == "KIRC":
	manifest = manifest[manifest["Indication"] == "KIRC"]
	manifest = manifest[manifest["Sample_ID"] != "Kidney_TC_DTC_0121"]
	manifest = manifest.reset_index(drop=True)
	ref = "SC"

elif IND == "AML":
	manifest = manifest[manifest["Indication"] == "AML"]
	manifest = manifest.reset_index(drop=True)
	ref = "SC"

elif IND == "ESCA":
	manifest = manifest[manifest["Indication"] == "ESCA"]
	manifest = manifest.reset_index(drop=True)
	ref = "SC"

elif IND == "OVCA":
	manifest = manifest[manifest["Indication"] == "OVCA"]
	manifest = manifest[manifest["Sample_ID"] != "C005_Aud_Ovary_CBP2838_Tumor_GEX_CB02"]
	manifest = manifest[~manifest["Sample_ID"].str.contains("ovary1|ovary2|r1|r2", na=False)]
	manifest = manifest.reset_index(drop=True)
	ref = "SC"

elif IND == "PDAC_FFPE":
	manifest = manifest[manifest["Indication"] == "PDAC_FFPE"]
	manifest = manifest.reset_index(drop=True)
	ref = "FFPE"


#Download Manifest
manifest = tid.utils.download_manifest(manifest=manifest, overwrite = False)

#Malig Data
malig_med_adata = tid.utils.get_malig_med_adata(manifest)
malig_adata = tid.utils.get_malig_ar_adata(manifest)

#Reference Data
if ref == "FFPE":
	df_off = tid.utils.get_ref_ffpe_off_target()
	ref_med_adata = tid.utils.get_ref_lv4_ffpe_med_adata()
	ref_adata = tid.utils.get_ref_lv4_ffpe_ar_adata()
elif ref == "SC":
	df_off = tid.utils.get_ref_sc_off_target()
	ref_med_adata = tid.utils.get_ref_lv4_sc_med_adata()
	ref_adata = tid.utils.get_ref_lv4_sc_ar_adata()

#Add Weights to Ref Data
ref_med_adata = tid.utils.add_ref_weights(ref_med_adata, df_off, "Off_Target.V0")
ref_adata = tid.utils.add_ref_weights(ref_adata, df_off, "Off_Target.V0")

#Run Single Target Workflow
single = tid.run.target_id_v1(malig_med_adata, ref_med_adata)

single.to_parquet(IND + '.Single.Results.20251015.parquet', engine='pyarrow', compression=None)

#Ready Up Multi Target Workflow
surface = tid.utils.surface_genes()

#Pairs
gene_pairs = tid.utils.create_gene_pairs(surface, surface)

#Run Multi Target Workflow
multi = tid.run.target_id_multi_v1_safe(
    malig_adata=malig_adata,
    ref_adata=ref_adata,
    gene_pairs=gene_pairs,
    malig_med_adata=malig_med_adata,
    batch_size=20000,
    use_fp16=True
)

multi.to_parquet(IND + '.Multi.Results.20251015.parquet', engine='pyarrow', compression=None)
















































ad = ref_adata[:, surface].to_memory()








#Malig Data
malig_med_adata = tid.utils.get_malig_med_adata(manifest)
malig_adata = tid.utils.get_malig_ar_adata(manifest)

#Reference Data
df_off = tid.utils.get_ref_ffpe_off_target()
ref_med_adata = tid.utils.get_ref_lv4_ffpe_med_adata()
ref_adata = tid.utils.get_ref_lv4_ffpe_ar_adata()

#Add Weights to Ref Data
ref_med_adata = tid.utils.add_ref_weights(ref_med_adata, df_off, "Off_Target.V0")
ref_adata = tid.utils.add_ref_weights(ref_adata, df_off, "Off_Target.V0")

# #Run Single Target Workflow
#single = tid.run.target_id_v1(malig_med_adata, ref_med_adata)

# #Ready Up Multi Target Workflow
# surface = tid.utils.surface_genes()

# #Pairs
# gene_pairs = tid.utils.create_gene_pairs(surface, surface)

# #Run Multi Target Workflow
# multi = tid.run.target_id_multi_v1(
#     malig_adata=malig_adata,
#     ha_adata=ref_adata,
#     gene_pairs=gene_pairs,
#     malig_med_adata=malig_med_adata,
#     batch_size=50000,
#     use_fp16=True
# )

# #Save
# single.to_parquet('20251010.PDAC.Single.Results.parquet', engine='pyarrow', compression=None)
# multi.to_parquet('20251010.PDAC.Multi.Results.parquet', engine='pyarrow', compression=None)
