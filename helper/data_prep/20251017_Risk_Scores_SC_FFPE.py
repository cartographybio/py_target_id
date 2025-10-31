#########################################################################################################
#########################################################################################################
import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
from importlib.resources import files
from py_target_id import utils

#1. Single Cell Weighted-Risk Scoring

#Ref
ref_sc = tid.utils.get_ref_lv4_sc_ar_adata()

#Combos
surface = tid.utils.surface_genes()

#Pairs
gene_pairs = tid.utils.create_gene_pairs(surface, surface)

#Compute On
results = tid.run.compute_ref_risk_scores(
    ref_adata=ref_sc,
    type="SC",
    gene_pairs=gene_pairs,
    device='cuda',
    batch_size=5000
)

results.to_parquet('SC_Multi_Risk_Scores.20251018.parquet', engine='pyarrow', compression=None)

#Single
all_results = []
gene_pairs = [(gene, gene) for gene in tid.utils.valid_genes()]
for i in range(0, len(gene_pairs), 5000):
    all_results.append(tid.run.compute_ref_risk_scores(ref_adata=ref_sc, type="SC", gene_pairs=gene_pairs[i:i+5000], device='cuda', batch_size=5000))
risk_single_sc = pd.concat(all_results, ignore_index=True)
risk_single_sc.loc[:,"gene_name"] = risk_single_sc['pair_name'].str.split('_', expand=True)[0]
risk_single_sc = risk_single_sc[['gene_name', 'hazard_weighted_risk']].copy()
risk_single_sc.to_parquet('SC_Single_Risk_Scores.20251017.parquet', engine='pyarrow', compression=None)

#########################################################################################################
#########################################################################################################
import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
from importlib.resources import files
from py_target_id import utils

#2. FFPE Weighted-Risk Scoring

#Ref
ref_ff = tid.utils.get_ref_lv4_ffpe_ar_adata()

#Combos
surface = tid.utils.surface_genes()

#Pairs
gene_pairs = tid.utils.create_gene_pairs(surface, surface)

#Compute On
results = tid.run.compute_ref_risk_scores(
    ref_adata=ref_ff,
    type="FFPE",
    gene_pairs=gene_pairs,
    device='cuda',
    batch_size=5000
)

results.to_parquet('FFPE_Multi_Risk_Scores.20251018.parquet', engine='pyarrow', compression=None)

#Single
all_results = []
gene_pairs = [(gene, gene) for gene in tid.utils.valid_genes()]
for i in range(0, len(gene_pairs), 5000):
    all_results.append(tid.run.compute_ref_risk_scores(ref_adata=ref_ff, type="FFPE", gene_pairs=gene_pairs[i:i+5000], device='cuda', batch_size=5000))
risk_ffpe_sc = pd.concat(all_results, ignore_index=True)
risk_ffpe_sc.loc[:,"gene_name"] = risk_ffpe_sc['pair_name'].str.split('_', expand=True)[0]
risk_ffpe_sc = risk_ffpe_sc[['gene_name', 'hazard_weighted_risk']].copy()
risk_ffpe_sc.to_parquet('FFPE_Single_Risk_Scores.20251017.parquet', engine='pyarrow', compression=None)

#########################################################################################################
#########################################################################################################

#Save
sc_haz = pd.read_parquet("SC_Single_Risk_Scores.20251017.parquet")
ff_haz = pd.read_parquet("FFPE_Single_Risk_Scores.20251017.parquet")
sc_haz.columns = ["gene_name", "Hazard_SC_v1"]
ff_haz.columns = ["gene_name", "Hazard_FFPE_v1"]

sc_m_haz = pd.read_parquet("SC_Multi_Risk_Scores.20251018.parquet")
ff_m_haz = pd.read_parquet("FFPE_Multi_Risk_Scores.20251018.parquet")
sc_m_haz.columns = ["gene_name", "Hazard_SC_v1"]
ff_m_haz.columns = ["gene_name", "Hazard_FFPE_v1"]

#Join
haz_sgl = pd.merge(sc_haz, ff_haz, on = "gene_name", how = "left")
haz_dbl = pd.merge(sc_m_haz, ff_m_haz, on = "gene_name", how = "left")

haz_sgl = haz_sgl.sort_values("gene_name").reset_index(drop=True)
haz_dbl = haz_dbl.sort_values("gene_name").reset_index(drop=True)

haz_sgl.to_parquet("Single_Risk_Scores.20251017.parquet", compression=None)
haz_dbl.to_parquet("Multi_Risk_Scores.20251018.parquet", compression=None)

cp Single_Risk_Scores.20251017.parquet gs://cartography_target_id_package/Other_Input/Risk
cp Multi_Risk_Scores.20251018.parquet gs://cartography_target_id_package/Other_Input/Risk

import subprocess
subprocess.run("gcloud storage cp Single_Risk_Scores.20251017.parquet gs://cartography_target_id_package/Other_Input/Risk/", shell=True) #Update Quick
subprocess.run("gcloud storage cp Multi_Risk_Scores.20251018.parquet gs://cartography_target_id_package/Other_Input/Risk/", shell=True) #Update Quick


#########################################################################################################
#########################################################################################################

import subprocess; import os; subprocess.run(["bash", os.path.expanduser("~/update_tid.sh")]) #Update Quick
import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
from importlib.resources import files
from py_target_id import utils

#Get
gtex = tid.utils.get_gtex_adata().to_memory()

#Single
gtex_single = tid.run.compute_gtex_risk_scores_single(gtex)

#Check
gene_pairs = [(name, name) for name in gtex_single['gene_name']]
gtex_multi_check = tid.run.compute_gtex_risk_scores_multi(gtex, gene_pairs)

# Compare
pair_scores = gtex_multi_check["Hazard_GTEX_v1"].values
single_scores = gtex_single["Hazard_GTEX_v1"].values

diff = pair_scores - single_scores
print(f"Pearson r: {np.corrcoef(pair_scores, single_scores)[0, 1]:.6f}")
print(f"MAE: {np.mean(np.abs(diff)):.6f}")
print(f"RMSE: {np.sqrt(np.mean(diff ** 2)):.6f}")
print(f"Max diff: {np.abs(diff).max():.6f}")

#Multi
risk = tid.utils.get_multi_risk_scores()
gene_pairs = [tuple(gene.split("_")) for gene in risk["gene_name"].tolist()]
gtex_multi = tid.run.compute_gtex_risk_scores_multi(gtex, gene_pairs)

#Let's Read In Current Ones
risk_sngl = pd.read_parquet("Single_Risk_Scores.20251017.parquet")
risk_sngl = pd.merge(risk_sngl, gtex_single, on ="gene_name", how = "left")

risk_dbl = pd.read_parquet("Multi_Risk_Scores.20251018.parquet")
risk_dbl = pd.merge(risk_dbl, gtex_multi, on ="gene_name", how = "left")

risk_sngl.to_parquet("Single_Risk_Scores.20251030.parquet", compression=None)
risk_dbl.to_parquet("Multi_Risk_Scores.20251030.parquet", compression=None)

import subprocess
subprocess.run("gcloud storage cp Single_Risk_Scores.20251030.parquet gs://cartography_target_id_package/Other_Input/Risk/", shell=True) #Update Quick
subprocess.run("gcloud storage cp Multi_Risk_Scores.20251030.parquet gs://cartography_target_id_package/Other_Input/Risk/", shell=True) #Update Quick

