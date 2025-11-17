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

#Load Cohort
IND = os.path.basename(os.getcwd())

#Load
manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata = utils.load_cohort(IND, nMalig = 100)

#Run Single Target Workflow
single = tid.run.target_id_v1(malig_med_adata, ref_med_adata)

#Ready Up Multi Target Workflow
surface = tid.utils.surface_genes()

#Pairs
gene_pairs = tid.utils.create_gene_pairs(surface, surface)

#Run Multi Target Workflow
multi = tid.run.target_id_multi_v1(
    malig_adata=malig_adata,
    ref_adata=ref_adata,
    gene_pairs=gene_pairs,
    ref_med_adata=ref_med_adata,
    malig_med_adata=malig_med_adata,
    batch_size=20000,
    use_fp16=True
)

#Get Risk
risk_sngl = tid.utils.get_single_risk_scores()
risk_multi = tid.utils.get_multi_risk_scores()

#Add
single = pd.merge(single, risk_sngl, how = "left", on = "gene_name")
multi = pd.merge(multi, risk_multi, how = "left", on = "gene_name")

single = tid.run.target_quality_v2_01(single)
multi = tid.run.target_quality_v2_01(multi)

single.to_parquet(IND + '.Single.Results.20251110.parquet', engine='pyarrow', compression=None)
multi.to_parquet(IND + '.Multi.Results.20251110.parquet', engine='pyarrow', compression=None)

#CLDN18.2
import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
from importlib.resources import files
from py_target_id import utils
from tqdm import tqdm

#Load Cohort
IND = os.path.basename(os.getcwd())

#Load
manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata = utils.load_cohort(IND, nMalig = 100)

#CLDN18
ref_med_18p2 = ref_med_adata[:,"CLDN18"].copy()
ref_med_18p2.obs.loc[ref_med_18p2.obs.Tissue=="Lung", "Weights"] = 0

ref_adata_18p2 = ref_adata.copy()
ref_adata_18p2.obs.loc[ref_adata_18p2.obs.Tissue=="Lung", "Weights"] = 0

#Run All Multis With 18.2
surface = tid.utils.surface_genes()
gene_pairs = tid.utils.create_gene_pairs(["CLDN18"], surface)

#Re-weight

#Run Single Target Workflo

#Run Multi Target Workflow
multi_18p2 = tid.run.target_id_multi_v1(
    malig_adata=malig_adata,
    ref_adata=ref_adata_18p2,
    gene_pairs=gene_pairs,
    ref_med_adata=ref_med_18p2,
    malig_med_adata=malig_med_adata,
    batch_size=20000,
    use_fp16=True
)

#RISK
ref_sc_18p2 = tid.utils.get_ref_lv4_sc_ar_adata()
obs = ref_sc_18p2.obs
ref_sc_18p2 = ref_sc_18p2[:, surface].to_memory()
ref_sc_18p2.obs = obs

ref_ff_18p2 = tid.utils.get_ref_lv4_ffpe_ar_adata()
obs = ref_ff_18p2.obs
ref_ff_18p2 = ref_ff_18p2[:, surface].to_memory()
ref_ff_18p2.obs = obs

#Set 0 Lung
lung_mask = ref_sc_18p2.obs["Tissue"] == "Lung"
ref_sc_18p2.X[lung_mask, ref_sc_18p2.var_names.get_loc("CLDN18")] = 0

lung_mask = ref_ff_18p2.obs["Tissue"] == "Lung"
ref_ff_18p2.X[lung_mask, ref_ff_18p2.var_names.get_loc("CLDN18")] = 0

#Compute Risk
haz_sc = tid.run.compute_ref_risk_scores(
    ref_adata=ref_sc_18p2,
    type="SC",
    gene_pairs=gene_pairs,
    device='cuda',
    batch_size=5000
)

haz_ff = tid.run.compute_ref_risk_scores(
    ref_adata=ref_ff_18p2,
    type="FFPE",
    gene_pairs=gene_pairs,
    device='cuda',
    batch_size=5000
)

gtex = tid.utils.get_gtex_adata().to_memory()
gtex = gtex[:,surface].copy()
gtex_mask = gtex.obs.GTEX.str.contains("lung",case=False)
gtex.X[gtex_mask, gtex.var_names.get_loc("CLDN18")] = 0

haz_gtex = tid.run.compute_gtex_risk_scores_multi(gtex, gene_pairs)

haz_sc.columns = ["gene_name", "Hazard_SC_v1"]
haz_ff.columns = ["gene_name", "Hazard_FFPE_v1"]

#Pull Data
multi_18p2 = pd.merge(multi_18p2, haz_sc, on = 'gene_name', how = 'left')
multi_18p2 = pd.merge(multi_18p2, haz_ff, on = 'gene_name', how = 'left')
multi_18p2 = pd.merge(multi_18p2, haz_gtex, on = 'gene_name', how = 'left')

#Rank
multi_18p2 = tid.run.target_quality_v2_01(multi_18p2)
multi_18p2["gene_name"] = multi_18p2["gene_name"].str.replace("CLDN18", "CLDN18.2")

single_18p2 = multi_18p2[multi_18p2["gene_name"]=="CLDN18.2_CLDN18.2"].copy()
single_18p2["gene_name"]="CLDN18.2"

single_18p2.to_parquet(IND + '.Single.CLDN18p2.Results.20251110.parquet', engine='pyarrow', compression=None)
multi_18p2.to_parquet(IND + '.Multi.CLDN18p2.Results.20251110.parquet', engine='pyarrow', compression=None)
