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
from tqdm import tqdm

#Load Cohort
IND = os.path.basename(os.getcwd())

#Load
manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata = utils.load_cohort(IND)

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

#Now Let's Pull All Data
multi = pd.read_parquet('PDAC_FFPE.Multi.Results.20251104.parquet')

#Risk
risk = tid.utils.get_multi_risk_scores()

#Join
multi = pd.merge(multi, risk, on = "gene_name", how = "left")
multi = tid.run.target_quality_v2_01(multi)

multi["gene_name"].head(50)

# Genes with high confidence of NOT being on the cell surface
not_cell_surface = [
    # Original list
    "PLEKHN1",  # Cytoplasmic scaffolding protein
    "PTK6",     # Cytoplasmic tyrosine kinase (BRK)
    "MYRF",     # Transcription factor, intracellular
    "ERN2",     # Endoplasmic reticulum protein (IRE1-beta)
    "TGM2",     # Tissue transglutaminase, primarily cytoplasmic/intracellular
    "STX1A",    # SNARE protein, intracellular (synaptic vesicles)
    "SYT8",     # Synaptotagmin-8, synaptic vesicle protein
    "CKLF",     # Chemokine-like factor, secreted (not surface-anchored)
    "LGALS4",   # Galectin-4, cytoplasmic/secreted
    "CGN",      # Cingulum gene product, cytoplasmic
    "MELTF",    # Melanotransferrin, controversial surface localization
    # Additional from second batch
    "ASAP2",    # Cytoplasmic Arf-GAP protein
    "PROM2",    # Intracellular profilin-associated protein
    "PLPP2",    # Intracellular phospholipid phosphatase
    "TMEM184A", # Transmembrane but primarily intracellular trafficking
    "AHNAK2",   # Cytoplasmic scaffold protein
    "MLKL",     # Intracellular pseudokinase (necroptosis pathway)
    "CYP3A5",   # Cytochrome P450, endoplasmic reticulum
    "LAMA5",    # Extracellular matrix protein (not cell surface)
    "GABRE",    # Intracellular GABA receptor subunit
    "PKP3",     # Plakophilin, primarily cytoplasmic/junctional
    "SCEL",     # Sciellin, primarily cytoplasmic
    "EPSTI1",   # Epithelial stromal interaction 1, intracellular
    "SFXN3",    # Sideroflexin, mitochondrial protein
    "LIF"
]

pattern = '|'.join(not_cell_surface)

multi2 = multi[(multi["Positive_Final_v2"] > 40) & (multi2["TargetQ_Final_v2"] > 40)].copy()
multi2 = multi2[~multi2.gene_name.str.contains(pattern)].copy()

#tabs
tabs = tid.utils.tabs_genes()
tabs.append("CLDN18.2")
multi3 = pd.concat([multi2, multi_18p2])
multi3["known"] = multi3.gene_name.str.split("_").str[0].isin(tabs).astype(int) + multi3.gene_name.str.split("_").str[1].isin(tabs).astype(int)
mulit3 = multi3[multi3["known"] > 0].copy()
multi3 = mulit3.sort_values("TargetQ_Final_v2", ascending=False)
multi3 = multi3[(multi3["Positive_Final_v2"] > 50) & (multi3["TargetQ_Final_v2"] > 50)].copy()
multi3 = multi3[~multi3.gene_name.str.contains(pattern)].copy()


multi3[multi3.known>0][["gene_name", "Positive_Final_v2", "TargetQ_Final_v1", "TargetQ_Final_v2", "known"]].head(25)

multi[multi.gene_name.str.contains("CEACAM6_CLDN18")].head(50)






















multi_18p2.head(25)

multi.head(25)



single_18p2.iloc[0]

df_m = malig_med_adata.obs
df_m["CLDN18"]= malig_med_adata[:, "CLDN18"].X[:, 0].flatten().copy()
df_m.sort_values("CLDN18", ascending=False)


single.to_parquet(IND + '.Single.Results.20251027.parquet', engine='pyarrow', compression=None)

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
multi.to_parquet(IND + '.Multi.Results.20251027.parquet', engine='pyarrow', compression=None)

df = pd.read_csv(
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/GTEx_Analysis_v10_RSEMv1.3.3_transcripts_tpm.txt.gz",
    sep='\t',
    compression='gzip',
    nrows=10
)