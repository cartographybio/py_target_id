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

