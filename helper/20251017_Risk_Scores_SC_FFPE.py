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

results.to_parquet('SC_Multi_Risk_Scores.20251017.parquet', engine='pyarrow', compression=None)

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

results.to_parquet('FFPE_Multi_Risk_Scores.20251017.parquet', engine='pyarrow', compression=None)
