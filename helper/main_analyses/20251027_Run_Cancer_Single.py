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
manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata = utils.load_cohort(IND)

#Run Single Target Workflow
single = tid.run.target_id_v1(malig_med_adata, ref_med_adata)
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


