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

df = pd.read_parquet('KIRC.Single.Results.20251029.parquet')
surface=tid.utils.surface_genes()
df = df[df["gene_name"].isin(surface)]
df = df[~df["gene_name"].str.contains("^UGT")]

df[(df["Positive_Final_v2"] > 90) & (df["TargetQ_Final_v1"] > 90)].shape[0]

p1_tq_vs_pp(df, out="KIRC_tq_vs_pp_labeled.20251104.pdf", label_top_interval=True, highlight_genes = ["CA9", "ENPP3"], target_q=90, ppos=90)
p1_tq_vs_pp(df, out="KIRC_tq_vs_pp_masked.20251104.pdf",  label_top_interval=False, highlight_genes = ["CA9", "ENPP3"], target_q=90, ppos=90)


multi = pd.read_parquet('KIRC.Multi.Results.20251029.parquet')
sub = multi[(multi["TargetQ_Final_v1"] > 40) & (multi["Positive_Final_v2"] > 40)]
sub2 = multi[(multi["TargetQ_Final_v1"] > 90) & (multi["Positive_Final_v2"] > 90)]

p1_tq_vs_pp(sub, 
	out="KIRC_tq_vs_pp_masked.multi.20251104.pdf",  label_top_interval=False, pdf_w=9, pdf_h=8, 
	highlight_genes = ["CA9", "ENPP3"], title_suffix = f"{sub2.shape[0]} multis", target_q=90, ppos=90)










sub = multi[(multi["TargetQ_Final_v1"] > 99) & (multi["Positive_Final_v2"] > 40)]

np.sum(sub["gene_name"].str.contains("ENPP3"))
np.sum(sub["gene_name"].str.contains("CA9"))
np.sum(sub["gene_name"].str.contains("CD70"))



# Or more concisely, convert all numeric to float16 except integers:
multi = multi.astype({col: 'int32' if multi[col].dtype.kind == 'i' else 'float16' 
                             for col in multi.select_dtypes(include=[np.number]).columns})

multi.memory_usage(deep=True).sum() / 1024**2