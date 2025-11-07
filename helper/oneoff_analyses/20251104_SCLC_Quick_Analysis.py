import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import sys
from importlib.resources import files
from py_target_id import utils
from scipy.stats import pearsonr

adata = sc.read_h5ad('5b7cbce2-ed53-47a4-99f8-10af741814e6.h5ad')
mask = adata.obs["cell_type_fine"].str.contains("^SCLC-A")
sclc = adata[mask, :].copy()

#CP10k
sclc.X = tid.utils.cp10k(sclc.X)

#Summarize Expression
sclc.var_names = sclc.var["feature_name"].values.astype(str)

#Summarize
malig_mean_adata = ad.AnnData(
    X=tid.utils.summarize_matrix(sclc.X, groups = sclc.obs["donor_id"].values, metric="mean", axis=0)
)
malig_mean_adata.var_names = sclc.var["feature_name"].values.astype(str)
malig_mean_adata.var_names_make_unique()

#Get Ref
ref_med_adata = tid.utils.get_ref_lv4_sc_med_adata()

#Join
malig_mean_adata = malig_mean_adata[:, malig_mean_adata.var_names.isin(ref_med_adata.var_names)]
malig_mean_adata = malig_mean_adata[malig_mean_adata.obs_names != "PleuralEffusion", :].copy()
ref_med_adata = ref_med_adata[:, malig_mean_adata.var_names]

#Positivity (CP10k > 0.5)
malig_mean_adata=tid.run.compute_positivity_matrix(malig_mean_adata, fallback_threshold = 0.5)

#Run
df = tid.run.target_id_v1(malig_adata=malig_mean_adata, ref_adata=ref_med_adata)

#Find
surface = tid.utils.surface_genes()
df2=df.loc[df["gene_name"].isin(surface),:]
df2[df2["Positive_Final_v2"] > 25].head(50)
df2[df2.gene_name=="DLL3"].iloc[0]






















df[df.gene_name=="DLL3"].iloc[0]
df[df["Positive_Final_v2"] > 50].head(50)

# Get DLL3 expression

dll3 = sclc[:, "ENSG00000090932"].X.toarray().flatten()
donor = sclc.obs["donor_id"].values
result = {d: np.mean(dll3[donor == d]) for d in np.unique(donor)}





# Group by donor and get mean
split(dll3, donor) %>% lapply(median) %>% unlist


result = {d: np.mean(dll3[donor == d]) for d in np.unique(donor)}




cp10k = sclc.raw.X
row_sums = np.array(cp10k.sum(axis=1)).flatten()
cp10k_normalized = cp10k.multiply(1 / row_sums.reshape(-1, 1))

def cp10k(m):
    """Normalize matrix rows to sum to 1 (CPM per 10k normalization)
    
    Preserves input matrix format (sparse or dense).
    """
    row_sums = np.array(m.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    
    scale = 10000 / row_sums.reshape(-1, 1)
    
    if hasattr(m, 'multiply'):  # Sparse matrix
        result = m.multiply(scale)
        # Preserve original format
        if hasattr(m, 'format'):
            result = result.asformat(m.format)
    else:  # Dense array
        result = m * scale
    
    return result



pearsonr(np.sum(cp10k, axis=1).flatten(), 10**sclc.obs["libsize"].values)

correlation, pvalue = pearsonr(
    np.array(np.sum(cp10k, axis=1)).flatten(),
    10**sclc.obs["libsize"].values
)

cp10k = sclc.X.copy()
cp10k.data = np.exp(cp10k.data)-1
sclc.X = cp10k

np.sum(np.exp(sclc.X[0,:].toarray().flatten())-1)


libsize_raw = 10 ** sclc.obs["libsize"]
raw_counts = np.expm1(sclc.X)
np.sum(2**sclc.X[0,:].toarray().flatten()-1)
np.sum(np.exp(sclc.X[1,:].toarray().flatten()-1))

dll3 = (sclc[:,"ENSG00000090932"].X.toarray())
sclc.obs["donor_id"]

# Get DLL3 expression
dll3 = sclc[:, "ENSG00000090932"].X.toarray().flatten()
np.sum(sclc.X, axis = 1)

# Add to obs
sclc.obs["DLL3"] = dll3

# Group by donor and get mean
dll3_per_donor = sclc.obs.groupby("donor_id")["DLL3"].mean()


Gene: DLL3 (ENSG00000090932) - Summary
