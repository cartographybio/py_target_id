import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
from importlib.resources import files
from py_target_id import utils
import glob
from py_target_id import utils, plot  # Relative import

#Load Cohort
IND = os.path.basename(os.getcwd())

#Load
manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata = utils.load_cohort(IND, nMalig = 0)

malig_med_adata.obs_names=malig_med_adata.obs["Patient"].values

sub_adata = malig_adata[:, ["CEACAM5", "HPN"]].to_memory()

m_sng = tid.utils.summarize_matrix(sub_adata.X, groups = sub_adata.obs["Patient"], metric = "median", axis = 0)
m_dbl = tid.utils.summarize_matrix(np.min(sub_adata.X, axis = 1), groups = sub_adata.obs["Patient"], metric = "median", axis = 0)
m_sng["CBP"] = m_sng.index
m_dbl["CBP"] = m_dbl.index
m_sng.columns = ["CEACAM5", "HPN", "CBP"]
m_dbl.columns = ["CEACAM5_HPN", "CBP"]

df = pd.read_csv(files('py_target_id').joinpath('data/cohorts/LUAD.Magellan.Stats.csv'))
df["CBP"] = df["ID"].str.replace(".250618", "")  # Clean sample IDs
df = pd.merge(df, m_sng, on = "CBP", how = "left")
df = pd.merge(df, m_dbl, on = "CBP", how = "left")
df.to_csv("LUAD.Magellan.csv")


tid.plot.pd2r("df", df)


r(f'''

df2 = df[df$Treatment %in% c("Post Tx", "Pre Tx"), ]

p1 <- ggplot(df, aes("ALL", log2(CEACAM5_HPN + 1))) +
    geom_jitter(pch=21,height=0, width=0.1, fill = "dodgerblue3") +
    theme_jg(xText90=TRUE)+
    geom_boxplot(fill = NA, outlier.shape = NA) +
    ylim(c(0,1.5))

p2 <- ggplot(df2, aes(Treatment, log2(CEACAM5_HPN + 1), fill = Treatment)) +
    geom_jitter(pch=21,height=0, width=0.1) +
    theme_jg(xText90=TRUE) +
    geom_boxplot(fill = NA, outlier.shape = NA) +
    ylim(c(0,1.5))

pdf("test.pdf", width = 6, height = 6)
print(p1 + p2)
dev.off()
''')

 
tcga = tid.utils.get_tcga_adata()
tcga = tcga[tcga.obs["TCGA"]=="LUAD", ["CEACAM5", "HPN"]].to_memory()
tcga.obs








































