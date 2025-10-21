import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
from importlib.resources import files
from py_target_id import utils

#https://cdn.amegroups.cn/static/public/atm-20-7005-1.pdf

#Load Cohort
IND = os.path.basename(os.getcwd())

#Load
manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata = utils.load_cohort(IND)

#Run Single Target Workflow
single = tid.run.target_id_v1(malig_med_adata, ref_med_adata)

#TCGA
tcga = tid.utils.get_tcga_adata()[:, ("LY6K", "PCDHB2", "NECTIN4", "PODXL2")].to_memory()
brca = tcga[tcga.obs["TCGA"]=="BRCA", :]

#We are going to join
df_brca = pd.read_csv("TCGA_BRCA_Subtypes_Claude.csv") #added to cohorts in package
df_brca["id"] = df_brca["ID"].str.replace("-01", "")
brca = brca[brca.obs_names.str.split("#").str[1].isin(df_brca["id"]),:]
brca.obs["id"] = brca.obs_names.str.split("#").str[1].values
brca_df = pd.merge(brca.obs, df_brca, on = "id", how = "left")
m_df = pd.DataFrame(brca.X.toarray())
m_df.columns = brca.var_names
m_df["NECTIN4_PODXL2"] = np.minimum(m_df["NECTIN4"], m_df["PODXL2"])
m_df = m_df.loc[:, ["LY6K", "PCDHB2", "NECTIN4_PODXL2"]]
df_all = pd.concat([brca_df, m_df], axis=1)
df_all["Subclass"] = df_all["Subclass"].astype(str)


tid.plot.pd2r("df", df_all.infer_objects())

r_code = f'''

pdf("test.pdf", width = 5, height =5)

p1 <- ggplot(df, aes(Subclass, log2(LY6K + 1), fill = Subclass)) +
	geom_jitter(height = 0, pch = 21) +
	geom_boxplot(outlier.shape = NA, fill = NA) 

print(p1)

p1 <- ggplot(df, aes(Subclass, log2(PCDHB2 + 1), fill = Subclass)) +
	geom_jitter(height = 0, pch = 21) +
	geom_boxplot(outlier.shape = NA, fill = NA) 

print(p1)


p1 <- ggplot(df, aes(Subclass, log2(NECTIN4_PODXL2 + 1), fill = Subclass)) +
	geom_jitter(height = 0, pch = 21) +
	geom_boxplot(outlier.shape = NA, fill = NA) 

print(p1)
dev.off()


'''

r(r_code)