pip uninstall py_target_id -y

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

#TCGA
tcga = tid.utils.get_tcga_adata()[:, ("LY6K", "PCDHB2", "NECTIN4", "PODXL2")].to_memory()
brca = tcga[tcga.obs["TCGA"]=="BRCA", :]

#We are going to join
df_brca_claude = pd.read_csv("TNBC_claude.csv")
df_brca_claude["id"] = df_brca_claude["ID"].str.replace("-01", "")
bnbrca.obs_names.str.split("#").str[1].isin(df_brca_claude["id"])

df_brca = pd.read_csv("BRCA.datafreeze.20120227.txt", sep = "\t")
brca = brca[brca.obs_names.str.split("#").str[1].isin(df_brca["bcr_patient_barcode"]),:]
brca.obs["bcr_patient_barcode"] = brca.obs_names.str.split("#").str[1].values
brca_df = pd.merge(brca.obs, df_brca, on = "bcr_patient_barcode", how = "left")
m_df = pd.DataFrame(brca.X.toarray())
m_df.columns = brca.var_names
m_df["NECTIN4_PODXL2"] = np.minimum(m_df["NECTIN4"], m_df["PODXL2"])
m_df = m_df.loc[:, ["LY6K", "PCDHB2", "NECTIN4_PODXL2"]]
df_all = pd.concat([brca_df, m_df], axis=1)
df_all["TN"]=df_all["Triple Negative"].astype(str)

brca_df[brca_df["PAM50"]=="Normal"]

tid.plot.pd2r("df", df_all.infer_objects())

r_code = f'''

pdf("test.pdf", width = 5, height =5)

p1 <- ggplot(df, aes(TN, log2(LY6K + 1), fill = TN)) +
	geom_jitter(height = 0, pch = 21) +
	geom_boxplot(outlier.shape = NA, fill = NA) 

print(p1)

p1 <- ggplot(df, aes(PAM50, log2(PCDHB2 + 1), fill = PAM50)) +
	geom_jitter(height = 0, pch = 21) +
	geom_boxplot(outlier.shape = NA, fill = NA) 

print(p1)


p1 <- ggplot(df, aes(PAM50, log2(NECTIN4_PODXL2 + 1), fill = PAM50)) +
	geom_jitter(height = 0, pch = 21) +
	geom_boxplot(outlier.shape = NA, fill = NA) 

print(p1)
dev.off()


'''

r(r_code)