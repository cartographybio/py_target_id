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

#Info
m = malig_med_adata[:, "LY6K"]
ref_ff = ref_med_adata[:, "LY6K"]
ref_sc = tid.utils.get_ref_lv4_sc_med_adata()[:, "LY6K"]

#Create Data Frames
df_malig = pd.DataFrame({"type" : "Malig_FFPE", "x" : m.obs["Patient"].values, "log2_y" : np.log2(m.X[:,0] + 1), "pos" : np.where(m.layers["positivity"][:, 0].astype(bool), "Positive", "Negative")})
df_ha_sc = pd.DataFrame({"type" : "Reference_SC", "x" : ref_sc.obs["CellType"], "log2_y" : np.log2(ref_sc.X[:,0] + 1), "tissue": ref_sc.obs["Tissue"] })
df_ha_ff = pd.DataFrame({"type" : "Reference_FFPE", "x" : ref_ff.obs["CellType"], "log2_y" : np.log2(ref_ff.X[:,0] + 1), "tissue": ref_ff.obs["Tissue"] })

#Pos
log2_med = np.median(df_malig[df_malig["pos"]=="Positive"]["log2_y"].values)

#Plot Each Side By Side
tissue = "Testes"
df_ha_sc_sub = df_ha_sc[df_ha_sc["tissue"]==tissue]
df_ha_ff_sub = df_ha_ff[df_ha_ff["tissue"]==tissue]

#Send
tid.plot.pd2r("df_malig", df_malig)
tid.plot.pd2r("df_ha_sc_sub", df_ha_sc_sub)
tid.plot.pd2r("df_ha_ff_sub", df_ha_ff_sub)

r_code = f'''

pdf("LY6K-{tissue}.pdf", width = 14, height = 7)

df_malig$pos <- factor(df_malig$pos, levels = c("Positive", "Negative"))

p1 <- ggplot(df_malig, aes(pos, log2_y, fill = pos)) +
	geom_jitter(height = 0, pch =21, width =0.25) +
	ylab("Log2(CP10k+1)") +
	ggtitle("Malignant FFPE")+
	ylim(c(0,2)) +
	theme(legend.position = "none")+
	xlab("") +
	geom_boxplot(outlier.shape = NA, fill = NA) +
	scale_fill_manual(values = c("Positive" = pal_cart[5], "Negative" = "lightgrey")) +
	geom_hline(yintercept={log2_med}, lty = 'dashed') +
	theme_small_margin()

p2 <- ggplot(df_ha_ff_sub, aes(x, log2_y)) +
	geom_point(pch =21, fill = pal_cart[2]) +
	ylab("Log2(CP10k+1)") +
	ggtitle("Reference FFPE")+
	theme_jg(xText90=TRUE) +
	ylim(c(0,2))+
	xlab("") +
	geom_hline(yintercept={log2_med}, lty = 'dashed') +
	theme_small_margin()

p3 <- ggplot(df_ha_sc_sub, aes(x, log2_y)) +
	geom_point(pch =21, fill = pal_cart[2]) +
	ylab("Log2(CP10k+1)") +
	ggtitle("Reference SC") +
	theme_jg(xText90=TRUE) +
	ylim(c(0,2)) +
	xlab("") +
	geom_hline(yintercept={log2_med}, lty = 'dashed') +
	theme_small_margin()

print(p1 + p3 + p2 + plot_layout(widths = c(1,2,3)) + plot_annotation(title = "{tissue}")) 

dev.off()
'''

r(r_code)

