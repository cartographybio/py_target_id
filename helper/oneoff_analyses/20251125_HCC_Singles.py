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

#Ready Up Multi Target Workflow
surface = tid.utils.surface_genes()
tabs = tid.utils.tabs_genes()

#Load
manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata = load_cohort(IND, nMalig = 100)

# Run Single Target Workflow
single_path = IND + '.Single.Results.20251125.parquet'
if os.path.exists(single_path):
    single = pd.read_parquet(single_path)
else:
    print(f"Running single target workflow for {IND}...")
    single = tid.run.target_id_v1(malig_med_adata, ref_med_adata)
    risk_single = tid.utils.get_single_risk_scores()
    single = pd.merge(single, risk_single, how = "left", on = "gene_name")
    single = tid.run.target_quality_v2_01(single)
    single.to_parquet(single_path, engine='pyarrow', compression=None)

#Let's Filter
files = glob.glob("../../Surface_Testing/processed/claude-haiku-4-5-20251001/prompt1/*.csv")
dfs = [pd.read_csv(f, index_col=0) for f in files]
combined = pd.concat(dfs, axis=0)
tallied = combined.groupby("gene_name").sum()
tallied = tallied[tallied.index.isin(surface)].copy()
nonsurface = tallied[(tallied["YES"]==0) & (tallied["NO"] > 4)].index.tolist()
nonsurface = tallied[(tallied["YES"]==0) & (tallied["NO"] > 2)].index.tolist()
clearsurface = tallied[(tallied["NO"]==0) & (tallied["YES"] > 2)].index.tolist()

#Subset
single["claude_surface"] = 1*(single["gene_name"].isin(clearsurface))
single["claude_not_surface"] = 1*(single["gene_name"].isin(nonsurface))
single = single[single["gene_name"].isin(surface)].copy()
single["claude_summary"] = 1*(single["claude_not_surface"]==0) + 10 * single["claude_surface"]

#Plot Top
top = single[(single["Positive_Final_v2"] > 15) & (single["TargetQ_Final_v2"] > 70)].sort_values("TargetQ_Final_v2", ascending=False).copy()
top = top.head(30)
top["TI_Tox"] = np.max(top[["TI_Brain", "TI_Heart", "TI_Lung"]], axis = 1)
top["Scaled TI"] = np.minimum(top["TI"]/(top["Positive_Final_v2"]/100), 1)
top["Scaled TI Tox"] = np.minimum(top["TI_Tox"]/(top["Positive_Final_v2"]/100), 1)
top["Log2 Median Exp"] = np.log2(top["On_Val_50"] + 1)
top_plot = top[["gene_name", "TargetQ_Final_v1", "TargetQ_Final_v2", "Positive_Final_v2", "N_Off_Targets", "Log2 Median Exp", "Scaled TI", "Scaled TI Tox", "Hazard_SC_v1", "Hazard_FFPE_v1", "Hazard_GTEX_v1", "claude_summary"]].copy()
top_plot["TABS"] = top_plot["gene_name"].isin(tabs).astype(int)

plot.pd2r("df", top_plot)
r(f'''
library(tidyr)
library(ggplot2)
library(patchwork)

df_long <- df %>%
  pivot_longer(
    cols = -gene_name,
    names_to = "metric",
    values_to = "value"
  )

df_long$gene_name <- factor(df_long$gene_name, levels = rev(unique(df_long$gene_name)))

# Create a mapping of gene_name to TABS value and corresponding color
tabs_colors <- df_long %>%
  filter(metric == "TABS") %>%
  select(gene_name, value) %>%
  distinct() %>%
  mutate(axis_color = case_when(
    value == 1 ~ "dodgerblue3",
    value == 2 ~ "firebrick3",
    TRUE ~ "black"
  )) %>%
  arrange(gene_name)

# Create a plotting function
plot_metric <- function(metric_name, y_max, show_axis = FALSE, dup_axis = FALSE) {{
 
  data <- df_long %>% 
    filter(metric == metric_name) %>%
    mutate(value_capped = pmin(value, y_max))
 
  p <- ggplot(data, aes(x = gene_name, y = value_capped, fill = gene_name)) +
    geom_col(color = "black", linewidth = 0.3) +
    geom_label(aes(label = round(value_capped, 2), y = y_max/2), vjust = 0.5, hjust = 0.5, size = 2.5, fill = "white", color = "black", label.size = 0.3) +
    coord_flip(ylim = c(0, y_max)) + theme_jg(xText90=TRUE) + theme_small_margin()+
    theme(
      axis.text.y = element_text(size = 8),
      plot.title = element_text(face = "bold", size = 8),
      legend.position = "none",
      axis.title.y = element_blank(),
      axis.title.x = element_blank()
    ) +
    ggtitle(metric_name)
  
  if (dup_axis) {{
    axis_colors <- tabs_colors %>%
      filter(gene_name %in% levels(data$gene_name)) %>%
      pull(axis_color, name = gene_name)
    
    p <- p + theme(
      axis.text.y = element_text(size = 8, color = axis_colors[as.character(levels(data$gene_name))])
    ) +
    guides(y = guide_axis(position = "right"))
  }} else if (show_axis) {{
    # Get the axis text colors in the correct order
    axis_colors <- tabs_colors %>%
      filter(gene_name %in% levels(data$gene_name)) %>%
      pull(axis_color, name = gene_name)
    
    p <- p + theme(
      axis.text.y = element_text(size = 8, color = axis_colors[as.character(levels(data$gene_name))])
    )
  }} else {{
    p <- p + theme(axis.text.y = element_blank())
  }}
  
  p
}}

# Create individual plots
plots <- list(
  plot_metric("TargetQ_Final_v1", 100, show_axis = TRUE),
  plot_metric("TargetQ_Final_v2", 100),
  plot_metric("Positive_Final_v2", 100),
  plot_metric("N_Off_Targets", 10),
  plot_metric("Log2 Median Exp", 4),
  plot_metric("Scaled TI", 1),
  plot_metric("Scaled TI Tox", 1),
  plot_metric("Hazard_SC_v1", 25),
  plot_metric("Hazard_FFPE_v1", 25),
  plot_metric("Hazard_GTEX_v1", 40),
  plot_metric("claude_summary", 11, dup_axis = TRUE)
)

p <- wrap_plots(plots, nrow = 1)
pdf("target_quality_barplots.pdf", width = 14, height = 8)
print(p)
dev.off()
''')

#Plot Top Singles Summary

tid.plot.single.axial_summary(
    genes=top_plot["gene_name"],
    malig_adata=malig_adata,
    malig_med_adata=malig_med_adata,
    ref_adata=ref_adata,
    gtex_adata=tid.utils.get_gtex_adata(),
    tcga_adata=tid.utils.get_tcga_adata(),
    show=20,
    width=12,
    height=8,
    dpi=300
)



















label_df = single[(single["TargetQ_Final_v2"] > 75) & (single["Positive_Final_v2"] > 10)]

plot.pd2r("df", single)
plot.pd2r("sub_df", label_df)
target_q = 50
ppos = 10

r(f'''
p1 <- ggplot(df, aes(Positive_Final_v2, TargetQ_Final_v2, 
                fill = pmin(log2(SC_2nd_Target_Val + 1), 2))) +
    geom_point(pch = 21, size = 2, color = "black") +
    geom_hline(yintercept = {target_q}, linetype = "dashed", color = "firebrick3", size = 0.5, alpha = 0.7) +
    geom_vline(xintercept = {ppos}, linetype = "dashed", color = "firebrick3", size = 0.5, alpha = 0.7) +
    geom_label_repel(
        data = sub_df, 
        aes(label = gene_name), 
        fill = "white", 
        max.overlaps = Inf,
        size = 1.75
    ) +
    labs(
        title = "Target Quality vs Positive Percentage",
        x = "Positive Percentage (v2)",
        y = "Target Quality (v2)",
        fill = "log2(SC 2nd Target + 1)"
    ) +
    xlim(0, 100) +
    ylim(50, 100) +
    theme_jg()

p2 <- ggplot(df, aes(Positive_Final_v2, TargetQ_Final_v2, 
                fill = paste0(claude_summary))) +
    geom_point(pch = 21, size = 2, color = "black") +
    geom_hline(yintercept = {target_q}, linetype = "dashed", color = "firebrick3", size = 0.5, alpha = 0.7) +
    geom_vline(xintercept = {ppos}, linetype = "dashed", color = "firebrick3", size = 0.5, alpha = 0.7) +
    geom_label_repel(
        data = sub_df, 
        aes(label = gene_name), 
        alpha = 0.9,
        max.overlaps = Inf,
        size = 1.75
    ) +
    scale_fill_manual(values = c("11" = "firebrick3", "1" = "dodgerblue3", "0" = "lightgrey")) +
    labs(
        title = "Target Quality vs Positive Percentage",
        x = "Positive Percentage (v2)",
        y = "Target Quality (v2)",
        fill = "log2(SC 2nd Target + 1)"
    ) +
    xlim(0, 100) +
    ylim(50, 100) +
    theme_jg()

pdf("HCC.pdf", width = 12, height = 7)
print(p1 + p2)
dev.off()

''')


























#Get Risk
risk_single = tid.utils.get_single_risk_scores()
single = pd.merge(single, risk_single, how = "left", on = "gene_name")
single = tid.run.target_quality_v2_01(single)
single = single[single["gene_name"].isin(surface)]
single[single["gene_name"]=="TMPRSS9"]




#Single Target
single = tid.run.target_id_v1(malig_med_adata, ref_med_adata)

#Multi
gene_pairs = tid.utils.create_gene_pairs(surface, surface)

multi = tid.run.target_id_multi_v1(
    malig_adata=malig_adata,
    ref_adata=ref_adata,
    gene_pairs=gene_pairs,
    ref_med_adata=ref_med_adata,
    malig_med_adata=malig_med_adata,
    batch_size=20000,
    use_fp16=True
)

#Get Risk
risk_single = tid.utils.get_single_risk_scores()
single = pd.merge(single, risk_single, how = "left", on = "gene_name")
single = tid.run.target_quality_v2_01(single)
single = single[single["gene_name"].isin(surface)]
single[single["gene_name"]=="TMPRSS9"]