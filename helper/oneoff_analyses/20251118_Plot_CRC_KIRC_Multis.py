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

import numpy as np
import pandas as pd

def analyze_multis_tpm(
    multis: list,
    tcga_adata = None,
    tpm_threshold: float = 10,
    pct_threshold: float = 0.20,
    main_indication: str = None
):
    """
    Analyze TCGA indications for gene combinations with co-expression above TPM threshold.
    
    Parameters
    ----------
    multis : list
        Gene combinations in format ["GENE1_GENE2", ...]
    tcga_adata : AnnData
        TCGA expression data (samples × genes)
    tpm_threshold : float
        TPM threshold for "positive" (default: 10)
    pct_threshold : float
        Percentage threshold for indication inclusion (default: 0.20 = 20%)
    main_indication : str
        Indication to report percentage for (e.g., "LUAD"). If None, skipped.
    
    Returns
    -------
    DataFrame with columns:
        - 'gene_pair': gene pair name
        - 'n_indications_above_threshold': number of indications with >= pct_threshold patients
        - 'n_patients_above_threshold': total patients meeting criteria across qualifying indications
        - '{main_indication}_pct': percentage for the specified indication (if provided)
    """
    
    from py_target_id import utils
    
    # Lazy-load TCGA data only if needed
    if tcga_adata is None:
        tcga_adata = utils.get_tcga_adata()
    
    # Parse gene combinations - convert to list if pandas Series
    if hasattr(multis, 'tolist'):
        multis = multis.tolist()
    
    multis_split = np.array([m.split("_") for m in multis])
    genes = np.unique(multis_split.ravel()).tolist()
    
    print(f"Analyzing {len(multis)} gene combinations with {len(genes)} unique genes")
    
    # Load TCGA data
    print("Reading in TCGA...")
    if hasattr(tcga_adata, 'to_memory'):
        print("Materializing TCGA VirtualAnnData...")
        tcga_adata = tcga_adata.to_memory()
    
    # Get all samples first, then extract indication info before subsetting genes
    tcga_samples = tcga_adata.obs_names.values
    tcga_id = np.array([s.split('#')[0] for s in tcga_samples])
    
    print(f"Found {len(np.unique(tcga_id))} indications: {sorted(np.unique(tcga_id))}")
    
    # Now subset to genes
    tcga_subset = tcga_adata[:, genes]
    tcga_mat = tcga_subset.X.toarray() if hasattr(tcga_subset.X, 'toarray') else tcga_subset.X
    
    # Process each gene combination
    results_list = []
    
    for idx, (gx, gy) in enumerate(multis_split):
        # Get gene indices
        idx_gx = genes.index(gx)
        idx_gy = genes.index(gy)
        
        # TCGA data
        df_tcga = pd.DataFrame({
            'x1': tcga_mat[:, idx_gx],
            'x2': tcga_mat[:, idx_gy],
            'facet': tcga_id
        })
        
        # Minimum of the pair
        df_tcga['xm'] = np.minimum(df_tcga['x1'], df_tcga['x2'])
        df_tcga['TPM_threshold'] = df_tcga['xm'] > tpm_threshold
        
        # Calculate percentage positive by cancer type
        pct_results = df_tcga.groupby('facet')['TPM_threshold'].agg(['mean', 'size']).reset_index()
        pct_results.columns = ['cancer_type', 'percentage', 'n']
        pct_results['percentage'] = pct_results['percentage'] * 100
        
        # Count indications meeting threshold
        pct_results['meets_threshold'] = pct_results['percentage'] >= (pct_threshold * 100)
        n_indications = pct_results['meets_threshold'].sum()
        n_patients = pct_results[pct_results['meets_threshold']]['n'].sum()
        
        # Get percentage for main indication if specified
        main_pct = None
        if main_indication:
            main_row = pct_results[pct_results['cancer_type'] == main_indication]
            if len(main_row) > 0:
                main_pct = main_row.iloc[0]['percentage']
        
        result_dict = {
            'gene_pair': multis[idx],
            'n_indications_above_threshold': int(n_indications),
            'n_patients_above_threshold': int(n_patients)
        }
        
        if main_indication:
            result_dict[f'{main_indication}_pct'] = main_pct
        
        results_list.append(result_dict)
        
        # Debug: show which indications pass
        if idx < 3:  # Show first 3 pairs
            print(f"\n{multis[idx]}:")
            print(pct_results[['cancer_type', 'percentage', 'n', 'meets_threshold']].head(10))
    
    # Create DataFrame in same order as input
    result_df = pd.DataFrame(results_list)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete: {len(result_df)} gene pairs")
    print(f"Threshold: ≥{pct_threshold*100:.1f}% patients per indication with min(gene1, gene2) >{tpm_threshold} TPM")
    print(f"{'='*60}\n")
    
    return result_df

#Load Cohort
IND = os.path.basename(os.getcwd())

#Ready Up Multi Target Workflow
surface = tid.utils.surface_genes()
surface.append("CLDN18.2")
tabs = tid.utils.tabs_genes()
tabs.append("CLDN18.2")

#Load
manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata = utils.load_cohort(IND, nMalig = 0)
gtex = tid.utils.get_gtex_adata()
tcga = tid.utils.get_tcga_adata()
tcga = tcga[tcga.obs["TCGA"]=="LUAD",:]

multi = pd.concat([
	#pd.read_parquet(IND + '.Multi.CLDN18p2.Results.20251107.parquet'),
	pd.read_parquet(IND + '.Multi.Results.20251111.parquet')
])

#Get Risk
risk_multi = tid.utils.get_multi_risk_scores()

#Add
multi = pd.merge(multi, risk_multi, how = "left", on = "gene_name")

multi = tid.run.target_quality_v2_01(multi)

#Split Out Singles
genes = multi["gene_name"].values
# Pre-allocate arrays
gene1 = np.empty(len(genes), dtype=object)
gene2 = np.empty(len(genes), dtype=object)

for i, g in enumerate(genes):
    parts = g.split("_")
    gene1[i] = parts[0]
    gene2[i] = parts[1]

mask_single = gene1==gene2
single = multi[mask_single].copy()
single["gene_name"] = single["gene_name"].str.split("_").str[0]

#Let's Filter
files = glob.glob("../../Surface_Testing/processed/claude-haiku-4-5-20251001/prompt1/*.csv")
dfs = [pd.read_csv(f, index_col=0) for f in files]
combined = pd.concat(dfs, axis=0)
tallied = combined.groupby("gene_name").sum()
tallied = tallied[tallied.index.isin(surface)].copy()
nonsurface = tallied[(tallied["YES"]==0) & (tallied["NO"] > 4)].index.tolist()
nonsurface = tallied[(tallied["YES"]==0) & (tallied["NO"] > 2)].index.tolist()
clearsurface = tallied[(tallied["NO"]==0) & (tallied["YES"] > 2)].index.tolist()
clearsurface.append("CLDN18.2")

#Multi
multi2 = multi[(multi["TargetQ_Final_v2"] > 25) & (multi["Positive_Final_v2"] > 25)].copy().reset_index()
multi2 = pd.concat([multi2, multi2["gene_name"].str.split("_", expand=True)], axis = 1)
multi2["claude_surface"] = 1*(multi2[0].isin(clearsurface)) + 1*(multi2[1].isin(clearsurface))
multi2["claude_not_surface"] = 1*(multi2[0].isin(nonsurface)) + 1*(multi2[1].isin(nonsurface))
multi2["claude_summary"] = 1*(multi2["claude_not_surface"]==0) + 10 * multi2["claude_surface"]
multi2 = multi2.sort_values("TargetQ_Final_v2", ascending=False)
multi3 = multi2[multi2["claude_summary"] >= 11].copy()
multi3["TABS"] = 1*multi3[0].isin(tabs) + 1*multi3[1].isin(tabs)
multi3 = multi3.reset_index()

#Multi Score
single0 = single.copy().set_index("gene_name").loc[multi3[0]].reset_index()
single1 = single.copy().set_index("gene_name").loc[multi3[1]].reset_index()
multi3["LFC_X_vs_Y"] = abs(np.log2(single0["On_Val_50"]+1) - np.log2(single1["On_Val_50"]+1))
multi3["TargetQ_vs_Sngl"] = np.maximum(multi3["TargetQ_Final_v2"] - np.maximum(single0["TargetQ_Final_v2"], single1["TargetQ_Final_v2"]), 0)
multi3["TI_vs_Sngl"] = np.maximum(multi3["TI"] - np.maximum(single0["TI"], single1["TI"]), 0)

multi4 = multi3.copy()
multi4 = multi4.reset_index(drop=True)
multi4 = multi4[~multi4["gene_name"].str.contains("LY6G6D")]

# multi2 = multi2[multi2["claude_not_surface"]==0].copy()
# multi2 = multi2[multi2["claude_surface"]==2].copy()
# multi2 = multi2[~multi2["gene_name"].str.contains("LIF")]
# multi2["TABS"] = 1*multi2[0].isin(tabs) + 1*multi2[1].isin(tabs)

# multi3[multi3["TABS"] == 0][["TargetQ_Final_v2", "Positive_Final_v2", "gene_name"]].head(25)
# multi3[multi3["TABS"] == 1][["TargetQ_Final_v2", "Positive_Final_v2", "gene_name"]].head(25)
# multi3[multi3["TABS"] == 2][["TargetQ_Final_v2", "Positive_Final_v2", "gene_name"]].head(25)
# multi2[["TargetQ_Final_v2", "Positive_Final_v2", "gene_name", "claude_summary"]].head(25)

print(multi4[multi4["gene_name"].str.contains("SLC17A3")].head().iloc[0])

top = multi4[(multi4["TABS"] >= 2) & (multi4["Positive_Final_v2"] > 40) & (multi4["TargetQ_Final_v2"] > 85)].sort_values("TargetQ_Final_v2", ascending=False).copy()
#top = pd.concat([multi4[multi4["gene_name"].isin(["TMPRSS4_ROS1", "TMPRSS4_SLC22A31"])], top])
top = top.sort_values("TargetQ_Final_v2", ascending=False).copy()
top = top[~top["gene_name"].str.contains("PRAME|VTCN1|LY6K|HIST1H1A|B4GALNT4|PCDHB2")]
#top = top[top["gene_name"].str.contains("VTCN1")]
top = top.head(50).copy()
top["TI_Tox"] = np.max(top[["TI_Brain", "TI_Heart", "TI_Lung"]], axis = 1)
top["Scaled TI"] = np.minimum(top["TI"]/(top["Positive_Final_v2"]/100), 1)
top["Scaled TI Tox"] = np.minimum(top["TI_Tox"]/(top["Positive_Final_v2"]/100), 1)
top["Log2 Median Exp"] = np.log2(top["On_Val_50"] + 1)
top_plot = top[["gene_name", "TargetQ_Final_v1", "TargetQ_Final_v2", "Positive_Final_v2", 
  "N_Off_Targets", "Log2 Median Exp", "Scaled TI", "Scaled TI Tox", "Hazard_SC_v1", "Hazard_FFPE_v1", "Hazard_GTEX_v1",
  "claude_summary", "LFC_X_vs_Y", "TargetQ_vs_Sngl", "TI_vs_Sngl", "TABS"
]]

tcga_info = analyze_multis_tpm(top_plot["gene_name"], tid.utils.get_tcga_adata(), pct_threshold = 0.25, main_indication = "COAD")
tcga_info.columns = ["gene_name", "TCGA_N_>_25%", "TCGA_Total", "TCGA_Main_%"]
top_plot = pd.merge(top_plot, tcga_info, on = "gene_name", how = "left")


title = IND + " TABS = 2 Multis"
pdf = IND + "_TABS_2_Multis.pdf"
plot.pd2r("df", top_plot)

r(f'''
library(tidyr)
library(dplyr)
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
  plot_metric("TargetQ_Final_v2", 110, show_axis = TRUE),
  plot_metric("Positive_Final_v2", 100),
  plot_metric("LFC_X_vs_Y", 2),
  plot_metric("N_Off_Targets", 10),
  plot_metric("Log2 Median Exp", 4),
  plot_metric("Scaled TI", 1),
  plot_metric("Scaled TI Tox", 1),
  plot_metric("Hazard_SC_v1", 25),
  plot_metric("Hazard_FFPE_v1", 25),
  plot_metric("Hazard_GTEX_v1", 40),
  plot_metric("TargetQ_Final_v1", 100),
  plot_metric("claude_summary", 11),
  plot_metric("TargetQ_vs_Sngl", 50),
  plot_metric("TI_vs_Sngl", 0.5),
  plot_metric("TABS", 2),
  plot_metric("TCGA_Main_%", 100),
  plot_metric("TCGA_N_>_25%", 33, dup_axis = TRUE)
)

p <- wrap_plots(plots, nrow = 1) +
  plot_annotation(title = "{title}", theme = theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold")))

pdf("{pdf}", width = 20, height = 10)
print(p)
dev.off()
''')

















#Plot
tid.plot.multi.biaxial_summary(
  multis = top["gene_name"].tolist(), 
  malig_adata=malig_adata, 
  malig_med_adata=malig_med_adata, 
  ref_adata=ref_adata,
  gtex_adata=gtex,
  tcga_adata=tcga,
  out_dir="multi/TABS_1_bi_axial"
)