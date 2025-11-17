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
surface.append("CLDN18.2")
tabs = tid.utils.tabs_genes()
tabs.append("CLDN18.2")

#Load
manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata = utils.load_cohort(IND, nMalig = 100)

#Read Previous
single = pd.concat([
	pd.read_parquet(IND + '.Single.CLDN18p2.Results.20251107.parquet'),
	pd.read_parquet(IND + '.Single.Results.20251107.parquet')
])

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

#Split
single["claude_surface"] = 1*(single["gene_name"].isin(clearsurface))
single["claude_not_surface"] = 1*(single["gene_name"].isin(nonsurface))
single = single[single["gene_name"].isin(surface)]
single[(single["claude_surface"]==1) & (single["claude_not_surface"]==0)].head(25)[["TargetQ_Final_v2", "Positive_Final_v2", "gene_name"]]
single.head(25)[["TargetQ_Final_v2", "Positive_Final_v2", "gene_name"]]

#Plot
single["claude_summary"] = 1*(single["claude_not_surface"]==0) + 10 * single["claude_surface"]
label_df = single[(single["TargetQ_Final_v1"] > 50) & (single["Positive_Final_v2"] > 10)]

plot.pd2r("df", single)
plot.pd2r("sub_df", label_df)
target_q = 50
ppos = 10

r(f'''
p1 <- ggplot(df, aes(Positive_Final_v2, TargetQ_Final_v1, 
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
        y = "Target Quality (v1)",
        fill = "log2(SC 2nd Target + 1)"
    ) +
    xlim(0, 100) +
    ylim(0, 100) +
    theme_jg()

p2 <- ggplot(df, aes(Positive_Final_v2, TargetQ_Final_v1, 
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
        y = "Target Quality (v1)",
        fill = "log2(SC 2nd Target + 1)"
    ) +
    xlim(0, 100) +
    ylim(0, 100) +
    theme_jg()

pdf("PDAC-Single-20251107.pdf", width = 12, height = 7)
print(p1 + p2)
dev.off()

''')


label_df = single[(single["TargetQ_Final_v2"] > 50) & (single["Positive_Final_v2"] > 10)]

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

pdf("PDAC-Single-TQ2-20251107.pdf", width = 12, height = 7)
print(p1 + p2)
dev.off()

''')

top = single[(single["claude_summary"]==11) & (single["Positive_Final_v2"] > 15) & (single["TargetQ_Final_v2"] > 65)].sort_values("TargetQ_Final_v2", ascending=False).head(20).copy()
top = single[(single["Positive_Final_v2"] > 15) & (single["TargetQ_Final_v2"] > 70)].sort_values("TargetQ_Final_v2", ascending=False).copy()
top["TI_Tox"] = np.max(top[["TI_Brain", "TI_Heart", "TI_Lung"]], axis = 1)
top["Scaled TI"] = np.minimum(top["TI"]/(top["Positive_Final_v2"]/100), 1)
top["Scaled TI Tox"] = np.minimum(top["TI_Tox"]/(top["Positive_Final_v2"]/100), 1)
top["Log2 Median Exp"] = np.log2(top["On_Val_50"] + 1)
top_plot = top[["gene_name", "TargetQ_Final_v1", "TargetQ_Final_v2", "Positive_Final_v2", "N_Off_Targets", "Log2 Median Exp", "Scaled TI", "Scaled TI Tox", "Hazard_SC_v1", "Hazard_FFPE_v1", "Hazard_GTEX_v1", "claude_summary"]]

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
# Create a plotting function
plot_metric <- function(metric_name, y_max, show_axis = FALSE) {{
 
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
  
  if (!show_axis) {{
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
  plot_metric("claude_summary", 11)
)
p <- wrap_plots(plots, nrow = 1)
pdf("target_quality_barplots.pdf", width = 14, height = 6)
print(p)
dev.off()
''')

top["P_Pos_Val_0.5"] = 100 * top["P_Pos_Val_0.5"]

create_gene_summary_pdf(top, top["gene_name"].tolist())

df_summary = single[single["gene_name"]=="LEMD1"].iloc[0]
df_summary["P_Pos_Val_0.5"] = 100 * df_summary["P_Pos_Val_0.5"]

import pandas as pd
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch

# Assuming df_summary is your Series
# df_summary = ...

# Convert Series to DataFrame
df_table = df_summary.to_frame().reset_index()
df_table.columns = ['Metric', 'Value']

# Format values
def format_value(x):
    try:
        # Try to convert to float
        val = float(x)
        
        if isinstance(x, (int, np.integer)):
            return str(int(val))
        elif isinstance(x, bool):
            return str(x)
        elif abs(val) < 0.001 and val != 0:
            return f"{val:.2e}"
        elif abs(val) >= 1:
            # For values >= 1, use fixed decimal places
            return f"{val:.3f}".rstrip('0').rstrip('.')
        else:
            # For values < 1, use 3 significant figures
            return f"{val:.3g}"
    except (ValueError, TypeError):
        return str(x)

df_table['Value'] = df_table['Value'].apply(format_value)

# Organize into sections
sections = {
    'Summary': ['TargetQ_Final_v1', 'TargetQ_Final_v2', 'Positive_Final_v2'],
    'Target Expr': ['Target_Val', 'SC_2nd_Target_Val', 'On_Val_25', 'On_Val_50', 'On_Val_75'],
    'Specificity & Log2FC': ['N_Off_Targets', 'Top_Off_Target_Val', 'Corrected_Top_Off_Target_Val', 
                              'Log2_Fold_Change', 'Corrected_Log2_Fold_Change'],
    'Off Target Distribution': ['N_Off_Targets_0.01', 'N_Off_Targets_0.05', 'N_Off_Targets_0.1',
                                 'N_Off_Targets_0.25', 'N_Off_Targets_0.5', 'N_Off_Targets_1.0'],
    'Positive Patients': ['Positive_Final_v2', 'P_Pos_Val_0.5', 'P_Pos_Val_0.1'],
    'Tissue Toxicity': ['TI', 'Tox_Brain', 'TI_Brain', 'Tox_Heart', 'TI_Heart', 'Tox_Lung', 'TI_Lung', 
                        'Tox_Immune', 'TI_Immune', 'Tox_NonImmune', 'TI_NonImmune'],
    'Target Risk': ['Hazard_SC_v1', 'Hazard_FFPE_v1', 'Hazard_GTEX_v1', 'GTEX_Tox_Tier1', 'GTEX_Tox_Tier2', 'GTEX_Tox_Tier3']
}

# Metric display name mapping
metric_names = {
    'On_Val_25': 'On_Expr_25th',
    'On_Val_50': 'On_Expr_50th',
    'On_Val_75': 'On_Expr_75th',
    'Target_Val': 'Top_Patient_Expr',
    'SC_2nd_Target_Val': '2nd_Top_Patient_Expr',
    'N_Off_Targets': 'N_Off_Targets_Specificity',
    'P_Pos_Val_0.5': 'Positive_Patients_Val_0.5',
    'P_Pos_Val_0.1': 'Positive_Patients_Val_0.1',
    'GTEX_Tox_Tier1': 'GTEX_Tox_Tier1_N_Tissue',
    'GTEX_Tox_Tier2': 'GTEX_Tox_Tier2_N_Tissue',
    'GTEX_Tox_Tier3': 'GTEX_Tox_Tier3_N_Tissue'
}

# Section colors
section_colors = {
    'Summary': '#1F4788',
    'Target Expr': '#2E5C8A',
    'Specificity & Log2FC': '#3D7B8C',
    'Off Target Distribution': '#4C99B8',
    'Positive Patients': '#5BA8D0',
    'Tissue Toxicity': '#D35400',
    'Target Risk': '#C0392B'
}

# Create PDF in portrait
pdf_file = "df_summary_table.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=letter, rightMargin=0.3*inch, 
                        leftMargin=0.3*inch, topMargin=0.3*inch, bottomMargin=0.3*inch)

elements = []

# Add title
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'Title',
    parent=styles['Heading1'],
    fontSize=9,
    textColor=colors.HexColor('#1F4788'),
    spaceAfter=4,
    alignment=1,
    fontName='Helvetica-Bold'
)

gene_name = df_summary.get('gene_name', 'N/A')
elements.append(Paragraph(f"Target Summary Statistics - {gene_name}", title_style))

# Build table data with section headers
table_data = [['Section', 'Metric', 'Value']]
for section_name, metrics in sections.items():
    section_data = df_table[df_table['Metric'].isin(metrics)].copy()
    section_data = section_data.set_index('Metric').loc[metrics].reset_index()
    
    for idx, (metric, value) in enumerate(section_data[['Metric', 'Value']].values):
        display_metric = metric_names.get(metric, metric)
        if idx == 0:
            table_data.append([section_name, display_metric, value])
        else:
            table_data.append(['', display_metric, value])

# Create table
table = Table(table_data, colWidths=[1.8*inch, 2.2*inch, 0.7*inch])

# Apply styling
table.setStyle(TableStyle([
    # Header
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4788')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 8),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
    ('TOPPADDING', (0, 0), (-1, 0), 4),
    
    # Body
    ('FONTSIZE', (0, 1), (-1, -1), 7),
    ('TOPPADDING', (0, 1), (-1, -1), 2),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
    ('LEFTPADDING', (0, 1), (-1, -1), 4),
    ('RIGHTPADDING', (0, 1), (-1, -1), 4),
    
    # Alignment
    ('ALIGN', (0, 1), (0, -1), 'LEFT'),
    ('ALIGN', (1, 1), (1, -1), 'LEFT'),
    ('ALIGN', (2, 1), (2, -1), 'RIGHT'),
    
    # Fonts
    ('FONTNAME', (1, 1), (1, -1), 'Courier'),
    ('FONTNAME', (2, 1), (2, -1), 'Courier'),
    
    # Borders
    ('GRID', (0, 0), (-1, -1), 0.2, colors.HexColor('#E8E8E8')),
    ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#1F4788')),
    ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor('#1F4788')),
    
    # Alternating row colors
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F9F9F9')]),
]))

elements.append(table)

# Build PDF
doc.build(elements)
print(f"Saved 1-page PDF with sections to {pdf_file}")

top_singles = single.sort_values("TargetQ_Final_v2", ascending=False)["gene_name"].str.replace("CLDN18.2", "CLDN18").head(50).tolist()

axial_summary(
    genes=top_singles,
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




"""
Single gene axial summary plot visualization across datasets.
Comprehensive view of target expression in malignant vs healthy tissues.
"""

__all__ = ['axial_summary']

import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import os
import time
from typing import Optional
from py_target_id import plot
from py_target_id import utils
from rpy2.robjects import r
from datetime import datetime


def axial_summary(
    gene: str,
    malig_adata,
    malig_med_adata,
    ref_adata,
    gtex_adata=None,
    tcga_adata=None,
    out_dir: str = "single/axial_summary",
    show: int = 15,
    width: float = 20,
    height: float = 8,
    dpi: int = 300,
    title: Optional[str] = None
):
    """
    Create comprehensive axial summary plot for a single gene target.
    
    This creates a 4-panel plot showing:
    - Panel 1: Malignant tumor expression (Positive vs Negative patients)
    - Panel 2: Reference tissues grouped by tissue type
    - Panel 3: Top off-target cell types
    - Panel 4: GTEx/TCGA bulk expression
    
    Parameters
    ----------
    gene : str
        Gene target name
    malig_adata : AnnData
        Malignant cell data with patient annotations
    malig_med_adata : AnnData
        Malignant median data (patients × genes)
        Must have layers['positivity'] with binary positivity data
    ref_adata : AnnData
        Healthy reference atlas with CellType annotations
    gtex_adata : AnnData, optional
        GTEx expression data (samples × genes) with obs['GTEX']
    tcga_adata : AnnData, optional
        TCGA expression data (samples × genes)
    out_dir : str
        Output directory for plots
    show : int
        Number of top off-target cell types to display
    width : float
        Plot width in inches
    height : float
        Plot height in inches
    dpi : int
        Resolution for output
    title : str, optional
        Custom title (defaults to gene name)
    """
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    if title is None:
        title = gene
    
    overall_start = time.time()
    
    print(f"Processing gene: {gene}")
    print(f"Output directory: {out_dir}")
    
    # Get cell types BEFORE materializing
    CT = ref_adata.obs['CellType'].values
    CT = pd.Series(CT).str.replace('Î±', 'a').str.replace('Î²', 'B').values
    
    # Load malignant data
    print("Loading malignant data...")
    malig_subset = malig_adata[:, [gene]]
    
    if hasattr(malig_subset, 'to_memory'):
        print("Materializing malignant subset...")
        malig_subset = malig_subset.to_memory()
    
    if sparse.issparse(malig_subset.X):
        malig_subset.X = malig_subset.X.toarray()
    
    malig_exp = malig_subset.X.ravel()
    patients = malig_subset.obs['Patient'].values
    
    # Load healthy atlas
    print("Loading healthy atlas...")
    ref_subset = ref_adata[:, [gene]]
    
    if hasattr(ref_subset, 'to_memory'):
        print("Materializing reference subset...")
        ref_subset = ref_subset.to_memory()
    
    if sparse.issparse(ref_subset.X):
        ref_subset.X = ref_subset.X.toarray()
    
    ref_exp = ref_subset.X.ravel()
    
    # Compute medians
    print("Computing medians...")
    malig_med = utils.summarize_matrix(
        malig_exp.reshape(-1, 1), patients, axis=0, metric="median", verbose=False
    )
    malig_med = malig_med.iloc[:, 0]
    
    ha_med = utils.summarize_matrix(
        ref_exp.reshape(-1, 1), CT, axis=0, metric="median", verbose=False
    )
    ha_med = ha_med.iloc[:, 0]
    
    # Get positivity data
    print("Extracting positivity data...")
    pos_data = malig_med_adata[:, gene].layers['positivity'].flatten()
    pos_patients = malig_med_adata.obs['Patient'].values[pos_data.astype(bool)]
    pos_patient_unique = np.unique(pos_patients)
    per_pos = round(100 * len(pos_patient_unique) / len(malig_med_adata.obs_names), 1)
    
    # Load GTEx and TCGA if provided
    gtex_data = None
    tcga_data = None
    
    if gtex_adata is not None:
        print("Loading GTEx data...")
        gtex_subset = gtex_adata[:, [gene]]
        if hasattr(gtex_subset, 'to_memory'):
            gtex_subset = gtex_subset.to_memory()
        gtex_mat = gtex_subset.X.toarray().ravel() if sparse.issparse(gtex_subset.X) else gtex_subset.X.ravel()
        gtex_id = gtex_subset.obs['GTEX'].values if 'GTEX' in gtex_subset.obs else gtex_subset.obs_names.values
        gtex_data = {'values': gtex_mat, 'id': gtex_id}
    
    if tcga_adata is not None:
        print("Loading TCGA data...")
        tcga_subset = tcga_adata[:, [gene]]
        if hasattr(tcga_subset, 'to_memory'):
            tcga_subset = tcga_subset.to_memory()
        tcga_mat = tcga_subset.X.toarray().ravel() if sparse.issparse(tcga_subset.X) else tcga_subset.X.ravel()
        # Use TCGA cancer type from obs if available, otherwise parse from index
        tcga_type = tcga_subset.obs['TCGA'].values if 'TCGA' in tcga_subset.obs else np.array([s.split('#')[0] for s in tcga_subset.obs_names.values])
        tcga_data = {'values': tcga_mat, 'type': tcga_type}
    
    # Prepare malignant dataframe
    dfT = pd.DataFrame({
        'log2_exp': np.log2(malig_med.values + 1),
        'ID': malig_med.index,
        'Positive': [p in pos_patient_unique for p in malig_med.index]
    })
    
    n_pos = (dfT['Positive']).sum()
    n_neg = (~dfT['Positive']).sum()
    
    # Prepare reference dataframe - sorted by expression
    dfR = pd.DataFrame({
        'log2_exp': np.log2(ha_med.values + 1),
        'CellType': ha_med.index,
        'Tissue': [ct.split(':')[0] for ct in ha_med.index]
    })
    
    dfR = dfR.sort_values('log2_exp', ascending=False).reset_index(drop=True)
    dfR['ID_display'] = [plot.add_smart_newlines(ct, max_length=30) for ct in dfR['CellType']]
    
    # Prepare bulk data if available
    df_bulk = None
    if tcga_data is not None and gtex_data is not None:
        df_bulk = pd.DataFrame({
            'log2_exp': np.concatenate([
                np.log2(tcga_data['values'] + 1),
                np.log2(gtex_data['values'] + 1)
            ]),
            'Type': np.concatenate([
                tcga_data['type'],
                gtex_data['id']
            ]),
            'Source': ['TCGA'] * len(tcga_data['values']) + ['GTEx'] * len(gtex_data['values'])
        })
    
    # Create output path
    out_path = os.path.join(out_dir, f"{gene}_summary.pdf")
    
    # Send data to R
    print("Sending data to R...")
    plot.pd2r("dfT", dfT)
    plot.pd2r("dfR", dfR)
    if df_bulk is not None:
        plot.pd2r("df_bulk", df_bulk)
    
    # Get limits from data
    max_exp = max(dfT['log2_exp'].max(), dfR['log2_exp'].max())
    limits = [0, max_exp]
    med_pos = dfT[dfT['Positive']]['log2_exp'].median()
    
    # For bulk data, use independent limits based on bulk data
    bulk_limit_tpm = 10
    bulk_line = np.log2(bulk_limit_tpm + 1)
    
    bulk_max = df_bulk['log2_exp'].max()
    bulk_limits = [0, bulk_max * 1.05]
    
    # Create plots in R
    print("Creating plots in R...")
    
    r_code = f'''
    library(ggplot2)
    library(patchwork)
    library(scales)
    
    # Parameters
    show <- {show}
    size <- 6
    limits <- c({limits[0]}, {limits[1]})
    med_pos <- {med_pos}
    n_pos <- {n_pos}
    n_neg <- {n_neg}
    per_pos <- {per_pos}
    bulk_limits <- c({bulk_limits[0]}, {bulk_limits[1]})
    bulk_line <- {bulk_line}
    
    # ========== PANEL 1: MALIGNANT EXPRESSION ==========
    dfT$pos_status <- ifelse(dfT$Positive, "Malig Pos.", "Malig Neg.")
    dfT$pos_status <- factor(dfT$pos_status, levels = c("Malig Pos.", "Malig Neg."))
    
    p1 <- ggplot(dfT, aes(pos_status, log2_exp)) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
        geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=pos_status), color="black", height=0, width=0.2) +
        theme(legend.position="none") +
        coord_cartesian(ylim=limits) +
        scale_x_discrete(
            drop=FALSE,
            labels=c("Malig Pos."=paste0("Malig Pos.\\n(N=", n_pos, ")"),
                    "Malig Neg."=paste0("Malig Neg.\\n(N=", n_neg, ")"))
        ) +
        xlab("") +
        geom_hline(yintercept=med_pos, lty="dashed", color="firebrick3") +
        scale_fill_manual(values=c("Malig Pos."="#28154C", "Malig Neg."="lightgrey")) +
        ylab("Log2(CP10k + 1)") +
        ggtitle(paste0(per_pos, "%"))
    
    # ========== PANEL 2: REFERENCE BY TISSUE ==========
    dfR$tissue_factor <- factor(dfR$Tissue, levels=unique(dfR$Tissue[order(dfR$log2_exp, decreasing=TRUE)]))
    
    p2 <- ggplot(dfR, aes(tissue_factor, squish(log2_exp, limits))) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
        geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=Tissue), color="black", height=0, width=0.2) +
        theme(legend.position="none") +
        coord_cartesian(ylim=limits) +
        geom_hline(yintercept=med_pos, lty="dashed", color="firebrick3") +
        xlab("") + ylab("Log2(CP10k + 1)") +
        theme(axis.text.x=element_text(size=size)) +
        scale_fill_manual(values=suppressMessages(create_pal_d(dfR$Tissue))) +
        ggtitle("Facet By Tissue")
    
    # ========== PANEL 3: TOP OFF-TARGET CELL TYPES ==========
    dfR_top <- head(dfR, show)
    dfR_top$ID_display <- factor(dfR_top$ID_display, levels=dfR_top$ID_display)
    
    p3 <- ggplot(dfR_top, aes(ID_display, squish(log2_exp, limits))) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_point(size=1.5, stroke=0.35, pch=21, aes(fill=Tissue), color="black") +
        theme(legend.position="none") +
        coord_cartesian(ylim=limits) +
        geom_hline(yintercept=med_pos, lty="dashed", color="firebrick3") +
        xlab("") + ylab("Log2(CP10k + 1)") +
        theme(axis.text.x=element_text(size=size*0.75)) +
        scale_fill_manual(values=suppressMessages(create_pal_d(dfR_top$Tissue))) +
        ggtitle("Top Off Targeted Cell Types")
    '''
    
    if df_bulk is not None:
        # Calculate percentage of samples with TPM > 10 for each group
        tcga_pct = (df_bulk[df_bulk['Source'] == 'TCGA']['log2_exp'] > bulk_line).sum() / len(df_bulk[df_bulk['Source'] == 'TCGA']) * 100
        gtex_pct = (df_bulk[df_bulk['Source'] == 'GTEx']['log2_exp'] > bulk_line).sum() / len(df_bulk[df_bulk['Source'] == 'GTEx']) * 100
        
        # Calculate percent per TCGA indication
        tcga_df = df_bulk[df_bulk['Source'] == 'TCGA'].copy()
        tcga_pcts_per_type = {}
        for tcga_type in tcga_df['Type'].unique():
            subset = tcga_df[tcga_df['Type'] == tcga_type]
            pct = (subset['log2_exp'] > bulk_line).sum() / len(subset) * 100
            tcga_pcts_per_type[tcga_type] = pct
        
        # Calculate percent per GTEx tissue
        gtex_df = df_bulk[df_bulk['Source'] == 'GTEx'].copy()
        gtex_pcts_per_type = {}
        for gtex_type in gtex_df['Type'].unique():
            subset = gtex_df[gtex_df['Type'] == gtex_type]
            pct = (subset['log2_exp'] > bulk_line).sum() / len(subset) * 100
            gtex_pcts_per_type[gtex_type] = pct
        
        # Convert to R list format in the R code
        tcga_pcts_r = ", ".join([f'"{k}"={v:.1f}' for k, v in tcga_pcts_per_type.items()])
        gtex_pcts_r = ", ".join([f'"{k}"={v:.1f}' for k, v in gtex_pcts_per_type.items()])
        
        r_code += f'''
    # ========== PANEL 4: TCGA/GTEX BULK EXPRESSION (BOTTOM) ==========
    # Split TCGA and GTEx into separate dataframes
    df_tcga <- df_bulk[df_bulk$Source == "TCGA", ]
    df_gtex <- df_bulk[df_bulk$Source == "GTEx", ]
    
    # Order TCGA by median expression
    df_tcga$Type <- factor(df_tcga$Type, 
                           levels=names(sort(tapply(df_tcga$log2_exp, df_tcga$Type, median), decreasing=TRUE)))
    
    # Order GTEx by median expression
    df_gtex$Type <- factor(df_gtex$Type, 
                           levels=names(sort(tapply(df_gtex$log2_exp, df_gtex$Type, median), decreasing=TRUE)))
    
    # Add percent positive per indication to dataframes
    tcga_pcts <- list({tcga_pcts_r})
    gtex_pcts <- list({gtex_pcts_r})
    
    df_tcga$pct <- sapply(as.character(df_tcga$Type), function(x) as.numeric(tcga_pcts[[x]]))
    df_gtex$pct <- sapply(as.character(df_gtex$Type), function(x) as.numeric(gtex_pcts[[x]]))
    
    # Calculate overall percentages FIRST
    tcga_overall_pct <- round((sum(df_tcga$log2_exp > bulk_line) / nrow(df_tcga)) * 100, 1)
    gtex_overall_pct <- round((sum(df_gtex$log2_exp > bulk_line) / nrow(df_gtex)) * 100, 1)
    
    # Create display labels with percentages for x-axis
    tcga_type_labels <- unique(df_tcga[, c("Type", "pct")])
    tcga_type_labels$label <- paste0(as.character(tcga_type_labels$Type), "\n(", round(tcga_type_labels$pct, 1), "%)")
    tcga_type_labels <- tcga_type_labels[order(tcga_type_labels$Type), ]
    
    # Map the labels back to df_tcga
    df_tcga$Type_label <- tcga_type_labels$label[match(as.character(df_tcga$Type), as.character(tcga_type_labels$Type))]
    df_tcga$Type_label <- factor(df_tcga$Type_label, levels=tcga_type_labels$label)
    
    # TCGA plot (left half)
    p4_tcga <- ggplot(df_tcga, aes(x=Type_label, y=log2_exp)) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_jitter(size=0.8, stroke=0.2, pch=21, fill="firebrick3", color="black", height=0, width=0.2, alpha=0.5) +
        geom_boxplot(fill=NA, color="black", outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.6) +
        theme(legend.position="none") +
        coord_cartesian(ylim=bulk_limits) +
        geom_hline(yintercept=bulk_line, lty="dashed", color="firebrick3") +
        xlab("") + ylab("Log2(TPM + 1)") +
        theme(axis.text.x=element_text(size=5, angle=90)) +
        ggtitle(paste0("TCGA (", tcga_overall_pct, "%)"))
    
    # GTEx plot (right half)
    p4_gtex <- ggplot(df_gtex, aes(Type, log2_exp)) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_jitter(size=0.8, stroke=0.2, pch=21, fill="grey", color="black", height=0, width=0.2, alpha=0.5) +
        geom_boxplot(fill=NA, color="black", outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.6) +
        theme(legend.position="none") +
        coord_cartesian(ylim=bulk_limits) +
        geom_hline(yintercept=bulk_line, lty="dashed", color="firebrick3") +
        xlab("") + ylab("Log2(TPM + 1)") +
        theme(axis.text.x=element_text(size=5, angle=90)) +
        ggtitle(paste0("GTEx (", gtex_overall_pct, "%)"))
    
    # Top row: single cell plots
    top_row <- p1 + p2 + p3 + plot_layout(widths=c(1, 3, 3))
    
    # Bottom row: bulk plots
    bottom_row <- p4_tcga + p4_gtex + plot_layout(widths=c(1, 1))
    
    # Combine vertically with equal heights
    final_plot <- top_row / bottom_row + plot_layout(heights=c(1, 1))
    '''
    else:
        r_code += f'''
    # Top row: single cell plots only
    final_plot <- p1 + p2 + p3 + plot_layout(widths=c(1, 3, 3))
    '''
    
    r_code += f'''
    suppressWarnings(ggsave("{out_path}", final_plot, width={width}, height={height}, dpi={dpi}))
    '''
    
    # Execute R code
    r(r_code)
    
    print(f"✓ Saved: {out_path}")
    
    # Convert to PNG
    png_path = out_path.replace('.pdf', '.png')
    try:
        plot.pdf_to_png(pdf_path=out_path, dpi=dpi, output_path=png_path)
        print(f"✓ Made PNG: {png_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not convert to PNG: {e}")
    
    # Write summary file
    summary_file = os.path.join(out_dir, f"{gene}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Gene: {gene}\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Positivity in patients: {per_pos}% ({len(pos_patient_unique)}/{len(malig_med_adata.obs_names)})\n")
        f.write(f"Median tumor expression (positive): {med_pos:.2f}\n")
        f.write(f"Number of reference cell types: {len(dfR)}\n")
    
    elapsed = (time.time() - overall_start) / 60
    print(f"\n{'='*60}")
    print(f"✓ Plot complete: {gene}")
    print(f"  Completed in {elapsed:.2f} minutes")
    print(f"{'='*60}")







sections = {
    'Target Metrics': ['On_Val_25' : "On_Expr_25th", 'On_Val_50' : 'On_Expr_50th', 'On_Val_75' : 'On_Expr_75th', 'Target_Val' : 'Top_Patient_Expr', 'SC_2nd_Target_Val' : '2nd_Top_Patient_Expr'],
    
    'Specificity & Off-Targets': ['Specificity', 'Corrected_Specificity', 'Top_Off_Target_Val', 
                                   'Corrected_Top_Off_Target_Val', 'Log2_Fold_Change', 'Corrected_Log2_Fold_Change',
                                   'N_Off_Targets', 'N_Off_Targets_0.01', 'N_Off_Targets_0.05', 'N_Off_Targets_0.1',
                                   'N_Off_Targets_0.25', 'N_Off_Targets_0.5', 'N_Off_Targets_1.0'],

    'On-Target Distribution': ['On_Val_25', 'On_Val_50', 'On_Val_75'],
    
    'Positive Cells': ['N_Pos_Val_0.5', 'P_Pos_Val_0.5', 'P_Pos_Val_0.1', 'N_Pos_Specific', 'P_Pos_Specific'],
    
    'Tissue Toxicity': ['Tox_Brain', 'TI_Brain', 'Tox_Heart', 'TI_Heart', 'Tox_Lung', 'TI_Lung', 
                       'Tox_Immune', 'TI_Immune', 'Tox_NonImmune', 'TI_NonImmune'],
    
    'GTEx & Quality': ['GTEX_Tox_Tier1', 'GTEX_Tox_Tier2', 'GTEX_Tox_Tier3', 'TargetQ_Final_v1', 'TargetQ_Final_v2', 
                       'Hazard_SC_v1', 'Hazard_FFPE_v1', 'Hazard_GTEX_v1'],
    
    'Summary': ['TI', 'Positive_Final_v2', 'gene_name', 'claude_surface', 'claude_not_surface', 'claude_summary']
}


t1 = summary[[]]



"""
Single gene axial summary plot visualization across datasets.
Comprehensive view of target expression in malignant vs healthy tissues.
"""

__all__ = ['axial_summary']

import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import os
import time
from typing import Optional
from py_target_id import plot
from py_target_id import utils
from rpy2.robjects import r
from datetime import datetime


def axial_summary(
    gene: str,
    malig_adata,
    malig_med_adata,
    ref_adata,
    gtex_adata=None,
    tcga_adata=None,
    out_dir: str = "single/axial_summary",
    show: int = 15,
    width: float = 20,
    height: float = 8,
    dpi: int = 300,
    title: Optional[str] = None
):
    """
    Create comprehensive axial summary plot for a single gene target.
    
    This creates a 4-panel plot showing:
    - Panel 1: Malignant tumor expression (Positive vs Negative patients)
    - Panel 2: Reference tissues grouped by tissue type
    - Panel 3: Top off-target cell types
    - Panel 4: GTEx/TCGA bulk expression
    
    Parameters
    ----------
    gene : str
        Gene target name
    malig_adata : AnnData
        Malignant cell data with patient annotations
    malig_med_adata : AnnData
        Malignant median data (patients × genes)
        Must have layers['positivity'] with binary positivity data
    ref_adata : AnnData
        Healthy reference atlas with CellType annotations
    gtex_adata : AnnData, optional
        GTEx expression data (samples × genes) with obs['GTEX']
    tcga_adata : AnnData, optional
        TCGA expression data (samples × genes)
    out_dir : str
        Output directory for plots
    show : int
        Number of top off-target cell types to display
    width : float
        Plot width in inches
    height : float
        Plot height in inches
    dpi : int
        Resolution for output
    title : str, optional
        Custom title (defaults to gene name)
    """
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    if title is None:
        title = gene
    
    overall_start = time.time()
    
    print(f"Processing gene: {gene}")
    print(f"Output directory: {out_dir}")
    
    # Get cell types BEFORE materializing
    CT = ref_adata.obs['CellType'].values
    CT = pd.Series(CT).str.replace('Î±', 'a').str.replace('Î²', 'B').values
    
    # Load malignant data
    print("Loading malignant data...")
    malig_subset = malig_adata[:, [gene]]
    
    if hasattr(malig_subset, 'to_memory'):
        print("Materializing malignant subset...")
        malig_subset = malig_subset.to_memory()
    
    if sparse.issparse(malig_subset.X):
        malig_subset.X = malig_subset.X.toarray()
    
    malig_exp = malig_subset.X.ravel()
    patients = malig_subset.obs['Patient'].values
    
    # Load healthy atlas
    print("Loading healthy atlas...")
    ref_subset = ref_adata[:, [gene]]
    
    if hasattr(ref_subset, 'to_memory'):
        print("Materializing reference subset...")
        ref_subset = ref_subset.to_memory()
    
    if sparse.issparse(ref_subset.X):
        ref_subset.X = ref_subset.X.toarray()
    
    ref_exp = ref_subset.X.ravel()
    
    # Compute medians
    print("Computing medians...")
    malig_med = utils.summarize_matrix(
        malig_exp.reshape(-1, 1), patients, axis=0, metric="median", verbose=False
    )
    malig_med = malig_med.iloc[:, 0]
    
    ha_med = utils.summarize_matrix(
        ref_exp.reshape(-1, 1), CT, axis=0, metric="median", verbose=False
    )
    ha_med = ha_med.iloc[:, 0]
    
    # Get positivity data
    print("Extracting positivity data...")
    pos_data = malig_med_adata[:, gene].layers['positivity'].flatten()
    pos_patients = malig_med_adata.obs['Patient'].values[pos_data.astype(bool)]
    pos_patient_unique = np.unique(pos_patients)
    per_pos = round(100 * len(pos_patient_unique) / len(malig_med_adata.obs_names), 1)
    
    # Load GTEx and TCGA if provided
    gtex_data = None
    tcga_data = None
    
    if gtex_adata is not None:
        print("Loading GTEx data...")
        gtex_subset = gtex_adata[:, [gene]]
        if hasattr(gtex_subset, 'to_memory'):
            gtex_subset = gtex_subset.to_memory()
        gtex_mat = gtex_subset.X.toarray().ravel() if sparse.issparse(gtex_subset.X) else gtex_subset.X.ravel()
        gtex_id = gtex_subset.obs['GTEX'].values if 'GTEX' in gtex_subset.obs else gtex_subset.obs_names.values
        gtex_data = {'values': gtex_mat, 'id': gtex_id}
    
    if tcga_adata is not None:
        print("Loading TCGA data...")
        tcga_subset = tcga_adata[:, [gene]]
        if hasattr(tcga_subset, 'to_memory'):
            tcga_subset = tcga_subset.to_memory()
        tcga_mat = tcga_subset.X.toarray().ravel() if sparse.issparse(tcga_subset.X) else tcga_subset.X.ravel()
        # Use TCGA cancer type from obs if available, otherwise parse from index
        tcga_type = tcga_subset.obs['TCGA'].values if 'TCGA' in tcga_subset.obs else np.array([s.split('#')[0] for s in tcga_subset.obs_names.values])
        tcga_data = {'values': tcga_mat, 'type': tcga_type}
    
    # Prepare malignant dataframe
    dfT = pd.DataFrame({
        'log2_exp': np.log2(malig_med.values + 1),
        'ID': malig_med.index,
        'Positive': [p in pos_patient_unique for p in malig_med.index]
    })
    
    n_pos = (dfT['Positive']).sum()
    n_neg = (~dfT['Positive']).sum()
    
    # Prepare reference dataframe - sorted by expression
    dfR = pd.DataFrame({
        'log2_exp': np.log2(ha_med.values + 1),
        'CellType': ha_med.index,
        'Tissue': [ct.split(':')[0] for ct in ha_med.index]
    })
    
    dfR = dfR.sort_values('log2_exp', ascending=False).reset_index(drop=True)
    dfR['ID_display'] = [plot.add_smart_newlines(ct, max_length=30) for ct in dfR['CellType']]
    
    # Prepare bulk data if available
    df_bulk = None
    if tcga_data is not None and gtex_data is not None:
        df_bulk = pd.DataFrame({
            'log2_exp': np.concatenate([
                np.log2(tcga_data['values'] + 1),
                np.log2(gtex_data['values'] + 1)
            ]),
            'Type': np.concatenate([
                tcga_data['type'],
                gtex_data['id']
            ]),
            'Source': ['TCGA'] * len(tcga_data['values']) + ['GTEx'] * len(gtex_data['values'])
        })
    
    # Create output path
    out_path = os.path.join(out_dir, f"{gene}_summary.pdf")
    
    # Send data to R
    print("Sending data to R...")
    plot.pd2r("dfT", dfT)
    plot.pd2r("dfR", dfR)
    if df_bulk is not None:
        plot.pd2r("df_bulk", df_bulk)
    
    # Get limits from data
    max_exp = max(dfT['log2_exp'].max(), dfR['log2_exp'].max())
    limits = [0, max_exp]
    med_pos = dfT[dfT['Positive']]['log2_exp'].median()
    
    # For bulk data, use independent limits based on bulk data
    bulk_limit_tpm = 10
    bulk_line = np.log2(bulk_limit_tpm + 1)
    
    bulk_max = df_bulk['log2_exp'].max()
    bulk_limits = [0, bulk_max * 1.05]
    
    print(f"DEBUG: limits = {limits}, bulk_limits = {bulk_limits}, bulk_line = {bulk_line}")
    
    # Create plots in R
    print("Creating plots in R...")
    
    r_code = f'''
    library(ggplot2)
    library(patchwork)
    library(scales)
    
    # Parameters
    show <- {show}
    size <- 6
    limits <- c({limits[0]}, {limits[1]})
    med_pos <- {med_pos}
    n_pos <- {n_pos}
    n_neg <- {n_neg}
    per_pos <- {per_pos}
    
    # ========== PANEL 1: MALIGNANT EXPRESSION ==========
    dfT$pos_status <- ifelse(dfT$Positive, "Malig Pos.", "Malig Neg.")
    dfT$pos_status <- factor(dfT$pos_status, levels = c("Malig Pos.", "Malig Neg."))
    
    p1 <- ggplot(dfT, aes(pos_status, log2_exp)) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
        geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=pos_status), color="black", height=0, width=0.2) +
        theme(legend.position="none") +
        coord_cartesian(ylim=limits) +
        scale_x_discrete(
            drop=FALSE,
            labels=c("Malig Pos."=paste0("Malig Pos.\\n(N=", n_pos, ")"),
                    "Malig Neg."=paste0("Malig Neg.\\n(N=", n_neg, ")"))
        ) +
        xlab("") +
        geom_hline(yintercept=med_pos, lty="dashed", color="firebrick3") +
        scale_fill_manual(values=c("Malig Pos."="#28154C", "Malig Neg."="lightgrey")) +
        ylab("{gene}") +
        ggtitle(paste0(per_pos, "%"))
    
    # ========== PANEL 2: REFERENCE BY TISSUE ==========
    dfR$tissue_factor <- factor(dfR$Tissue, levels=unique(dfR$Tissue[order(dfR$log2_exp, decreasing=TRUE)]))
    
    p2 <- ggplot(dfR, aes(tissue_factor, squish(log2_exp, limits))) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
        geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=Tissue), color="black", height=0, width=0.2) +
        theme(legend.position="none") +
        coord_cartesian(ylim=limits) +
        geom_hline(yintercept=med_pos, lty="dashed", color="firebrick3") +
        xlab("") + ylab("") +
        theme(axis.text.x=element_text(size=size)) +
        scale_fill_manual(values=suppressMessages(create_pal_d(dfR$Tissue))) +
        ggtitle("Facet By Tissue")
    
    # ========== PANEL 3: TOP OFF-TARGET CELL TYPES ==========
    dfR_top <- head(dfR, show)
    dfR_top$ID_display <- factor(dfR_top$ID_display, levels=dfR_top$ID_display)
    
    p3 <- ggplot(dfR_top, aes(ID_display, squish(log2_exp, limits))) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_point(size=1.5, stroke=0.35, pch=21, aes(fill=Tissue), color="black") +
        theme(legend.position="none") +
        coord_cartesian(ylim=limits) +
        geom_hline(yintercept=med_pos, lty="dashed", color="firebrick3") +
        xlab("") + ylab("") +
        theme(axis.text.x=element_text(size=size*0.75)) +
        scale_fill_manual(values=suppressMessages(create_pal_d(dfR_top$Tissue))) +
        ggtitle("Top Off Targeted Cell Types")
    '''
    
    # Calculate percentage of samples with TPM > 10 for each group
    tcga_pct = (df_bulk[df_bulk['Source'] == 'TCGA']['log2_exp'] > bulk_line).sum() / len(df_bulk[df_bulk['Source'] == 'TCGA']) * 100
    gtex_pct = (df_bulk[df_bulk['Source'] == 'GTEx']['log2_exp'] > bulk_line).sum() / len(df_bulk[df_bulk['Source'] == 'GTEx']) * 100
    
    r_code += f'''
    # ========== PANEL 4: TCGA/GTEX BULK EXPRESSION (BOTTOM) ==========
    # Split TCGA and GTEx into separate dataframes
    df_tcga <- df_bulk[df_bulk$Source == "TCGA", ]
    df_gtex <- df_bulk[df_bulk$Source == "GTEx", ]
    
    # Order TCGA by median expression
    df_tcga$Type <- factor(df_tcga$Type, 
                           levels=names(sort(tapply(df_tcga$log2_exp, df_tcga$Type, median), decreasing=TRUE)))
    
    # Order GTEx by median expression
    df_gtex$Type <- factor(df_gtex$Type, 
                           levels=names(sort(tapply(df_gtex$log2_exp, df_gtex$Type, median), decreasing=TRUE)))
    
    # TCGA plot (left half)
    p4_tcga <- ggplot(df_tcga, aes(Type, log2_exp)) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_boxplot(fill="firebrick3", outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.6, alpha=0.7) +
        geom_jitter(size=0.8, stroke=0.2, pch=21, fill="firebrick3", color="black", height=0, width=0.2, alpha=0.5) +
        theme(legend.position="none") +
        coord_cartesian(ylim=bulk_limits) +
        geom_hline(yintercept=bulk_line, lty="dashed", color="firebrick3") +
        xlab("") + ylab("") +
        theme(axis.text.x=element_text(size=5, angle=90)) +
        ggtitle("TCGA ({tcga_pct:.1f}%)")
    
    # GTEx plot (right half)
    p4_gtex <- ggplot(df_gtex, aes(Type, log2_exp)) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_boxplot(fill="grey", outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.6, alpha=0.7) +
        geom_jitter(size=0.8, stroke=0.2, pch=21, fill="grey", color="black", height=0, width=0.2, alpha=0.5) +
        theme(legend.position="none") +
        coord_cartesian(ylim=bulk_limits) +
        geom_hline(yintercept=bulk_line, lty="dashed", color="firebrick3") +
        xlab("") + ylab("") +
        theme(axis.text.x=element_text(size=5, angle=90)) +
        ggtitle("GTEx ({gtex_pct:.1f}%)")
    
    # Top row: single cell plots
    top_row <- p1 + p2 + p3 + plot_layout(widths=c(1, 3, 3))
    
    # Bottom row: bulk plots
    bottom_row <- p4_tcga + p4_gtex + plot_layout(widths=c(1, 1))
    
    # Combine vertically with bulk taking up less space
    final_plot <- top_row / bottom_row + plot_layout(heights=c(2, 1))
    '''
    
    r_code += f'''
    suppressWarnings(ggsave("{out_path}", final_plot, width={width}, height={height}, dpi={dpi}))
    '''
    
    # Execute R code
    r(r_code)
    
    print(f"✓ Saved: {out_path}")
    
    # Convert to PNG
    png_path = out_path.replace('.pdf', '.png')
    try:
        plot.pdf_to_png(pdf_path=out_path, dpi=dpi, output_path=png_path)
        print(f"✓ Made PNG: {png_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not convert to PNG: {e}")
    
    # Write summary file
    summary_file = os.path.join(out_dir, f"{gene}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Gene: {gene}\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Positivity in patients: {per_pos}% ({len(pos_patient_unique)}/{len(malig_med_adata.obs_names)})\n")
        f.write(f"Median tumor expression (positive): {med_pos:.2f}\n")
        f.write(f"Number of reference cell types: {len(dfR)}\n")
    
    elapsed = (time.time() - overall_start) / 60
    print(f"\n{'='*60}")
    print(f"✓ Plot complete: {gene}")
    print(f"  Completed in {elapsed:.2f} minutes")
    print(f"{'='*60}")




"""
Single gene axial summary plot visualization across datasets.
Comprehensive view of target expression in malignant vs healthy tissues.
"""

__all__ = ['axial_summary']

import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import os
import time
from typing import Optional
from py_target_id import plot
from py_target_id import utils
from rpy2.robjects import r
from datetime import datetime


def axial_summary(
    gene: str,
    malig_adata,
    malig_med_adata,
    ref_adata,
    gtex_adata=None,
    tcga_adata=None,
    out_dir: str = "single/axial_summary",
    show: int = 15,
    width: float = 20,
    height: float = 8,
    dpi: int = 300,
    title: Optional[str] = None
):
    """
    Create comprehensive axial summary plot for a single gene target.
    
    This creates a 4-panel plot showing:
    - Panel 1: Malignant tumor expression (Positive vs Negative patients)
    - Panel 2: Reference tissues grouped by tissue type
    - Panel 3: Top off-target cell types
    - Panel 4: GTEx/TCGA bulk expression
    
    Parameters
    ----------
    gene : str
        Gene target name
    malig_adata : AnnData
        Malignant cell data with patient annotations
    malig_med_adata : AnnData
        Malignant median data (patients × genes)
        Must have layers['positivity'] with binary positivity data
    ref_adata : AnnData
        Healthy reference atlas with CellType annotations
    gtex_adata : AnnData, optional
        GTEx expression data (samples × genes) with obs['GTEX']
    tcga_adata : AnnData, optional
        TCGA expression data (samples × genes)
    out_dir : str
        Output directory for plots
    show : int
        Number of top off-target cell types to display
    width : float
        Plot width in inches
    height : float
        Plot height in inches
    dpi : int
        Resolution for output
    title : str, optional
        Custom title (defaults to gene name)
    """
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    if title is None:
        title = gene
    
    overall_start = time.time()
    
    print(f"Processing gene: {gene}")
    print(f"Output directory: {out_dir}")
    
    # Get cell types BEFORE materializing
    CT = ref_adata.obs['CellType'].values
    CT = pd.Series(CT).str.replace('Î±', 'a').str.replace('Î²', 'B').values
    
    # Load malignant data
    print("Loading malignant data...")
    malig_subset = malig_adata[:, [gene]]
    
    if hasattr(malig_subset, 'to_memory'):
        print("Materializing malignant subset...")
        malig_subset = malig_subset.to_memory()
    
    if sparse.issparse(malig_subset.X):
        malig_subset.X = malig_subset.X.toarray()
    
    malig_exp = malig_subset.X.ravel()
    patients = malig_subset.obs['Patient'].values
    
    # Load healthy atlas
    print("Loading healthy atlas...")
    ref_subset = ref_adata[:, [gene]]
    
    if hasattr(ref_subset, 'to_memory'):
        print("Materializing reference subset...")
        ref_subset = ref_subset.to_memory()
    
    if sparse.issparse(ref_subset.X):
        ref_subset.X = ref_subset.X.toarray()
    
    ref_exp = ref_subset.X.ravel()
    
    # Compute medians
    print("Computing medians...")
    malig_med = utils.summarize_matrix(
        malig_exp.reshape(-1, 1), patients, axis=0, metric="median", verbose=False
    )
    malig_med = malig_med.iloc[:, 0]
    
    ha_med = utils.summarize_matrix(
        ref_exp.reshape(-1, 1), CT, axis=0, metric="median", verbose=False
    )
    ha_med = ha_med.iloc[:, 0]
    
    # Get positivity data
    print("Extracting positivity data...")
    pos_data = malig_med_adata[:, gene].layers['positivity'].flatten()
    pos_patients = malig_med_adata.obs['Patient'].values[pos_data.astype(bool)]
    pos_patient_unique = np.unique(pos_patients)
    per_pos = round(100 * len(pos_patient_unique) / len(malig_med_adata.obs_names), 1)
    
    # Load GTEx and TCGA if provided
    gtex_data = None
    tcga_data = None
    
    if gtex_adata is not None:
        print("Loading GTEx data...")
        gtex_subset = gtex_adata[:, [gene]]
        if hasattr(gtex_subset, 'to_memory'):
            gtex_subset = gtex_subset.to_memory()
        gtex_mat = gtex_subset.X.toarray().ravel() if sparse.issparse(gtex_subset.X) else gtex_subset.X.ravel()
        gtex_id = gtex_subset.obs['GTEX'].values if 'GTEX' in gtex_subset.obs else gtex_subset.obs_names.values
        gtex_data = {'values': gtex_mat, 'id': gtex_id}
    
    if tcga_adata is not None:
        print("Loading TCGA data...")
        tcga_subset = tcga_adata[:, [gene]]
        if hasattr(tcga_subset, 'to_memory'):
            tcga_subset = tcga_subset.to_memory()
        tcga_mat = tcga_subset.X.toarray().ravel() if sparse.issparse(tcga_subset.X) else tcga_subset.X.ravel()
        # Use TCGA cancer type from obs if available, otherwise parse from index
        tcga_type = tcga_subset.obs['TCGA'].values if 'TCGA' in tcga_subset.obs else np.array([s.split('#')[0] for s in tcga_subset.obs_names.values])
        tcga_data = {'values': tcga_mat, 'type': tcga_type}
    
    # Prepare malignant dataframe
    dfT = pd.DataFrame({
        'log2_exp': np.log2(malig_med.values + 1),
        'ID': malig_med.index,
        'Positive': [p in pos_patient_unique for p in malig_med.index]
    })
    
    n_pos = (dfT['Positive']).sum()
    n_neg = (~dfT['Positive']).sum()
    
    # Prepare reference dataframe - sorted by expression
    dfR = pd.DataFrame({
        'log2_exp': np.log2(ha_med.values + 1),
        'CellType': ha_med.index,
        'Tissue': [ct.split(':')[0] for ct in ha_med.index]
    })
    
    dfR = dfR.sort_values('log2_exp', ascending=False).reset_index(drop=True)
    dfR['ID_display'] = [plot.add_smart_newlines(ct, max_length=30) for ct in dfR['CellType']]
    
    # Prepare bulk data if available
    df_bulk = None
    if tcga_data is not None and gtex_data is not None:
        df_bulk = pd.DataFrame({
            'log2_exp': np.concatenate([
                np.log2(tcga_data['values'] + 1),
                np.log2(gtex_data['values'] + 1)
            ]),
            'Type': np.concatenate([
                tcga_data['type'],
                gtex_data['id']
            ]),
            'Source': ['TCGA'] * len(tcga_data['values']) + ['GTEx'] * len(gtex_data['values'])
        })
    
    # Create output path
    out_path = os.path.join(out_dir, f"{gene}_summary.pdf")
    
    # Send data to R
    print("Sending data to R...")
    plot.pd2r("dfT", dfT)
    plot.pd2r("dfR", dfR)
    if df_bulk is not None:
        plot.pd2r("df_bulk", df_bulk)
    
    # Get limits from data
    max_exp = max(dfT['log2_exp'].max(), dfR['log2_exp'].max())
    limits = [0, max_exp]
    med_pos = dfT[dfT['Positive']]['log2_exp'].median()
    
    # For bulk data, use independent limits based on bulk data
    bulk_limit_tpm = 10
    bulk_line = np.log2(bulk_limit_tpm + 1)
    
    bulk_max = df_bulk['log2_exp'].max()
    bulk_limits = [0, bulk_max * 1.05]
    
    print(f"DEBUG: limits = {limits}, bulk_limits = {bulk_limits}, bulk_line = {bulk_line}")
    
    # Create plots in R
    print("Creating plots in R...")
    
    r_code = f'''
    library(ggplot2)
    library(patchwork)
    library(scales)
    
    # Parameters
    show <- {show}
    size <- 6
    limits <- c({limits[0]}, {limits[1]})
    med_pos <- {med_pos}
    n_pos <- {n_pos}
    n_neg <- {n_neg}
    per_pos <- {per_pos}
    
    # ========== PANEL 1: MALIGNANT EXPRESSION ==========
    dfT$pos_status <- ifelse(dfT$Positive, "Malig Pos.", "Malig Neg.")
    dfT$pos_status <- factor(dfT$pos_status, levels = c("Malig Pos.", "Malig Neg."))
    
    p1 <- ggplot(dfT, aes(pos_status, log2_exp)) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
        geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=pos_status), color="black", height=0, width=0.2) +
        theme(legend.position="none") +
        coord_cartesian(ylim=limits) +
        scale_x_discrete(
            drop=FALSE,
            labels=c("Malig Pos."=paste0("Malig Pos.\\n(N=", n_pos, ")"),
                    "Malig Neg."=paste0("Malig Neg.\\n(N=", n_neg, ")"))
        ) +
        xlab("") +
        geom_hline(yintercept=med_pos, lty="dashed", color="firebrick3") +
        scale_fill_manual(values=c("Malig Pos."="#28154C", "Malig Neg."="lightgrey")) +
        ylab("{gene}") +
        ggtitle(paste0(per_pos, "%"))
    
    # ========== PANEL 2: REFERENCE BY TISSUE ==========
    dfR$tissue_factor <- factor(dfR$Tissue, levels=unique(dfR$Tissue[order(dfR$log2_exp, decreasing=TRUE)]))
    
    p2 <- ggplot(dfR, aes(tissue_factor, squish(log2_exp, limits))) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
        geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=Tissue), color="black", height=0, width=0.2) +
        theme(legend.position="none") +
        coord_cartesian(ylim=limits) +
        geom_hline(yintercept=med_pos, lty="dashed", color="firebrick3") +
        xlab("") + ylab("") +
        theme(axis.text.x=element_text(size=size)) +
        scale_fill_manual(values=suppressMessages(create_pal_d(dfR$Tissue))) +
        ggtitle("Facet By Tissue")
    
    # ========== PANEL 3: TOP OFF-TARGET CELL TYPES ==========
    dfR_top <- head(dfR, show)
    dfR_top$ID_display <- factor(dfR_top$ID_display, levels=dfR_top$ID_display)
    
    p3 <- ggplot(dfR_top, aes(ID_display, squish(log2_exp, limits))) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_point(size=1.5, stroke=0.35, pch=21, aes(fill=Tissue), color="black") +
        theme(legend.position="none") +
        coord_cartesian(ylim=limits) +
        geom_hline(yintercept=med_pos, lty="dashed", color="firebrick3") +
        xlab("") + ylab("") +
        theme(axis.text.x=element_text(size=size*0.75)) +
        scale_fill_manual(values=suppressMessages(create_pal_d(dfR_top$Tissue))) +
        ggtitle("Top Off Targeted Cell Types")
    '''
    
    # Calculate percentage of samples with TPM > 10 for each group
    tcga_pct = (df_bulk[df_bulk['Source'] == 'TCGA']['log2_exp'] > bulk_line).sum() / len(df_bulk[df_bulk['Source'] == 'TCGA']) * 100
    gtex_pct = (df_bulk[df_bulk['Source'] == 'GTEx']['log2_exp'] > bulk_line).sum() / len(df_bulk[df_bulk['Source'] == 'GTEx']) * 100
    
    r_code += f'''
    # ========== PANEL 4: TCGA/GTEX BULK EXPRESSION (BOTTOM) ==========
    # Split TCGA and GTEx into separate dataframes
    df_tcga <- df_bulk[df_bulk$Source == "TCGA", ]
    df_gtex <- df_bulk[df_bulk$Source == "GTEx", ]
    
    # Order TCGA by median expression
    df_tcga$Type <- factor(df_tcga$Type, 
                           levels=names(sort(tapply(df_tcga$log2_exp, df_tcga$Type, median), decreasing=TRUE)))
    
    # Order GTEx by median expression
    df_gtex$Type <- factor(df_gtex$Type, 
                           levels=names(sort(tapply(df_gtex$log2_exp, df_gtex$Type, median), decreasing=TRUE)))
    
    # TCGA plot (left half)
    p4_tcga <- ggplot(df_tcga, aes(Type, log2_exp)) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_boxplot(fill="firebrick3", outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.6, alpha=0.7) +
        geom_jitter(size=0.8, stroke=0.2, pch=21, fill="firebrick3", color="black", height=0, width=0.2, alpha=0.5) +
        theme(legend.position="none") +
        coord_cartesian(ylim={bulk_limits}) +
        geom_hline(yintercept=bulk_line, lty="dashed", color="firebrick3") +
        xlab("") + ylab("") +
        theme(axis.text.x=element_text(size=5, angle=90)) +
        ggtitle("TCGA ({tcga_pct:.1f}%)")
    
    # GTEx plot (right half)
    p4_gtex <- ggplot(df_gtex, aes(Type, log2_exp)) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_boxplot(fill="grey", outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.6, alpha=0.7) +
        geom_jitter(size=0.8, stroke=0.2, pch=21, fill="grey", color="black", height=0, width=0.2, alpha=0.5) +
        theme(legend.position="none") +
        coord_cartesian(ylim=bulk_limits) +
        geom_hline(yintercept=bulk_line, lty="dashed", color="firebrick3") +
        xlab("") + ylab("") +
        theme(axis.text.x=element_text(size=5, angle=90)) +
        ggtitle("GTEx ({gtex_pct:.1f}%)")
    
    # Top row: single cell plots
    top_row <- p1 + p2 + p3 + plot_layout(widths=c(1, 3, 3))
    
    # Bottom row: bulk plots
    bottom_row <- p4_tcga + p4_gtex + plot_layout(widths=c(1, 1))
    
    # Combine vertically with bulk taking up less space
    final_plot <- top_row / bottom_row + plot_layout(heights=c(2, 1))
    '''
    
    r_code += f'''
    suppressWarnings(ggsave("{out_path}", final_plot, width={width}, height={height}, dpi={dpi}))
    '''
    
    # Execute R code
    r(r_code)
    
    print(f"✓ Saved: {out_path}")
    
    # Convert to PNG
    png_path = out_path.replace('.pdf', '.png')
    try:
        plot.pdf_to_png(pdf_path=out_path, dpi=dpi, output_path=png_path)
        print(f"✓ Made PNG: {png_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not convert to PNG: {e}")
    
    # Write summary file
    summary_file = os.path.join(out_dir, f"{gene}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Gene: {gene}\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Positivity in patients: {per_pos}% ({len(pos_patient_unique)}/{len(malig_med_adata.obs_names)})\n")
        f.write(f"Median tumor expression (positive): {med_pos:.2f}\n")
        f.write(f"Number of reference cell types: {len(dfR)}\n")
    
    elapsed = (time.time() - overall_start) / 60
    print(f"\n{'='*60}")
    print(f"✓ Plot complete: {gene}")
    print(f"  Completed in {elapsed:.2f} minutes")
    print(f"{'='*60}")


"""
Single gene axial summary plot visualization across datasets.
Comprehensive view of target expression in malignant vs healthy tissues.
"""

__all__ = ['axial_summary']

import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import os
import time
from typing import Optional
from py_target_id import plot
from py_target_id import utils
from rpy2.robjects import r
from datetime import datetime


def axial_summary(
    gene: str,
    malig_adata,
    malig_med_adata,
    ref_adata,
    gtex_adata=None,
    tcga_adata=None,
    out_dir: str = "single/axial_summary",
    show: int = 15,
    width: float = 20,
    height: float = 8,
    dpi: int = 300,
    title: Optional[str] = None
):
    """
    Create comprehensive axial summary plot for a single gene target.
    
    This creates a 4-panel plot showing:
    - Panel 1: Malignant tumor expression (Positive vs Negative patients)
    - Panel 2: Reference tissues grouped by tissue type
    - Panel 3: Top off-target cell types
    - Panel 4: GTEx/TCGA bulk expression
    
    Parameters
    ----------
    gene : str
        Gene target name
    malig_adata : AnnData
        Malignant cell data with patient annotations
    malig_med_adata : AnnData
        Malignant median data (patients × genes)
        Must have layers['positivity'] with binary positivity data
    ref_adata : AnnData
        Healthy reference atlas with CellType annotations
    gtex_adata : AnnData, optional
        GTEx expression data (samples × genes) with obs['GTEX']
    tcga_adata : AnnData, optional
        TCGA expression data (samples × genes)
    out_dir : str
        Output directory for plots
    show : int
        Number of top off-target cell types to display
    width : float
        Plot width in inches
    height : float
        Plot height in inches
    dpi : int
        Resolution for output
    title : str, optional
        Custom title (defaults to gene name)
    """
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    if title is None:
        title = gene
    
    overall_start = time.time()
    
    print(f"Processing gene: {gene}")
    print(f"Output directory: {out_dir}")
    
    # Get cell types BEFORE materializing
    CT = ref_adata.obs['CellType'].values
    CT = pd.Series(CT).str.replace('Î±', 'a').str.replace('Î²', 'B').values
    
    # Load malignant data
    print("Loading malignant data...")
    malig_subset = malig_adata[:, [gene]]
    
    if hasattr(malig_subset, 'to_memory'):
        print("Materializing malignant subset...")
        malig_subset = malig_subset.to_memory()
    
    if sparse.issparse(malig_subset.X):
        malig_subset.X = malig_subset.X.toarray()
    
    malig_exp = malig_subset.X.ravel()
    patients = malig_subset.obs['Patient'].values
    
    # Load healthy atlas
    print("Loading healthy atlas...")
    ref_subset = ref_adata[:, [gene]]
    
    if hasattr(ref_subset, 'to_memory'):
        print("Materializing reference subset...")
        ref_subset = ref_subset.to_memory()
    
    if sparse.issparse(ref_subset.X):
        ref_subset.X = ref_subset.X.toarray()
    
    ref_exp = ref_subset.X.ravel()
    
    # Compute medians
    print("Computing medians...")
    malig_med = utils.summarize_matrix(
        malig_exp.reshape(-1, 1), patients, axis=0, metric="median", verbose=False
    )
    malig_med = malig_med.iloc[:, 0]
    
    ha_med = utils.summarize_matrix(
        ref_exp.reshape(-1, 1), CT, axis=0, metric="median", verbose=False
    )
    ha_med = ha_med.iloc[:, 0]
    
    # Get positivity data
    print("Extracting positivity data...")
    pos_data = malig_med_adata[:, gene].layers['positivity'].flatten()
    pos_patients = malig_med_adata.obs['Patient'].values[pos_data.astype(bool)]
    pos_patient_unique = np.unique(pos_patients)
    per_pos = round(100 * len(pos_patient_unique) / len(malig_med_adata.obs_names), 1)
    
    # Load GTEx and TCGA if provided
    gtex_data = None
    tcga_data = None
    
    if gtex_adata is not None:
        print("Loading GTEx data...")
        gtex_subset = gtex_adata[:, [gene]]
        if hasattr(gtex_subset, 'to_memory'):
            gtex_subset = gtex_subset.to_memory()
        gtex_mat = gtex_subset.X.toarray().ravel() if sparse.issparse(gtex_subset.X) else gtex_subset.X.ravel()
        gtex_id = gtex_subset.obs['GTEX'].values if 'GTEX' in gtex_subset.obs else gtex_subset.obs_names.values
        gtex_data = {'values': gtex_mat, 'id': gtex_id}
    
    if tcga_adata is not None:
        print("Loading TCGA data...")
        tcga_subset = tcga_adata[:, [gene]]
        if hasattr(tcga_subset, 'to_memory'):
            tcga_subset = tcga_subset.to_memory()
        tcga_mat = tcga_subset.X.toarray().ravel() if sparse.issparse(tcga_subset.X) else tcga_subset.X.ravel()
        # Use TCGA cancer type from obs if available, otherwise parse from index
        tcga_type = tcga_subset.obs['TCGA'].values if 'TCGA' in tcga_subset.obs else np.array([s.split('#')[0] for s in tcga_subset.obs_names.values])
        tcga_data = {'values': tcga_mat, 'type': tcga_type}
    
    # Prepare malignant dataframe
    dfT = pd.DataFrame({
        'log2_exp': np.log2(malig_med.values + 1),
        'ID': malig_med.index,
        'Positive': [p in pos_patient_unique for p in malig_med.index]
    })
    
    n_pos = (dfT['Positive']).sum()
    n_neg = (~dfT['Positive']).sum()
    
    # Prepare reference dataframe - sorted by expression
    dfR = pd.DataFrame({
        'log2_exp': np.log2(ha_med.values + 1),
        'CellType': ha_med.index,
        'Tissue': [ct.split(':')[0] for ct in ha_med.index]
    })
    
    dfR = dfR.sort_values('log2_exp', ascending=False).reset_index(drop=True)
    dfR['ID_display'] = [plot.add_smart_newlines(ct, max_length=30) for ct in dfR['CellType']]
    
    # Prepare bulk data if available
    df_bulk = None
    if tcga_data is not None and gtex_data is not None:
        df_bulk = pd.DataFrame({
            'log2_exp': np.concatenate([
                np.log2(tcga_data['values'] + 1),
                np.log2(gtex_data['values'] + 1)
            ]),
            'Type': np.concatenate([
                tcga_data['type'],
                gtex_data['id']
            ]),
            'Source': ['TCGA'] * len(tcga_data['values']) + ['GTEx'] * len(gtex_data['values'])
        })
    
    # Create output path
    out_path = os.path.join(out_dir, f"{gene}_summary.pdf")
    
    # Send data to R
    print("Sending data to R...")
    plot.pd2r("dfT", dfT)
    plot.pd2r("dfR", dfR)
    if df_bulk is not None:
        plot.pd2r("df_bulk", df_bulk)
    
    # Get limits from data
    max_exp = max(dfT['log2_exp'].max(), dfR['log2_exp'].max())
    limits = [0, max_exp]
    med_pos = dfT[dfT['Positive']]['log2_exp'].median()
    
    # For bulk data, use independent limits based on bulk data
    bulk_limit_tpm = 10
    bulk_line = np.log2(bulk_limit_tpm + 1)
    
    bulk_max = df_bulk['log2_exp'].max()
    bulk_limits = [0, bulk_max * 1.05]
    
    print(f"DEBUG: limits = {limits}, bulk_limits = {bulk_limits}, bulk_line = {bulk_line}")
    
    # Create plots in R
    print("Creating plots in R...")
    
    r_code = f'''
    library(ggplot2)
    library(patchwork)
    library(scales)
    
    # Parameters
    show <- {show}
    size <- 6
    limits <- c({limits[0]}, {limits[1]})
    med_pos <- {med_pos}
    n_pos <- {n_pos}
    n_neg <- {n_neg}
    per_pos <- {per_pos}
    
    # ========== PANEL 1: MALIGNANT EXPRESSION ==========
    dfT$pos_status <- ifelse(dfT$Positive, "Malig Pos.", "Malig Neg.")
    dfT$pos_status <- factor(dfT$pos_status, levels = c("Malig Pos.", "Malig Neg."))
    
    p1 <- ggplot(dfT, aes(pos_status, log2_exp)) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
        geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=pos_status), color="black", height=0, width=0.2) +
        theme(legend.position="none") +
        coord_cartesian(ylim=limits) +
        scale_x_discrete(
            drop=FALSE,
            labels=c("Malig Pos."=paste0("Malig Pos.\\n(N=", n_pos, ")"),
                    "Malig Neg."=paste0("Malig Neg.\\n(N=", n_neg, ")"))
        ) +
        xlab("") +
        geom_hline(yintercept=med_pos, lty="dashed", color="firebrick3") +
        scale_fill_manual(values=c("Malig Pos."="#28154C", "Malig Neg."="lightgrey")) +
        ylab("{gene}") +
        ggtitle(paste0(per_pos, "%"))
    
    # ========== PANEL 2: REFERENCE BY TISSUE ==========
    dfR$tissue_factor <- factor(dfR$Tissue, levels=unique(dfR$Tissue[order(dfR$log2_exp, decreasing=TRUE)]))
    
    p2 <- ggplot(dfR, aes(tissue_factor, squish(log2_exp, limits))) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
        geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=Tissue), color="black", height=0, width=0.2) +
        theme(legend.position="none") +
        coord_cartesian(ylim=limits) +
        geom_hline(yintercept=med_pos, lty="dashed", color="firebrick3") +
        xlab("") + ylab("") +
        theme(axis.text.x=element_text(size=size)) +
        scale_fill_manual(values=suppressMessages(create_pal_d(dfR$Tissue))) +
        ggtitle("Facet By Tissue")
    
    # ========== PANEL 3: TOP OFF-TARGET CELL TYPES ==========
    dfR_top <- head(dfR, show)
    dfR_top$ID_display <- factor(dfR_top$ID_display, levels=dfR_top$ID_display)
    
    p3 <- ggplot(dfR_top, aes(ID_display, squish(log2_exp, limits))) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_point(size=1.5, stroke=0.35, pch=21, aes(fill=Tissue), color="black") +
        theme(legend.position="none") +
        coord_cartesian(ylim=limits) +
        geom_hline(yintercept=med_pos, lty="dashed", color="firebrick3") +
        xlab("") + ylab("") +
        theme(axis.text.x=element_text(size=size*0.75)) +
        scale_fill_manual(values=suppressMessages(create_pal_d(dfR_top$Tissue))) +
        ggtitle("Top Off Targeted Cell Types")
    '''
    
    # Calculate percentage of samples with TPM > 10 for each group
    tcga_pct = (df_bulk[df_bulk['Source'] == 'TCGA']['log2_exp'] > bulk_line).sum() / len(df_bulk[df_bulk['Source'] == 'TCGA']) * 100
    gtex_pct = (df_bulk[df_bulk['Source'] == 'GTEx']['log2_exp'] > bulk_line).sum() / len(df_bulk[df_bulk['Source'] == 'GTEx']) * 100
    
    r_code += f'''
    # ========== PANEL 4: TCGA/GTEX BULK EXPRESSION (BOTTOM) ==========
    # Split TCGA and GTEx into separate dataframes
    df_tcga <- df_bulk[df_bulk$Source == "TCGA", ]
    df_gtex <- df_bulk[df_bulk$Source == "GTEx", ]
    
    # Order TCGA by median expression
    df_tcga$Type <- factor(df_tcga$Type, 
                           levels=names(sort(tapply(df_tcga$log2_exp, df_tcga$Type, median), decreasing=TRUE)))
    
    # Order GTEx by median expression
    df_gtex$Type <- factor(df_gtex$Type, 
                           levels=names(sort(tapply(df_gtex$log2_exp, df_gtex$Type, median), decreasing=TRUE)))
    
    # TCGA plot (left half)
    p4_tcga <- ggplot(df_tcga, aes(Type, log2_exp)) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_boxplot(fill="firebrick3", outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.6, alpha=0.7) +
        geom_jitter(size=0.8, stroke=0.2, pch=21, fill="firebrick3", color="black", height=0, width=0.2, alpha=0.5) +
        theme(legend.position="none") +
        coord_cartesian(ylim=c({bulk_limits[0]:.4f}, {bulk_limits[1]:.4f})) +
        geom_hline(yintercept=bulk_line, lty="dashed", color="firebrick3") +
        xlab("") + ylab("") +
        theme(axis.text.x=element_text(size=5, angle=90)) +
        ggtitle("TCGA ({tcga_pct:.1f}%)")
    
    # GTEx plot (right half)
    p4_gtex <- ggplot(df_gtex, aes(Type, log2_exp)) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.01) +
        geom_boxplot(fill="grey", outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.6, alpha=0.7) +
        geom_jitter(size=0.8, stroke=0.2, pch=21, fill="grey", color="black", height=0, width=0.2, alpha=0.5) +
        theme(legend.position="none") +
        coord_cartesian(ylim=c({bulk_limits[0]:.4f}, {bulk_limits[1]:.4f})) +
        geom_hline(yintercept=bulk_line, lty="dashed", color="firebrick3") +
        xlab("") + ylab("") +
        theme(axis.text.x=element_text(size=5, angle=90)) +
        ggtitle("GTEx ({gtex_pct:.1f}%)")
    
    # Top row: single cell plots
    top_row <- p1 + p2 + p3 + plot_layout(widths=c(1, 3, 3))
    
    # Bottom row: bulk plots
    bottom_row <- p4_tcga + p4_gtex + plot_layout(widths=c(1, 1))
    
    # Combine vertically with bulk taking up less space
    final_plot <- top_row / bottom_row + plot_layout(heights=c(2, 1))
    '''
    
    r_code += f'''
    suppressWarnings(ggsave("{out_path}", final_plot, width={width}, height={height}, dpi={dpi}))
    '''
    
    # Execute R code
    r(r_code)
    
    print(f"✓ Saved: {out_path}")
    
    # Convert to PNG
    png_path = out_path.replace('.pdf', '.png')
    try:
        plot.pdf_to_png(pdf_path=out_path, dpi=dpi, output_path=png_path)
        print(f"✓ Made PNG: {png_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not convert to PNG: {e}")
    
    # Write summary file
    summary_file = os.path.join(out_dir, f"{gene}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Gene: {gene}\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Positivity in patients: {per_pos}% ({len(pos_patient_unique)}/{len(malig_med_adata.obs_names)})\n")
        f.write(f"Median tumor expression (positive): {med_pos:.2f}\n")
        f.write(f"Number of reference cell types: {len(dfR)}\n")
    
    elapsed = (time.time() - overall_start) / 60
    print(f"\n{'='*60}")
    print(f"✓ Plot complete: {gene}")
    print(f"  Completed in {elapsed:.2f} minutes")
    print(f"{'='*60}")


import pandas as pd
from plotnine import *

# Assuming top10_plot is your dataframe
# top10_plot = pd.read_csv("your_data.csv", index_col=0)

# Get numeric columns (exclude gene_name if present)
numeric_cols = top10_plot.select_dtypes(include=['float64', 'int64']).columns

# Create a barplot for each numeric column
plots = []
for col in numeric_cols:
    p = (
        ggplot(top10_plot, aes(x='reorder(gene_name, -' + col + ')', y=col)) +
        geom_col(fill='steelblue', color='black', size=0.3) +
        labs(
            title=f'{col}',
            x='Gene',
            y=col
        ) +
        theme_minimal() +
        theme(
            axis_text_x=element_text(angle=45, hjust=1, size=9),
            plot_title=element_text(hjust=0.5, face='bold', size=10)
        )
    )
    plots.append(p)

# Combine all plots (2 columns layout)
from patchwork import wrap_plots
combined = wrap_plots(*plots, ncol=2)
combined.save('target_quality_barplots.pdf', width=14, height=12)
print(f"Saved plots for {len(numeric_cols)} columns")


single[single["gene_name"]=="CLDN18.2"]



multi = pd.concat([
	pd.read_parquet(IND + '.Multi.CLDN18p2.Results.20251107.parquet'),
	pd.read_parquet(IND + '.Multi.Results.20251107.parquet')
])

#Multi
multi2 = multi[(multi["TargetQ_Final_v2"] > 25) & (multi["Positive_Final_v2"] > 25)].copy().reset_index()
multi2 = pd.concat([multi2, multi2["gene_name"].str.split("_", expand=True)], axis = 1)
multi2["claude_surface"] = 1*(multi2[0].isin(clearsurface)) + 1*(multi2[1].isin(clearsurface))
multi2["claude_not_surface"] = 1*(multi2[0].isin(nonsurface)) + 1*(multi2[1].isin(nonsurface))
multi2 = multi2.sort_values("TargetQ_Final_v2", ascending=False)
multi2 = multi2[multi2["claude_not_surface"]==0].copy()
multi2 = multi2[multi2["claude_surface"]==2].copy()
multi2 = multi2[~multi2["gene_name"].str.contains("LIF")]
multi2["TABS"] = 1*multi2[0].isin(tabs) + 1*multi2[1].isin(tabs)

multi2[multi2["TABS"] == 2][["TargetQ_Final_v2", "Positive_Final_v2", "gene_name"]].head(50)
multi2[["TargetQ_Final_v2", "Positive_Final_v2", "gene_name"]].head(50)

#Check















