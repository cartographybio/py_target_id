"""
Multi-target radar plot visualization.
"""

__all__ = ['plot_multi_radar']

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from py_target_id import plot
from rpy2.robjects import r
import os
from datetime import datetime

def plot_multi_radar(
    multis: List[str],
    df_single: pd.DataFrame,
    df_multi: pd.DataFrame,
    out_dir: str = "multi/multi_radar",
    width: float = 8,
    height: float = 8,
    dpi: int = 300
):
    """
    Create radar plots comparing single genes and multi-gene combinations.
    
    Parameters
    ----------
    multis : List[str]
        Gene combinations in format ["GENE1_GENE2", ...]
    df_single : pd.DataFrame
        Single gene target ID results with gene_name as index or column
    df_multi : pd.DataFrame
        Multi-gene combination target ID results with gene_name as index or column
    out_dir : str
        Output directory for plots
    width : float
        Plot width in inches
    height : float
        Plot height in inches
    dpi : int
        Resolution for output
    """
    
    # Create output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Check and install ggradar if needed
    r('''
    if (!require("ggradar", quietly = TRUE)) {
        message("Installing ggradar from GitHub...")
        if (!require("remotes", quietly = TRUE)) {
            install.packages("remotes", repos = "https://cloud.r-project.org")
        }
        remotes::install_github("ricardo-bion/ggradar", upgrade = "never")
    }
    library(ggradar)
    ''')
    
    # Check required columns
    required_cols = ['gene_name', 'Corrected_Specificity', 'N_Off_Targets', 
                     'Surface_Prob', 'Target_Val', 'Positive_Final_v2']
    
    missing_single = [col for col in required_cols if col not in df_single.columns]
    if missing_single:
        raise ValueError(f"Missing required columns in df_single: {', '.join(missing_single)}")
    
    missing_multi = [col for col in required_cols if col not in df_multi.columns]
    if missing_multi:
        raise ValueError(f"Missing required columns in df_multi: {', '.join(missing_multi)}")
    
    # Parse gene combinations
    multis_split = np.array([m.split("_") for m in multis])
    genes = np.unique(multis_split.ravel()).tolist()
    
    # Set gene_name as index if not already
    if 'gene_name' in df_single.columns and df_single.index.name != 'gene_name':
        df_single = df_single.set_index('gene_name')
    if 'gene_name' in df_multi.columns and df_multi.index.name != 'gene_name':
        df_multi = df_multi.set_index('gene_name')
    
    # Prepare single gene radar data
    list_sng = {}
    for gene in genes:
        if gene not in df_single.index:
            print(f"Warning: Gene {gene} not found in df_single")
            continue
            
        dfj = df_single.loc[gene]
        
        # Specificity update
        v1 = dfj['Corrected_Specificity'] / 10
        v2 = max(9 - dfj['N_Off_Targets'], 0) / 10
        v3 = v1 + v2
        
        # Create radar data
        radar_data = pd.DataFrame({
            'Group': ['Target'],
            'Surface\nEvidence': [dfj['Surface_Prob']],
            'Target\nExpression': [min(dfj['Target_Val'], 2) / 2],
            'Patients\nPositive\n(0-0.5)': [min(dfj['Positive_Final_v2']/100 * 2, 1)],
            'Low\nOff-Tumor\nToxicity': [1 - min(dfj['N_Off_Targets'], 5) / 5],
            'Target\nSpecificity': [v3]
        })
        
        list_sng[gene] = radar_data
    
    # Prepare multi-gene radar data
    list_multi = {}
    for multi in multis:
        if multi not in df_multi.index:
            print(f"Warning: Multi {multi} not found in df_multi")
            continue
            
        dfj = df_multi.loc[multi]
        
        # Specificity update
        v1 = dfj['Corrected_Specificity'] / 10
        v2 = max(9 - dfj['N_Off_Targets'], 0) / 10
        v3 = v1 + v2
        
        # Create radar data
        radar_data = pd.DataFrame({
            'Group': ['Target'],
            'Surface\nEvidence': [dfj['Surface_Prob']],
            'Target\nExpression': [min(dfj['Target_Val'], 2) / 2],
            'Patients\nPositive\n(0-0.5)': [min(dfj['Positive_Final_v2']/100 * 2, 1)],
            'Low\nOff-Tumor\nToxicity': [1 - min(dfj['N_Off_Targets'], 5) / 5],
            'Target\nSpecificity': [v3]
        })
        
        list_multi[multi] = radar_data
    
    # Process each multi combination
    for i, multi in enumerate(multis):
        print(f"Processing target {i+1} of {len(multis)}: {multi}")
        
        gene1 = multis_split[i, 0]
        gene2 = multis_split[i, 1]
        
        # Check if all data available
        if gene1 not in list_sng or gene2 not in list_sng or multi not in list_multi:
            print(f"  Skipping {multi} - missing data")
            continue
        
        # Send data to R
        plot.pd2r("radar_gene1", list_sng[gene1])
        plot.pd2r("radar_gene2", list_sng[gene2])
        plot.pd2r("radar_combo", list_multi[multi])
        
        # Create output path
        out_path = os.path.join(out_dir, f"{multi}.pdf")
        
        # Generate plots in R
        r(f'''
        library(ggplot2)
        library(patchwork)
        library(ggradar)
        
        # Create radar plot for gene 1
        p1 <- ggradar(
          radar_gene1,
          values.radar = c("0", "0.5", "1"),
          axis.label.size = 3,
          grid.label.size = 4.5,
          grid.line.width = 0.5,
          group.line.width = 0.5,
          group.point.size = 4,
          gridline.min.colour = "#6BC291",
          gridline.mid.colour = "#18B5CB",
          gridline.max.colour = "#2E95D2",
          group.colours = c("#28154c"),
          background.circle.colour = "#eeeeee"
        ) + ggtitle("  {gene1}")
        
        # Add green circle
        p1 <- p1 +
          annotate("path",
                   x = 0.86 * cos(seq(0, 4 * pi, length.out = 100)),
                   y = 0.86 * sin(seq(0, 4 * pi, length.out = 100)),
                   linewidth = 18, color = "#228B22", alpha = 0.2)
        
        # Create radar plot for gene 2
        p2 <- ggradar(
          radar_gene2,
          values.radar = c("0", "0.5", "1"),
          axis.label.size = 3,
          grid.label.size = 4.5,
          grid.line.width = 0.5,
          group.line.width = 0.5,
          group.point.size = 4,
          gridline.min.colour = "#6BC291",
          gridline.mid.colour = "#18B5CB",
          gridline.max.colour = "#2E95D2",
          group.colours = c("#28154c"),
          background.circle.colour = "#eeeeee"
        ) + ggtitle("  {gene2}")
        
        # Add green circle
        p2 <- p2 +
          annotate("path",
                   x = 0.86 * cos(seq(0, 4 * pi, length.out = 100)),
                   y = 0.86 * sin(seq(0, 4 * pi, length.out = 100)),
                   linewidth = 18, color = "#228B22", alpha = 0.2)
        
        # Create radar plot for combo
        p12 <- ggradar(
          radar_combo,
          values.radar = c("0", "0.5", "1"),
          axis.label.size = 3,
          grid.label.size = 4.5,
          grid.line.width = 0.5,
          group.line.width = 0.5,
          group.point.size = 4,
          gridline.min.colour = "#6BC291",
          gridline.mid.colour = "#18B5CB",
          gridline.max.colour = "#2E95D2",
          group.colours = c("#28154c"),
          background.circle.colour = "#eeeeee"
        ) + ggtitle("  {multi}")
        
        # Add green circle
        p12 <- p12 +
          annotate("path",
                   x = 0.86 * cos(seq(0, 4 * pi, length.out = 100)),
                   y = 0.86 * sin(seq(0, 4 * pi, length.out = 100)),
                   linewidth = 18, color = "#228B22", alpha = 0.2)
        
        # Remove margins
        p1 <- p1 + theme_void() +
          theme(plot.margin = unit(c(-1,-1,-1,-1), "cm"),
                legend.position = "none")
        
        p2 <- p2 + theme_void() +
          theme(plot.margin = unit(c(-1,-1,-1,-1), "cm"),
                legend.position = "none")
        
        p12 <- p12 + theme_void() +
          theme(plot.margin = unit(c(-1,-1,-1,-1), "cm"),
                legend.position = "none")
        
        # Combine plots
        final_plot <- (p1 + p2) / (plot_spacer() + p12) +
          plot_layout(widths = c(1, 1), heights = c(1, 1)) &
          theme(plot.margin = unit(c(0,0,0,0), "cm"))
        
        # Save
        suppressWarnings(ggsave("{out_path}", final_plot, width = {width}, height = {height}))
        ''')
        
        print(f"  ✓ Saved: {out_path}")
        
        # Convert to PNG
        png_path = out_path.replace('.pdf', '.png')
        try:
            plot.pdf_to_png(pdf_path=out_path, dpi=dpi, output_path=png_path)
            print(f"  ✓ Made PNG: {png_path}")
        except Exception as e:
            print(f"  ⚠ Warning: Could not convert to PNG: {e}")

    #Write custom done
    out_file = out_dir + "/finished.txt"

    with open(out_file, 'w') as f:
        f.write(f"Finished: {datetime.now()}\n\n")
    
    print(f"\n{'='*60}")
    print(f"✓ All {len(multis)} radar plots saved to {out_dir}/")
    print(f"{'='*60}")