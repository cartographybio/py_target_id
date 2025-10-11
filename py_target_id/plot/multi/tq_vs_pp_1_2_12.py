"""
Multi-target TargetQ vs Positive Patients scatter plots.
"""

__all__ = ['tq_vs_pp_1_2_12']

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from py_target_id import plot
from rpy2.robjects import r
import os

def tq_vs_pp_1_2_12(
    multis: List[str],
    df_single: pd.DataFrame,
    df_multi: pd.DataFrame,
    known: List[str] = ["DLL3", "MET", "EGFR", "TACSTD2", "CEACAM5", "ERBB3", "MSLN"],
    out_dir: str = "multi/multi_tq_vs_pp",
    width: float = 8,
    height: float = 8,
    dpi: int = 300
):
    """
    Create TargetQ vs Positive Patients scatter plots with arrows showing improvement.
    
    Parameters
    ----------
    multis : List[str]
        Gene combinations in format ["GENE1_GENE2", ...]
    df_single : pd.DataFrame
        Single gene target ID results with gene_name column
        Note: Positive_Final_v2 should already be in percentage (0-100)
    df_multi : pd.DataFrame
        Multi-gene combination target ID results with gene_name column
        Note: Positive_Final_v2 should already be in percentage (0-100)
    known : List[str]
        Known target genes to highlight
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
    
    # Check required columns
    required_cols = ['gene_name', 'TargetQ_Final_v1', 'Positive_Final_v2']
    
    for col in required_cols:
        if col not in df_single.columns:
            raise ValueError(f"Missing required column '{col}' in df_single")
        if col not in df_multi.columns:
            raise ValueError(f"Missing required column '{col}' in df_multi")
    
    # Parse gene combinations
    multis_split = np.array([m.split("_") for m in multis])
    
    # Add type column
    df_single = df_single.copy()
    df_multi = df_multi.copy()
    df_single['type'] = 'single'
    df_multi['type'] = 'combo'
    
    # Helper function to swap gene pairs
    def swap_pair(gene_pair):
        """Swap order of genes in pair: GENE1_GENE2 -> GENE2_GENE1"""
        if '_' not in gene_pair:
            return gene_pair
        parts = gene_pair.split('_')
        return f"{parts[1]}_{parts[0]}"
    
    # Process each multi combination
    for i, multi in enumerate(multis):
        print(f"Plotting {i+1} of {len(multis)}: {multi}")
        
        gene1, gene2 = multis_split[i]
        
        # Combine data for this plot
        cols_use = ['TargetQ_Final_v1', 'Positive_Final_v2', 'gene_name', 'type']
        
        # Get single gene rows
        df_single_subset = df_single[df_single['gene_name'].isin([gene1, gene2])][cols_use]
        
        # Get combo row
        if multi in df_multi['gene_name'].values:
            df_combo_subset = df_multi[df_multi['gene_name'] == multi][cols_use]
        else:
            print(f"  Warning: {multi} not found in df_multi")
            continue
        
        # Get all other single genes for background
        df_all_singles = df_single[cols_use]
        
        # Combine
        df_plot = pd.concat([df_all_singles, df_combo_subset], ignore_index=True)
        
        # Create highlight column
        highlight_genes = [gene1, gene2, multi, swap_pair(multi)]
        df_plot['highlight'] = df_plot['gene_name'].apply(
            lambda x: 'yes' if x in highlight_genes else 
                     ('true' if x in known else 'no')
        )
        
        # Sort by highlight so highlighted points are on top
        df_plot = df_plot.sort_values('highlight')
        
        # Get points for arrow
        single_points = df_plot[(df_plot['highlight'] == 'yes') & (df_plot['type'] == 'single')]
        combo_point = df_plot[(df_plot['highlight'] == 'yes') & (df_plot['type'] == 'combo')]
        
        if len(single_points) < 2 or len(combo_point) < 1:
            print(f"  Warning: Missing data points for {multi}")
            continue
        
        # Create title with gene stats
        single1 = single_points.iloc[0]
        single2 = single_points.iloc[1]
        combo = combo_point.iloc[0]
        
        title = (f"{single1['gene_name']} Target Q: {single1['TargetQ_Final_v1']:.1f} "
                f"Pos: {single1['Positive_Final_v2']:.2f}%\n"
                f"{single2['gene_name']} Target Q: {single2['TargetQ_Final_v1']:.1f} "
                f"Pos: {single2['Positive_Final_v2']:.2f}%\n"
                f"{combo['gene_name']} Target Q: {combo['TargetQ_Final_v1']:.1f} "
                f"Pos: {combo['Positive_Final_v2']:.2f}%")
        
        # Send data to R
        plot.pd2r("df_plot", df_plot)
        
        # Get known genes for labeling
        df_label = df_plot[df_plot['highlight'] == 'true']
        plot.pd2r("df_label", df_label)
        
        # Arrow coordinates
        arrow_data = pd.DataFrame({
            'x': single_points['Positive_Final_v2'].values,
            'y': single_points['TargetQ_Final_v1'].values,
            'xend': [combo['Positive_Final_v2']] * len(single_points),
            'yend': [combo['TargetQ_Final_v1']] * len(single_points)
        })
        plot.pd2r("arrow_data", arrow_data)
        
        # Create output path
        out_path = os.path.join(out_dir, f"{multi}.pdf")
        
        # Generate plot in R
        r(f'''
        library(ggplot2)
        library(ggrepel)
        
        # Create combined highlight:type variable
        df_plot$fill_var <- paste0(df_plot$highlight, ":", df_plot$type)
        
        p <- ggplot(df_plot, aes(Positive_Final_v2, TargetQ_Final_v1, 
                                 size = highlight, fill = fill_var)) +
            geom_point(pch = 21, alpha = 0.9, stroke = 0.25) +
            scale_size_manual(values = c("yes" = 3, "no" = 1, "true" = 2)) +
            scale_fill_manual(values = c(
                "yes:single" = "dodgerblue3",
                "yes:combo" = "firebrick3",
                "true:single" = "black",
                "no:single" = "lightgrey",
                "no:combo" = "lightgrey"
            )) +
            xlim(c(0, 100)) +
            ylim(c(0, 100)) +
            ggtitle("{title}") +
            ylab("Target Quality (v1)") +
            xlab("Patients Positive % (v2)") +
            theme(legend.position = "none")
        
        # Add arrows
        if(nrow(arrow_data) > 0) {{
            p <- p + geom_segment(data = arrow_data,
                                 aes(x = x, y = y, xend = xend, yend = yend),
                                 arrow = arrow(length = unit(0.2, "cm")),
                                 color = "black",
                                 linewidth = 0.25,
                                 inherit.aes = FALSE)
        }}
        
        # Add labels for known genes
        if(nrow(df_label) > 0) {{
            p <- p + geom_label_repel(data = df_label,
                                     aes(label = gene_name),
                                     size = 2,
                                     fill = "white",
                                     inherit.aes = TRUE)
        }}
        
        ggsave("{out_path}", p, width = {width}, height = {height})
        ''')
        
        print(f"  ✓ Saved: {out_path}")
        
        # Convert to PNG
        png_path = out_path.replace('.pdf', '.png')
        try:
            plot.pdf_to_png(pdf_path=out_path, dpi=dpi, output_path=png_path)
            print(f"  ✓ Made PNG: {png_path}")
        except Exception as e:
            print(f"  ⚠ Warning: Could not convert to PNG: {e}")
    
    print(f"\n{'='*60}")
    print(f"✓ All {len(multis)} plots saved to {out_dir}/")
    print(f"{'='*60}")