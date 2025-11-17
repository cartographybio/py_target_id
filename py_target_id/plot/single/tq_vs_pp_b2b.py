"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['tq_vs_pp_b2b']

import numpy as np
import pandas as pd
from py_target_id import utils, plot  # Relative import

from rpy2.robjects.packages import importr
from rpy2 import robjects as r
from rpy2.robjects import pandas2ri
import pandas as pd

def tq_vs_pp_b2b(
    df,
    target_q_col="TargetQ_Final_v1",
    positive_col="Positive_Final_v2",
    p2_color_col="claude_summary",
    gene_col="gene_name",
    target_q_threshold=50,
    positive_threshold=10,
    y_min=0,
    y_max=100,
    output_file="target_quality_plot.pdf",
    p2_color_dict=None,
    figsize=(12, 7)
):
    """
    Create side-by-side ggplot2 scatter plots comparing target quality metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with all required columns
    target_q_col : str
        Column name for target quality scores
    positive_col : str
        Column name for positive percentage
    p1_color_col : str
        Column name for continuous coloring in plot 1 (default SC_2nd_Target_Val)
    p2_color_col : str
        Column name for categorical coloring in plot 2 (default claude_summary)
    gene_col : str
        Column name for gene names
    target_q_threshold : float
        Threshold line for target quality (default 50)
    positive_threshold : float
        Threshold line for positive percentage (default 10)
    y_min : float
        Minimum y-axis value (0 for v1, 50 for v2)
    y_max : float
        Maximum y-axis value (100 for v1 and v2)
    output_file : str
        Output PDF filename
    p1_color_dict : dict, optional
        Dictionary for plot 1 coloring. For continuous scales, use None (default viridis).
        For discrete scales, map values to colors.
    p2_color_dict : dict, optional
        Dictionary mapping p2_color_col values to colors. 
        If None, defaults to {"11": "firebrick3", "1": "dodgerblue3", "0": "lightgrey"}
    figsize : tuple
        Figure size (width, height)
    """
    
    # Filter data for labels
    label_df = df[(df[target_q_col] > target_q_threshold) & 
                  (df[positive_col] > positive_threshold)]
    
    # Convert to R dataframes
    plot.pd2r("df", df)
    plot.pd2r("sub_df", label_df)

    p1_color_col="SC_2nd_Target_Val"
  
    # Set default color dict for p2 if not provided
    if p2_color_dict is None:
        p2_color_dict = {"11": "firebrick3", "1": "dodgerblue3", "0": "lightgrey"}
    
    # Format p2 color dict for R
    p2_color_str = ", ".join([f'"{k}" = "{v}"' for k, v in p2_color_dict.items()])
    
    # R script
    r(f'''
    
    p1 <- ggplot(df, aes({positive_col}, {target_q_col}, 
                    fill = pmin(log2({p1_color_col} + 1), 2))) +
        geom_point(pch = 21, size = 2, color = "black") +
        geom_hline(yintercept = {target_q_threshold}, linetype = "dashed", color = "firebrick3", size = 0.5, alpha = 0.7) +
        geom_vline(xintercept = {positive_threshold}, linetype = "dashed", color = "firebrick3", size = 0.5, alpha = 0.7) +
        geom_label_repel(
            data = sub_df, 
            aes(label = {gene_col}), 
            fill = "white", 
            max.overlaps = Inf,
            size = 1.75
        ) +
        labs(
            title = "Target Quality vs Positive Percentage",
            x = "Positive Percentage (v2)",
            y = "Target Quality ({target_q_col})",
            fill = "log2({p1_color_col} + 1)"
        ) +
        xlim(0, 100) +
        ylim({y_min}, {y_max}) +
        theme_jg()
    
    p2 <- ggplot(df, aes({positive_col}, {target_q_col}, 
                    fill = paste0({p2_color_col}))) +
        geom_point(pch = 21, size = 2, color = "black") +
        geom_hline(yintercept = {target_q_threshold}, linetype = "dashed", color = "firebrick3", size = 0.5, alpha = 0.7) +
        geom_vline(xintercept = {positive_threshold}, linetype = "dashed", color = "firebrick3", size = 0.5, alpha = 0.7) +
        geom_label_repel(
            data = sub_df, 
            aes(label = {gene_col}), 
            alpha = 0.9,
            max.overlaps = Inf,
            size = 1.75
        ) +
        scale_fill_manual(values = c({p2_color_str})) +
        labs(
            title = "Target Quality vs Positive Percentage",
            x = "Positive Percentage (v2)",
            y = "Target Quality ({target_q_col})",
            fill = "Classification"
        ) +
        xlim(0, 100) +
        ylim({y_min}, {y_max}) +
        theme_jg()
    
    library(patchwork)
    pdf("{output_file}", width = {figsize[0]}, height = {figsize[1]})
    print(p1 + p2)
    dev.off()
    ''')

    print(f"Plot saved to {output_file}")
