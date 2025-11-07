"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['p1_tq_vs_pp']

import numpy as np
import pandas as pd
from py_target_id import utils, plot  # Relative import

def p1_tq_vs_pp(
    df, 
    out="tq_vs_pp.pdf", 
    target_q=80, 
    ppos=15, 
    label_top_interval=True, 
    top_n=2, 
    interval=10, 
    highlight_genes=None,
    pdf_w=9,
    pdf_h=8,
    title_suffix=None
):
    # Subset high-quality targets for labeling
    genes_list = []
    
    if label_top_interval:
        g1 = plot.get_top_n_per_interval(
            df, 
            x_col='Positive_Final_v2', 
            y_col='TargetQ_Final_v1',
            label_col='gene_name',
            n=top_n,
            interval=interval
        )["gene_name"].values
        genes_list.append(g1)
        
        g2 = df[(df["TargetQ_Final_v1"] >= target_q) & (df["Positive_Final_v2"] * 100 >= ppos)]["gene_name"].values
        genes_list.append(g2)
    
    if highlight_genes is not None:
        genes_list.append(np.array(highlight_genes))
    
    # Only label if genes_list is not empty
    if genes_list:
        genes = np.unique(np.concatenate(genes_list))
        sub_df = df[df["gene_name"].isin(genes)]
    else:
        sub_df = df.iloc[0:0]  # Empty dataframe with same structure
    
    # Calculate TargetQ thresholds
    thresholds = [40, 50, 60, 70, 80, 90, 95]
    counts = [df[(df["TargetQ_Final_v1"] > t) & (df["Positive_Final_v2"] >= ppos)].shape[0] for t in thresholds]
    threshold_df_py = pd.DataFrame({
        "TargetQ_Threshold": [f"> {t}" for t in thresholds],
        "Count": counts
    })

    # Send data to R
    plot.pd2r("df", df)
    plot.pd2r("sub_df", sub_df)
    plot.pd2r("threshold_df", threshold_df_py)
    
    # Create plots in R
    r(f'''
    p_main <- ggplot(df, aes(Positive_Final_v2, TargetQ_Final_v1, 
                        fill = pmin(log2(SC_2nd_Target_Val + 1), 2))) +
        geom_point(pch = 21, size = 2, color = "black") +
        geom_hline(yintercept = {target_q}, linetype = "dashed", color = "firebrick3", size = 0.5, alpha = 0.7) +
        geom_vline(xintercept = {ppos}, linetype = "dashed", color = "firebrick3", size = 0.5, alpha = 0.7) +
        geom_label_repel(
            data = sub_df, 
            aes(label = gene_name), 
            fill = "white", 
            max.overlaps = Inf,
            size = 2
        ) +
        labs(
            title = "Target Quality vs Positive Percentage{f' - {title_suffix}' if title_suffix else ''}",
            x = "Positive Percentage (v2)",
            y = "Target Quality (v1)",
            fill = "log2(SC 2nd Target + 1)"
        ) +
        xlim(0, 100) +
        ylim(0, 100) +
        theme_jg()
    
    p_bar <- ggplot(threshold_df, aes(x = factor(TargetQ_Threshold, levels = TargetQ_Threshold), y = Count)) +
        geom_bar(stat = "identity", fill = "steelblue", color = "black") +
        geom_text(aes(label = Count), vjust = -0.5, size = 3) +
        geom_hline(yintercept = 0, linetype = "solid", color = "black", size = 0.3) +
        labs(x = "TargetQ Threshold Pos% > {ppos})", y = "Count") +
        theme_jg() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    p_combined <- p_main + p_bar + plot_layout(widths = c(3, 1.5))
    
    ggsave("{out}", p_combined, width = {pdf_w + 3}, height = {pdf_h}, dpi = 300)
    
    ''')
