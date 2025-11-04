"""
TCGA/GTEx biaxial plot visualization.
"""

__all__ = ['biaxial_tcga_gtex']

import numpy as np
import pandas as pd
from pathlib import Path
import os
from py_target_id import plot
from rpy2.robjects import r
from py_target_id import utils
from datetime import datetime

def biaxial_tcga_gtex(
    multis: list,
    main: str = "LUAD",
    gtex_adata = None,  # Changed from utils.get_gtex_adata()
    tcga_adata = None,  # Changed from utils.get_tcga_adata()
    out_dir: str = "multi/biaxial_tcga_gtex",
    width: float = 24,
    height: float = 12,
    dpi: int = 300
):
    """
    Create biaxial TCGA/GTEx plots for gene combinations.
    
    Parameters
    ----------
    multis : list
        Gene combinations in format ["GENE1_GENE2", ...]
    main : str
        Main cancer type to highlight (default: "LUAD")
    gtex_adata : AnnData
        GTEx expression data (samples × genes) with obs['GTEX']
    tcga_adata : AnnData
        TCGA expression data (samples × genes)
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
    
    # Lazy-load reference data only if needed
    if gtex_adata is None:
        gtex_adata = utils.get_gtex_adata()
    
    if tcga_adata is None:
        tcga_adata = utils.get_tcga_adata()
    
    # Parse gene combinations
    multis_split = np.array([m.split("_") for m in multis])
    genes = np.unique(multis_split.ravel()).tolist()
    
    print(f"Processing {len(multis)} gene combinations with {len(genes)} unique genes")
    print(f"Output directory: {out_dir}")
    
    # Essential tissues (critical for survival)
    gtex_essential = [
        "Brain.Hippo", "Brain.Cortex", "Brain.Putamen", "Brain.ACC_BA24",
        "Brain.Cerebellar", "Brain.Frontal_BA9", "Brain.Spinal_C1", "Brain.SubNigra",
        "Brain.Basal_G_NAcc", "Brain.Hypothal", "Brain.Cerebellum", "Brain.Basal_G_Caud",
        "Brain.Amygdala",
        "Heart.Atrium", "Heart.Ventr",
        "Lung",
        "Liver"
    ]
    
    # Non-essential tissues (reproductive/secondary organs)
    gtex_non_essential = [
        "Testis", "Ovary", "Prostate", "Uterus", "Cervix.Endo", "Cervix.Ecto",
        "Fallopian", "Vagina",
        "Breast.Mammary",
        "Adipose.Subcut", "Adipose.Visc"
    ]
    
    # Load GTEx data
    print("Reading in GTEx...")
    gtex_subset = gtex_adata[:, genes]
    if hasattr(gtex_subset, 'to_memory'):
        print("Materializing GTEx VirtualAnnData...")
        gtex_subset = gtex_subset.to_memory()
    gtex_mat = gtex_subset.X.toarray() if hasattr(gtex_subset.X, 'toarray') else gtex_subset.X
    gtex_id = gtex_subset.obs['GTEX'].values
    
    # Load TCGA data
    print("Reading in TCGA...")
    tcga_subset = tcga_adata[:, genes]
    if hasattr(tcga_subset, 'to_memory'):
        print("Materializing TCGA VirtualAnnData...")
        tcga_subset = tcga_subset.to_memory()
    tcga_mat = tcga_subset.X.toarray() if hasattr(tcga_subset.X, 'toarray') else tcga_subset.X
    tcga_samples = tcga_subset.obs_names.values
    tcga_id = np.array([s.split('#')[0] for s in tcga_samples])
    
    # Process each gene combination
    for idx, (gx, gy) in enumerate(multis_split):
        print(f"\n{'='*60}")
        print(f"Processing {idx+1}/{len(multis)}: {gx}_{gy}")
        print(f"{'='*60}")
        
        # Get gene indices
        idx_gx = genes.index(gx)
        idx_gy = genes.index(gy)
        
        # TCGA data
        df_tcga = pd.DataFrame({
            'x1': tcga_mat[:, idx_gx],
            'x2': tcga_mat[:, idx_gy],
            'Type': tcga_id,
            'facet': tcga_id,
            'sample': tcga_samples
        })
        df_tcga['xm'] = np.minimum(df_tcga['x1'], df_tcga['x2'])
        
        # Calculate limits based on third highest value
        all_values = np.concatenate([df_tcga['x1'].values, df_tcga['x2'].values])
        third_highest = np.sort(all_values)[-3] if len(all_values) >= 3 else np.max(all_values)
        bulk_limits = [0, third_highest * 1.05]
        
        # Cap and log transform
        df_tcga['Capped'] = (df_tcga['x1'] > bulk_limits[1]) | (df_tcga['x2'] > bulk_limits[1])
        df_tcga['lx1'] = np.log2(np.minimum(df_tcga['x1'], bulk_limits[1]) + 1)
        df_tcga['lx2'] = np.log2(np.minimum(df_tcga['x2'], bulk_limits[1]) + 1)
        df_tcga['TPM_10'] = df_tcga['xm'] >= 10
        
        # GTEx data - calculate median per tissue
        df_gtex_raw = pd.DataFrame({
            'T1': gtex_mat[:, idx_gx],
            'T2': gtex_mat[:, idx_gy],
            'Type': 'GTEX',
            'ID': gtex_id
        })
        
        # Calculate median per tissue
        df_gtex = df_gtex_raw.groupby('ID').agg({
            'T1': 'median',
            'T2': 'median'
        }).reset_index()
        df_gtex.columns = ['ID', 'x1', 'x2']
        df_gtex['Type'] = 'GTEX'
        df_gtex['xm'] = np.minimum(df_gtex['x1'], df_gtex['x2'])
        
        # Cap and log transform
        df_gtex['Capped'] = (df_gtex['x1'] > bulk_limits[1]) | (df_gtex['x2'] > bulk_limits[1])
        df_gtex['lx1'] = np.log2(np.minimum(df_gtex['x1'], bulk_limits[1]) + 1)
        df_gtex['lx2'] = np.log2(np.minimum(df_gtex['x2'], bulk_limits[1]) + 1)
        
        # Sort by xm
        df_gtex = df_gtex.sort_values('xm', ascending=False)
        
        # Split into essential and non-essential
        df_gtex_ess = df_gtex[df_gtex['ID'].isin(gtex_essential)].copy()
        df_gtex_non_ess = df_gtex[df_gtex['ID'].isin(gtex_non_essential)].copy()
        df_gtex_other = df_gtex[~df_gtex['ID'].isin(gtex_essential + gtex_non_essential)].copy()
        
        # Calculate percentage positive by cancer type
        pct_results = df_tcga.groupby('facet')['TPM_10'].agg(['mean', 'size']).reset_index()
        pct_results.columns = ['cancer_type', 'percentage', 'n']
        pct_results['percentage'] = pct_results['percentage'] * 100
        pct_results = pct_results.sort_values('percentage')
        pct_results['label'] = pct_results['cancer_type'] + ' (N=' + pct_results['n'].astype(str) + ')'
        
        # Get main cancer type subset
        df_tcga_sub = df_tcga[df_tcga['facet'] == main].copy()
        
        # Create output path
        out_path = os.path.join(out_dir, f"{multis[idx]}.pdf")
        
        # Send data to R
        print("Sending data to R...")
        plot.pd2r("df_tcga", df_tcga)
        plot.pd2r("df_tcga_sub", df_tcga_sub)
        plot.pd2r("df_gtex_ess", df_gtex_ess)
        plot.pd2r("df_gtex_non_ess", df_gtex_non_ess)
        plot.pd2r("df_gtex_other", df_gtex_other)
        plot.pd2r("pct_results", pct_results)
        
        # Create plots in R
        print("Creating plots in R...")
        r(f'''
        library(ggplot2)
        library(patchwork)
        
        # Parameters
        bulk_limits <- c({bulk_limits[0]}, {bulk_limits[1]})
        log_limits <- log2(bulk_limits + 1)
        
        # Make factors for ordering
        pct_results$label <- factor(pct_results$label, levels = pct_results$label)
        
        # Plot 1: Sub by cancer type (faceted)
        p1_1 <- ggplot() +
            geom_point(data=df_tcga_sub, aes(lx1, lx2), fill = pal_cart[5], pch = 21, size = 2) +
            geom_point(data=df_gtex_non_ess, aes(lx1, lx2), fill = "grey", pch = 24, size = 3) +
            geom_point(data=df_gtex_other, aes(lx1, lx2), fill = pal_cart[2], pch = 24, size = 3) +
            geom_point(data=df_gtex_ess, aes(lx1, lx2), fill = "firebrick3", pch = 24, size = 3) +
            geom_abline(slope = 1, intercept = 0, lty = "dashed", color = "black") +
            xlim(log_limits) + 
            ylim(log_limits) + 
            facet_wrap(~facet, nrow = 5) + 
            theme(legend.position = "none") +
            geom_vline(xintercept = log2(11), lty = "dashed", color = "firebrick3", linewidth = 0.5) +
            geom_hline(yintercept = log2(11), lty = "dashed", color = "firebrick3", linewidth = 0.5) +
            xlab(paste0("{gx} Log2(TPM + 1)")) +
            ylab(paste0("{gy} Log2(TPM + 1)")) +
            theme_small_margin(0.2) +
            ggtitle("{multis[idx]}") +
            theme(strip.text = element_text(size = 14), 
                  axis.text = element_text(size = 14), 
                  axis.title = element_text(size = 14), 
                  plot.title = element_text(size = 14))
        
        # Plot 1_2: Percentage bar plot
        p1_2 <- ggplot(pct_results, aes(x = label, y = percentage)) +
            geom_bar(stat = "identity", fill = pal_cart[5], color = "black", alpha = 0.8) +
            geom_text(aes(label = sprintf("%.1f%%", percentage)), 
                      hjust = -0.1, size = 3) +
            labs(title = "TCGA 10 TPM Percent Co-Expressed",
                 x = "Cancer Type",
                 y = "Percentage Positive (%)") +
            coord_flip() +
            theme(plot.title = element_text(hjust = 0.5, size = 14),
                  panel.grid.minor = element_blank(),
                  axis.text.y = element_text(size = 8)) +
            theme(legend.position="none") +
            ylim(c(0, 100)) +
            theme_small_margin(0.2) +
            theme(strip.text = element_text(size = 14), 
                  axis.text = element_text(size = 14), 
                  axis.title = element_text(size = 14))
        
        # Plot 2: All TCGA samples
        p2 <- ggplot() +
            geom_point(data=df_tcga, aes(lx1, lx2), fill = pal_cart[5], pch = 21, size = 2) +
            geom_point(data=df_gtex_non_ess, aes(lx1, lx2), fill = "grey", pch = 24, size = 3) +
            geom_point(data=df_gtex_other, aes(lx1, lx2), fill = pal_cart[2], pch = 24, size = 3) +
            geom_point(data=df_gtex_ess, aes(lx1, lx2), fill = "firebrick3", pch = 24, size = 3) +
            geom_abline(slope = 1, intercept = 0, lty = "dashed", color = "black") +
            xlim(log_limits) + 
            ylim(log_limits) + 
            facet_wrap(~facet, nrow = 5) + 
            theme(legend.position = "none", plot.caption = element_text(size = 11, hjust = 0)) +
            geom_vline(xintercept = log2(11), lty = "dashed", color = "firebrick3", linewidth = 0.5) +
            geom_hline(yintercept = log2(11), lty = "dashed", color = "firebrick3", linewidth = 0.5) +
            xlab(paste0("{gx} Log2(TPM + 1)")) +
            ylab(paste0("{gy} Log2(TPM + 1)")) +
            theme(strip.text = element_text(size = 14), 
                  axis.text = element_text(size = 14), 
                  axis.title = element_text(size = 14)) +
            labs(caption = "Circle: TCGA Patient Expression \nTriangle: GTEx Tissue Expression (Red = High Essential, Green = Medium Essential, Grey = Low Essential)")
    
        # Combine plots
        final_plot <- wrap_plots(
            wrap_plots(p1_1, p1_2, ncol = 1, heights = c(1, 1)),
            p2,
            ncol = 2, 
            widths = c(1, 3)  
        )
        
        # Save
        suppressWarnings(ggsave("{out_path}", final_plot, width={width}, height={height}))
        ''')
        
        print(f"✓ Saved: {out_path}")
        
        # Convert to PNG
        png_path = out_path.replace('.pdf', '.png')
        try:
            plot.pdf_to_png(pdf_path=out_path, dpi=dpi, output_path=png_path)
            print(f"✓ Made PNG: {png_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not convert to PNG: {e}")

    #Write custom done
    out_file = out_dir + "/finished.txt"

    with open(out_file, 'w') as f:
        f.write(f"Finished: {datetime.now()}\n\n")

    print(f"\n{'='*60}")
    print(f"✓ All {len(multis)} plots saved to {out_dir}/")
    print(f"{'='*60}")

