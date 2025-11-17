"""
Multi-specific target dot plot visualization.
"""

# Define what gets exported
__all__ = ['axial_1_2_12']

import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import os
import time
from typing import Literal, Union
from py_target_id import plot
from py_target_id import utils
from rpy2.robjects import r
from datetime import datetime

def axial_1_2_12(
    multis: list,
    malig_adata,  # Malignant AnnData (cells × genes)
    malig_med_adata,  # Malignant median AnnData (patients × genes) with positivity layer
    ref_adata,  # Healthy atlas AnnData
    out_dir: str = "multi/multi_axial",
    show: int = 15,
    width: float = 16,
    height: float = 10,
    dpi: int = 300
):
    """
    Create multi-axial plots for gene combinations.
    
    Parameters
    ----------
    multis : list
        Gene combinations in format ["GENE1_GENE2", ...]
    malig_adata : AnnData
        Malignant cell data with patient annotations
    malig_med_adata : AnnData
        Malignant median data (patients × genes)
        Must have layers['positivity'] with binary positivity data
    ref_adata : AnnData
        Healthy reference atlas with CellType annotations
    out_dir : str
        Output directory for plots
    show : int
        Number of top off-targets to display
    width : float
        Plot width in inches
    height : float
        Plot height in inches
    dpi : int
        Resolution for output
    """
    
    # Create output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Parse gene combinations
    multis_split = np.array([m.split("_") for m in multis])
    genes = np.unique(multis_split.ravel()).tolist()
    combo1 = multis_split[:, 0].tolist()
    combo2 = multis_split[:, 1].tolist()
    
    print(f"Processing {len(multis)} gene combinations with {len(genes)} unique genes")
    print(f"Output directory: {out_dir}")
    
    # Load malignant data
    print("Loading malignant data...")
    malig_subset = malig_adata[:, genes]
    
    if hasattr(malig_subset, 'to_memory'):
        print("Materializing malignant VirtualAnnData...")
        malig_subset = malig_subset.to_memory()
    
    #Dense
    malig_subset.X = malig_subset.X.toarray()

    # Get positivity matrix from malig_med_adata
    print("Extracting positivity data...")
    pos_mat = malig_med_adata[:, genes].layers['positivity']
    pos_mat = pos_mat.toarray().astype(int)
    
    # Load healthy atlas
    print("Loading healthy atlas...")
    ref_subset = ref_adata[:, genes]
    
    if hasattr(ref_subset, 'to_memory'):
        print("Materializing reference VirtualAnnData...")
        ref_subset = ref_subset.to_memory()
    
    # Dense
    if sparse.issparse(ref_subset.X):
        ref_subset.X = ref_subset.X.toarray()

    ref_subset.obs['CellType'] = ref_adata.obs['CellType']
 
    # Compute single gene medians
    print("Computing malignant single gene medians...")
    malig_sng_med = utils.summarize_matrix(malig_subset.X, malig_subset.obs['Patient'].values, axis = 0, metric="median", verbose=False)
    malig_sng_med.columns = genes

    print("Computing healthy single gene medians...")
    ha_sng_med = utils.summarize_matrix(ref_subset.X, ref_subset.obs['CellType'], axis = 0, metric="median", verbose=False)
    ha_sng_med.columns = genes

    print("Computing malignant combo medians...")
    malig_dbl_med = utils.summarize_matrix(np.minimum(malig_subset[:, combo1].X, malig_subset[:, combo2].X), 
        malig_subset.obs['Patient'].values, axis = 0, metric="median", verbose=False)
    malig_dbl_med.columns = multis

    print("Computing healthy combo medians...")
    ref_dbl_med = utils.summarize_matrix(np.minimum(ref_subset[:, combo1].X, ref_subset[:, combo2].X), 
        ref_subset.obs['CellType'].values, axis = 0, metric="median", verbose=False)
    ref_dbl_med.columns = multis

    print("Data preparation complete. Starting plotting...")
    
    # Process each gene combination
    for idx, (gene1, gene2) in enumerate(multis_split):
        print(f"\n{'='*60}")
        print(f"Processing {idx+1}/{len(multis)}: {gene1}_{gene2}")
        print(f"{'='*60}")
        
        multi_name = multis[idx]
        
        # Get positivity for each gene
        pos_gene1 = malig_med_adata[:, gene1].layers['positivity'].flatten()
        pos_gene2 = malig_med_adata[:, gene2].layers['positivity'].flatten()
        pos_combo = pos_gene1 & pos_gene2
        
        # Get patient names for positive patients
        pos_patients_1 = set(malig_med_adata.obs['Patient'].values[pos_gene1])
        pos_patients_2 = set(malig_med_adata.obs['Patient'].values[pos_gene2])
        pos_patients_12 = set(malig_med_adata.obs['Patient'].values[pos_combo])
        
        # Calculate percentages
        n_patients = len(malig_med_adata.obs['Patient'])
        per_1 = round(100 * len(pos_patients_1) / n_patients, 1)
        per_2 = round(100 * len(pos_patients_2) / n_patients, 1)
        per_12 = round(100 * len(pos_patients_12) / n_patients, 1)
        
        # Prepare malignant dataframes
        df1_malig = pd.DataFrame({
            'pos_patient': ['Pos.' if p in pos_patients_1 else 'Neg.' for p in malig_sng_med.index],
            'log2_exp': np.log2(malig_sng_med[gene1].values + 1)
        })
        df1_malig['pos_patient'] = pd.Categorical(df1_malig['pos_patient'], categories=['Pos.', 'Neg.'])
        med1_malig = df1_malig[df1_malig['pos_patient']=='Pos.']['log2_exp'].median()
        
        df2_malig = pd.DataFrame({
            'pos_patient': ['Pos.' if p in pos_patients_2 else 'Neg.' for p in malig_sng_med.index],
            'log2_exp': np.log2(malig_sng_med[gene2].values + 1)
        })
        df2_malig['pos_patient'] = pd.Categorical(df2_malig['pos_patient'], categories=['Pos.', 'Neg.'])
        med2_malig = df2_malig[df2_malig['pos_patient']=='Pos.']['log2_exp'].median()

        df12_malig = pd.DataFrame({
            'pos_patient': ['Pos.' if p in pos_patients_12 else 'Neg.' for p in malig_dbl_med.index],
            'log2_exp': np.log2(malig_dbl_med[multi_name].values + 1)
        })
        df12_malig['pos_patient'] = pd.Categorical(df12_malig['pos_patient'], categories=['Pos.', 'Neg.'])
        med12_malig = df12_malig[df12_malig['pos_patient']=='Pos.']['log2_exp'].median()
        
        # Prepare healthy dataframes
        cell_types = ha_sng_med.index
        tissues_list = [ct.split(':')[0] for ct in cell_types]
        
        df1_healthy = pd.DataFrame({
            'tissue': tissues_list,
            'full': [plot.add_smart_newlines(ct) for ct in cell_types],
            'log2_exp': np.log2(ha_sng_med[gene1].values + 1)
        })
        
        df2_healthy = pd.DataFrame({
            'tissue': tissues_list,
            'full': [plot.add_smart_newlines(ct) for ct in cell_types],
            'log2_exp': np.log2(ha_sng_med[gene2].values + 1)
        })
        
        df12_healthy = pd.DataFrame({
            'tissue': tissues_list,
            'full': [plot.add_smart_newlines(ct) for ct in cell_types],
            'log2_exp': np.log2(ref_dbl_med[multi_name].values + 1)
        })
        
        # Calculate limits
        max_val = max(
            df1_malig['log2_exp'].max(),
            df2_malig['log2_exp'].max(),
            df12_malig['log2_exp'].max()
        )
        
        # Create output path
        out_path = os.path.join(out_dir, f"{multi_name}.pdf")
        
        # Send data to R
        print("Sending data to R...")
        plot.pd2r("df1_malig", df1_malig)
        plot.pd2r("df2_malig", df2_malig)
        plot.pd2r("df12_malig", df12_malig)
        plot.pd2r("df1_healthy", df1_healthy)
        plot.pd2r("df2_healthy", df2_healthy)
        plot.pd2r("df12_healthy", df12_healthy)
        
        # Create plots in R
        print("Creating plots in R...")
        r(f'''
        library(ggplot2)
        library(patchwork)
        library(scales)
        
        # Parameters
        show <- {show}
        max_val <- {max_val}
        limits <- c(0, max_val)
        size <- 6
        
        med1 <- {med1_malig}
        med2 <- {med2_malig}
        med12 <- {med12_malig}
        
        per_1 <- {per_1}
        per_2 <- {per_2}
        per_12 <- {per_12}
        
        # Sort dataframes for column 3 (by gene1)
        df1h_sort1 <- df1_healthy[order(df1_healthy$log2_exp, decreasing=TRUE),]
        df1h_sort1$full <- factor(df1h_sort1$full, levels=df1h_sort1$full)
        
        df2h_sort1 <- df2_healthy
        df2h_sort1$full <- factor(df2h_sort1$full, levels=df1h_sort1$full)
        df2h_sort1 <- df2h_sort1[order(df2h_sort1$full),]
        
        df12h_sort1 <- df12_healthy
        df12h_sort1$full <- factor(df12h_sort1$full, levels=df1h_sort1$full)
        df12h_sort1 <- df12h_sort1[order(df12h_sort1$full),]
        
        # Sort dataframes for column 4 (by gene2)
        df2h_sort2 <- df2_healthy[order(df2_healthy$log2_exp, decreasing=TRUE),]
        df2h_sort2$full <- factor(df2h_sort2$full, levels=df2h_sort2$full)
        
        df1h_sort2 <- df1_healthy
        df1h_sort2$full <- factor(df1h_sort2$full, levels=df2h_sort2$full)
        df1h_sort2 <- df1h_sort2[order(df1h_sort2$full),]
        
        df12h_sort2 <- df12_healthy
        df12h_sort2$full <- factor(df12h_sort2$full, levels=df2h_sort2$full)
        df12h_sort2 <- df12h_sort2[order(df12h_sort2$full),]
        
        # Sort dataframes for column 5 (by combo)
        df12h_sort12 <- df12_healthy[order(df12_healthy$log2_exp, decreasing=TRUE),]
        df12h_sort12$full <- factor(df12h_sort12$full, levels=df12h_sort12$full)
        
        df1h_sort12 <- df1_healthy
        df1h_sort12$full <- factor(df1h_sort12$full, levels=df12h_sort12$full)
        df1h_sort12 <- df1h_sort12[order(df1h_sort12$full),]
        
        df2h_sort12 <- df2_healthy
        df2h_sort12$full <- factor(df2h_sort12$full, levels=df12h_sort12$full)
        df2h_sort12 <- df2h_sort12[order(df2h_sort12$full),]
        
        # ROW 1: Gene 1
        p1_1 <- ggplot(df1_malig, aes(pos_patient, log2_exp)) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
            geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=pos_patient), color="black", height=0, width=0.2) +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) + scale_x_discrete(drop=FALSE) +
            xlab("") +
            geom_hline(yintercept=med1, lty="dashed", color="firebrick3") +
            scale_fill_manual(values=c("Pos."="#28154C", "Neg."="lightgrey")) +
            ylab("{gene1}") + ggtitle(paste0(per_1, "%"))
        
        p1_2 <- ggplot(df1_healthy, aes(tissue, squish(log2_exp, limits))) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
            geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=tissue), color="black", height=0, width=0.2) +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) +
            geom_hline(yintercept=med1, lty="dashed", color="firebrick3") +
            xlab("") + ylab("") +
            theme(axis.text.x=element_text(size=size)) +
            scale_fill_manual(values=suppressMessages(create_pal_d(df1_healthy$tissue)))
        
        p1_3 <- ggplot(head(df1h_sort1, show), aes(full, squish(log2_exp, limits))) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_point(size=1.5, stroke=0.35, pch=21, aes(fill=tissue), color="black") +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) +
            geom_hline(yintercept=med1, lty="dashed", color="firebrick3") +
            xlab("") + ylab("") +
            theme(axis.text.x=element_text(size=size*0.75)) +
            scale_fill_manual(values=suppressMessages(create_pal_d(df1h_sort1$tissue))) +
            ggtitle("Top Pair 1 Off Targets")
        
        p1_4 <- ggplot(head(df1h_sort2, show), aes(full, squish(log2_exp, limits))) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_point(size=1.5, stroke=0.35, pch=21, aes(fill=tissue), color="black") +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) +
            geom_hline(yintercept=med1, lty="dashed", color="firebrick3") +
            xlab("") + ylab("") +
            theme(axis.text.x=element_text(size=size*0.75)) +
            scale_fill_manual(values=suppressMessages(create_pal_d(df1h_sort2$tissue))) +
            ggtitle("Top Pair 2 Off Targets")
        
        p1_5 <- ggplot(head(df1h_sort12, show), aes(full, squish(log2_exp, limits))) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_point(size=1.5, stroke=0.35, pch=21, aes(fill=tissue), color="black") +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) +
            geom_hline(yintercept=med1, lty="dashed", color="firebrick3") +
            xlab("") + ylab("") +
            theme(axis.text.x=element_text(size=size*0.75)) +
            scale_fill_manual(values=suppressMessages(create_pal_d(df1h_sort12$tissue))) +
            ggtitle("Top Combo 1&2 Off Targets")
        
        # ROW 2: Gene 2
        p2_1 <- ggplot(df2_malig, aes(pos_patient, log2_exp)) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
            geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=pos_patient), color="black", height=0, width=0.2) +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) + scale_x_discrete(drop=FALSE) +
            xlab("") +
            geom_hline(yintercept=med2, lty="dashed", color="firebrick3") +
            scale_fill_manual(values=c("Pos."="#28154C", "Neg."="lightgrey")) +
            ylab("{gene2}") + ggtitle(paste0(per_2, "%"))
        
        p2_2 <- ggplot(df2_healthy, aes(tissue, squish(log2_exp, limits))) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
            geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=tissue), color="black", height=0, width=0.2) +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) +
            geom_hline(yintercept=med2, lty="dashed", color="firebrick3") +
            xlab("") + ylab("") +
            theme(axis.text.x=element_text(size=size)) +
            scale_fill_manual(values=suppressMessages(create_pal_d(df2_healthy$tissue)))
        
        p2_3 <- ggplot(head(df2h_sort1, show), aes(full, squish(log2_exp, limits))) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_point(size=1.5, stroke=0.35, pch=21, aes(fill=tissue), color="black") +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) +
            geom_hline(yintercept=med2, lty="dashed", color="firebrick3") +
            xlab("") + ylab("") +
            theme(axis.text.x=element_text(size=size*0.75)) +
            scale_fill_manual(values=suppressMessages(create_pal_d(df2h_sort1$tissue)))
        
        p2_4 <- ggplot(head(df2h_sort2, show), aes(full, squish(log2_exp, limits))) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_point(size=1.5, stroke=0.35, pch=21, aes(fill=tissue), color="black") +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) +
            geom_hline(yintercept=med2, lty="dashed", color="firebrick3") +
            xlab("") + ylab("") +
            theme(axis.text.x=element_text(size=size*0.75)) +
            scale_fill_manual(values=suppressMessages(create_pal_d(df2h_sort2$tissue)))
        
        p2_5 <- ggplot(head(df2h_sort12, show), aes(full, squish(log2_exp, limits))) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_point(size=1.5, stroke=0.35, pch=21, aes(fill=tissue), color="black") +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) +
            geom_hline(yintercept=med2, lty="dashed", color="firebrick3") +
            xlab("") + ylab("") +
            theme(axis.text.x=element_text(size=size*0.75)) +
            scale_fill_manual(values=suppressMessages(create_pal_d(df2h_sort12$tissue)))
        
        # ROW 3: Combo
        p3_1 <- ggplot(df12_malig, aes(pos_patient, log2_exp)) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
            geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=pos_patient), color="black", height=0, width=0.2) +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) + scale_x_discrete(drop=FALSE) +
            xlab("") +
            geom_hline(yintercept=med12, lty="dashed", color="firebrick3") +
            scale_fill_manual(values=c("Pos."="#28154C", "Neg."="lightgrey")) +
            ylab("{gene1}:{gene2}") + ggtitle(paste0(per_12, "%"))
        
        p3_2 <- ggplot(df12_healthy, aes(tissue, squish(log2_exp, limits))) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
            geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=tissue), color="black", height=0, width=0.2) +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) +
            geom_hline(yintercept=med12, lty="dashed", color="firebrick3") +
            xlab("") + ylab("") +
            theme(axis.text.x=element_text(size=size)) +
            scale_fill_manual(values=suppressMessages(create_pal_d(df12_healthy$tissue)))
        
        p3_3 <- ggplot(head(df12h_sort1, show), aes(full, squish(log2_exp, limits))) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_point(size=1.5, stroke=0.35, pch=21, aes(fill=tissue), color="black") +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) +
            geom_hline(yintercept=med12, lty="dashed", color="firebrick3") +
            xlab("") + ylab("") +
            theme(axis.text.x=element_text(size=size*0.75)) +
            scale_fill_manual(values=suppressMessages(create_pal_d(df12h_sort1$tissue)))
        
        p3_4 <- ggplot(head(df12h_sort2, show), aes(full, squish(log2_exp, limits))) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_point(size=1.5, stroke=0.35, pch=21, aes(fill=tissue), color="black") +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) +
            geom_hline(yintercept=med12, lty="dashed", color="firebrick3") +
            xlab("") + ylab("") +
            theme(axis.text.x=element_text(size=size*0.75)) +
            scale_fill_manual(values=suppressMessages(create_pal_d(df12h_sort2$tissue)))
        
        p3_5 <- ggplot(head(df12h_sort12, show), aes(full, squish(log2_exp, limits))) +
            theme_jg(xText90=TRUE) +
            theme_small_margin(0.01) +
            geom_point(size=1.5, stroke=0.35, pch=21, aes(fill=tissue), color="black") +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) +
            geom_hline(yintercept=med12, lty="dashed", color="firebrick3") +
            xlab("") + ylab("") +
            theme(axis.text.x=element_text(size=size*0.75)) +
            scale_fill_manual(values=suppressMessages(create_pal_d(df12h_sort12$tissue)))
        
        # Combine with widths
        row1 <- p1_1 + p1_2 + p1_3 + p1_4 + p1_5 + plot_layout(widths=c(1, 5, 3, 3, 3))
        row2 <- p2_1 + p2_2 + p2_3 + p2_4 + p2_5 + plot_layout(widths=c(1, 5, 3, 3, 3))
        row3 <- p3_1 + p3_2 + p3_3 + p3_4 + p3_5 + plot_layout(widths=c(1, 5, 3, 3, 3))
        
        # Stack all rows
        final_plot <- row1 / row2 / row3
        
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
        f.write(f"Total Patients: {len(malig_med_adata.obs_names)}\n\n")
        f.write("Patient Names:\n")
        f.write('\n'.join(malig_med_adata.obs_names))
  
    print(f"\n{'='*60}")
    print(f"✓ All {len(multis)} plots saved to {out_dir}/")
    print(f"{'='*60}")


