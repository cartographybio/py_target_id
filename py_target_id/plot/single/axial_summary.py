"""
Single gene axial summary plot visualization across datasets.
Comprehensive view of target expression in malignant vs healthy tissues.
"""

__all__ = ['axial_summary', 'axial_summary_single']

import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import os
import time
from typing import Optional, List
from py_target_id import plot
from py_target_id import utils
from rpy2.robjects import r
from datetime import datetime


def axial_summary(
    genes: List[str],
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
    titles: Optional[List[str]] = None
):
    """
    Create axial summary plots for multiple gene targets.
    
    Parameters
    ----------
    genes : List[str]
        List of gene target names
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
    titles : List[str], optional
        Custom titles for each gene (defaults to gene names)
    """
    
    if titles is None:
        titles = genes
    
    print(f"Processing {len(genes)} genes")
    overall_start = time.time()
    
    for idx, (gene, title) in enumerate(zip(genes, titles), 1):
        try:
            print(f"\n{'='*60}")
            print(f"Processing {idx}/{len(genes)}: {gene}")
            print(f"{'='*60}")
            
            axial_summary_single(
                gene=gene,
                malig_adata=malig_adata,
                malig_med_adata=malig_med_adata,
                ref_adata=ref_adata,
                gtex_adata=gtex_adata,
                tcga_adata=tcga_adata,
                out_dir=out_dir,
                show=show,
                width=width,
                height=height,
                dpi=dpi,
                title=title
            )
        except Exception as e:
            print(f"✗ Error processing {gene}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    elapsed = (time.time() - overall_start) / 60
    print(f"\n{'='*60}")
    print(f"✓ All {len(genes)} plots completed!")
    print(f"  Total time: {elapsed:.2f} minutes")
    print(f"  Average per gene: {elapsed/len(genes):.2f} minutes")
    print(f"  Output directory: {out_dir}/")
    print(f"{'='*60}")


def axial_summary_single(
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
        theme_small_margin(0.1) +
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
        theme_small_margin(0.1) +
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
        theme_small_margin(0.1) +
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
    tcga_type_labels$label <- paste0(as.character(tcga_type_labels$Type), " (", round(tcga_type_labels$pct, 1), "%)")
    tcga_type_labels <- tcga_type_labels[order(tcga_type_labels$Type), ]
    
    # Map the labels back to df_tcga
    df_tcga$Type_label <- tcga_type_labels$label[match(as.character(df_tcga$Type), as.character(tcga_type_labels$Type))]
    df_tcga$Type_label <- factor(df_tcga$Type_label, levels=tcga_type_labels$label)
    
    # TCGA plot (left half)
    p4_tcga <- ggplot(df_tcga, aes(x=Type_label, y=log2_exp)) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.1) +
        geom_jitter(size=0.8, stroke=0.2, pch=21, fill="firebrick3", color="black", height=0, width=0.2, alpha=0.5) +
        geom_boxplot(fill=NA, color="black", outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.6) +
        theme(legend.position="none") +
        coord_cartesian(ylim=bulk_limits) +
        geom_hline(yintercept=bulk_line, lty="dashed", color="firebrick3") +
        xlab("") + ylab("Log2(TPM + 1)") +
        theme(axis.text.x=element_text(size=5, angle=90)) +
        ggtitle(paste0("TCGA"))
    
    # GTEx plot (right half)
    p4_gtex <- ggplot(df_gtex, aes(Type, log2_exp)) +
        theme_jg(xText90=TRUE) +
        theme_small_margin(0.1) +
        geom_jitter(size=0.8, stroke=0.2, pch=21, fill="grey", color="black", height=0, width=0.2, alpha=0.5) +
        geom_boxplot(fill=NA, color="black", outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.6) +
        theme(legend.position="none") +
        coord_cartesian(ylim=bulk_limits) +
        geom_hline(yintercept=bulk_line, lty="dashed", color="firebrick3") +
        xlab("") + ylab("Log2(TPM + 1)") +
        theme(axis.text.x=element_text(size=5, angle=90)) +
        ggtitle(paste0("GTEx"))
    
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