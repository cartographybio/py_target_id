"""
Multi-specific biaxial plot visualization with density plots and comprehensive panels.
"""

__all__ = ['biaxial_summary']

import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import os
import time
from typing import List, Optional
from py_target_id import plot
from py_target_id import utils
from rpy2.robjects import r
from datetime import datetime

def biaxial_summary(
    multis: List[str],
    malig_adata,  # Malignant AnnData (cells × genes)
    malig_med_adata,  # Malignant median AnnData (patients × genes) with positivity layer
    ref_adata,  # Healthy atlas AnnData
    gtex_adata = None,  # Changed from utils.get_gtex_adata()
    tcga_adata = None,  # Changed from utils.get_tcga_adata()
    out_dir: str = "multi/biaxial_summary",
    show: int = 15,
    width: float = 28,
    height: float = 8,
    dpi: int = 300,
    titles: Optional[List[str]] = None
):
    """
    Create comprehensive biaxial plots for gene combinations.
    
    This creates a 5-panel plot showing:
    1. Legend of top off-targets
    2. Biaxial plot with density plots
    3. Co-expression distribution
    4. Single gene expression
    5. TCGA/GTEx bulk comparison
    
    Parameters
    ----------
    multis : List[str]
        Gene combinations in format ["GENE1_GENE2", ...]
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
        Number of top off-targets to display
    width : float
        Plot width in inches
    height : float
        Plot height in inches
    dpi : int
        Resolution for output
    titles : List[str], optional
        Custom titles for each combination (defaults to multis)
    """
    
    # Create output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Lazy-load reference data only if needed
    if gtex_adata is None:
        gtex_adata = utils.get_gtex_adata()
    
    if tcga_adata is None:
        tcga_adata = utils.get_tcga_adata()
    
    if titles is None:
        titles = multis
    
    # Start timing
    overall_start = time.time()
    
    # Parse gene combinations
    multis_split = np.array([m.split("_") for m in multis])
    combo_genes = np.unique(multis_split.ravel()).tolist()
    
    print(f"Processing {len(multis)} gene combinations with {len(combo_genes)} unique genes")
    print(f"Output directory: {out_dir}")
    
    # Load malignant data
    print("Loading malignant data...")
    malig_subset = malig_adata[:, combo_genes]
    
    if hasattr(malig_subset, 'to_memory'):
        print("Materializing malignant VirtualAnnData...")
        malig_subset = malig_subset.to_memory()
    
    # Dense
    if sparse.issparse(malig_subset.X):
        malig_subset.X = malig_subset.X.toarray()
    
    # Get patient IDs
    patients = malig_subset.obs['Patient'].values
    
    # Load healthy atlas
    print("Loading healthy atlas...")
    ref_subset = ref_adata[:, combo_genes]
    
    if hasattr(ref_subset, 'to_memory'):
        print("Materializing reference VirtualAnnData...")
        ref_subset = ref_subset.to_memory()
    
    # Dense
    if sparse.issparse(ref_subset.X):
        ref_subset.X = ref_subset.X.toarray()
    
    # Get cell types and clean special characters
    CT = ref_subset.obs['CellType'].values
    CT = pd.Series(CT).str.replace('α', 'a').str.replace('β', 'B').values
    
    # Compute single gene medians
    print("Computing single gene medians...")
    malig_sng_med = utils.summarize_matrix(
        malig_subset.X, 
        patients, 
        axis=0, 
        metric="median", 
        verbose=False
    )
    malig_sng_med.columns = combo_genes
    
    ha_sng_med = utils.summarize_matrix(
        ref_subset.X, 
        CT, 
        axis=0, 
        metric="median", 
        verbose=False
    )
    ha_sng_med.columns = combo_genes
    
    # Compute combo medians
    print("Computing combo medians...")
    combo1_idx = [combo_genes.index(g) for g in multis_split[:, 0]]
    combo2_idx = [combo_genes.index(g) for g in multis_split[:, 1]]
    
    malig_dbl = np.minimum(
        malig_subset.X[:, combo1_idx],
        malig_subset.X[:, combo2_idx]
    )
    
    ha_dbl = np.minimum(
        ref_subset.X[:, combo1_idx],
        ref_subset.X[:, combo2_idx]
    )
    
    malig_dbl_med = utils.summarize_matrix(
        malig_dbl, 
        patients, 
        axis=0, 
        metric="median", 
        verbose=False
    )
    malig_dbl_med.columns = multis
    
    ha_dbl_med = utils.summarize_matrix(
        ha_dbl, 
        CT, 
        axis=0, 
        metric="median", 
        verbose=False
    )
    ha_dbl_med.columns = multis
    
    # Get positivity matrix
    print("Extracting positivity data...")
    pos_mat = malig_med_adata[:, combo_genes].layers['positivity']
    if sparse.issparse(pos_mat):
        pos_mat = pos_mat.toarray()
    pos_mat = pos_mat.astype(int)
    
    # Load GTEx and TCGA if provided
    gtex_data = None
    tcga_data = None
    
    if gtex_adata is not None:
        print("Loading GTEx data...")
        
        # Materialize if it's a view to avoid double indexing
        if gtex_adata.is_view:
            print("  Materializing GTEx view...")
            gtex_adata = gtex_adata.to_memory()
        
        gtex_subset = gtex_adata[:, combo_genes]
        if hasattr(gtex_subset, 'to_memory'):
            gtex_subset = gtex_subset.to_memory()
        gtex_mat = gtex_subset.X.toarray() if sparse.issparse(gtex_subset.X) else gtex_subset.X
        gtex_id = gtex_subset.obs['GTEX'].values if 'GTEX' in gtex_subset.obs else gtex_subset.obs_names.values
        gtex_data = {'matrix': gtex_mat, 'id': gtex_id, 'genes': combo_genes}
    
    if tcga_adata is not None:
        print("Loading TCGA data...")
        
        # Materialize if it's a view to avoid double indexing
        if tcga_adata.is_view:
            print("  Materializing TCGA view...")
            tcga_adata = tcga_adata.to_memory()
        
        tcga_subset = tcga_adata[:, combo_genes]
        if hasattr(tcga_subset, 'to_memory'):
            tcga_subset = tcga_subset.to_memory()
        tcga_mat = tcga_subset.X.toarray() if sparse.issparse(tcga_subset.X) else tcga_subset.X
        tcga_samples = tcga_subset.obs_names.values
        tcga_id = np.array([s.split('#')[0] for s in tcga_samples])
        tcga_data = {'matrix': tcga_mat, 'id': tcga_id, 'samples': tcga_samples, 'genes': combo_genes}
    
    # Define essential tissues
    essential_survival = ["Brain", "Heart", "Lung", "Liver"]
    non_essential_survival = [
        "Testes", "Ovary", "Prostate", "Uterus", "Cervix", 
        "Fallopian_Tube", "Breast", "Adipose", "Appendix"
    ]
    
    # GTEx tissue categories
    gtex_essential = [
        "Brain.Hippo", "Brain.Cortex", "Brain.Putamen", "Brain.ACC_BA24",
        "Brain.Cerebellar", "Brain.Frontal_BA9", "Brain.Spinal_C1", "Brain.SubNigra",
        "Brain.Basal_G_NAcc", "Brain.Hypothal", "Brain.Cerebellum", "Brain.Basal_G_Caud",
        "Brain.Amygdala", "Heart.Atrium", "Heart.Ventr", "Lung", "Liver"
    ]
    
    gtex_non_essential = [
        "Testis", "Ovary", "Prostate", "Uterus", "Cervix.Endo", "Cervix.Ecto",
        "Fallopian", "Vagina", "Breast.Mammary", "Adipose.Subcut", "Adipose.Visc"
    ]
    
    elapsed = (time.time() - overall_start) / 60
    print(f"------ Data preparation complete - | Total: {elapsed:.1f}min")
    
    # Process each gene combination
    for idx, (gene1, gene2) in enumerate(multis_split):
        print(f"\n{'='*60}")
        print(f"Processing {idx+1}/{len(multis)}: {gene1}_{gene2}")
        print(f"{'='*60}")
        
        multi_name = multis[idx]
        
        # Get positivity for this combination
        gene1_idx = combo_genes.index(gene1)
        gene2_idx = combo_genes.index(gene2)
        
        pos_1 = pos_mat[:, gene1_idx].astype(bool)
        pos_2 = pos_mat[:, gene2_idx].astype(bool)
        pos_12 = pos_1 & pos_2
        
        pos_patients_12 = malig_med_adata.obs['Patient'].values[pos_12]
        
        # Prepare tumor data
        dfT2 = _calculate_and_gate_by_id(
            malig_subset[:, [gene1_idx, gene2_idx]].X,
            patients
        )
        dfT2['Type'] = 'Malignant'
        dfT2['Capped'] = False
        
        # Prepare reference data
        dfR2 = _calculate_and_gate_by_id(
            ref_subset[:, [gene1_idx, gene2_idx]].X,
            CT
        )
        dfR2['Type'] = 'Reference'
        dfR2 = dfR2.sort_values('xm', ascending=False)  # Don't reset index
        
        # Add essential classification
        tissues = [ct.split(':')[0] for ct in dfR2.index]
        dfR2['Essential'] = ~pd.Series(tissues).isin(non_essential_survival).values
        dfR2['Critical'] = pd.Series(tissues).isin(essential_survival).values
        dfR2['Capped'] = False
        
        # Add smart newlines to cell type names
        dfR2['ID'] = [plot.add_smart_newlines(ct, max_length=30) for ct in dfR2.index]
        dfR2['ID_2'] = range(1, len(dfR2) + 1)
        
        # Get limits from tumor
        values = np.concatenate([dfT2['x1'].values, dfT2['x2'].values])
        third_highest = np.sort(values)[-3] if len(values) >= 3 else np.max(values)
        limits = [0, third_highest * 1.05]
        
        # Cap values
        dfT2['Capped'] = (dfT2['x1'] > limits[1]) | (dfT2['x2'] > limits[1])
        dfT2['x1'] = np.minimum(dfT2['x1'], limits[1])
        dfT2['x2'] = np.minimum(dfT2['x2'], limits[1])
        
        dfR2['Capped'] = (dfR2['x1'] > limits[1]) | (dfR2['x2'] > limits[1])
        dfR2['x1'] = np.minimum(dfR2['x1'], limits[1])
        dfR2['x2'] = np.minimum(dfR2['x2'], limits[1])
        
        # Log2 transform
        dfT2['lx1'] = np.log2(dfT2['x1'] + 1)
        dfT2['lx2'] = np.log2(dfT2['x2'] + 1)
        dfT2['lxm'] = np.log2(dfT2['xm'] + 1)
        
        dfR2['lx1'] = np.log2(dfR2['x1'] + 1)
        dfR2['lx2'] = np.log2(dfR2['x2'] + 1)
        dfR2['lxm'] = np.log2(dfR2['xm'] + 1)
        
        # Prepare single gene data for panel 2_12
        df_sgl_list = []
        
        # Malignant data
        for gene_idx, gene_name in enumerate([gene1, gene2]):
            df_gene = pd.DataFrame({
                'lx': np.log2(malig_sng_med[gene_name].values + 1),
                'Pos': malig_sng_med.index.isin(pos_patients_12),
                'Type': 'Tumor',
                'var2': f"{gene_idx+1}.{gene_name}"
            })
            df_sgl_list.append(df_gene)
        
        # Reference data - fix the Pos column logic
        top_ref_cell_types = dfR2.head(show).index.tolist()  # Original cell type names without newlines
        for gene_idx, gene_name in enumerate([gene1, gene2]):
            df_gene = pd.DataFrame({
                'lx': np.log2(ha_sng_med[gene_name].values + 1),
                'Pos': ha_sng_med.index.isin(top_ref_cell_types),  # Compare against original names
                'Type': 'Reference',
                'var2': f"{gene_idx+1}.{gene_name}"
            })
            df_sgl_list.append(df_gene)
        
        df_sgl = pd.concat(df_sgl_list, ignore_index=True)
        
        # Create pos_type classification - FIXED
        # Convert boolean to string properly
        df_sgl['pos_type'] = df_sgl['Pos'].map({True: 'True', False: 'False'}) + df_sgl['Type']
        df_sgl['pos_type2'] = '3.Reference'
        df_sgl.loc[df_sgl['pos_type'] == 'FalseReference', 'pos_type2'] = '3.Reference'
        df_sgl.loc[df_sgl['pos_type'] == 'FalseTumor', 'pos_type2'] = '2.Malig\n Neg. Multi'
        df_sgl.loc[df_sgl['pos_type'] == 'TrueTumor', 'pos_type2'] = '1.Malig\n Pos. Multi'
        df_sgl = df_sgl.sort_values('pos_type')
        
        # Prepare TCGA/GTEx data if available
        df_tcga = None
        df_gtex_ess = None
        df_gtex_non_ess = None
        df_gtex_other = None
        
        if tcga_data is not None and gtex_data is not None:
            # TCGA data
            df_tcga = pd.DataFrame({
                'x1': tcga_data['matrix'][:, gene1_idx],
                'x2': tcga_data['matrix'][:, gene2_idx],
                'Type': tcga_data['id'],
                'ID': tcga_data['id']
            })
            df_tcga['xm'] = np.minimum(df_tcga['x1'], df_tcga['x2'])
            
            # Calculate bulk limits
            bulk_values = np.concatenate([df_tcga['x1'].values, df_tcga['x2'].values])
            bulk_third_highest = np.sort(bulk_values)[-3] if len(bulk_values) >= 3 else np.max(bulk_values)
            bulk_limits = [0, bulk_third_highest * 1.05]
            
            # Cap and log transform TCGA
            df_tcga['Capped'] = (df_tcga['x1'] > bulk_limits[1]) | (df_tcga['x2'] > bulk_limits[1])
            df_tcga['lx1'] = np.log2(np.minimum(df_tcga['x1'], bulk_limits[1]) + 1)
            df_tcga['lx2'] = np.log2(np.minimum(df_tcga['x2'], bulk_limits[1]) + 1)
            
            # GTEx data - calculate median per tissue
            df_gtex_raw = pd.DataFrame({
                'T1': gtex_data['matrix'][:, gene1_idx],
                'T2': gtex_data['matrix'][:, gene2_idx],
                'Type': 'GTEX',
                'ID': gtex_data['id']
            })
            
            df_gtex = df_gtex_raw.groupby('ID').agg({
                'T1': 'median',
                'T2': 'median'
            }).reset_index()
            df_gtex.columns = ['ID', 'x1', 'x2']
            df_gtex['Type'] = 'GTEX'
            df_gtex['xm'] = np.minimum(df_gtex['x1'], df_gtex['x2'])
            
            # Cap and log transform GTEx
            df_gtex['Capped'] = (df_gtex['x1'] > bulk_limits[1]) | (df_gtex['x2'] > bulk_limits[1])
            df_gtex['lx1'] = np.log2(np.minimum(df_gtex['x1'], bulk_limits[1]) + 1)
            df_gtex['lx2'] = np.log2(np.minimum(df_gtex['x2'], bulk_limits[1]) + 1)
            
            # Sort by expression
            df_gtex = df_gtex.sort_values('xm', ascending=False)
            
            # Split into categories
            df_gtex_ess = df_gtex[df_gtex['ID'].isin(gtex_essential)].copy()
            df_gtex_non_ess = df_gtex[df_gtex['ID'].isin(gtex_non_essential)].copy()
            df_gtex_other = df_gtex[~df_gtex['ID'].isin(gtex_essential + gtex_non_essential)].copy()
        
        # Create output path
        out_path = os.path.join(out_dir, f"{multi_name}.pdf")
        
        # Send data to R for plotting
        print("Sending data to R...")
        plot.pd2r("dfT2", dfT2)
        plot.pd2r("dfR2", dfR2)
        plot.pd2r("df_sgl", df_sgl)
        
        if df_tcga is not None:
            plot.pd2r("df_tcga", df_tcga)
            plot.pd2r("df_gtex_ess", df_gtex_ess)
            plot.pd2r("df_gtex_non_ess", df_gtex_non_ess)
            plot.pd2r("df_gtex_other", df_gtex_other)
        
        # Create plots in R
        print("Creating plots in R...")
        
        # Build R plotting code
        r_code = f'''
        library(ggplot2)
        library(patchwork)
        
        # Parameters
        limits <- c({limits[0]}, {limits[1]})
        log_limits <- log2(limits + 1)
        show <- {show}
        
        # Get positive patients
        pos_patients_12 <- dfT2$ID[dfT2$ID %in% c({", ".join([f'"{p}"' for p in pos_patients_12])})]
        
        # Essential/non-essential organs
        essential_survival <- c({", ".join([f'"{x}"' for x in essential_survival])})
        non_essential_survival <- c({", ".join([f'"{x}"' for x in non_essential_survival])})
        
        # Get top reference entries for legend
        legend_entries <- head(dfR2, show)
        organs <- sapply(strsplit(as.character(legend_entries$ID), ":"), "[", 1)
        legend_entries$text_color <- "black"
        
        for(essential_organ in essential_survival) {{
            legend_entries$text_color[grepl(paste0("^", essential_organ), organs)] <- "red"
        }}
        
        for(non_essential_organ in non_essential_survival) {{
            legend_entries$text_color[grepl(paste0("^", non_essential_organ), organs)] <- "grey50"
        }}
        
        legend_df <- data.frame(
            Number = legend_entries$ID_2,
            CellType = legend_entries$ID,
            Color = legend_entries$text_color,
            stringsAsFactors = FALSE
        )
        
        # Legend plot
        p_legend <- ggplot(legend_df, aes(x = 0, y = rev(1:nrow(legend_df)))) +
            geom_text(aes(label = paste0(Number, ": ", CellType), color = Color), 
                     hjust = 0, size = 3.25, lineheight = 0.9) +
            scale_color_identity() +
            xlim(0, 1) +
            ylim(0.5, nrow(legend_df) + 0.5) +
            theme_void() +
            theme(plot.margin = unit(c(10,10,10,10), "pt"),
                  plot.title = element_text(size = 15.6, hjust = 0.5)) +
            ggtitle("Top Off-Targets")
        
        # Main biaxial plot (p1)
        dfT2_pos <- dfT2[dfT2$ID %in% pos_patients_12, ]
        dfR2_show <- head(dfR2, show)
        dfR2_rest <- tail(dfR2, -show)
        
        p1 <- ggplot() +
            geom_vline(xintercept = median(dfT2_pos$lx1), lty = "dashed", color = "firebrick3", linewidth = 0.5) +
            geom_hline(yintercept = median(dfT2_pos$lx2), lty = "dashed", color = "firebrick3", linewidth = 0.5) +
            geom_point(data = dfR2_rest, aes(lx1, lx2, shape = Capped), fill = pal_cart[2], size = 1) +
            geom_point(data = dfR2_show, aes(lx1, lx2, shape = Capped), fill = pal_cart[2], size = 2) +
            geom_point(data = dfT2[!(dfT2$ID %in% pos_patients_12), ], aes(lx1, lx2, shape = Capped), fill = pal_cart[5], size = 1) +
            geom_point(data = dfT2_pos, aes(lx1, lx2, shape = Capped), fill = pal_cart[5], size = 2) +
            geom_abline(slope = 1, intercept = 0, lty = "dashed") +
            scale_shape_manual(values = c("TRUE" = 24, "FALSE" = 21)) +
            xlim(log_limits) +
            ylim(log_limits) +
            xlab("Log2({gene1} + 1)") +
            ylab("Log2({gene2} + 1)") +
            theme(legend.position = "none") +
            theme_small_margin()
        
        # Add colored labels
        dfR2_show$text_color <- legend_df$Color
        p1 <- p1 +
            ggrepel::geom_label_repel(
                data = dfR2_show, aes(lx1, lx2, label = ID_2, color = text_color),
                size = 2, alpha = 0.8, max.overlaps = 99) +
            scale_color_identity()
        
        # Co-expression plot (p2)
        pos_jitter <- position_jitter(width = 0.2, height = 0, seed = 42)
        
        p2 <- ggplot() +
            geom_hline(yintercept = median(dfT2_pos$lxm), lty = "dashed", color = "firebrick3", linewidth = 0.5) +
            geom_point(data = dfR2_rest, aes(Type, lxm), fill = pal_cart[2], size = 1, shape = 21, position = pos_jitter) +
            geom_point(data = dfR2_show, aes(Type, lxm), fill = pal_cart[2], size = 2, shape = 21, position = pos_jitter) +
            geom_point(data = dfT2[!(dfT2$ID %in% pos_patients_12), ], aes(Type, lxm), fill = pal_cart[5], size = 1, shape = 21, position = pos_jitter) +
            geom_point(data = dfT2_pos, aes(Type, lxm), fill = pal_cart[5], size = 2, shape = 21, position = pos_jitter) +
            geom_boxplot(data = dfT2_pos, aes(Type, lxm), outlier.size = 0, outlier.stroke = 0, color = pal_cart[5], fill = NA, width = 0.2) +
            ylab("Log2(Co-Expression + 1)") +
            theme_small_margin()
        
        # Add colored labels for p2
        p2 <- p2 +
            ggrepel::geom_label_repel(
                data = dfR2_show, aes(Type, lxm, label = ID_2, color = text_color),
                size = 2, alpha = 0.8, max.overlaps = 99, position = pos_jitter) +
            scale_color_identity()
        
        # Single gene expression plot (p2_12) - FIXED COLOR MAPPING
        df_sgl_pos <- df_sgl[df_sgl$pos_type2 == "1.Malig\\n Pos. Multi", ]
        
        p2_12 <- ggplot() +
            geom_point(data = df_sgl, aes(pos_type2, lx, fill = pos_type, size = Pos), shape = 21, position = pos_jitter) +
            facet_wrap(~var2, ncol = 1, strip.position = "right", scales = "free") +
            scale_size_manual(values = c("TRUE" = 2, "FALSE" = 1)) +
            ylab("Log2(Expression + 1)") +
            theme_small_margin() +
            geom_boxplot(data = df_sgl_pos, aes(pos_type2, lx), outlier.size = 0, outlier.stroke = 0, color = pal_cart[5], fill = NA, width = 0.2) +
            scale_fill_manual(values = c(
                "TrueReference" = "firebrick3", 
                "TrueTumor" = pal_cart[5], 
                "FalseTumor" = "lightgrey", 
                "FalseReference" = pal_cart[2]
            )) +
            geom_hline(data = df_sgl_pos, aes(yintercept = median(lx)), color = "firebrick3", linetype = "dashed", linewidth = 0.5) +
            xlab("") +
            theme(legend.position = "none")
        
        # Density plots for p1
        left_density <- ggplot(dfT2_pos, aes(x = lx2)) +
            geom_density(fill = pal_cart[5], alpha = 0.6) +
            xlim(log_limits) +
            coord_flip() +
            scale_y_reverse() +
            theme_void() +
            theme(plot.margin = unit(c(0,15,0,0), "pt"))
        
        bottom_density <- ggplot(dfT2_pos, aes(x = lx1)) +
            geom_density(fill = pal_cart[5], alpha = 0.6) +
            xlim(log_limits) +
            scale_y_reverse() +
            theme_void() +
            theme(plot.margin = unit(c(0,0,0,0), "pt"))
        
        # Apply consistent themes
        p1 <- p1 + theme(plot.margin = unit(c(0,0,0,20), "pt"),
                         axis.text = element_text(size = 12),
                         axis.title = element_text(size = 12))
        
        p2 <- p2 + theme(plot.margin = unit(c(0,0,0,0), "pt"),
                         axis.text = element_text(size = 12),
                         axis.title = element_text(size = 12))
        
        p2_12 <- p2_12 + theme(plot.margin = unit(c(0,0,0,0), "pt"),
                               axis.text = element_text(size = 12),
                               axis.title = element_text(size = 12),
                               axis.text.x = element_text(size = 8))
        
        # Create layouts
        p1_plus <- left_density + plot_spacer() + p1 +
                   plot_spacer() + plot_spacer() + bottom_density +
                   plot_layout(ncol = 3, nrow = 2, 
                              widths = c(0.3, 0.1, 6),
                              heights = c(6, 0.4))
        
        p2_plus <- plot_spacer() + p2 +
                   plot_spacer() + plot_spacer() +
                   plot_layout(ncol = 2, nrow = 2,
                              widths = c(0.4, 6),
                              heights = c(6, 0.4))
        
        p2_12_plus <- plot_spacer() + p2_12 +
                      plot_spacer() + plot_spacer() +
                      plot_layout(ncol = 2, nrow = 2,
                                 widths = c(0.4, 6),
                                 heights = c(6, 0.4))
        '''
        
        # Add TCGA/GTEx plot if available
        # Add TCGA/GTEx plot if available
        if df_tcga is not None:
            r_code += f'''
            # TCGA/GTEx plot (p3)
            bulk_limits <- c({bulk_limits[0]}, {bulk_limits[1]})
            bulk_log_limits <- log2(bulk_limits + 1)
            
            # Create legend for top GTEx tissues
            df_gtex_all <- rbind(df_gtex_ess, df_gtex_non_ess, df_gtex_other)
            df_gtex_all <- df_gtex_all[order(-df_gtex_all$xm), ]
            gtex_legend_entries <- head(df_gtex_all, show)
            gtex_legend_entries$ID_2 <- 1:nrow(gtex_legend_entries)
            
            # Assign colors based on category
            gtex_legend_entries$text_color <- "black"
            gtex_legend_entries$text_color[gtex_legend_entries$ID %in% c({", ".join([f'"{x}"' for x in gtex_essential])})] <- "red"
            gtex_legend_entries$text_color[gtex_legend_entries$ID %in% c({", ".join([f'"{x}"' for x in gtex_non_essential])})] <- "grey50"
            
            gtex_legend_df <- data.frame(
                Number = gtex_legend_entries$ID_2,
                Tissue = gtex_legend_entries$ID,
                Color = gtex_legend_entries$text_color,
                stringsAsFactors = FALSE
            )
            
            # GTEx Legend plot
            p_gtex_legend <- ggplot(gtex_legend_df, aes(x = 0, y = rev(1:nrow(gtex_legend_df)))) +
                geom_text(aes(label = paste0(Number, ": ", Tissue), color = Color), 
                         hjust = 0, size = 3.25, lineheight = 0.9) +
                scale_color_identity() +
                xlim(0, 1) +
                ylim(0.5, nrow(gtex_legend_df) + 0.5) +
                theme_void() +
                theme(plot.margin = unit(c(10,10,10,10), "pt"),
                      plot.title = element_text(size = 15.6, hjust = 0.5)) +
                ggtitle("Top GTEx Off-Targets")
            
            # Main TCGA/GTEx plot
            p3 <- ggplot() +
                geom_point(data = df_tcga, aes(lx1, lx2), fill = pal_cart[5], pch = 21, size = 2) +
                geom_point(data = df_gtex_non_ess, aes(lx1, lx2), fill = "grey", pch = 24, size = 3) +
                geom_point(data = df_gtex_other, aes(lx1, lx2), fill = pal_cart[2], pch = 24, size = 3) +
                geom_point(data = df_gtex_ess, aes(lx1, lx2), fill = "firebrick3", pch = 24, size = 3) +
                geom_abline(slope = 1, intercept = 0, lty = "dashed", color = "black") +
                xlim(bulk_log_limits) +
                ylim(bulk_log_limits) +
                geom_vline(xintercept = log2(11), lty = "dashed", color = "firebrick3", linewidth = 0.5) +
                geom_hline(yintercept = log2(11), lty = "dashed", color = "firebrick3", linewidth = 0.5) +
                xlab("Log2({gene1} TPM + 1)") +
                ylab("Log2({gene2} TPM + 1)") +
                theme(legend.position = "none") +
                theme(plot.margin = unit(c(0,0,0,0), "pt"),
                      axis.text = element_text(size = 12),
                      axis.title = element_text(size = 12))
            
            # Add numbered labels to top GTEx tissues
            p3 <- p3 +
                ggrepel::geom_label_repel(
                    data = gtex_legend_entries, 
                    aes(lx1, lx2, label = ID_2, color = text_color),
                    size = 2, alpha = 0.8, max.overlaps = 99) +
                scale_color_identity()
            
            p3_plus <- plot_spacer() + p3 +
                       plot_spacer() + plot_spacer() +
                       plot_layout(ncol = 2, nrow = 2,
                                  widths = c(0.4, 6),
                                  heights = c(6, 0.4))
            
            # Final plot with 6 panels (added GTEx legend on far right)
            final_plot <- wrap_plots(
                p_legend & theme(plot.margin = unit(c(5,5,5,5), "pt")),
                p1_plus & theme(plot.margin = unit(c(5,0,5,0), "pt")),
                p2_plus & theme(plot.margin = unit(c(5,5,5,-20), "pt")),
                p2_12_plus & theme(plot.margin = unit(c(5,5,5,-20), "pt")),
                p3_plus & theme(plot.margin = unit(c(5,5,5,-20), "pt")),
                p_gtex_legend & theme(plot.margin = unit(c(5,5,5,5), "pt")),
                ncol = 6,
                widths = c(1.5, 3, 2, 2, 3, 1.5)
            ) + plot_annotation(title = "{titles[idx]}")
            '''
        else:
            # Final plot with 4 panels (no TCGA/GTEx)
            r_code += f'''
            # Final plot with 4 panels
            final_plot <- wrap_plots(
                p_legend & theme(plot.margin = unit(c(5,5,5,5), "pt")),
                p1_plus & theme(plot.margin = unit(c(5,0,5,0), "pt")),
                p2_plus & theme(plot.margin = unit(c(5,5,5,-20), "pt")),
                p2_12_plus & theme(plot.margin = unit(c(5,5,5,-20), "pt")),
                ncol = 4,
                widths = c(1.5, 3, 2, 2)
            ) + plot_annotation(title = "{titles[idx]}")
            '''
        
        r_code += f'''
        # Save plot
        suppressWarnings(ggsave("{out_path}", final_plot, width = {width}, height = {height}))
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
        
        elapsed = (time.time() - overall_start) / 60
        print(f"------ Processed {idx+1}/{len(multis)} - | Total: {elapsed:.1f}min")

    #Write custom done
    out_file = out_dir + "/finished.txt"

    with open(out_file, 'w') as f:
        f.write(f"Finished: {datetime.now()}\n\n")
        f.write(f"Total Patients: {len(malig_med_adata.obs_names)}\n\n")
        f.write("Patient Names:\n")
        f.write('\n'.join(malig_med_adata.obs_names))    

    elapsed = (time.time() - overall_start) / 60
    print(f"\n{'='*60}")
    print(f"✓ All {len(multis)} plots saved to {out_dir}/")
    print(f"  Completed in {elapsed:.2f} minutes")
    print(f"{'='*60}")


def _calculate_and_gate_by_id(matrix, groups):
    """
    Calculate AND gate (minimum) for each group.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix of shape (n_cells, 2) with expression values for 2 genes
    groups : np.ndarray
        Group labels for each cell
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: x1, x2, xm (minimum) for each unique group
    """
    df = pd.DataFrame({
        'x1': matrix[:, 0],
        'x2': matrix[:, 1],
        'group': groups
    })
    
    # Calculate median AND gate for each group
    results = []
    for group_name, group_data in df.groupby('group'):
        result = _calculate_and_gate_median(group_data['x1'].values, group_data['x2'].values)
        result['group'] = group_name
        results.append(result)
    
    result_df = pd.concat(results, axis=1).T
    result_df = result_df.set_index('group')
    result_df['ID'] = result_df.index
    
    # Ensure numeric columns are numeric
    for col in ['x1', 'x2', 'xm']:
        result_df[col] = pd.to_numeric(result_df[col])
    
    return result_df


def _calculate_and_gate_median(x1, x2):
    """
    Calculate median AND gate (minimum) values.
    
    Parameters
    ----------
    x1 : np.ndarray
        Expression values for gene 1
    x2 : np.ndarray
        Expression values for gene 2
    
    Returns
    -------
    pd.Series
        Series with x1, x2, and xm (minimum) median values
    """
    xm = np.minimum(x1, x2)
    
    # Sort by minimum
    sort_idx = np.argsort(xm)
    x1_sorted = x1[sort_idx]
    x2_sorted = x2[sort_idx]
    xm_sorted = xm[sort_idx]
    
    # Get median index
    med_idx = len(xm) // 2
    
    result_x1 = x1_sorted[med_idx]
    result_x2 = x2_sorted[med_idx]
    result_xm = xm_sorted[med_idx]
    
    # Calculate actual median of xm
    median_xm = np.median(xm)
    
    if result_xm != median_xm:
        result_xm = median_xm
        
        # Determine which gene has lower average (tie goes to x1)
        if np.mean(x1) <= np.mean(x2):
            # x1 is limiting factor
            result_x1 = result_xm
            result_x2 = max(result_x2, result_xm)
        else:
            # x2 is limiting factor
            result_x2 = result_xm
            result_x1 = max(result_x1, result_xm)
    
    return pd.Series({
        'x1': result_x1,
        'x2': result_x2,
        'xm': result_xm
    })