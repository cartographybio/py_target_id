"""
Multi-specific target vs known targets dot plot visualization.
"""

# Define what gets exported
__all__ = ['axial_vs_known']

import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import os
import time
from typing import Literal, Union, List
from py_target_id import plot
from py_target_id import utils
from rpy2.robjects import r
from datetime import datetime

def axial_vs_known(
    multis: list,
    malig_adata,  # Malignant AnnData (cells × genes)
    malig_med_adata,  # Malignant median AnnData (patients × genes) with positivity layer
    ref_adata,  # Healthy atlas AnnData
    out_dir: str = "multi/multi_axial_vs_known",
    show: int = 15,
    known: List[str] = ["DLL3", "MET", "EGFR", "TACSTD2", "CEACAM5", "ERBB3", "MSLN"],
    width: float = 20,
    height: float = 10,
    dpi: int = 300
):
    """
    Create multi-axial plots comparing gene combinations to known targets.
    
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
    known : List[str]
        List of known target genes to compare against
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
    combo_genes = np.unique(multis_split.ravel()).tolist()
    genes = combo_genes + known
    genes = list(np.unique(genes))  # Remove duplicates
    combo1 = multis_split[:, 0].tolist()
    combo2 = multis_split[:, 1].tolist()
    
    print(f"Processing {len(multis)} gene combinations")
    print(f"Unique genes in combos: {len(combo_genes)}")
    print(f"Known targets: {len(known)}")
    print(f"Total unique genes: {len(genes)}")
    print(f"Output directory: {out_dir}")
    
    # Load malignant data
    print("Loading malignant data...")
    malig_subset = malig_adata[:, genes]
    
    if hasattr(malig_subset, 'to_memory'):
        print("Materializing malignant VirtualAnnData...")
        malig_subset = malig_subset.to_memory()
    
    # Dense
    malig_subset.X = malig_subset.X.toarray()

    # Get positivity matrix from malig_med_adata
    print("Extracting positivity data...")
    pos_mat = malig_med_adata[:, genes].layers['positivity']
    pos_mat = pos_mat.toarray().astype(int)
    patient_names = malig_med_adata.obs['Patient'].values
    
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
    malig_sng_med = utils.summarize_matrix(
        malig_subset.X, 
        malig_subset.obs['Patient'].values, 
        axis=0, 
        metric="median", 
        verbose=False
    )
    malig_sng_med.columns = genes

    print("Computing healthy single gene medians...")
    ha_sng_med = utils.summarize_matrix(
        ref_subset.X, 
        ref_subset.obs['CellType'], 
        axis=0, 
        metric="median", 
        verbose=False
    )
    ha_sng_med.columns = genes

    print("Computing malignant combo medians...")
    malig_dbl_med = utils.summarize_matrix(
        np.minimum(malig_subset[:, combo1].X, malig_subset[:, combo2].X), 
        malig_subset.obs['Patient'].values, 
        axis=0, 
        metric="median", 
        verbose=False
    )
    malig_dbl_med.columns = multis

    print("Computing healthy combo medians...")
    ref_dbl_med = utils.summarize_matrix(
        np.minimum(ref_subset[:, combo1].X, ref_subset[:, combo2].X), 
        ref_subset.obs['CellType'].values, 
        axis=0, 
        metric="median", 
        verbose=False
    )
    ref_dbl_med.columns = multis

    print("Data preparation complete. Starting plotting...")
    
    # Process each gene combination
    for idx, (gene1, gene2) in enumerate(multis_split):
        print(f"\n{'='*60}")
        print(f"Processing {idx+1}/{len(multis)}: {gene1}_{gene2}")
        print(f"{'='*60}")
        
        multi_name = multis[idx]
        
        # Get positivity for combination
        pos_gene1 = malig_med_adata[:, gene1].layers['positivity'].flatten()
        pos_gene2 = malig_med_adata[:, gene2].layers['positivity'].flatten()
        pos_combo = pos_gene1 & pos_gene2
        
        # Get patient names for positive patients
        pos_patients_12 = set(patient_names[pos_combo])
        
        # Calculate percentage
        n_patients = len(patient_names)
        per_12 = round(100 * len(pos_patients_12) / n_patients, 1)
        
        # Prepare malignant combo dataframe
        df12_malig = pd.DataFrame({
            'pos_patient': ['Pos.' if p in pos_patients_12 else 'Neg.' for p in malig_dbl_med.index],
            'log2_exp': np.log2(malig_dbl_med[multi_name].values + 1)
        })
        df12_malig['pos_patient'] = pd.Categorical(df12_malig['pos_patient'], categories=['Pos.', 'Neg.'])
        med12_malig = df12_malig[df12_malig['pos_patient']=='Pos.']['log2_exp'].median()
        
        # Prepare healthy dataframes
        cell_types = ha_sng_med.index
        tissues_list = [ct.split(':')[0] for ct in cell_types]
        
        # Combo healthy data
        df12_healthy = pd.DataFrame({
            'tissue': tissues_list,
            'full': [plot.add_smart_newlines(ct) for ct in cell_types],
            'log2_exp': np.log2(ref_dbl_med[multi_name].values + 1),
            'id': multi_name
        })
        
        # Sort by expression for combo
        df12_healthy = df12_healthy.sort_values('log2_exp', ascending=False)
        df12_healthy['full'] = pd.Categorical(df12_healthy['full'], categories=df12_healthy['full'])
        df12_healthy = df12_healthy.sort_values('full')
        
        # Other genes (gene1, gene2, and known targets)
        other_genes = [gene1, gene2] + known
        
        df_other_list = []
        for gene in other_genes:
            df_gene = pd.DataFrame({
                'tissue': tissues_list,
                'full': [plot.add_smart_newlines(ct) for ct in cell_types],
                'log2_exp': np.log2(ha_sng_med[gene].values + 1),
                'id': gene
            })
            df_other_list.append(df_gene)
        
        df_all_healthy = pd.concat([df12_healthy] + df_other_list, ignore_index=True)
        
        # Rename IDs for display
        df_all_healthy.loc[df_all_healthy['id'] == gene1, 'id'] = 'Gene 1'
        df_all_healthy.loc[df_all_healthy['id'] == gene2, 'id'] = 'Gene 2'
        df_all_healthy.loc[df_all_healthy['id'] == multi_name, 'id'] = 'Gene 1: Gene2'
        
        # Set factor order
        id_order = ['Gene 1: Gene2', 'Gene 1', 'Gene 2'] + known
        df_all_healthy['id'] = pd.Categorical(df_all_healthy['id'], categories=id_order)
        
        # Calculate limits
        max_val = df12_malig['log2_exp'].max()
        
        # Create output path
        out_path = os.path.join(out_dir, f"{multi_name}.pdf")
        
        # Send data to R
        print("Sending data to R...")
        plot.pd2r("df12_malig", df12_malig)
        plot.pd2r("dfh", df_all_healthy)
        
        # Create plots in R
        print("Creating plots in R...")
        r(f'''
        library(ggplot2)
        library(patchwork)
        library(scales)
        
        # Parameters
        max_val <- {max_val}
        limits <- c(0, max_val)
        size <- 6
        med12 <- {med12_malig}
        per_12 <- {per_12}
        
        # Malignant plot
        pm <- ggplot(df12_malig, aes(pos_patient, log2_exp)) +
            theme_jg(xText90=TRUE) +
            theme_small_margin() +
            geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
            geom_jitter(size=3, stroke=0.35, pch=21, aes(fill=pos_patient), color="black", height=0, width=0.2) +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) + scale_x_discrete(drop=FALSE) +
            xlab("") +
            geom_hline(yintercept=med12, lty="dashed", color="firebrick3") +
            scale_fill_manual(values=c("Pos."="#28154C", "Neg."="lightgrey")) +
            ylab("{gene1}:{gene2}") +
            ggtitle(paste0(per_12, " %")) +
            theme(axis.text=element_text(size=14), axis.title=element_text(size=14))
        
        # Healthy plot
        ph <- ggplot(dfh, aes(tissue, squish(log2_exp, limits))) +
            theme_jg(xText90=TRUE) +
            theme_small_margin() +
            geom_boxplot(fill=NA, outlier.size=NA, outlier.stroke=NA, outlier.shape=NA, width=0.5) +
            geom_jitter(size=1.5, stroke=0.35, pch=21, aes(fill=tissue), color="black", height=0, width=0.2) +
            theme(legend.position="none") +
            coord_cartesian(ylim=limits) +
            geom_hline(yintercept=med12, lty="dashed", color="firebrick3") +
            xlab("") + ylab("") +
            theme(axis.text.x=element_text(size=size)) +
            scale_fill_manual(values=suppressMessages(create_pal_d(dfh$tissue))) +
            facet_wrap(~id, nrow=2) +
            theme(strip.text=element_text(size=20)) +
            theme(axis.text.y=element_text(size=14), axis.title.y=element_text(size=14)) +
            ylab("Log2(CP10k + 1)")
        
        # Combine plots
        final_plot <- pm + ph + plot_layout(widths=c(1, 6))
        
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
