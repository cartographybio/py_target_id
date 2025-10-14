"""
Multi-specific target dot plot visualization.
"""

# Define what gets exported
__all__ = ['dot_plot']

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

def dot_plot(
    multis: list,
    ref_adata,  # AnnData or VirtualAnnData: cells × genes with obs['CellType']
    ref_med_adata,  # AnnData with metadata
    out_dir: str = "multi/multi_dot_plot",
    max_log2: float = 3.0,
    width: float = 24,
    height: float = 8,
    dpi: int = 300
) -> None:
    """
    Create multi-panel dot plot for gene combinations across cell types.
    
    Parameters
    ----------
    multis : list
        Gene combinations in format ["GENE1_GENE2", "GENE3_GENE4", ...]
    ref_adata : AnnData or VirtualAnnData
        Reference atlas (cells × genes) with obs['CellType']
        Can be VirtualAnnData backed by zarr/h5ad
    ref_med_adata : AnnData
        Metadata with columns: Combo_Lv1, Combo_Lv4, Tissue, CellType
    out_dir : str
        Output directory for plots
    max_log2 : float
        Maximum log2 value for color scale
    width : float
        Plot width in inches
    height : float
        Plot height in inches
    dpi : int
        Resolution
    """
    
    # Create output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Parse gene combinations
    multis_split = np.array([m.split("_") for m in multis])
    genes = np.unique(multis_split.ravel())
    
    print(f"Processing {len(multis)} gene combinations with {len(genes)} unique genes")
    print(f"Output directory: {out_dir}")
    
    # Subset reference atlas to genes of interest
    print(f"Loading expression data for {len(genes)} genes...")
    gene_mask = np.isin(ref_adata.var_names, genes)
    
    # Subset the data
    ref_subset = ref_adata[:, gene_mask]
    
    # Materialize if VirtualAnnData
    if hasattr(ref_subset, 'to_memory'):
        print("Materializing VirtualAnnData to memory...")
        ref_subset = ref_subset.to_memory()
    
    # Get the expression matrix
    ha = ref_subset.X  # (cells × genes_subset)
    
    # Transpose to (genes × cells) for efficient summarization
    print("Transposing to (genes × cells)...")
    if sparse.issparse(ha):
        ha = ha.T.tocsr()  # CSR for efficient row operations
    else:
        ha = ha.T
    
    # Get gene names in the subset
    gene_names = ref_subset.var_names
    
    # Create gene name to index mapping (index within the subset)
    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
    
    # Get cell type labels (always in memory)
    CT = ref_adata.obs['CellType'].values
    
    # Prepare metadata dataframe
    df_template = _prepare_tissue_groups(ref_med_adata.obs.copy())
    
    # Fix special characters in CT to match df
    CT = pd.Series(CT).str.replace('α', 'a').str.replace('β', 'B').values
    
    # Validate alignment
    assert all(ct in df_template['Combo_Lv4'].values for ct in np.unique(CT)), \
        "Some cell types in CT not found in ref_med_adata"
    
    # Process each gene combination
    for i, (gene1, gene2) in enumerate(multis_split):
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(multis_split)}: {gene1}_{gene2}")
        print(f"{'='*60}")
        
        # Make a fresh copy of the dataframe for this iteration
        df = df_template.copy()
        
        # Get gene indices in the subset
        idx1 = gene_to_idx.get(gene1)
        idx2 = gene_to_idx.get(gene2)
        
        if idx1 is None or idx2 is None:
            print(f"Warning: Genes not found - {gene1}: {idx1}, {gene2}: {idx2}")
            continue
        
        # Extract expression for each gene (already in memory as genes × cells)
        ha_1 = ha[[idx1], :]  # (1 × cells)
        ha_2 = ha[[idx2], :]  # (1 × cells)
        
        # Compute minimum (for multi-specific targeting)
        if sparse.issparse(ha_1):
            ha_12 = ha_1.minimum(ha_2)
        else:
            ha_12 = np.minimum(ha_1, ha_2)
        
        # Summarize by cell type
        print("Summarizing gene 1...")
        med_1 = utils.summarize_matrix(ha_1, CT, metric="median", verbose=False)
        
        print("Summarizing gene 2...")
        med_2 = utils.summarize_matrix(ha_2, CT, metric="median", verbose=False)
        
        print("Summarizing combined...")
        med_12 = utils.summarize_matrix(ha_12, CT, metric="median", verbose=False)
        
        # Align with metadata (ensure same order)
        cell_type_order = df['Combo_Lv4'].values
        
        # Reindex to match df order
        med_1_aligned = med_1.reindex(columns=cell_type_order, fill_value=0).iloc[0].values
        med_2_aligned = med_2.reindex(columns=cell_type_order, fill_value=0).iloc[0].values
        med_12_aligned = med_12.reindex(columns=cell_type_order, fill_value=0).iloc[0].values
        
        # Transform expression values: log2(x + 0.95), clip to [0, max_log2], set 0 to NaN
        taa_1 = np.clip(np.log2(med_1_aligned + 0.95), 0, max_log2)
        taa_1[taa_1 == 0] = np.nan
        df['TAA_1'] = taa_1
        
        taa_2 = np.clip(np.log2(med_2_aligned + 0.95), 0, max_log2)
        taa_2[taa_2 == 0] = np.nan
        df['TAA_2'] = taa_2
        
        taa_12 = np.clip(np.log2(med_12_aligned + 0.95), 0, max_log2)
        taa_12[taa_12 == 0] = np.nan
        df['TAA_12'] = taa_12
        
        # Create output path
        out_path = os.path.join(out_dir, f"{multis[i]}.pdf")
        #print(f"Output: {out_path}")
        
        # Create three-panel plot in R
        print("Creating plot...")
        plot.pd2r("df", df)
                
        r(f'''
        library(ggplot2)
        library(patchwork)
        library(ggh4x)

        # Define cube transformation
        cube <- function(x) x^3
        cube_root <- function(x) x^(1/3)
        trans_cube <- scales::trans_new(
            name = "cube",
            transform = cube,
            inverse = cube_root
        )

        # Calculate panel sizes
        tissue_counts <- table(unique(df[, c("Tissue2", "TissueGroup")])[, "TissueGroup"])[levels(df$TissueGroup)]
        lv1_counts <- table(unique(df[, c("CellType", "Lv1")])[, "Lv1"])[levels(df$Lv1)]
        col_adjust <- c(0, 0, 0, 10, 0, 30)

        # Common theme elements
        common_theme <- list(
            theme(panel.grid.minor = element_blank(), 
                  panel.grid.major.x = element_blank()),
            theme(axis.ticks.x = element_blank()),
            theme(axis.text.x = element_blank()),
            scale_x_discrete(expand = c(0.05, 0.05)),
            scale_y_discrete(limits = rev),
            theme(plot.title = element_text(hjust = 0.5, size = 16)),
            theme(axis.title.x = element_text(hjust = 0.5, size = 12)),
            theme(strip.text.y = element_text(margin = margin(0, 0.1, 0, 0.1, "cm"))),
            scale_fill_gradientn(colors = pal_cart2, limits = c(0, {max_log2})),
            scale_size(limits = c(0, {max_log2}), trans = trans_cube),
            ylab(""),
            xlab("Distinct Cell Types")
        )

        # Plot 1: Gene 1
        p1 <- ggplot(df, aes(CellType, Tissue2)) +
            geom_point(aes(size = TAA_1, fill = TAA_1), color = "black", pch = 21, alpha = 0.8) +
            facet_grid(TissueGroup ~ Lv1, scales = "free") +
            force_panelsizes(rows = tissue_counts, cols = lv1_counts + col_adjust) +
            ggtitle("{gene1}") +
            theme(axis.text.y = element_text(size = 6.5)) +
            theme_small_margin(0.01) +
            common_theme +
            theme(strip.text.y = element_blank(), strip.background.y = element_blank()) 

        # Plot 2: Gene 2
        p2 <- ggplot(df, aes(CellType, Tissue2)) +
            geom_point(aes(size = TAA_2, fill = TAA_2), color = "black", pch = 21, alpha = 0.8) +
            facet_grid(TissueGroup ~ Lv1, scales = "free") +
            force_panelsizes(rows = tissue_counts, cols = lv1_counts + col_adjust) +
            ggtitle("{gene2}") +
            theme(axis.text.y = element_blank()) +
            theme_small_margin(0.01) +
            common_theme +
            theme(strip.text.y = element_blank(), strip.background.y = element_blank()) 

        # Plot 3: Combined (minimum)
        p12 <- ggplot(df, aes(CellType, Tissue2)) +
            geom_point(aes(size = TAA_12, fill = TAA_12), color = "black", pch = 21, alpha = 0.8) +
            facet_grid(TissueGroup ~ Lv1, scales = "free") +
            force_panelsizes(rows = tissue_counts, cols = lv1_counts + col_adjust) +
            ggtitle("{multis[i]}") +
            theme(axis.text.y = element_blank()) +
            theme_small_margin(0.01) +
            common_theme

        # Combine and save
        p_combined <- p1 + p2 + p12
        suppressWarnings(ggsave("{out_path}", p_combined, width = {width}, height = {height}))
        ''')
        
        print(f"✓ Saved: {out_path}")

        png_path = out_path.replace('.pdf', '.png')
       
        try:
            plot.pdf_to_png(pdf_path=out_path, dpi=dpi, output_path=png_path)
            print(f"✓ Made PNG: {png_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not convert to PNG: {e}")
            print(f"  (PDF saved successfully at {out_path})")

    #Write custom done
    out_file = out_dir + "/finished.txt"

    with open(out_file, 'w') as f:
        f.write(f"Finished: {datetime.now()}\n\n")
        f.write(f"Total Cell Types: {len(ref_med_adata.obs_names)}\n\n")
        f.write("Cell Type Names:\n")
        f.write('\n'.join(ref_med_adata.obs_names))

    print(f"\n{'='*60}")
    print(f"✓ All {len(multis)} plots saved to {out_dir}/")
    print(f"{'='*60}")


def _prepare_tissue_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare tissue and cell type groupings."""
    
    df['Tissue2'] = df['Tissue'].copy()
    tissues = df['Tissue2'].unique()
    
    # Define tissue groups - convert to Series for string operations
    groups = pd.Series('Other', index=tissues)
    tissues_series = pd.Series(tissues)
    
    # Apply tissue group assignments
    groups[tissues_series.str.contains('brain|phalon', case=False, na=False, regex=True).values] = 'Brain'
    groups[tissues_series.str.contains('testes|uterus|ovary|cervix|prostate|fallopian', 
                                 case=False, na=False, regex=True).values] = 'Repro.'
    groups[tissues_series.str.contains('blood|thymus|lymph_node|pbmc|spleen|bone_marrow', 
                                 case=False, na=False, regex=True).values] = 'Immune'
    groups[tissues_series.str.contains('heart|kidney|liver|lung|pancreas|peripheral|skeletal|spine', 
                                 case=False, na=False, regex=True).values] = 'High Risk\nOrgan'
    
    # Process Lv1 cell type categories
    df['Lv1_Old'] = df['Combo_Lv1'].str.split(':', n=1, expand=True)[1]
    
    # Consolidate neural categories
    neural_terms = ['Neuron/glia', 'Glial', 'Glia', 'Neuron']
    df.loc[df['Lv1_Old'].isin(neural_terms), 'Lv1_Old'] = 'Neural'
    
    # Vasculature
    vasc_pattern = 'Endothelial|Vascular|Pericyte'
    vasc_mask = df.index.str.contains(vasc_pattern, case=False, na=False, regex=True)
    df.loc[vasc_mask, 'Lv1_Old'] = 'Vasculature'
    
    # Neural (additional)
    epen_mask = df.index.str.contains('Ependymal', case=False, na=False)
    df.loc[epen_mask, 'Lv1_Old'] = 'Neural'
    
    # Epithelial
    epi_mask = df.index.str.contains('epithelial', case=False, na=False)
    df.loc[epi_mask, 'Lv1_Old'] = 'Epithelial'
    
    # Create ordered categorical
    lv1_order = ['Neural', 'Epithelial', 'Stromal', 'Vasculature', 'Immune', 'Somatic']
    df['Lv1'] = pd.Categorical(df['Lv1_Old'], categories=lv1_order, ordered=True)
    
    # Add tissue group
    df['TissueGroup'] = df['Tissue2'].map(groups)
    tissue_group_order = ['Brain', 'High Risk\nOrgan', 'Other', 'Immune', 'Repro.']
    df['TissueGroup'] = pd.Categorical(df['TissueGroup'], 
                                        categories=tissue_group_order, ordered=True)
    
    # Clean tissue names
    df['Tissue2'] = df['Tissue2'].str.replace('Brain_', '', regex=False).str.replace('_', ' ', regex=False)
    
    # Fix special characters
    df['Combo_Lv4'] = df['Combo_Lv4'].str.replace('α', 'a').str.replace('β', 'B')
    
    return df
