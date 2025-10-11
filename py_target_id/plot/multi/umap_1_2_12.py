"""
Multi-gene UMAP visualization functions.
"""

__all__ = ['umap_1_2_12']

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import time
from typing import List, Optional
from scipy import sparse
from tqdm import tqdm
from py_target_id import plot, utils, infra
from rpy2.robjects import r
import os

def umap_1_2_12(
    multis: List[str],
    malig_adata,
    manifest: pd.DataFrame,
    show: int = 10,
    out_dir: str = "multi/multi_umaps",
    width: float = 17,
    height: float = 12,
    dpi: int = 300
):
    """
    Create UMAP plots for multi-gene combinations across top expressing samples.
    
    Parameters
    ----------
    multis : List[str]
        Gene combinations in format ["GENE1_GENE2", ...]
    malig_adata : AnnData
        Malignant cell data with patient annotations
    manifest : pd.DataFrame
        Manifest with columns: Sample_ID, Local_h5map
    show : int
        Number of top samples to show per combination
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
    
    overall_start = time.time()
    
    # Parse gene combinations
    multis_split = np.array([m.split("_") for m in multis])
    genes = np.unique(multis_split.ravel()).tolist()
    print(f"Found {len(genes)} unique genes from {len(multis)} multi-gene combinations")
    
    # Process malignancy data
    print("Processing malignancy data...")
    malig_subset = malig_adata[:, genes]
    
    # Handle VirtualAnnData
    if hasattr(malig_subset, 'to_memory'):
        print("  Materializing VirtualAnnData to memory...")
        malig_subset = malig_subset.to_memory()
    
    # Ensure dense matrix
    if sparse.issparse(malig_subset.X):
        malig_subset.X = malig_subset.X.toarray()
        
    # Compute medians for all combinations
    print("Computing medians for gene combinations...")
    combo1_idx = [genes.index(g) for g in multis_split[:, 0]]
    combo2_idx = [genes.index(g) for g in multis_split[:, 1]]
        
    # Vectorized minimum computation
    malig_combo = np.minimum(
        malig_subset.X[:, combo1_idx],
        malig_subset.X[:, combo2_idx]
    )
    
    # Summarize by patient
    malig_top_multi = utils.summarize_matrix(
        malig_combo,
        malig_subset.obs['Patient'].values, 
        axis=0, 
        metric="median", 
        verbose=False
    )
    malig_top_multi.columns = multis
    
    # Pre-compute top samples for all combinations
    print("Pre-computing top samples for all combinations...")
    top_samples_list = [
        malig_top_multi.iloc[:, i].nlargest(show).index.tolist()
        for i in range(len(multis))
    ]
    
    # Get unique samples - use set for O(1) lookup
    all_unique_samples = list(set().union(*[set(samples) for samples in top_samples_list]))
    print(f"Will process {len(all_unique_samples)} unique samples")
    
    # Filter manifest
    man = manifest[manifest['Sample_ID'].isin(all_unique_samples)].copy()
    path_lookup = dict(zip(man['Sample_ID'], man['Local_h5map']))
    
    # Batch read h5 files
    print(f"Batch reading {len(all_unique_samples)} h5 files...")
    h5_cache = {}
    
    for sample_id in tqdm(all_unique_samples, desc="Reading H5 files"):
        h5_path = path_lookup[sample_id]
        
        with h5py.File(h5_path, 'r') as f:
            barcodes = np.char.decode(f['coldata']['barcodes'][:].astype('S'), 'utf-8')
            coldata = pd.DataFrame({
                'nCount_RNA': f['coldata']['nCount_RNA'][:],
                'malig': np.char.decode(f['coldata']['malig'][:].astype('S'), 'utf-8'),
                'UMAP_1': f['coldata']['UMAP_1'][:],
                'UMAP_2': f['coldata']['UMAP_2'][:]
            }, index=barcodes)
        
        # Read only needed genes
        adata = infra.read_h5(h5_path)
        adata_subset = adata[:, genes].to_memory()
        
        h5_cache[sample_id] = {
            'coldata': coldata,
            'rna_matrix': adata_subset.X.T.toarray(),  # Genes x cells for faster access
            'sample_id': sample_id
        }
    
    print("H5 files loaded with gene subsetting")
    
    # Process combinations
    print(f"Processing {len(multis)} multi-gene combinations...")
    
    for x, multi in enumerate(multis):
        print(f"\nProcessing combination {x+1}/{len(multis)}: {multi}")
        
        # Get top samples and genes
        sample_ids = top_samples_list[x]
        gx_idx = combo1_idx[x]
        gy_idx = combo2_idx[x]
        
        # Extract data from cache - vectorized operations
        all_data = []
        for i, sample_id in enumerate(sample_ids, 1):
            cached_data = h5_cache[sample_id]
            coldata = cached_data['coldata']
            rna_matrix = cached_data['rna_matrix']
            
            # Get expression values - already indexed
            val1 = rna_matrix[gx_idx, :]
            val2 = rna_matrix[gy_idx, :]
            
            # Vectorized normalization
            ncount = coldata['nCount_RNA'].values
            scale_factor = 10000 / ncount
            
            val1_norm = np.log2(scale_factor * val1 + 1)
            val2_norm = np.log2(scale_factor * val2 + 1)
            val12_norm = np.log2(scale_factor * np.minimum(val1, val2) + 1)
            
            # Create dataframe
            df = pd.DataFrame({
                'ID': sample_id,
                'UMAP_1': coldata['UMAP_1'].values,
                'UMAP_2': coldata['UMAP_2'].values,
                'val1': val1_norm,
                'val2': val2_norm,
                'val12': val12_norm,
                'Target': np.where(coldata['malig'] == 'malig', 'Target', 'Other'),
                'sample_num': i
            })
            
            all_data.append(df)
        
        # Combine all samples
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create plots in R
        print("  Creating UMAP plots...")
        
        # Send data to R
        plot.pd2r("umap_data", combined_df)
        
        # Create output path
        out_path = os.path.join(out_dir, f"{multi}.pdf")
        
        # Get gene names for titles
        gx = multis_split[x, 0]
        gy = multis_split[x, 1]
        
        # Generate plots
        r(f'''
        library(ggplot2)
        library(patchwork)
               
        # Define quantile_cut function
        quantile_cut <- function(x = NULL, lo = 0.025, hi = 0.975, maxIf0 = TRUE) {{
            q <- quantile(x, probs = c(lo, hi))
            if (q[2] == 0) {{
                if (maxIf0) {{
                    q[2] <- mean(x[x!=0])
                }}
            }}
            x[x < q[1]] <- q[1]
            x[x > q[2]] <- q[2]
            return(x)
        }}

        # Color palette
        pal_cart <- c("lightgrey", "#e0f3db", "#6BC291", "#18B5CB", "#2E95D2", "#28154C", "#000000")
        
        # Split by sample
        sample_ids <- unique(umap_data$ID)
        
        # Function to create individual UMAP
        create_umap <- function(dtx, value_col, plot_title = NULL, threads = length(dtx)) {{
              plots <- parallel::mclapply(seq_along(dtx), function(i) {{
                df <- dtx[[i]]
                
                p <- ggplot(df, aes(UMAP_1, UMAP_2)) +
                  stat_summary_hex(aes(z = .data[[value_col]]), 
                                   fun = function(x) max(quantile(x, 0.9), mean(x)),
                                   bins = 75)

                p <- p +
                  scale_fill_gradientn(
                    colors = pal_cart,
                    limits = range(quantile_cut(ggplot_build(p)$data[[1]]$value, lo = 0, hi = 0.95)), 
                    oob = scales::squish
                  ) +
                  theme(legend.position = "none") +
                  theme_small_margin() +
                  theme(axis.text = element_blank(), axis.ticks = element_blank()) +
                  theme(panel.grid = element_blank()) +
                  xlab("") + ylab("")

                # Density Contour
                dfT <- df[df$Target == "Target", , drop = FALSE]
                
                o <- geom_density_2d(data = dfT, bins = 4, color = "red", size = 0.3)

                p <- tryCatch({{
                  ggplot_build(p + o)
                  p + o
                }}, error = function(e) {{
                  dx <- diff(range(df$UMAP_1)) * 0.001
                  dy <- diff(range(df$UMAP_2)) * 0.001
                  
                  dfT2 <- data.frame(
                    UMAP_1 = sample(dfT$UMAP_1, min(201, nrow(dfT)), replace = TRUE) + 
                             seq(-dx, dx, length.out = min(201, nrow(dfT))),
                    UMAP_2 = sample(dfT$UMAP_2, min(201, nrow(dfT)), replace = TRUE) + 
                             seq(-dy, dy, length.out = min(201, nrow(dfT))),
                    Target = "Target"
                  )
                  dfT2 <- rbind(dfT[, c("UMAP_1", "UMAP_2", "Target")], dfT2)
                  p + geom_density_2d(data = dfT2, bins = 2, color = "red", size = 0.3)
                }})

                # Add sample number
                x_range <- range(df$UMAP_1)
                y_range <- range(df$UMAP_2)
                
                p + annotate("text", 
                            x = x_range[1] + diff(x_range) * 0.05,
                            y = y_range[2] - diff(y_range) * 0.05,
                            label = as.character(i), 
                            size = 3, color = "black")
              }}, mc.cores = threads)
              
              combined_plots <- Reduce("+", plots) + plot_layout(ncol = 2)
              
              if (!is.null(plot_title)) {{
                combined_plots[[1]] <- combined_plots[[1]] + ggtitle(plot_title)
              }}
              
              combined_plots
        }}

        umap_data_list <- split(umap_data, umap_data$sample_num)
        
        # Create three panels
        p1 <- create_umap(umap_data_list, "val1", "{gx}")
        p2 <- create_umap(umap_data_list, "val2", "{gy}")
        p3 <- create_umap(umap_data_list, "val12", "{multi}")
        
        # Vertical separator
        vline <- ggplot() + 
            geom_vline(xintercept = 0.75, color = "black", size = 0.25) +
            theme_void() + xlim(-1, 1)
        
        # Combine
        id_text <- paste("Samples:", paste(sample_ids, collapse = ", "))
        final_plot <- (p1 | vline | p2 | vline | p3) + 
            plot_layout(ncol = 5, widths = c(2, 0.1, 2, 0.1, 2)) +
            plot_annotation(caption = id_text)
        
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
    
    overall_end = time.time()
    print(f"\n{'='*60}")
    print(f"✓ All {len(multis)} plots saved to {out_dir}/")
    print(f"  Completed in {(overall_end - overall_start)/60:.2f} minutes")
    print(f"{'='*60}")