"""
Multi-gene UMAP visualization functions.
"""

__all__ = ['umap_1_2_12']


import numpy as np
import pandas as pd
import polars as pl
import h5py
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List
from tqdm import tqdm
from scipy import sparse
from datetime import datetime
import os
import time
from py_target_id import plot, utils, infra
from rpy2.robjects import r
from concurrent.futures import ThreadPoolExecutor

def _load_h5_single(sample_id, h5_path, genes):
    """Helper to load H5 file into memory for a given sample."""
    with h5py.File(h5_path, "r") as f:
        barcodes = np.char.decode(f["coldata"]["barcodes"][:].astype("S"), "utf-8")
        coldata = pd.DataFrame({
            "nCount_RNA": f["coldata"]["nCount_RNA"][:],
            "malig": np.char.decode(f["coldata"]["malig"][:].astype("S"), "utf-8"),
            "UMAP_1": f["coldata"]["UMAP_1"][:],
            "UMAP_2": f["coldata"]["UMAP_2"][:]
        }, index=barcodes)

    # Load AnnData subset
    adata = infra.read_h5(h5_path)
    adata_subset = adata[:, genes].to_memory()

    # Return dense gene expression matrix (Genes x Cells)
    X = adata_subset.X
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    return sample_id, {
        "coldata": coldata,
        "rna_matrix": X.T,
        "sample_id": sample_id
    }

def umap_1_2_12(
    multis: List[str],
    malig_adata,
    manifest: pd.DataFrame,
    show: int = 10,
    out_dir: str = "multi/multi_umaps",
    width: float = 17,
    height: float = 12,
    dpi: int = 300,
    n_jobs: int = 8
):
    """
    Optimized UMAP visualization for multi-gene combinations.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    overall_start = time.time()

    multis_split = np.array([m.split("_") for m in multis])
    genes = np.unique(multis_split.ravel()).tolist()
    print(f"Found {len(genes)} unique genes from {len(multis)} combinations")

    # Malignant subset
    malig_subset = malig_adata[:, genes]
    if hasattr(malig_subset, "to_memory"):
        malig_subset = malig_subset.to_memory()
    if sparse.issparse(malig_subset.X):
        malig_subset.X = malig_subset.X.toarray()

    combo1_idx = [genes.index(g) for g in multis_split[:, 0]]
    combo2_idx = [genes.index(g) for g in multis_split[:, 1]]

    print("Computing patient-level medians...")
    malig_combo = np.minimum(
        malig_subset.X[:, combo1_idx],
        malig_subset.X[:, combo2_idx]
    )

    malig_top_multi = utils.summarize_matrix(
        malig_combo,
        malig_subset.obs["Patient"].values,
        axis=0,
        metric="median",
        verbose=False
    )
    malig_top_multi.columns = multis

    top_samples_list = [
        malig_top_multi.iloc[:, i].nlargest(show).index.tolist()
        for i in range(len(multis))
    ]

    all_unique_samples = list(set().union(*top_samples_list))
    print(f"Will process {len(all_unique_samples)} unique samples")

    # Map Sample_ID to path
    manifest = manifest[manifest["Sample_ID"].isin(all_unique_samples)]
    path_lookup = dict(zip(manifest["Sample_ID"], manifest["Local_h5map"]))

    print(f"Parallel reading {len(all_unique_samples)} H5 files (n_jobs={n_jobs})...")
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        results = list(
            tqdm(
                ex.map(_load_h5_single,
                       all_unique_samples,
                       [path_lookup[s] for s in all_unique_samples],
                       [genes] * len(all_unique_samples)),
                total=len(all_unique_samples)
            )
        )

    h5_cache = dict(results)
    print("H5 files loaded and cached.\n")

    # Main loop per multi-gene combination
    for x, multi in enumerate(multis):
        gx, gy = multis_split[x]
        gx_idx, gy_idx = combo1_idx[x], combo2_idx[x]
        sample_ids = top_samples_list[x]

        print(f"\n[{x+1}/{len(multis)}] Processing {multi}...")

        dfs = []
        for i, sample_id in enumerate(sample_ids, 1):
            cached = h5_cache[sample_id]
            col = cached["coldata"]
            rna = cached["rna_matrix"]

            val1 = rna[gx_idx, :]
            val2 = rna[gy_idx, :]

            ncount = col["nCount_RNA"].values
            scale_factor = 10000 / ncount

            # Vectorized normalization
            val1n = np.log2(scale_factor * val1 + 1)
            val2n = np.log2(scale_factor * val2 + 1)
            val12n = np.log2(scale_factor * np.minimum(val1, val2) + 1)

            # Polars DataFrame (faster)
            df = pl.DataFrame({
                "ID": sample_id,
                "UMAP_1": col["UMAP_1"].values,
                "UMAP_2": col["UMAP_2"].values,
                "val1": val1n,
                "val2": val2n,
                "val12": val12n,
                "Target": np.where(col["malig"] == "malig", "Target", "Other"),
                "sample_num": i
            })
            dfs.append(df)

        # Concatenate efficiently
        combined_df = pl.concat(dfs).to_pandas(use_pyarrow_extension_array=True)

        # Send to R
        # Create plots in R
        print("  Creating UMAP plots...")
                
        # --- FIX ArrowDtype before sending to R ---
        arrow_cols = ['UMAP_1','UMAP_2','val1','val2','val12','sample_num']
        for c in arrow_cols:
            if c in combined_df.columns:
                combined_df[c] = pd.to_numeric(combined_df[c], errors='coerce').astype(float)

        # Ensure categorical columns are string, not ArrowDtype
        combined_df['ID'] = combined_df['ID'].astype(str)
        combined_df['Target'] = combined_df['Target'].astype(str)

        # Drop any rows with missing numeric values just in case
        combined_df = combined_df.dropna(subset=arrow_cols)

        # Send data to R
        plot.pd2r("umap_data", combined_df)
        
        # Create output path
        out_path = os.path.join(out_dir, f"{multi}.pdf")
        
        # Get gene names for titles
        gx = multis_split[x, 0]
        gy = multis_split[x, 1]

        # --- R plotting code unchanged ---
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
        
        par_apply <- function(X, FUN, threads = parallel::detectCores()) {{
          if (requireNamespace("pbapply", quietly = TRUE)) {{
            pbapply::pblapply(X, FUN, cl = threads)
          }} else {{
            parallel::mclapply(X, FUN, mc.cores = threads)
          }}
        }}

        # Color palette
        pal_cart <- c("lightgrey", "#e0f3db", "#6BC291", "#18B5CB", "#2E95D2", "#28154C", "#000000")
        
        # Split by sample
        sample_ids <- unique(umap_data$ID)
        
        # Function to create individual UMAP
        create_umap <- function(dtx, value_col, plot_title = NULL, threads = length(dtx)) {{
              plots <- par_apply(seq_along(dtx), function(i) {{
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
              }}, threads = threads)
              
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

        # Optional PNG
        png_path = out_path.replace(".pdf", ".png")
        try:
            plot.pdf_to_png(pdf_path=out_path, dpi=dpi, output_path=png_path)
            print(f"  ✓ Made PNG: {png_path}")
        except Exception as e:
            print(f"  ⚠ PNG conversion failed: {e}")

    # Write summary file
    with open(os.path.join(out_dir, "finished.txt"), "w") as f:
        f.write(f"Finished: {datetime.now()}\n")
        f.write(f"Total Patients: {len(malig_adata.obs_names)}\n\n")
        f.write("Patient Names:\n")
        f.write("\n".join(malig_adata.obs_names))

    print("\n" + "="*60)
    print(f"✓ All {len(multis)} plots saved to {out_dir}")
    print(f"Completed in {(time.time() - overall_start)/60:.2f} min")
    print("="*60)
