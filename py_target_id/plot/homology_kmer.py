"""
Topology and k-mer homology visualization across species.
"""

__all__ = ['plot_homology_kmer']

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import time
from typing import List, Optional
from rpy2.robjects import r
from py_target_id import plot

def plot_homology_kmer(
    tx: Optional[List[str]] = None,
    genes: Optional[List[str]] = None,
    hide_axis: bool = False,
    output_dir: str = "homology/kmer",
    width: float = 10,
    height: float = 8,
    dpi: int = 300
):
    """
    Create topology and k-mer homology plots for transcripts.
    
    Shows:
    - Topology prediction (Inner/Membrane/Outer/Signal)
    - K-mer matches to human proteome
    - Gene-level matches to human proteome  
    - K-mer matches to mouse proteome
    - Gene-level matches to mouse proteome
    - K-mer matches to cyno proteome
    - Gene-level matches to cyno proteome
    
    Parameters
    ----------
    tx : List[str], optional
        List of transcript IDs to analyze
    genes : List[str], optional
        List of gene symbols (will get transcripts via get_human_topology)
    hide_axis : bool
        If True, hide gene names on y-axis
    output_dir : str
        Output directory for plots
    width : float
        Plot width in inches
    height : float
        Plot height in inches
    dpi : int
        Resolution for output
    """
    
    start_time = time.time()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get topology data (assumes get_human_topology is already available)
    print("Loading topology data...")
    
    df_all = get_human_topology(genes=None, outer_only=False, verbose=False)
    
    # Get transcript IDs
    if tx is None:
        if genes is not None:
            df_subset = get_human_topology(genes=genes, outer_only=True, verbose=False)
            tx = df_subset['transcript_id'].tolist()
        else:
            print("Error: Must provide either tx or genes")
            return None
    
    # Filter to valid transcripts
    tx = [t for t in tx if t in df_all['transcript_id'].values]
    if len(tx) == 0:
        print("No valid transcripts found")
        return None
    
    elapsed = (time.time() - start_time) / 60
    print(f"------  Loading kmer files in parallel | Total: {elapsed:.1f}min")
    
    # Download k-mer data files from GCS
    gcs_base = "gs://cartography_target_id_package/Other_Input/Homology_Kmer"
    local_dir = Path("temp/kmer")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    kmer_files = {
        'human': 'human_vs_human.8mer.20250505.parquet',
        'mouse': 'human_vs_mouse.8mer.20250505.parquet',
        'cyno': 'human_vs_cyno.8mer.20250505.parquet'
    }
    
    # Download files if needed
    for species, filename in kmer_files.items():
        local_path = local_dir / filename
        if not local_path.exists():
            print(f"Downloading {filename}...")
            gcs_path = f"{gcs_base}/{filename}"
            cmd = ["gsutil", "cp", gcs_path, str(local_path)]
            subprocess.run(cmd, check=True, capture_output=True)
    
    # Load k-mer data
    print("Loading k-mer data...")
    kmer_data = {}
    for species, filename in kmer_files.items():
        kmer_data[species] = pd.read_parquet(local_dir / filename)
    
    elapsed = (time.time() - start_time) / 60
    print(f"------  Data files loaded successfully | Total: {elapsed:.1f}min")
    
    # Process each transcript
    for i, txi in enumerate(tx):
        elapsed = (time.time() - start_time) / 60
        
        # Get gene and transcript info
        tx_row = df_all[df_all['transcript_id'] == txi].iloc[0]

        # Get the actual gene from the k-mer data (human file should have it)
        species_data = kmer_data['human']
        tx_col = 'TX' if 'TX' in species_data.columns else 'transcript_id'
        dtxi_human = species_data[species_data[tx_col] == txi]

        if len(dtxi_human) > 0 and 'gene_name' in dtxi_human.columns:
            gene = dtxi_human['gene_name'].iloc[0]
        else:
            # Fallback to topology data
            gene = tx_row['gene_name']

        gene_i = gene

        print(f"------  Processing {i+1}/{len(tx)}: {gene} {txi} | Total: {elapsed:.1f}min")
        
        # Extract base transcript ID (remove version)
        txi_base = txi.split('.')[0]
        
        # Get topology
        topology_str = tx_row['Predict']
        topology_codes = list(topology_str)
        topology_y = [{'I': 1, 'M': 2, 'O': 3, 'S': 4}.get(c, 0) for c in topology_codes]
        
        # Process each species
        plot_data = {}
        
        for species in ['human', 'mouse', 'cyno']:
            # Get k-mer data for this transcript
            species_data = kmer_data[species]
            
            # Try to match transcript ID (handle column name variations)
            tx_col = 'TX' if 'TX' in species_data.columns else 'transcript_id'
            dtxi = species_data[species_data[tx_col] == txi].copy()
            
            if len(dtxi) == 0:
                print(f"  Warning: No k-mer data for {txi} in {species}")
                # Create empty data
                plot_data[species] = {
                    'kmer': pd.DataFrame({'Start': [0], 'N_Genes': [0]}),
                    'genes': pd.DataFrame({'Position': [1], 'Gene': ['No_Data'], 'Present': [0]}),
                    'width': len(topology_str)
                }
                continue
            
            # Process Genes column
            if 'Genes' in dtxi.columns:
                # Clean up genes column
                dtxi['Genes'] = dtxi['Genes'].fillna('')
                dtxi['Genes'] = dtxi['Genes'].replace('NA', '')
                
                # Count genes per k-mer if not already present
                if 'N_Genes' not in dtxi.columns:
                    dtxi['N_Genes'] = dtxi['Genes'].apply(
                        lambda x: len([g for g in str(x).split(';') if g.strip()])
                    )
                
                # Get all unique genes across all k-mers
                all_genes = []
                for genes_str in dtxi['Genes']:
                    if genes_str:
                        all_genes.extend([g.strip() for g in str(genes_str).split(';') if g.strip()])
                
                # Count gene frequencies and get top 25 with stable sorting for ties
                from collections import Counter
                gene_counts = Counter(all_genes)
                
                # Sort by count (descending), then by gene name (ascending) for stable tie-breaking
                sorted_genes = sorted(gene_counts.items(), key=lambda x: (-x[1], x[0]))
                top_genes = [gene for gene, count in sorted_genes[:25]]
                
                # Create presence matrix
                if top_genes:
                    rows = []
                    for idx, row in dtxi.iterrows():
                        genes_at_pos = set([g.strip() for g in str(row['Genes']).split(';') if g.strip()])
                        for gene in top_genes:
                            if gene in genes_at_pos:
                                rows.append({
                                    'Position': row['Start'],
                                    'Gene': gene,
                                    'Present': 1
                                })
                    
                    if rows:
                        df_melted = pd.DataFrame(rows)
                        # Reverse order for plotting (to match R)
                        df_melted['Gene'] = pd.Categorical(
                            df_melted['Gene'], 
                            categories=top_genes[::-1], 
                            ordered=True
                        )
                    else:
                        df_melted = pd.DataFrame({
                            'Position': [1], 
                            'Gene': ['No_Genes'], 
                            'Present': [0]
                        })
                else:
                    # No genes found
                    df_melted = pd.DataFrame({
                        'Position': [1], 
                        'Gene': ['No_Genes'], 
                        'Present': [0]
                    })
            else:
                # Genes column missing
                dtxi['N_Genes'] = 0
                df_melted = pd.DataFrame({
                    'Position': [1], 
                    'Gene': ['No_Data'], 
                    'Present': [0]
                })
            
            # Get width
            if 'Width' in dtxi.columns:
                width_val = dtxi['Width'].iloc[0] if len(dtxi) > 0 else len(topology_str)
            else:
                width_val = len(topology_str)
            
            # Store data
            plot_data[species] = {
                'kmer': dtxi[['Start', 'N_Genes']],
                'genes': df_melted,
                'width': width_val
            }
        
        # Get max width
        max_width = max([plot_data[s]['width'] for s in plot_data.keys()])
        
        # Create topology dataframe
        df_topology = pd.DataFrame({
            'x': list(range(len(topology_codes))),
            'y': topology_y
        })
        
        # Send data to R
        plot.pd2r("df_topology", df_topology)
        
        for species in ['human', 'mouse', 'cyno']:
            if species in plot_data:
                plot.pd2r(f"df_{species}_kmer", plot_data[species]['kmer'])
                plot.pd2r(f"df_{species}_genes", plot_data[species]['genes'])
        
        # Create plot in R
        out_path = Path(output_dir) / f"{gene_i}_{txi_base}_kmer.pdf"
        
        hide_y_axis = "TRUE" if hide_axis else "FALSE"
        
        r(f'''
        library(jgplot2)
        library(patchwork)
        
        max_width <- {max_width}
        
        # Topology plot
        p1h <- ggplot(df_topology, aes(x, y)) +
            geom_line(color = "#6BC291", size = 1) +
            scale_x_continuous(limits = c(0, max_width), breaks = scales::pretty_breaks(n = 10)) +
            scale_y_continuous(breaks = 1:4, limits = c(1, 4), 
                             labels = c("Inner", "Membrane", "Outer", "Signal")) +
            theme(panel.grid.minor = element_blank(),
                  panel.grid.major.x = element_blank(),
                  panel.grid.minor.x = element_blank(),
                  axis.text.x = element_blank(),
                  axis.title = element_blank(),
                  axis.ticks.x = element_blank(),
                  plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm")) +
            ggtitle(paste0("{gene_i} : {txi} : {len(topology_str)}"))
        
        # Human k-mer plot
        p2h <- ggplot(df_human_kmer, aes(Start, pmin(N_Genes, 10))) +
            geom_point(color = "#18B5CB", size = 1, alpha = 0.6) +
            scale_x_continuous(limits = c(0, max_width), breaks = scales::pretty_breaks(n = 10)) +
            scale_y_continuous(breaks = seq(0, 10, 2), limits = c(0, 10), 
                             labels = c(seq(0, 8, 2), "10+")) +
            ylab("Human Proteome") +
            theme(panel.grid.minor = element_blank(),
                  panel.grid.major.x = element_blank(),
                  panel.grid.minor.x = element_blank(),
                  axis.text.x = element_blank(),
                  axis.title.x = element_blank(),
                  axis.ticks.x = element_blank(),
                  plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm"))
        
        # Human genes plot
        text_size <- 4
        theme_genes <- if ({hide_y_axis}) {{
            theme(axis.text.y = element_blank())
        }} else {{
            theme(axis.text.y = element_text(size = text_size))
        }}
        
        p3h <- ggplot(df_human_genes, aes(Position, Gene)) +
            geom_point(color = "#18B5CB", size = 1, alpha = 0.6) +
            scale_x_continuous(limits = c(0, max_width), breaks = scales::pretty_breaks(n = 10)) +
            ylab("") +
            theme(panel.grid.minor = element_blank(),
                  panel.grid.major.x = element_blank(),
                  panel.grid.minor.x = element_blank(),
                  axis.text.x = element_blank(),
                  axis.ticks.x = element_blank(),
                  plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm")) +
            theme_genes
        
        # Mouse k-mer plot
        p2m <- ggplot(df_mouse_kmer, aes(Start, pmin(N_Genes, 10))) +
            geom_point(color = "#2E95D2", size = 1, alpha = 0.6) +
            scale_x_continuous(limits = c(0, max_width), breaks = scales::pretty_breaks(n = 10)) +
            scale_y_continuous(breaks = seq(0, 10, 2), limits = c(0, 10), 
                             labels = c(seq(0, 8, 2), "10+")) +
            ylab("Mouse Proteome") +
            theme(panel.grid.minor = element_blank(),
                  panel.grid.major.x = element_blank(),
                  panel.grid.minor.x = element_blank(),
                  axis.text.x = element_blank(),
                  axis.title.x = element_blank(),
                  axis.ticks.x = element_blank(),
                  plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm"))
        
        # Mouse genes plot
        p3m <- ggplot(df_mouse_genes, aes(Position, Gene)) +
            geom_point(color = "#2E95D2", size = 1, alpha = 0.6) +
            scale_x_continuous(limits = c(0, max_width), breaks = scales::pretty_breaks(n = 10)) +
            ylab("") +
            xlab("AA Position") +
            theme(panel.grid.minor = element_blank(),
                  panel.grid.major.x = element_blank(),
                  panel.grid.minor.x = element_blank(),
                  axis.text.x = element_blank(),
                  axis.ticks.x = element_blank(),
                  plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm")) +
            theme_genes
        
        # Cyno k-mer plot
        p2c <- ggplot(df_cyno_kmer, aes(Start, pmin(N_Genes, 10))) +
            geom_point(color = "#28154C", size = 1, alpha = 0.6) +
            scale_x_continuous(limits = c(0, max_width), breaks = scales::pretty_breaks(n = 10)) +
            scale_y_continuous(breaks = seq(0, 10, 2), limits = c(0, 10), 
                             labels = c(seq(0, 8, 2), "10+")) +
            ylab("Cyno Proteome") +
            xlab("AA Position") +
            theme(panel.grid.minor = element_blank(),
                  panel.grid.major.x = element_blank(),
                  panel.grid.minor.x = element_blank(),
                  axis.text.x = element_blank(),
                  axis.title.x = element_blank(),
                  axis.ticks.x = element_blank(),
                  plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm"))
        
        # Cyno genes plot  
        p3c <- ggplot(df_cyno_genes, aes(Position, Gene)) +
            geom_point(color = "#28154C", size = 1, alpha = 0.6) +
            scale_x_continuous(limits = c(0, max_width), breaks = scales::pretty_breaks(n = 10)) +
            ylab("") +
            xlab("AA Position") +
            theme(panel.grid.minor = element_blank(),
                  panel.grid.major.x = element_blank(),
                  panel.grid.minor.x = element_blank(),
                  axis.text.x = element_blank(),
                  axis.ticks.x = element_blank(),
                  plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm")) +
            theme_genes
        
        # Combine all plots
        final_plot <- p1h / p2h / p3h / p2m / p3m / p2c / p3c + 
            plot_layout(heights = c(6, 4, 3, 4, 3, 4, 3))
        
        # Save
        suppressWarnings(ggsave("{str(out_path)}", final_plot, width={width}, height={height}))
        ''')
        
        print(f"✓ Saved: {out_path}")
        
        # Convert to PNG
        png_path = str(out_path).replace('.pdf', '.png')
        try:
            plot.pdf_to_png(pdf_path=str(out_path), dpi=dpi, output_path=png_path)
            print(f"✓ Made PNG: {png_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not convert to PNG: {e}")
    
    elapsed = (time.time() - start_time) / 60
    print(f"------  All plots completed | Total: {elapsed:.1f}min")