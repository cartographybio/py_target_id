"""
BLAST-based homology analysis and visualization for protein sequences.
"""

__all__ = ['check_blast_installed', 'get_blast_db', 'get_human_topology', 'plot_homology_blast']

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import shutil

def check_blast_installed() -> bool:
    """
    Check if BLAST+ is installed and install if missing.
    
    Returns
    -------
    bool
        True if BLAST+ is available, False otherwise
    """
    # Check if blastp is available
    if shutil.which("blastp") is not None:
        try:
            result = subprocess.run(["blastp", "-version"], 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            print(f"BLAST+ found: {result.stdout.split()[1]}")
            return True
        except subprocess.CalledProcessError:
            pass
    
    # BLAST not found, try to install via conda
    print("BLAST+ not found. Attempting to install via conda...")
    try:
        result = subprocess.run(
            ["conda", "install", "-c", "bioconda", "blast", "-y"],
            capture_output=True,
            text=True,
            check=True
        )
        print("BLAST+ installed successfully!")
        
        # Verify installation
        if shutil.which("blastp") is not None:
            result = subprocess.run(["blastp", "-version"], 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            print(f"BLAST+ version: {result.stdout.split()[1]}")
            return True
        else:
            print("Installation completed but blastp not found in PATH")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Failed to install BLAST+: {e.stderr}")
        print("\nPlease install BLAST+ manually:")
        print("  conda install -c bioconda blast")
        print("  or")
        print("  apt-get install ncbi-blast+  (Ubuntu/Debian)")
        print("  brew install blast  (macOS)")
        return False
    except FileNotFoundError:
        print("Conda not found. Please install BLAST+ manually:")
        print("  Ubuntu/Debian: sudo apt-get install ncbi-blast+")
        print("  macOS: brew install blast")
        print("  CentOS/RHEL: sudo yum install ncbi-blast+")
        return False


def _download_file(args):
    """Helper function for parallel file downloads."""
    filename, gcs_base, local_dir = args
    gcs_path = f"{gcs_base}/{filename}"
    local_path = local_dir / filename
    cmd = ["gsutil", "cp", gcs_path, str(local_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return f"Downloaded: {filename}"
    except subprocess.CalledProcessError as e:
        return f"Failed: {filename} - {e.stderr.decode()}"


def get_blast_db(overwrite: bool = False) -> int:
    """
    Download BLAST database files from GCS.
    
    Parameters
    ----------
    overwrite : bool
        If True, download files even if they exist locally
        
    Returns
    -------
    int
        0 on success
    """
    
    filenames = [
        "df_uniprot_names.parquet",
        "human_db.fasta", "human_db.phr", "human_db.pin", "human_db.psq",
        "macfa_db.fasta", "macfa_db.phr", "macfa_db.pin", "macfa_db.psq",
        "macmu_db.fasta", "macmu_db.phr", "macmu_db.pin", "macmu_db.psq",
        "mouse_db.fasta", "mouse_db.phr", "mouse_db.pin", "mouse_db.psq"
    ]
    
    gcs_base = "gs://cartography_target_id_package/Other_Input/Homology_Blast"
    local_dir = Path("temp/blast")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which files need downloading
    files_to_download = []
    for filename in filenames:
        local_path = local_dir / filename
        if not local_path.exists() or overwrite:
            files_to_download.append(filename)
    
    if not files_to_download:
        print("All files already exist locally")
        return 0
    
    # Download files in parallel
    print(f"Downloading {len(files_to_download)} files...")
    download_args = [(f, gcs_base, local_dir) for f in files_to_download]
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_download_file, args): args[0] for args in download_args}
        for future in as_completed(futures):
            print(future.result())
    
    return 0


def get_human_topology(genes: List[str] = None, outer_only: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Get human protein topology information for given genes.
    
    Parameters
    ----------
    genes : List[str], optional
        List of gene symbols to filter. If None, returns all genes.
    outer : bool
        If True, subset to transcripts with outer domains (containing 'O' in Predict column)
    verbose : bool
        If True, print progress messages
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gene_name, transcript_id, AA, Predict (topology), N_AA, etc.
    """
    # Download topology file if needed
    gcs_path = "gs://cartography_target_id_package/Other_Input/Homology_Kmer/Transcript_Topology_Summary.Human_v29.parquet"
    local_path = Path("temp/topology/Transcript_Topology_Summary.Human_v29.parquet")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not local_path.exists():
        if verbose:
            print(f"Downloading topology file...")
        cmd = ["gsutil", "cp", gcs_path, str(local_path)]
        subprocess.run(cmd, check=True, capture_output=True)
    
    # Load topology data
    human = pd.read_parquet(local_path)
    
    # Filter by genes if provided
    if genes is not None:
        human = human[human['gene_name'].isin(genes)].copy()
    
    # Filter by outer domains if requested
    if outer_only:
        if verbose:
            print("Subsetting Transcripts with Predicted Outer Domain...")
        human = human[human['Predict'].str.contains('O', na=False)].copy()
    else :
        print("Using All transcripts...")

    # Sort by gene name and N_AA (descending)
    human = human.sort_values(['gene_name', 'N_AA'], ascending=[True, False])
    
    return human


def run_blast(params: Dict) -> pd.DataFrame:
    """
    Run BLAST search for a single database.
    
    Parameters
    ----------
    params : dict
        Dictionary with keys: query, db, out, search_name
        
    Returns
    -------
    pd.DataFrame
        BLAST results with search column added
    """
    
    query_file = params['query']
    db_path = params['db']
    out_file = params['out']
    search_name = params['search_name']
    
    cmd = [
        "blastp",
        "-query", query_file,
        "-db", db_path,
        "-out", out_file,
        "-outfmt", "6",
        "-max_target_seqs", "25"
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Read results
    if Path(out_file).exists() and Path(out_file).stat().st_size > 0:
        result = pd.read_csv(
            out_file,
            sep='\t',
            header=None,
            names=['query_id', 'subject_id', 'pct_identity', 'alignment_length',
                   'mismatches', 'gap_opens', 'q_start', 'q_end', 's_start', 's_end',
                   'evalue', 'bit_score']
        )
    else:
        # Return empty dataframe with correct columns
        result = pd.DataFrame(columns=[
            'query_id', 'subject_id', 'pct_identity', 'alignment_length',
            'mismatches', 'gap_opens', 'q_start', 'q_end', 's_start', 's_end',
            'evalue', 'bit_score'
        ])
    
    result['search'] = search_name
    return result


def plot_homology_blast(
    df: pd.DataFrame,
    output_dir: str = "homology/blast",
    figsize: tuple = (20, 10),
    dpi: int = 300,
    verbose: bool = False
) -> List[pd.DataFrame]:
    """
    Create homology plots comparing protein sequences across species.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: gene_name, transcript_id, AA (amino acid sequence)
    output_dir : str
        Directory to save output plots
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Resolution for output images
        
    Returns
    -------
    List[pd.DataFrame]
        List of BLAST results for each query
    """
    
    # Check if BLAST+ is installed
    if not check_blast_installed():
        raise RuntimeError(
            "BLAST+ is required but not installed. "
            "Please install it manually and try again."
        )
    
    # Ensure BLAST databases are available
    get_blast_db()
    
    # Load UniProt names
    df_uni = pd.read_parquet("temp/blast/df_uniprot_names.parquet")
    
    # Load FASTA files
    human_fa = {rec.id: str(rec.seq) for rec in SeqIO.parse("temp/blast/human_db.fasta", "fasta")}
    mouse_fa = {rec.id: str(rec.seq) for rec in SeqIO.parse("temp/blast/mouse_db.fasta", "fasta")}
    macmu_fa = {rec.id: str(rec.seq) for rec in SeqIO.parse("temp/blast/macmu_db.fasta", "fasta")}
    macfa_fa = {rec.id: str(rec.seq) for rec in SeqIO.parse("temp/blast/macfa_db.fasta", "fasta")}
    
    # Database paths
    db_paths = {
        'Human': "temp/blast/human_db",
        'Mouse': "temp/blast/mouse_db",
        'Macmu': "temp/blast/macmu_db",
        'Macfa': "temp/blast/macfa_db"
    }
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    out_list = []
    
    # Reset index to ensure clean iteration
    df = df.reset_index(drop=True)
    
    # Process each row
    for idx, row in df.iterrows():
        print(f"Processing row {idx + 1} of {len(df)}: {row['gene_name']} {row['transcript_id']}")
        
        seq = row['AA']
        seq_len = len(seq)
        
        # Create query FASTA
        query_name = f"FULL_AA[1-{seq_len}]"
        fa_file = f"temp/query.{row['transcript_id']}.{query_name.replace(' ', '_')}.fasta"
        
        record = SeqRecord(Seq(seq), id=query_name, description="")
        SeqIO.write([record], fa_file, "fasta")
        
        # Prepare BLAST parameters
        blast_params = [
            {
                'query': fa_file,
                'db': db_paths['Human'],
                'out': fa_file.replace('.fasta', '_human.tsv'),
                'search_name': 'Human'
            },
            {
                'query': fa_file,
                'db': db_paths['Mouse'],
                'out': fa_file.replace('.fasta', '_mouse.tsv'),
                'search_name': 'Mouse'
            },
            {
                'query': fa_file,
                'db': db_paths['Macmu'],
                'out': fa_file.replace('.fasta', '_macmu.tsv'),
                'search_name': 'Macmu'
            },
            {
                'query': fa_file,
                'db': db_paths['Macfa'],
                'out': fa_file.replace('.fasta', '_macfa.tsv'),
                'search_name': 'Macfa'
            }
        ]
        
        # Run BLAST in parallel
        with ProcessPoolExecutor(max_workers=4) as executor:
            blast_results = list(executor.map(run_blast, blast_params))
        
        # Combine results
        df_summary = pd.concat(blast_results, ignore_index=True)
        df_summary['gene'] = row['gene_name']
        df_summary['id'] = row['transcript_id']
        df_summary['sub_id'] = query_name
        df_summary['N_AA'] = seq_len
        
        if verbose:
            print(f"\nBLAST results per search:")
            print(df_summary['search'].value_counts())
        
        # Extract accession
        df_summary['Accession'] = df_summary['subject_id'].str.split('|').str[1]
        
        # Merge with UniProt data (drop 'search' column from df_uni to avoid conflict)
        df_uni_subset = df_uni.drop(columns=['search'], errors='ignore')
        df_summary = df_summary.merge(df_uni_subset, on='Accession', how='left')
        
        if verbose:
            print(f"\nAfter merge, results per search:")
            print(df_summary['search'].value_counts())
            print(f"\nResults with valid Organism field:")
            print(df_summary['Organism'].value_counts())
        
        # Filter results
        dt_blast = df_summary.copy()
        
        # Remove self-hits in Human
        if 'GeneName' in dt_blast.columns:
            dt_blast = dt_blast[~((dt_blast['GeneName'] == row['gene_name']) & 
                                   (dt_blast['search'] == 'Human'))]
            if 'EntryName' in dt_blast.columns:
                dt_blast = dt_blast[dt_blast['EntryName'].notna()]
        else:
            print(f"Warning: 'GeneName' column not found in UniProt data for {row['gene_name']}")
        
        if len(dt_blast) == 0:
            print(f"No valid BLAST results for {row['gene_name']} - skipping")
            continue
        
        # Calculate alignment metrics
        dt_blast['alignment_proportion'] = dt_blast['alignment_length'] / dt_blast['N_AA']
        dt_blast['alignment_coverage'] = dt_blast['alignment_proportion'] * dt_blast['pct_identity']
        
        # Get top 10 results per species
        dt_full = dt_blast[dt_blast['sub_id'].str.contains('FULL', na=False)].copy()
        
        if verbose:
            print(f"Total FULL results: {len(dt_full)}")
            print(f"Results per organism:\n{dt_full['Organism'].value_counts()}")
        
        dt_full = dt_full.groupby('search', sort=False).head(10).reset_index(drop=True)
        
        if verbose:
            print(f"After grouping by search:\n{dt_full.groupby('search').size()}")
        
        if 'Organism' not in dt_full.columns:
            print(f"Warning: 'Organism' column not found for {row['gene_name']} - skipping visualization")
            out_list.append(dt_blast)
            continue
        
        # Get top hit per species for comparison
        top_hits = []
        for organism in ['Homo sapiens', 'Mus musculus', 'Macaca mulatta', 'Macaca fascicularis']:
            org_data = dt_full[dt_full['Organism'] == organism]
            if len(org_data) > 0:
                if organism == 'Homo sapiens':
                    non_perfect = org_data[org_data['pct_identity'] != 100]
                    if len(non_perfect) > 0:
                        top_hits.append(non_perfect.iloc[0])
                else:
                    top_hits.append(org_data.iloc[0])
        
        if len(top_hits) < 2:
            print(f"Insufficient data for comparison matrix for {row['gene_name']} - skipping")
            continue
        
        top_hits_df = pd.DataFrame(top_hits)
        
        # Extract sequences for pairwise comparison
        seqs = [seq]
        seq_labels = [f"{row['transcript_id']} (Human)"]
        
        for _, hit in top_hits_df.iterrows():
            subject_id = hit['subject_id']
            if hit['Organism'] == 'Mus musculus':
                matching_keys = [k for k in mouse_fa.keys() if subject_id in k]
                if matching_keys:
                    seqs.append(mouse_fa[matching_keys[0]])
                    seq_labels.append(f"{subject_id} (Mouse)")
                    if verbose:
                        print(f"Mouse: {matching_keys[0]}, length: {len(mouse_fa[matching_keys[0]])}")
            elif hit['Organism'] == 'Macaca mulatta':
                matching_keys = [k for k in macmu_fa.keys() if subject_id in k]
                if matching_keys:
                    seqs.append(macmu_fa[matching_keys[0]])
                    seq_labels.append(f"{subject_id} (Macmu)")
                    if verbose:
                        print(f"Macmu: {matching_keys[0]}, length: {len(macmu_fa[matching_keys[0]])}")
            elif hit['Organism'] == 'Macaca fascicularis':
                matching_keys = [k for k in macfa_fa.keys() if subject_id in k]
                if matching_keys:
                    seqs.append(macfa_fa[matching_keys[0]])
                    seq_labels.append(f"{subject_id} (Macfa)")
                    if verbose:
                        print(f"Macfa: {matching_keys[0]}, length: {len(macfa_fa[matching_keys[0]])}")
        
        if verbose:
            print(f"\nQuery (Human): length {len(seq)}")
            print(f"Total sequences for comparison: {len(seqs)}")
        
        # Create species labels for heatmap BEFORE the alignment loop
        # Y-axis: Full FASTA IDs (or transcript ID for human)
        species_labels_y = []
        # X-axis: Species names only
        species_labels_x = []
        
        for label in seq_labels:
            if '(Human)' in label:
                # Y-axis: use transcript ID for human
                species_labels_y.append(row['transcript_id'])
                # X-axis: species name
                species_labels_x.append('Homo sapiens')
            elif '(Mouse)' in label:
                # Y-axis: FASTA ID
                fasta_id = label.split(' (Mouse)')[0]
                species_labels_y.append(fasta_id)
                # X-axis: species name
                species_labels_x.append('Mus musculus')
            elif '(Macmu)' in label:
                fasta_id = label.split(' (Macmu)')[0]
                species_labels_y.append(fasta_id)
                species_labels_x.append('Macaca mulatta')
            elif '(Macfa)' in label:
                fasta_id = label.split(' (Macfa)')[0]
                species_labels_y.append(fasta_id)
                species_labels_x.append('Macaca fascicularis')
            else:
                species_labels_y.append('Unknown')
                species_labels_x.append('Unknown')
        
        # Calculate pairwise identity matrix
        n = len(seqs)
        pid_matrix = np.zeros((n, n))
        
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = -5
        aligner.extend_gap_score = -2
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    pid_matrix[i, j] = 100.0
                else:
                    alignments = aligner.align(seqs[i], seqs[j])
                    if alignments:
                        alignment = alignments[0]
                        
                        # Get the actual aligned sequences (not the formatted string)
                        # Use format() to get just the sequences without formatting
                        aligned_seq1 = alignment.format().split('\n')[0]
                        aligned_seq2 = alignment.format().split('\n')[2]
                        
                        # Actually, better to use the alignment coordinates directly
                        # Get aligned sequences using alignment indices
                        seq1_aligned = str(alignment).replace('\n', '').replace(' ', '')
                        
                        # Better approach: use Bio.Align's aligned property
                        # This gives us the actual alignment including gaps
                        from Bio.Align import substitution_matrices
                        
                        # Convert alignment to strings with gaps
                        aligned_seq1 = ""
                        aligned_seq2 = ""
                        for (start1, end1), (start2, end2) in zip(alignment.aligned[0], alignment.aligned[1]):
                            # Add the aligned regions
                            aligned_seq1 += seqs[i][start1:end1]
                            aligned_seq2 += seqs[j][start2:end2]
                            # Check if we need to add gaps
                        
                        # Actually, let's use a simpler approach - format the alignment properly
                        # Extract sequences from the full alignment string
                        align_str = format(alignment)
                        lines = align_str.split('\n')
                        
                        # Reconstruct full aligned sequences from all lines
                        aligned_seq1 = ''
                        aligned_seq2 = ''
                        for line_idx in range(0, len(lines), 4):  # Alignment comes in blocks of 4 lines
                            if line_idx < len(lines) and len(lines[line_idx]) > 20:
                                # Extract sequence after the position number
                                seq1_part = lines[line_idx].split()[-1] if len(lines[line_idx].split()) > 1 else ''
                                aligned_seq1 += seq1_part
                            if line_idx + 2 < len(lines) and len(lines[line_idx + 2]) > 20:
                                seq2_part = lines[line_idx + 2].split()[-1] if len(lines[line_idx + 2].split()) > 1 else ''
                                aligned_seq2 += seq2_part
                        
                        # Count ALL matching positions
                        matches = sum(a == b for a, b in zip(aligned_seq1, aligned_seq2))
                        total_len = len(aligned_seq1)
                        pid = (matches / total_len) * 100
                        
                        if verbose and i == 0 and j == 1:
                            print(f"\nDebug alignment seq {i} ({species_labels_y[i]}) vs seq {j} ({species_labels_y[j]}):")
                            print(f"Alignment length: {total_len}")
                            print(f"Matches: {matches}")
                            print(f"Percent identity: {pid:.1f}%")
                            print(f"Seq lengths: {len(seqs[i])}, {len(seqs[j])}")
                        
                        pid_matrix[i, j] = pid
                        pid_matrix[j, i] = pid
        
        # Create figure
        fig = plt.figure(figsize=figsize, facecolor='white', dpi=dpi)
        gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.0], hspace=0.3, wspace=0.4,
                            left=0.05, right=0.98, top=0.90, bottom=0.15)
        
        # Left panel: Table
        ax_table = fig.add_subplot(gs[0])
        ax_table.axis('off')
        ax_table.set_xlim(0, 1)
        ax_table.set_ylim(0, 1)
        
        organisms = ['Homo sapiens', 'Mus musculus', 'Macaca mulatta', 'Macaca fascicularis']
        organism_colors = {
            'Homo sapiens': '#28154C',
            'Mus musculus': '#2E95D2', 
            'Macaca mulatta': '#18B5CB',
            'Macaca fascicularis': '#6BC291'
        }
        organism_bg_colors = {
            'Homo sapiens': '#E8E3F0',
            'Mus musculus': '#E3F2FD', 
            'Macaca mulatta': '#E0F7FA',
            'Macaca fascicularis': '#E8F5E9'
        }
        
        y_position = 0.98
        cell_height = 0.016  # Reduced from 0.018 to fit all 10 rows per organism
        header_height = 0.032
        section_gap = 0.015  # Slightly reduced gap
        
        for organism in organisms:
            org_data = dt_full[dt_full['Organism'] == organism].copy()
            if len(org_data) == 0:
                continue
            
            if verbose:
                print(f"{organism}: {len(org_data)} results")
            
            bbox_props = dict(boxstyle='round,pad=0.5', 
                            facecolor=organism_colors.get(organism, '#666666'),
                            edgecolor='none',
                            alpha=0.9)
            ax_table.text(0.015, y_position, f"  {organism}  ", 
                         fontsize=11, fontweight='bold', 
                         verticalalignment='top',
                         color='white',
                         family='sans-serif',
                         bbox=bbox_props)
            y_position -= header_height
            
            headers = [('Subject ID', 0.015, 'left'), 
                      ('Gene', 0.30, 'left'),
                      ('% ID', 0.42, 'center'),
                      ('Align Len', 0.52, 'center'),
                      ('Description', 0.64, 'left')]
            
            for header, x_pos, align in headers:
                ax_table.text(x_pos, y_position, header,
                            fontsize=8, fontweight='bold',
                            verticalalignment='top',
                            horizontalalignment=align,
                            color='#444444',
                            family='sans-serif')
            
            y_position -= cell_height
            
            rows_displayed = 0
            for idx_row, (_, row_data) in enumerate(org_data.head(10).iterrows()):
                if y_position < 0.05:
                    if verbose:
                        print(f"Warning: Ran out of space for {organism} after {rows_displayed} rows")
                    break
                
                bg_color = organism_bg_colors.get(organism, '#F8F8F8')
                ax_table.add_patch(plt.Rectangle((0.01, y_position - cell_height + 0.003), 
                                                0.98, cell_height,
                                                facecolor=bg_color, 
                                                edgecolor='none',
                                                zorder=0))
                
                values = [
                    (str(row_data['subject_id'])[:32], 0.015, 'left', '#000000', 'normal'),
                    (str(row_data['GeneName'])[:12] if pd.notna(row_data.get('GeneName')) else '', 0.30, 'left', '#333333', 'normal'),
                    (f"{row_data['pct_identity']:.1f}", 0.42, 'center', '#1a5490', 'bold'),
                    (str(int(row_data['alignment_length'])), 0.52, 'center', '#666666', 'normal'),
                    (str(row_data.get('Description', ''))[:50], 0.64, 'left', '#555555', 'normal')
                ]
                
                for val, x_pos, align, color, weight in values:
                    ax_table.text(x_pos, y_position, val,
                                fontsize=7,
                                verticalalignment='top',
                                horizontalalignment=align,
                                color=color,
                                fontweight=weight,
                                family='sans-serif')
                y_position -= cell_height
                rows_displayed += 1
            
            if verbose:
                print(f"{organism}: displayed {rows_displayed} of {len(org_data.head(10))} rows")
            
            y_position -= section_gap
        
        # Right panel: Heatmap
        ax_heatmap = fig.add_subplot(gs[1])
        
        # Perform clustering
        if n > 2:
            distance_matrix = 100 - pid_matrix
            np.fill_diagonal(distance_matrix, 0)
            condensed_dist = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_dist, method='average')
            dendro = dendrogram(linkage_matrix, no_plot=True)
            cluster_order = dendro['leaves']
            
            # Reorder matrix and labels according to clustering
            pid_matrix_ordered = pid_matrix[np.ix_(cluster_order, cluster_order)]
            species_labels_y_ordered = [species_labels_y[i] for i in cluster_order]
            species_labels_x_ordered = [species_labels_x[i] for i in cluster_order]
        else:
            # No clustering needed for 2 or fewer sequences
            pid_matrix_ordered = pid_matrix
            species_labels_y_ordered = species_labels_y
            species_labels_x_ordered = species_labels_x
        
        mask = ~np.eye(n, dtype=bool)
        off_diag_values = pid_matrix_ordered[mask]
        
        if len(off_diag_values) > 0:
            vmin = np.floor(off_diag_values.min() / 10) * 10
            if vmin == 100:
                vmin = 90
        else:
            vmin = 90
        vmax = 100
        
        print(f"Heatmap scale: {vmin} to {vmax}")
        
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['#ffffff', '#d0e1f2', '#4292c6', '#08519c', '#08306b']
        cmap = LinearSegmentedColormap.from_list('custom_blues', colors_list, N=100)
        
        im = ax_heatmap.imshow(pid_matrix_ordered, cmap=cmap, vmin=vmin, vmax=vmax, 
                              aspect='equal', interpolation='nearest')
        
        for i in range(n):
            for j in range(n):
                value = pid_matrix_ordered[i, j]
                normalized = (value - vmin) / (vmax - vmin)
                
                if normalized > 0.6:
                    color = 'white'
                    weight = 'bold'
                elif normalized > 0.3:
                    color = 'white'
                    weight = 'normal'
                else:
                    color = 'black'
                    weight = 'normal'
                    
                ax_heatmap.text(j, i, f'{value:.1f}',
                              ha="center", va="center", 
                              color=color, fontsize=11, 
                              fontweight=weight,
                              family='sans-serif')
        
        ax_heatmap.set_xticks(range(n))
        ax_heatmap.set_yticks(range(n))
        ax_heatmap.set_xticklabels(species_labels_x_ordered, rotation=45, ha='right', 
                                   fontsize=9, family='sans-serif')
        ax_heatmap.set_yticklabels(species_labels_y_ordered, fontsize=9, 
                                   family='sans-serif')
        
        ax_heatmap.set_xticks(np.arange(n) - 0.5, minor=True)
        ax_heatmap.set_yticks(np.arange(n) - 0.5, minor=True)
        ax_heatmap.grid(which='minor', color='white', linestyle='-', linewidth=2.5)
        ax_heatmap.tick_params(which='minor', size=0)
        ax_heatmap.tick_params(which='major', length=0)
        
        for spine in ax_heatmap.spines.values():
            spine.set_visible(False)
        
        title_text = f"{row['gene_name']} {row['transcript_id']} ({seq_len} AA)\nSequence Identity (%)"
        ax_heatmap.set_title(title_text, fontsize=11, fontweight='bold', 
                           pad=15, family='sans-serif', color='#333333')
        
        cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.04, pad=0.06, 
                          shrink=0.75, aspect=18)
        cbar.set_label('% Identity', rotation=270, labelpad=20, 
                      fontsize=9, family='sans-serif')
        cbar.ax.tick_params(labelsize=8, length=2, width=0.5)
        cbar.outline.set_linewidth(0.5)
        cbar.outline.set_edgecolor('#CCCCCC')
        
        fig.suptitle(f"{row['gene_name']} - Homology Analysis", 
                    fontsize=18, fontweight='bold', y=0.96,
                    family='sans-serif', color='#222222')
        
        output_file = Path(output_dir) / f"{row['gene_name']}_{row['transcript_id']}_homology.png"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_file}")
        
        out_list.append(dt_blast)
    
    return out_list