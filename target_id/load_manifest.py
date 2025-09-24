"""
Manifest loading and processing functions.
"""

import pandas as pd
import subprocess
import re
import os
from multiprocessing import Pool, cpu_count
import warnings

from .utils import google_copy, set_google_copy_version

# File extension patterns for regex matching
PATTERNS = {
    'h5map': r'\.h5$',
    'archr': r'\.ArchRCells\.rds$', 
    'stats': r'\.stats\.csv$',
    'archr_malig': r'\.ArchRCells\.h5$',
    'se_malig': r'\.malig\.se\.rds$',
    'metadata': r'\.mdata\.rds$'
}

def process_path(path_info):
    """Function to list and process files for each path"""
    path_name, path = path_info
    pattern = PATTERNS[path_name]
    
    # List files from GCS
    try:
        cmd = f"{google_copy()} ls {path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        files = result.stdout.strip().split('\n')
        files = [f for f in files if f]  # Remove empty strings
    except subprocess.CalledProcessError:
        return pd.DataFrame()
    
    if not files:
        return pd.DataFrame()
    
    # Extract sample IDs by removing file extensions
    basenames = [os.path.basename(f) for f in files]
    sample_ids = [re.sub(pattern, '', basename) for basename in basenames]
    
    # Return DataFrame with file type and paths
    return pd.DataFrame({
        'type': [path_name] * len(files),
        'sample_id': sample_ids,
        'path': files
    })

def load_manifest(latest=True, parallel=True):
    """
    Load and process manifest files from Google Cloud Storage.
    
    Args:
        latest (bool): If True, filter to latest versions only
        parallel (bool): If True, use parallel processing
    
    Returns:
        pandas.DataFrame: Manifest with sample information and file paths
    """
    
    # Define paths
    paths = {
        'h5map': 'gs://cartography_target_id_samples/Samples_v2/h5map/',
        'archr': 'gs://cartography_target_id_samples/Samples_v2/ArchRCells/',
        'stats': 'gs://cartography_target_id_samples/Samples_v2/Stats/',
        'archr_malig': 'gs://cartography_target_id_samples/Samples_v2/ArchRCells_Malig/',
        'se_malig': 'gs://cartography_target_id_samples/Samples_v2/SE_Malig/',
        'metadata': 'gs://cartography_target_id_samples/Samples_v2/Metadata/'
    }
    
    print("Loading manifest files...")
    
    # Process all paths (parallel or sequential)
    path_items = list(paths.items())
    
    if parallel and os.name != 'nt':  # Unix-like systems
        # Use parallel processing
        with Pool(processes=min(6, cpu_count())) as pool:
            file_lists = pool.map(process_path, path_items)
    else:
        # Sequential processing
        file_lists = [process_path(item) for item in path_items]
    
    # Combine all file information
    all_files = pd.concat(file_lists, ignore_index=True)
    
    if all_files.empty:
        warnings.warn("No files found in any directory")
        return pd.DataFrame()
    
    # Count files per sample ID across all types
    file_counts = all_files.groupby('sample_id').size().reset_index(name='count')
    complete_samples = file_counts[file_counts['count'] == 6]['sample_id'].tolist()
    
    if not complete_samples:
        warnings.warn("No samples found with all 6 required file types")
        return pd.DataFrame()
    
    print(f"Found {len(complete_samples)} samples in total")
    
    # Filter to complete samples and reshape to wide format
    complete_files = all_files[all_files['sample_id'].isin(complete_samples)]
    wide_files = complete_files.pivot(index='sample_id', columns='type', values='path').reset_index()
    
    # Parse sample IDs
    id_parts = wide_files['sample_id'].str.split('\\._.')
    
    # Create final manifest
    df = pd.DataFrame({
        'Indication': [parts[0] if len(parts) > 0 else '' for parts in id_parts],
        'Sample_ID': [parts[1] if len(parts) > 1 else '' for parts in id_parts],
        'DOC': [parts[2] if len(parts) > 2 else '' for parts in id_parts],
        'Full_ID': wide_files['sample_id'],
        'Cloud_h5map': wide_files.get('h5map', ''),
        'Cloud_Archr': wide_files.get('archr', ''),
        'Cloud_Stats': wide_files.get('stats', ''),
        'Cloud_Archr_Malig': wide_files.get('archr_malig', ''),
        'Cloud_SE_Malig': wide_files.get('se_malig', ''),
        'Cloud_Metadata': wide_files.get('metadata', '')
    })
    
    # Filter to latest versions if requested
    if latest:
        # Convert DOC to numeric for sorting (handle potential non-numeric values)
        df['DOC_numeric'] = pd.to_numeric(df['DOC'], errors='coerce')
        df = (df.sort_values('DOC_numeric', ascending=False)
                .drop_duplicates(subset='Sample_ID', keep='first')
                .sort_values(['Indication', 'Sample_ID'])
                .drop('DOC_numeric', axis=1)
                .reset_index(drop=True))
    
    print(f"Returning manifest with {len(df)} samples")
    return df