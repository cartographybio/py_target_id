"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['load_manifest']

import pandas as pd
import subprocess
import re
import os
from multiprocessing import Pool, cpu_count
import warnings

# File extension patterns for regex matching
PATTERNS = {
    'h5map': r'\.h5$',
    'archr': r'\.ArchRCells\.rds$', 
    'stats': r'\.stats\.csv$',
    'archr_malig': r'\.ArchRCells\.h5$',
    'se_malig': r'\.malig\.se\.rds$',
    'ad_malig': r'\.malig\.h5ad$',
    'metadata': r'\.mdata\.rds$'
}

# Global variable for GCS command - set once
_GCS_CMD = None

def _init_gcs_cmd():
    """Initialize GCS command once"""
    global _GCS_CMD
    if _GCS_CMD is None:
        try:
            from py_target_id.utils.google import GOOGLE_COPY_VERSION
            if GOOGLE_COPY_VERSION == "gsutil":
                _GCS_CMD = "gsutil"
            else:
                _GCS_CMD = "gcloud storage"
        except:
            _GCS_CMD = "gcloud storage"
    return _GCS_CMD

def process_path(path_info):
    """Function to list and process files for each path"""
    path_name, path = path_info
    pattern = PATTERNS[path_name]
    
    # Get GCS command (fast - uses global)
    gcp_cmd = _init_gcs_cmd()
    
    # List files from GCS
    try:
        cmd = f"{gcp_cmd} ls {path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        files = result.stdout.strip().split('\n')
        files = [f for f in files if f]  # Remove empty strings
        
        if not files:
            return pd.DataFrame()
            
    except subprocess.CalledProcessError:
        return pd.DataFrame()
    except Exception:
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

def load_manifest(latest=True, parallel=True, verbose=True):
    """
    Load and process manifest files from Google Cloud Storage.
    
    Args:
        latest (bool): If True, filter to latest versions only
        parallel (bool): If True, use parallel processing
        verbose (bool): If True, print detailed debugging information
    
    Returns:
        pandas.DataFrame: Manifest with sample information and file paths
    """
    
    # Initialize GCS command
    gcp_cmd = _init_gcs_cmd()
    
    # Define paths
    paths = {
        'h5map': 'gs://cartography_target_id_package/Sample_Input/20251008/processed/h5map/',
        'archr': 'gs://cartography_target_id_package/Sample_Input/20251008/processed/ArchRCells/',
        'stats': 'gs://cartography_target_id_package/Sample_Input/20251008/processed/Stats/',
        'archr_malig': 'gs://cartography_target_id_package/Sample_Input/20251008/processed/ArchRCells_Malig/',
        'se_malig': 'gs://cartography_target_id_package/Sample_Input/20251008/processed/SE_Malig/',
        'ad_malig': 'gs://cartography_target_id_package/Sample_Input/20251008/processed/AD_Malig/',
        'metadata': 'gs://cartography_target_id_package/Sample_Input/20251008/processed/Metadata/'
    }
    
    if verbose:
        print("="*70)
        print("Loading manifest files from GCS...")
        print(f"Using GCS command: {gcp_cmd}")
        print("="*70)
    
    # Process all paths
    path_items = list(paths.items())
    
    if parallel and os.name != 'nt':
        if verbose:
            print("Processing directories in parallel...\n")
        with Pool(processes=min(7, cpu_count())) as pool:
            file_lists = pool.map(process_path, path_items)
    else:
        if verbose:
            print("Processing directories sequentially...\n")
        file_lists = [process_path(item) for item in path_items]
    
    # Combine all file information
    all_files = pd.concat([df for df in file_lists if not df.empty], ignore_index=True)
    
    if all_files.empty:
        if verbose:
            print("\n❌ ERROR: No files found in any directory")
        warnings.warn("No files found in any directory")
        return pd.DataFrame()
    
    if verbose:
        print(f"\n✓ Successfully loaded {len(all_files)} total files")
    
    # Count files per sample ID across all types
    file_counts = all_files.groupby('sample_id').size().reset_index(name='count')
    complete_samples = file_counts[file_counts['count'] == 7]['sample_id'].tolist()
    
    if not complete_samples:
        if verbose:
            print("\n❌ ERROR: No samples found with all 7 required file types")
            print("\nFile type distribution:")
            print(all_files['type'].value_counts().to_string())
        warnings.warn("No samples found with all 7 required file types")
        return pd.DataFrame()
    
    if verbose:
        print(f"✓ Found {len(complete_samples)} samples with complete data (all 7 file types)")
    
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
        'Cloud_AD_Malig': wide_files.get('ad_malig', ''),
        'Cloud_Metadata': wide_files.get('metadata', '')
    })
    
    # Filter to latest versions if requested
    if latest:
        df['DOC_numeric'] = pd.to_numeric(df['DOC'], errors='coerce')
        df = (df.sort_values('DOC_numeric', ascending=False)
                .drop_duplicates(subset='Sample_ID', keep='first')
                .sort_values(['Indication', 'Sample_ID'])
                .drop('DOC_numeric', axis=1)
                .reset_index(drop=True))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ Successfully loaded manifest with {len(df)} samples")
        print(f"{'='*70}\n")
        print("Sample breakdown by indication:")
        print(df['Indication'].value_counts().to_string())
        print()
    
    return df