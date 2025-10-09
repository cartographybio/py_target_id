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

from py_target_id.utils import google_copy, set_google_copy_version

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

def process_path(path_info):
    """Function to list and process files for each path"""
    path_name, path = path_info
    pattern = PATTERNS[path_name]
    
    # List files from GCS
    try:
        gcp_cmd = google_copy()
        cmd = f"{gcp_cmd} ls {path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        files = result.stdout.strip().split('\n')
        files = [f for f in files if f]  # Remove empty strings
        
        if not files:
            print(f"  Warning: No files found in {path_name} directory")
            print(f"    Path: {path}")
            return pd.DataFrame()
            
    except subprocess.CalledProcessError as e:
        print(f"  Error accessing {path_name} directory:")
        print(f"    Path: {path}")
        print(f"    Command: {cmd}")
        print(f"    Return code: {e.returncode}")
        if e.stderr:
            print(f"    Error output: {e.stderr}")
        if e.stdout:
            print(f"    Standard output: {e.stdout}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  Unexpected error processing {path_name}:")
        print(f"    Path: {path}")
        print(f"    Error: {str(e)}")
        return pd.DataFrame()
    
    # Extract sample IDs by removing file extensions
    basenames = [os.path.basename(f) for f in files]
    sample_ids = [re.sub(pattern, '', basename) for basename in basenames]
    
    print(f"  Found {len(files)} {path_name} files")
    
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
        print(f"Using GCS command: {google_copy()}")
        print("="*70)
    
    # Test GCS connectivity first
    if verbose:
        test_cmd = f"{google_copy()} ls gs://cartography_target_id_package/"
        test_result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
        if test_result.returncode != 0:
            print("\n⚠️  WARNING: Cannot access GCS bucket")
            print(f"Command: {test_cmd}")
            print(f"Error: {test_result.stderr}")
            print("\nTroubleshooting steps:")
            print("  1. Check authentication: gcloud auth list")
            print("  2. Login if needed: gcloud auth application-default login")
            print("  3. Check bucket permissions")
            print("  4. Try switching GCS command: tid.utils.set_google_copy_version('gsutil')")
            print("="*70 + "\n")
        else:
            print("✓ GCS bucket accessible\n")
    
    # Process all paths (parallel or sequential)
    path_items = list(paths.items())
    
    if parallel and os.name != 'nt':  # Unix-like systems
        if verbose:
            print("Processing directories in parallel...\n")
        # Use parallel processing
        with Pool(processes=min(7, cpu_count())) as pool:  # Changed from 6 to 7
            file_lists = pool.map(process_path, path_items)
    else:
        if verbose:
            print("Processing directories sequentially...\n")
        # Sequential processing
        file_lists = [process_path(item) for item in path_items]
    
    # Combine all file information
    all_files = pd.concat(file_lists, ignore_index=True)
    
    if all_files.empty:
        print("\n" + "="*70)
        print("❌ ERROR: No files found in any directory")
        print("="*70)
        print("\nPossible causes:")
        print("  1. Wrong date in paths (currently: 20251008)")
        print("  2. Authentication issues")
        print("  3. Incorrect GCS command (currently: {})".format(google_copy()))
        print("  4. Files haven't been uploaded yet")
        print("\nDebugging steps:")
        print("  # Check available dates:")
        print("  subprocess.run('gcloud storage ls gs://cartography_target_id_package/Sample_Input/', shell=True)")
        print("\n  # Try different GCS command:")
        print("  tid.utils.set_google_copy_version('gsutil')  # or 'gcloud storage'")
        print("  manifest = tid.utils.load_manifest()")
        print("="*70)
        warnings.warn("No files found in any directory")
        return pd.DataFrame()
    
    if verbose:
        print(f"\n✓ Successfully loaded {len(all_files)} total files")
    
    # Count files per sample ID across all types
    file_counts = all_files.groupby('sample_id').size().reset_index(name='count')
    complete_samples = file_counts[file_counts['count'] == 7]['sample_id'].tolist()  # Changed from 6 to 7
    incomplete_samples = file_counts[file_counts['count'] != 7]  # Changed from 6 to 7
    
    if verbose and not incomplete_samples.empty:
        print(f"\n⚠️  Warning: {len(incomplete_samples)} samples have incomplete files:")
        print(incomplete_samples.to_string(index=False))
    
    if not complete_samples:
        print("\n" + "="*70)
        print("❌ ERROR: No samples found with all 7 required file types")  # Changed from 6 to 7
        print("="*70)
        print("\nFile type distribution:")
        print(all_files['type'].value_counts().to_string())
        print("\nSamples and their file counts:")
        print(file_counts.sort_values('count').to_string(index=False))
        print("="*70)
        warnings.warn("No samples found with all 7 required file types")  # Changed from 6 to 7
        return pd.DataFrame()
    
    if verbose:
        print(f"✓ Found {len(complete_samples)} samples with complete data (all 7 file types)")  # Changed from 6 to 7
    
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
        # Convert DOC to numeric for sorting (handle potential non-numeric values)
        df['DOC_numeric'] = pd.to_numeric(df['DOC'], errors='coerce')
        pre_filter_count = len(df)
        df = (df.sort_values('DOC_numeric', ascending=False)
                .drop_duplicates(subset='Sample_ID', keep='first')
                .sort_values(['Indication', 'Sample_ID'])
                .drop('DOC_numeric', axis=1)
                .reset_index(drop=True))
        if verbose and pre_filter_count > len(df):
            print(f"✓ Filtered to latest versions: {pre_filter_count} → {len(df)} samples")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ Successfully loaded manifest with {len(df)} samples")
        print(f"{'='*70}\n")
        print("Sample breakdown by indication:")
        print(df['Indication'].value_counts().to_string())
        print()
    
    return df