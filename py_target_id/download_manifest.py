
"""
Download functions for manifest files.
"""

# Define what gets exported
__all__ = ['download_manifest']

import os
import subprocess
import time
from pathlib import Path
import pandas as pd
import warnings

from py_target_id.google import google_copy

def download_manifest(
    manifest: pd.DataFrame, 
    dest_dir: str = "temp", 
    create_subdirs: bool = True, 
    overwrite: bool = False, 
    verbose: bool = True
) -> pd.DataFrame:
    """
    Download files from cloud storage based on a manifest DataFrame.
    
    Parameters:
    -----------
    manifest : pd.DataFrame
        DataFrame containing cloud file paths in Cloud_* columns
    dest_dir : str
        Destination directory for downloads
    create_subdirs : bool
        Whether to create subdirectories for different file types
    overwrite : bool
        Whether to overwrite existing files
    verbose : bool
        Whether to print progress messages
        
    Returns:
    --------
    pd.DataFrame
        Modified manifest with Local_* columns added
    """
    
    if len(manifest) == 0:
        if verbose:
            print("No files to download")
        return manifest
    
    # Create a copy to avoid modifying the original
    manifest_copy = manifest.copy()
    
    start_time = time.time()
    
    # Create directories
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    dest_dir = str(dest_path.resolve())
    
    if create_subdirs:
        subdirs = ["h5map", "ArchRCells", "Stats", "ArchRCells_Malig", "SE_Malig", "Metadata"]
        for subdir in subdirs:
            subdir_path = dest_path / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)
    
    # Check if required Cloud_ columns exist
    required_cols = ["Cloud_h5map", "Cloud_Archr", "Cloud_Stats", 
                    "Cloud_Archr_Malig", "Cloud_SE_Malig", "Cloud_Metadata"]
    missing_cols = [col for col in required_cols if col not in manifest_copy.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check for any NULL or non-string values in Cloud_ columns
    for col in required_cols:
        if not pd.api.types.is_string_dtype(manifest_copy[col]):
            # Try to convert to string
            manifest_copy[col] = manifest_copy[col].astype(str)
        
        if manifest_copy[col].isna().any():
            warnings.warn(f"Found NA values in {col} column")
    
    # Add Local_ columns to manifest based on Cloud_ columns
    if create_subdirs:
        manifest_copy['Local_h5map'] = manifest_copy['Cloud_h5map'].apply(
            lambda x: os.path.join(dest_dir, "h5map", os.path.basename(str(x)))
        )
        manifest_copy['Local_Archr'] = manifest_copy['Cloud_Archr'].apply(
            lambda x: os.path.join(dest_dir, "ArchRCells", os.path.basename(str(x)))
        )
        manifest_copy['Local_Stats'] = manifest_copy['Cloud_Stats'].apply(
            lambda x: os.path.join(dest_dir, "Stats", os.path.basename(str(x)))
        )
        manifest_copy['Local_Archr_Malig'] = manifest_copy['Cloud_Archr_Malig'].apply(
            lambda x: os.path.join(dest_dir, "ArchRCells_Malig", os.path.basename(str(x)))
        )
        manifest_copy['Local_SE_Malig'] = manifest_copy['Cloud_SE_Malig'].apply(
            lambda x: os.path.join(dest_dir, "SE_Malig", os.path.basename(str(x)))
        )
        manifest_copy['Local_Metadata'] = manifest_copy['Cloud_Metadata'].apply(
            lambda x: os.path.join(dest_dir, "Metadata", os.path.basename(str(x)))
        )
    else:
        manifest_copy['Local_h5map'] = manifest_copy['Cloud_h5map'].apply(
            lambda x: os.path.join(dest_dir, os.path.basename(str(x)))
        )
        manifest_copy['Local_Archr'] = manifest_copy['Cloud_Archr'].apply(
            lambda x: os.path.join(dest_dir, os.path.basename(str(x)))
        )
        manifest_copy['Local_Stats'] = manifest_copy['Cloud_Stats'].apply(
            lambda x: os.path.join(dest_dir, os.path.basename(str(x)))
        )
        manifest_copy['Local_Archr_Malig'] = manifest_copy['Cloud_Archr_Malig'].apply(
            lambda x: os.path.join(dest_dir, os.path.basename(str(x)))
        )
        manifest_copy['Local_SE_Malig'] = manifest_copy['Cloud_SE_Malig'].apply(
            lambda x: os.path.join(dest_dir, os.path.basename(str(x)))
        )
        manifest_copy['Local_Metadata'] = manifest_copy['Cloud_Metadata'].apply(
            lambda x: os.path.join(dest_dir, os.path.basename(str(x)))
        )
    
    # Define file types
    file_types = {
        'h5map': {
            'cloud_files': manifest_copy['Cloud_h5map'].tolist(),
            'local_files': manifest_copy['Local_h5map'].tolist(),
            'subdir': 'h5map'
        },
        'archr': {
            'cloud_files': manifest_copy['Cloud_Archr'].tolist(),
            'local_files': manifest_copy['Local_Archr'].tolist(),
            'subdir': 'ArchRCells'
        },
        'stats': {
            'cloud_files': manifest_copy['Cloud_Stats'].tolist(),
            'local_files': manifest_copy['Local_Stats'].tolist(),
            'subdir': 'Stats'
        },
        'archr_malig': {
            'cloud_files': manifest_copy['Cloud_Archr_Malig'].tolist(),
            'local_files': manifest_copy['Local_Archr_Malig'].tolist(),
            'subdir': 'ArchRCells_Malig'
        },
        'se_malig': {
            'cloud_files': manifest_copy['Cloud_SE_Malig'].tolist(),
            'local_files': manifest_copy['Local_SE_Malig'].tolist(),
            'subdir': 'SE_Malig'
        },
        'metadata': {
            'cloud_files': manifest_copy['Cloud_Metadata'].tolist(),
            'local_files': manifest_copy['Local_Metadata'].tolist(),
            'subdir': 'Metadata'
        }
    }
    
    # Check for existing files and filter if not overwriting
    if not overwrite:
        if verbose:
            print("Checking for existing files...")
        
        for file_type in file_types:
            cloud_files = file_types[file_type]['cloud_files']
            local_files = file_types[file_type]['local_files']
            
            existing = [os.path.exists(f) for f in local_files]
            if any(existing):
                n_existing = sum(existing)
                if verbose:
                    print(f"Found {n_existing} existing {file_type} files (skipping)")
                
                # Filter out existing files
                file_types[file_type]['cloud_files'] = [
                    cloud for cloud, exists in zip(cloud_files, existing) if not exists
                ]
                file_types[file_type]['local_files'] = [
                    local for local, exists in zip(local_files, existing) if not exists
                ]
    
    # Count total files to download after filtering
    total_files = sum(len(ft['cloud_files']) for ft in file_types.values())
    
    if total_files == 0:
        if verbose:
            print("No files to download (all files already exist)")
        return manifest_copy
    
    if verbose:
        print(f"Will download {total_files} files total")
    
    completed_files = 0
    
    for i, (file_type, file_info) in enumerate(file_types.items(), 1):
        cloud_files = file_info['cloud_files']
        subdir = file_info['subdir']
        
        # Skip if no files to download for this type
        if len(cloud_files) == 0:
            if verbose:
                print(f"Skipping {file_type} files (all already exist)")
            continue
        
        if create_subdirs:
            dest_path_str = os.path.join(dest_dir, subdir)
        else:
            dest_path_str = dest_dir
        
        # Progress update
        elapsed_mins = (time.time() - start_time) / 60
        if completed_files > 0 and elapsed_mins > 0:
            rate = completed_files / elapsed_mins
            remaining = total_files - completed_files
            eta_mins = remaining / rate
            
            if eta_mins < 60:
                eta_str = f"{eta_mins:.1f} minutes"
            else:
                eta_str = f"{eta_mins/60:.1f} hours"
            
            pct = round(100 * completed_files / total_files, 1)
            if verbose:
                print(f"[{pct}%] Starting {file_type} files ({completed_files}/{total_files}). "
                      f"Rate: {rate:.1f} files/min. ETA: {eta_str}")
        else:
            if verbose:
                print(f"[{i}/{len(file_types)}] Starting {file_type} files ({len(cloud_files)} files)...")
        
        # Create gsutil command for all files of this type
        quoted_files = [f"'{f}'" for f in cloud_files]
        files_str = " ".join(quoted_files)
        cmd = f"{google_copy()} cp {files_str} '{dest_path_str}/'"
        
        # Execute with timing
        batch_start = time.time()
        result = subprocess.run(cmd, shell=True)
        batch_elapsed = time.time() - batch_start
        
        if result.returncode == 0:
            completed_files += len(cloud_files)
            rate_this_batch = len(cloud_files) / (batch_elapsed / 60)  # files per minute
            if verbose:
                print(f"✓ Completed {len(cloud_files)} {file_type} files in {batch_elapsed:.1f} seconds "
                      f"({rate_this_batch:.1f} files/min)")
        else:
            warnings.warn(f"✗ Failed to download {file_type} files")
    
    total_elapsed = (time.time() - start_time) / 60
    overall_rate = completed_files / total_elapsed if total_elapsed > 0 else 0
    if verbose:
        print(f"\n🎉 Download complete! {completed_files} files in {total_elapsed:.1f} minutes "
              f"({overall_rate:.1f} files/min average)")
    
    manifest_copy = manifest_copy.reset_index(drop=True)
    
    return manifest_copy