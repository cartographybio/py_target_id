"""
Download functions for manifest files.
"""

# Define what gets exported
__all__ = ['download_manifest', 'parallel_h5_to_zarr']

import os
import subprocess
import time
from pathlib import Path
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from py_target_id.google import google_copy
from py_target_id.zarr import h5_to_zarr


def _convert_single_h5_to_zarr(args):
    """Worker function for parallel h5 to zarr conversion"""
    local_h5, zarr_path, h5_path = args
    try:
        h5_to_zarr(local_h5, zarr_path, h5_path=h5_path)
        return True, os.path.basename(local_h5), None
    except Exception as e:
        return False, os.path.basename(local_h5), str(e)


def parallel_h5_to_zarr(local_files, zarr_files, h5_path, n_workers=None, verbose=True):
    """
    Convert multiple h5 files to zarr in parallel
    
    Parameters
    ----------
    local_files : list
        List of h5 file paths to convert
    zarr_files : list
        List of output zarr paths
    h5_path : str
        Path within h5 file (e.g., 'RNA' or 'assays/RNA.counts')
    n_workers : int, optional
        Number of parallel workers. Default: CPU count - 1
    verbose : bool
        Print progress messages
        
    Raises
    ------
    RuntimeError
        If any conversions fail
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Prepare arguments, only for files that exist
    args_list = [(h5, zarr, h5_path) for h5, zarr in zip(local_files, zarr_files) 
                 if os.path.exists(h5)]
    
    if len(args_list) == 0:
        if verbose:
            print("No files to convert")
        return
    
    if verbose:
        print(f"Converting {len(args_list)} files using {n_workers} workers...")
    
    completed = 0
    failed = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all conversion jobs
        futures = {executor.submit(_convert_single_h5_to_zarr, args): args 
                   for args in args_list}
        
        # Process results as they complete
        for future in as_completed(futures):
            success, filename, error = future.result()
            completed += 1
            
            if success:
                if verbose:
                    print(f"  ✓ [{completed}/{len(args_list)}] {filename}")
            else:
                failed.append((filename, error))
                if verbose:
                    print(f"  ✗ [{completed}/{len(args_list)}] {filename}: {error}")
    
    if failed:
        error_msg = f"Failed to convert {len(failed)} files: " + ", ".join([f[0] for f in failed])
        raise RuntimeError(error_msg)
    
    if verbose:
        print(f"  ✓ All {len(args_list)} files converted successfully")


def download_manifest(
    manifest: pd.DataFrame, 
    dest_dir: str = "temp", 
    create_subdirs: bool = True, 
    overwrite: bool = False,
    verbose: bool = True,
    n_workers: int = None
) -> pd.DataFrame:
    """
    Download files from cloud storage based on a manifest DataFrame.
    Automatically converts h5map files to zarr format after downloading.
    
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
    n_workers : int, optional
        Number of parallel workers for zarr conversion. Default: CPU count - 1
        
    Returns:
    --------
    pd.DataFrame
        Modified manifest with Local_* and Zarr_* columns added
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
        subdirs = ["h5map", "ArchRCells", "Stats", "ArchRCells_Malig", "SE_Malig", "Metadata", "Zarr_h5map", "Zarr_Archr_Malig"]
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
        
        # Add Zarr path columns (ALWAYS created)
        manifest_copy['Local_Zarr_h5map'] = manifest_copy['Cloud_h5map'].apply(
            lambda x: os.path.join(dest_dir, "Zarr_h5map", os.path.basename(str(x)).replace('.h5', '.zarr'))
        )
        manifest_copy['Local_Zarr_Archr_Malig'] = manifest_copy['Cloud_Archr_Malig'].apply(
            lambda x: os.path.join(dest_dir, "Zarr_Archr_Malig", os.path.basename(str(x)).replace('.h5', '.zarr'))
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
        
        # Add Zarr path columns (ALWAYS created)
        manifest_copy['Local_Zarr_h5map'] = manifest_copy['Cloud_h5map'].apply(
            lambda x: os.path.join(dest_dir, os.path.basename(str(x)).replace('.h5', '.zarr'))
        )
        manifest_copy['Local_Zarr_Archr_Malig'] = manifest_copy['Cloud_Archr_Malig'].apply(
            lambda x: os.path.join(dest_dir, os.path.basename(str(x)).replace('.h5', '.zarr'))
        )
    
    # Define file types
    file_types = {
        'h5map': {
            'cloud_files': manifest_copy['Cloud_h5map'].tolist(),
            'local_files': manifest_copy['Local_h5map'].tolist(),
            'subdir': 'h5map',
            'convert_to_zarr': True,
            'h5_path': 'assays/RNA.counts',
            'zarr_files': manifest_copy['Local_Zarr_h5map'].tolist()
        },
        'archr': {
            'cloud_files': manifest_copy['Cloud_Archr'].tolist(),
            'local_files': manifest_copy['Local_Archr'].tolist(),
            'subdir': 'ArchRCells',
            'convert_to_zarr': False
        },
        'stats': {
            'cloud_files': manifest_copy['Cloud_Stats'].tolist(),
            'local_files': manifest_copy['Local_Stats'].tolist(),
            'subdir': 'Stats',
            'convert_to_zarr': False
        },
        'archr_malig': {
            'cloud_files': manifest_copy['Cloud_Archr_Malig'].tolist(),
            'local_files': manifest_copy['Local_Archr_Malig'].tolist(),
            'subdir': 'ArchRCells_Malig',
            'convert_to_zarr': True,
            'h5_path': 'RNA',
            'zarr_files': manifest_copy['Local_Zarr_Archr_Malig'].tolist()
        },
        'se_malig': {
            'cloud_files': manifest_copy['Cloud_SE_Malig'].tolist(),
            'local_files': manifest_copy['Local_SE_Malig'].tolist(),
            'subdir': 'SE_Malig',
            'convert_to_zarr': False
        },
        'metadata': {
            'cloud_files': manifest_copy['Cloud_Metadata'].tolist(),
            'local_files': manifest_copy['Local_Metadata'].tolist(),
            'subdir': 'Metadata',
            'convert_to_zarr': False
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
                
                # Also filter zarr files if applicable
                if file_types[file_type].get('convert_to_zarr') and file_types[file_type].get('zarr_files'):
                    zarr_files = file_types[file_type]['zarr_files']
                    file_types[file_type]['zarr_files'] = [
                        zarr for zarr, exists in zip(zarr_files, existing) if not exists
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
            
            # Convert to zarr in parallel (MANDATORY for h5map and archr_malig)
            if file_info.get('convert_to_zarr'):
                if verbose:
                    print(f"Converting {len(cloud_files)} {file_type} files to zarr (parallel)...")
                
                local_files = file_info['local_files']
                zarr_files = file_info['zarr_files']
                h5_path = file_info['h5_path']
                
                zarr_start = time.time()
                
                try:
                    parallel_h5_to_zarr(
                        local_files, 
                        zarr_files, 
                        h5_path, 
                        n_workers=n_workers,
                        verbose=verbose
                    )
                except Exception as e:
                    error_msg = f"Zarr conversion failed: {str(e)}"
                    warnings.warn(f"  ✗ {error_msg}")
                    raise RuntimeError(error_msg)
                
                zarr_elapsed = time.time() - zarr_start
                if verbose:
                    print(f"  ✓ Zarr conversion completed in {zarr_elapsed:.1f} seconds")
        else:
            warnings.warn(f"✗ Failed to download {file_type} files")
    
    total_elapsed = (time.time() - start_time) / 60
    overall_rate = completed_files / total_elapsed if total_elapsed > 0 else 0
    if verbose:
        print(f"\nDownload complete! {completed_files} files in {total_elapsed:.1f} minutes "
              f"({overall_rate:.1f} files/min average)")
    
    manifest_copy = manifest_copy.reset_index(drop=True)
    
    return manifest_copy