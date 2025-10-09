"""
Utility functions for target_id package.
"""
# Define what gets exported with "from google import *"
__all__ = ['google_copy', 'set_google_copy_version', 'list_gcs_versions', 'select_version', 'download_gcs_file']

import subprocess
import os
from typing import List, Optional

# Global variable to store the google_copy version
GOOGLE_COPY_VERSION = None

def set_google_copy_version(version):
    """Set the global google_copy version."""
    global GOOGLE_COPY_VERSION
    GOOGLE_COPY_VERSION = version

def google_copy(version=None):
    """
    Get the appropriate Google Cloud copy command.
    
    Args:
        version (str, optional): Version to use. If None, uses global setting or defaults to "gcloud"
    
    Returns:
        str: The command string to use for Google Cloud operations
    """
    if version is None:
        version = GOOGLE_COPY_VERSION
    
    if version is None:
        version = "gcloud"
    
    if version == "gcloud":
        return "gcloud storage"
    else:
        return "gsutil"

def list_gcs_versions(gcs_path: str) -> List[str]:
    """
    List date-formatted versions (YYYYMMDD) in a GCS path.
    
    Args:
        gcs_path: GCS path to list (e.g., 'gs://bucket/path/')
    
    Returns:
        List of version strings (date format YYYYMMDD)
    """
    cmd = f"{google_copy()} ls {gcs_path}"
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list GCS path: {gcs_path}\n{result.stderr}")
    
    # Extract only date-formatted folders (8 digits)
    versions = [
        os.path.basename(p.rstrip('/')) 
        for p in result.stdout.strip().split('\n')
        if p.endswith('/') and os.path.basename(p.rstrip('/')).isdigit() 
        and len(os.path.basename(p.rstrip('/'))) == 8
    ]
    
    return sorted(versions)

def select_version(available_versions: List[str], requested_version: str = "latest", verbose: bool = True) -> str:
    """
    Select a version from available versions.
    
    Args:
        available_versions: List of available version strings
        requested_version: Either "latest" or a specific version string
        verbose: If True, print selected version
    
    Returns:
        Selected version string
    """
    if not available_versions:
        raise ValueError("No versions available")
    
    if requested_version == "latest":
        selected = available_versions[-1]  # Already sorted
        if verbose:
            print(f"Using latest version: {selected}")
    else:
        if requested_version not in available_versions:
            raise ValueError(
                f"Version {requested_version} not found. "
                f"Available versions: {available_versions}"
            )
        selected = requested_version
        if verbose:
            print(f"Using specified version: {selected}")
    
    return selected

def download_gcs_file(gcs_path: str, local_path: str, overwrite: bool = False, verbose: bool = True) -> str:
    """
    Download a file from GCS to local filesystem.
    
    Args:
        gcs_path: Full GCS path to file
        local_path: Local path where file should be saved
        overwrite: If True, overwrite existing file
        verbose: If True, print progress messages
    
    Returns:
        Local path to downloaded file
    """
    # Create local directory if needed
    local_dir = os.path.dirname(local_path)
    if local_dir:  # Only create if there's a directory component
        os.makedirs(local_dir, exist_ok=True)
    
    # Check if file exists
    if os.path.exists(local_path) and not overwrite:
        if verbose:
            print(f"✓ File already exists: {local_path}")
            print("  Set overwrite=True to re-download")
        return local_path
    
    # Download file
    if verbose:
        print(f"Downloading from: {gcs_path}")
        print(f"Saving to: {local_path}")
    
    cmd = f"{google_copy()} cp {gcs_path} {local_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        if verbose:
            print("✓ Download complete!")
        return local_path
    else:
        if verbose:
            print(f"✗ Download failed!")
            print(f"Error: {result.stderr}")
        raise RuntimeError(f"Failed to download file from {gcs_path}")
