"""
Utility functions for target_id package.
"""

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
        return "gcloud alpha storage"
    else:
        return "gsutil -m"