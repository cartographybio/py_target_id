"""
Update
"""
__all__ = ['update']

import subprocess
import os

def update():
    """Update TID by running ~/update_tid.sh"""
    subprocess.run(["bash", os.path.expanduser("~/update_tid.sh")])