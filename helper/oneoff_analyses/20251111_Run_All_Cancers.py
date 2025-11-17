import subprocess
import os
import gc
import psutil
import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import sys
from importlib.resources import files
from py_target_id import utils

# Update Quick
#subprocess.run(["bash", os.path.expanduser("~/update_tid.sh")])

# Memory monitoring utilities
def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)

def print_memory_status(label=""):
    """Print current memory usage"""
    mem_gb = get_memory_usage()
    print(f"[Memory {label}] {mem_gb:.2f} GB")

def aggressive_cleanup():
    """Aggressive garbage collection and memory cleanup"""
    gc.collect()
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass

date = "20251111"
inds = ["AML", "KIRC", "CRC", "LUAD.Magellan", "TNBC.Magellan"]
base_dir = os.getcwd()

def cleanup_adata(*adata_objects):
    """Explicitly free AnnData objects"""
    for adata in adata_objects:
        if adata is not None:
            if hasattr(adata, 'X') and hasattr(adata.X, 'data'):
                del adata.X.data
            del adata
    gc.collect()

# Pre-cache static data once (outside loop)
print("\n" + "="*60)
print("Pre-loading static reference data...")
print("="*60)
print_memory_status("Start")

surface_genes_cached = tid.utils.surface_genes()
print(f"✓ Cached {len(surface_genes_cached)} surface genes")
print_memory_status("After surface genes cache")

for ind_idx, ind in enumerate(inds, 1):
    ind_path = os.path.join(base_dir, ind)
    
    if not os.path.isdir(ind_path):
        print(f"Warning: {ind_path} does not exist, skipping...")
        continue
    
    print(f"\n{'='*60}")
    print(f"[{ind_idx}/{len(inds)}] Processing: {ind}")
    print(f"{'='*60}")
    print_memory_status("Iteration start")
    
    try:
        os.chdir(ind_path)
        IND = os.path.basename(os.getcwd())
        
        # Load Cohort
        print(f"Loading cohort for {IND}...")
        manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata = utils.load_cohort(IND)
        print_memory_status("After load_cohort")
        
        # ===== SINGLE TARGET WORKFLOW =====
        single_path = IND + '.Single.Results.' + date + '.parquet'
        if os.path.exists(single_path):
            print(f"⊘ Single target results already exist, skipping...")
        else:
            print(f"Running single target workflow for {IND}...")
            print_memory_status("Before single target")
            
            single = tid.run.target_id_v1(malig_med_adata, ref_med_adata)
            single.to_parquet(single_path, engine='pyarrow', compression=None)
            print(f"✓ Single target results saved to {single_path}")
            
            # Cleanup single results
            del single
            aggressive_cleanup()
            print_memory_status("After single target cleanup")
        
        # ===== MULTI TARGET WORKFLOW =====
        multi_path = IND + '.Multi.Results.' + date + '.parquet'
        if os.path.exists(multi_path):
            print(f"⊘ Multi-target results already exist, skipping...")
        else:
            print(f"Preparing multi-target workflow for {IND}...")
            print_memory_status("Before gene pair creation")
            
            # Create gene pairs (memory-intensive)
            gene_pairs = tid.utils.create_gene_pairs(
                surface_genes_cached, 
                surface_genes_cached
            )
            print(f"✓ Created {len(gene_pairs):,} gene pairs")
            print_memory_status("After gene pair creation")
            
            # Run Multi Target Workflow
            print(f"Running multi-target workflow for {IND}...")
            print_memory_status("Before multi target")
            
            multi = tid.run.target_id_multi_v1(
                malig_adata=malig_adata,
                ref_adata=ref_adata,
                gene_pairs=gene_pairs,
                ref_med_adata=ref_med_adata,
                malig_med_adata=malig_med_adata,
                batch_size=20000,
                use_fp16=True
            )
            
            multi.to_parquet(multi_path, engine='pyarrow', compression=None)
            print(f"✓ Multi-target results saved to {multi_path}")
            print_memory_status("After multi target save")
            
            # Cleanup multi results and gene pairs
            del multi, gene_pairs
            aggressive_cleanup()
            print_memory_status("After multi target cleanup")
        
        print(f"✓ Successfully completed {IND}")
        
    except Exception as e:
        print(f"✗ Error processing {ind}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Aggressive cleanup after each indication
        try:
            cleanup_adata(
                manifest if 'manifest' in locals() else None,
                malig_adata if 'malig_adata' in locals() else None,
                malig_med_adata if 'malig_med_adata' in locals() else None,
                ref_adata if 'ref_adata' in locals() else None,
                ref_med_adata if 'ref_med_adata' in locals() else None
            )
        except:
            pass
        
        # Force cleanup
        aggressive_cleanup()
        print_memory_status("After iteration cleanup")
        
        # Return to base directory
        os.chdir(base_dir)

print(f"\n{'='*60}")
print("All indications processed!")
print(f"{'='*60}")
print_memory_status("Final")