import subprocess; import os; subprocess.run(["bash", os.path.expanduser("~/update_tid.sh")]) #Update Quick
import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
from importlib.resources import files
from py_target_id import utils

inds = ["AML", "KIRC", "CRC", "LUAD.Magellan", "TNBC.Magellan", "PDAC_FFPE"]
base_dir = os.getcwd()

for ind in inds:
    ind_path = os.path.join(base_dir, ind)
    
    # Check if the folder exists
    if not os.path.isdir(ind_path):
        print(f"Warning: {ind_path} does not exist, skipping...")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing: {ind}")
    print(f"{'='*60}")
    
    try:
        # Change to the indication directory
        os.chdir(ind_path)
        
        # Load Cohort
        IND = os.path.basename(os.getcwd())
        print(f"Loading cohort for {IND}...")
        manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata = utils.load_cohort(IND)
        
        # Run Single Target Workflow
        single_path = IND + '.Single.Results.20251029.parquet'
        if os.path.exists(single_path):
            print(f"⊘ Single target results already exist, skipping...")
        else:
            print(f"Running single target workflow for {IND}...")
            single = tid.run.target_id_v1(malig_med_adata, ref_med_adata)
            single.to_parquet(single_path, engine='pyarrow', compression=None)
            print(f"✓ Single target results saved to {single_path}")
        
        # Ready Up Multi Target Workflow
        multi_path = IND + '.Multi.Results.20251029.parquet'
        if os.path.exists(multi_path):
            print(f"⊘ Multi-target results already exist, skipping...")
        else:
            print(f"Preparing multi-target workflow for {IND}...")
            surface = tid.utils.surface_genes()
            gene_pairs = tid.utils.create_gene_pairs(surface, surface)
            
            # Run Multi Target Workflow
            print(f"Running multi-target workflow for {IND}...")
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
        
        print(f"✓ Successfully completed {IND}")
        
    except Exception as e:
        print(f"✗ Error processing {ind}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Change back to base directory for next iteration
        os.chdir(base_dir)

print(f"\n{'='*60}")
print("All indications processed!")
print(f"{'='*60}")