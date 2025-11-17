
# Memory-efficient loading and processing with streaming to disk

import pandas as pd
import numpy as np
from pathlib import Path
import py_target_id as tid

# Define all data sources
all_indications = {
    # Single-target
    "AML_Single": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/AML/AML.Single.Results.20251029.parquet",
    "CRC_Single": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/CRC/CRC.Single.Results.20251029.parquet",
    "KIRC_Single": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/KIRC/KIRC.Single.Results.20251029.parquet",
    "LUAD_Single": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/LUAD.Magellan/LUAD.Magellan.Single.Results.20251029.parquet",
    "TNBC_Single": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/TNBC.Magellan/TNBC.Magellan.Single.Results.20251029.parquet",
    "PDAC_Single": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/PDAC_FFPE/PDAC_FFPE.Single.Results.20251029.parquet",
    # Multi-target
    "CRC_Multi": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/CRC/CRC.Multi.Results.20251029.parquet",
    "LUAD_Multi": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/LUAD.Magellan/LUAD.Magellan.Multi.Results.20251029.parquet",
    "TNBC_Multi": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/TNBC.Magellan/TNBC.Magellan.Multi.Results.20251029.parquet",
}

# Load surface genes and hazard scores ONCE
print("Loading reference data...")
surface = tid.utils.surface_genes()
haz_sgl = tid.utils.get_single_risk_scores()
haz_dbl = tid.utils.get_multi_risk_scores()

print(f"Surface genes: {len(surface):,}")
print(f"Single hazard scores: {len(haz_sgl):,}")
print(f"Multi hazard scores: {len(haz_dbl):,}\n")

# =====================================================================
# STREAMING APPROACH: Process and save immediately
# =====================================================================

output_dir = Path("target_quality_v2_01")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "df_combined_streaming.parquet"
row_count = 0
first_write = True

print("Processing files with streaming to disk...")
print("=" * 70)

for indication, file_path in all_indications.items():
    print(f"\nProcessing: {indication}")
    print(f"  Path: {file_path}")
    
    # Determine which hazard scores to use (single vs multi)
    is_multi = "Multi" in indication
    haz = haz_dbl if is_multi else haz_sgl
    
    # Load only this file
    df = pd.read_parquet(file_path)
    initial_rows = len(df)
    
    # Filter to surface genes (SINGLE targets only)
    if not is_multi:
        df = df.loc[df["gene_name"].isin(surface)].copy()
    
    filtered_rows = len(df)
    print(f"  Rows: {initial_rows:,} → {filtered_rows:,} (surface filter)")
    
    # Merge hazard scores
    df = pd.merge(df, haz, on="gene_name", how="left")
    
    # Apply scoring
    df = tid.run.target_quality_v2_01(df)
    
    # Add indication column
    df["indication"] = indication
    
    # Stream to disk (append or create)
    if first_write:
        df.to_parquet(output_path, index=False, compression='snappy')
        first_write = False
        print(f"  ✓ Created: {output_path}")
    else:
        # Append by reading existing, concatenating, and overwriting
        # (For true streaming with pyarrow table, use:)
        df_existing = pd.read_parquet(output_path)
        df_combined_temp = pd.concat([df_existing, df], ignore_index=True)
        df_combined_temp.to_parquet(output_path, index=False, compression='snappy')
        print(f"  ✓ Appended to parquet")
    
    row_count += len(df)
    
    # Explicitly delete to free memory
    del df
    if 'df_existing' in locals():
        del df_existing
    
    file_size = output_path.stat().st_size / (1024**3)
    print(f"  Total rows written: {row_count:,} | File size: {file_size:.2f} GB")

print("\n" + "=" * 70)
print(f"✓ Complete: {row_count:,} rows written")
print(f"  Path: {output_path}")
print(f"  Size: {output_path.stat().st_size / (1024**3):.2f} GB")
print("=" * 70)

# Load final result
print("\nLoading combined dataset...")
df_combined = pd.read_parquet(output_path)
print(f"Shape: {df_combined.shape}")
print(f"Indications: {df_combined['indication'].value_counts().to_dict()}")



# Use Zarr for true streaming appends - much faster than Parquet concat

import pandas as pd
import numpy as np
from pathlib import Path
import zarr
import py_target_id as tid
import gc

all_indications = {
    "AML_Single": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/AML/AML.Single.Results.20251029.parquet",
    "CRC_Single": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/CRC/CRC.Single.Results.20251029.parquet",
    "KIRC_Single": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/KIRC/KIRC.Single.Results.20251029.parquet",
    "LUAD_Single": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/LUAD.Magellan/LUAD.Magellan.Single.Results.20251029.parquet",
    "TNBC_Single": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/TNBC.Magellan/TNBC.Magellan.Single.Results.20251029.parquet",
    "PDAC_Single": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/PDAC_FFPE/PDAC_FFPE.Single.Results.20251029.parquet",
    "CRC_Multi": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/CRC/CRC.Multi.Results.20251029.parquet",
    "LUAD_Multi": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/LUAD.Magellan/LUAD.Magellan.Multi.Results.20251029.parquet",
    "TNBC_Multi": "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/TNBC.Magellan/TNBC.Magellan.Multi.Results.20251029.parquet",
}

print("Loading reference data...")
surface = tid.utils.surface_genes()
haz_sgl = tid.utils.get_single_risk_scores()
haz_dbl = tid.utils.get_multi_risk_scores()

output_dir = Path("target_quality_v2_01")
output_dir.mkdir(parents=True, exist_ok=True)
zarr_path = output_dir / "df_combined.zarr"

print("\nProcessing with Zarr streaming...")
print("=" * 70)

chunk_size = 100000
row_count = 0
zarr_initialized = False

for indication, file_path in all_indications.items():
    print(f"\nProcessing: {indication}")
    
    is_multi = "Multi" in indication
    haz = haz_dbl if is_multi else haz_sgl
    
    df = pd.read_parquet(file_path)
    initial_rows = len(df)
    
    if not is_multi:
        df = df.loc[df["gene_name"].isin(surface)].copy()
    
    df = pd.merge(df, haz, on="gene_name", how="left")
    df["indication"] = indication
    
    print(f"  Rows: {initial_rows:,}")
    
    # Process in chunks
    for chunk_start in range(0, len(df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(df))
        chunk = df.iloc[chunk_start:chunk_end].copy()
        
        chunk = tid.run.target_quality_v2_01(chunk)
        
        # Initialize Zarr on first chunk
        if not zarr_initialized:
            root = zarr.open_group(str(zarr_path), mode='w')
            
            # Create resizable arrays
            for col in chunk.columns:
                if chunk[col].dtype == 'object':
                    dtype = str
                else:
                    dtype = chunk[col].dtype
                
                root.create_dataset(
                    col, 
                    data=chunk[col].values, 
                    chunks=(chunk_size,),
                    dtype=dtype
                )
            
            zarr_initialized = True
            print(f"    ✓ Initialized Zarr with {len(chunk.columns)} columns")
        else:
            # Append to existing arrays
            root = zarr.open_group(str(zarr_path), mode='r+')
            
            for col in chunk.columns:
                arr = root[col]
                arr.append(chunk[col].values)
        
        row_count += len(chunk)
        del chunk
        gc.collect()
        
        if (chunk_start + chunk_size) % (chunk_size * 10) == 0:
            print(f"    ✓ {row_count:,} rows written")
    
    del df
    gc.collect()

print("\n" + "=" * 70)
print(f"✓ COMPLETE: {row_count:,} rows written to Zarr")
print(f"  Path: {zarr_path}")

# Verify by loading back
print("\nVerifying Zarr dataset...")
root = zarr.open_group(str(zarr_path), mode='r')
print(f"Columns: {list(root.keys())}")
print(f"Total rows: {len(root[list(root.keys())[0]])}")

# Convert to parquet if needed
print("\nConverting Zarr to Parquet for easier use...")
df_combined = pd.DataFrame({
    col: root[col][:] for col in root.keys()
})

parquet_path = output_dir / "df_combined_final.parquet"
df_combined.to_parquet(parquet_path, index=False, compression='snappy')
print(f"✓ Saved to Parquet: {parquet_path}")
print(f"  Size: {parquet_path.stat().st_size / (1024**3):.2f} GB")

print(f"\nIndications breakdown:")
for ind, count in df_combined['indication'].value_counts().sort_index().items():
    print(f"  {ind:20s}: {count:12,}")