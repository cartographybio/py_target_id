library(data.table)

files <- list.files(pattern = "Cyno|Human|Mouse")

# Just read once and write
lapply(files, function(file) {
  df <- data.frame(readRDS(file))
  fwrite(df, gsub(".rds", ".csv", file), sep = ",")
})

import pandas as pd
from pathlib import Path

files = [
    "Transcript_Topology_Summary.Mouse_vM32.csv",
    "Transcript_Topology_Summary.Cyno_v6.0.csv",
    "Transcript_Topology_Summary.Human_v29.csv",
    "Mouse_ECD_8mer_DT.csv",
    "Human_ECD_8mer_DT.csv",
    "Cyno_ECD_8mer_DT.csv"
]

for file in files:
    print(f"Converting {file}...")
    df = pd.read_csv(file)
    parquet_file = file.replace('.csv', '.parquet')
    df.to_parquet(parquet_file, compression='zstd', index=False)
    print(f"  → {parquet_file} ({len(df):,} rows)")