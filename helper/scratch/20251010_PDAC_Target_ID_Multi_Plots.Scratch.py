pip uninstall py_target_id -y

import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os

#Manifest
manifest = tid.utils.load_manifest()
pdac = manifest["Indication"] == "PDAC_FFPE"
manifest = manifest[pdac].reset_index(drop=True)
manifest = tid.utils.download_manifest(manifest=manifest, overwrite = False)

#Malig Data
malig_med_adata = tid.utils.get_malig_med_adata(manifest)
malig_adata = tid.utils.get_malig_ar_adata(manifest)

#Reference Data
df_off = tid.utils.get_ref_ffpe_off_target()
ref_med_adata = tid.utils.get_ref_lv4_ffpe_med_adata()
ref_adata = tid.utils.get_ref_lv4_ffpe_ar_adata()

#Add Weights to Ref Data
ref_med_adata = tid.utils.add_ref_weights(ref_med_adata, df_off, "Off_Target.V0")
ref_adata = tid.utils.add_ref_weights(ref_adata, df_off, "Off_Target.V0")

# #Run Single Target Workflow
#single = tid.run.target_id_v1(malig_med_adata, ref_med_adata)

# #Ready Up Multi Target Workflow
# surface = tid.utils.surface_genes()

# #Pairs
# gene_pairs = tid.utils.create_gene_pairs(surface, surface)

# #Run Multi Target Workflow
# multi = tid.run.target_id_multi_v1(
#     malig_adata=malig_adata,
#     ha_adata=ref_adata,
#     gene_pairs=gene_pairs,
#     malig_med_adata=malig_med_adata,
#     batch_size=50000,
#     use_fp16=True
# )

# #Save
# single.to_parquet('20251010.PDAC.Single.Results.parquet', engine='pyarrow', compression=None)
# multi.to_parquet('20251010.PDAC.Multi.Results.parquet', engine='pyarrow', compression=None)

#Load
single = pd.read_parquet('20251010.PDAC.Single.Results.parquet', engine='pyarrow')
multi = pd.read_parquet('20251010.PDAC.Multi.Results.parquet', engine='pyarrow')

#Plot Time!
indication = "PDAC"
multis = ["ADAM8_ITGB6", "ADAM8_MSLN", "C4BPB_CDH3"]
genes = sorted(set(gene for multi in multis for gene in multi.split('_')))

tid.plot.multi.biaxial_summary(
	multis=multis,
	malig_adata=malig_adata,
	malig_med_adata=malig_med_adata,
	ref_adata=ref_adata,
	tcga_adata=tid.utils.get_tcga_adata()[tid.utils.get_tcga_adata().obs["TCGA"]=="PAAD",:],
    out_dir = indication + "/multi/biaxial_summary"
)

tid.plot.multi.dot_plot(
	multis = multis, 
	ref_adata = ref_adata, 
	ref_med_adata = ref_med_adata,
    out_dir = indication + "/multi/dot_plot"
)

tid.plot.multi.axial_1_2_12(
    multis=multis,
    malig_adata=malig_adata,
    malig_med_adata=malig_med_adata,
    ref_adata=ref_adata,
    out_dir = indication + "/multi/axial_1_2_12"
)

tid.plot.multi.axial_vs_known(
    multis=multis,
    malig_adata=malig_adata,
    malig_med_adata=malig_med_adata,
    ref_adata=ref_adata,
    out_dir = indication + "/multi/axial_vs_known"
)

tid.plot.multi.biaxial_tcga_gtex(
    multis=multis,
    main="PAAD",
    out_dir = indication + "/multi/biaxial_tcga_gtex"
)


tid.plot.multi.umap_1_2_12(
	multis = multis,
	malig_adata = malig_adata,
	manifest = manifest,
    out_dir = indication + "/multi/umap_1_2_12"
)


tid.plot.multi.radar_1_2_12(
    multis=multis,
    df_single=single,
    df_multi=multi,
    out_dir = indication + "/multi/radar_1_2_12"
)

tid.plot.multi.tq_vs_pp_1_2_12(
    multis=multis,
    df_single=single,
    df_multi=multi,
    known=["DLL3", "MET", "EGFR", "TACSTD2", "CEACAM5", "ERBB3", "MSLN"],
    out_dir = indication + "/multi/tq_vs_pp_1_2_12"
)

#Top TX By N_AA
df_topology = tid.plot.get_human_topology(genes=genes)
df_topology = df_topology.sort_values('N_AA', ascending=False)
df_topology = df_topology.drop_duplicates(subset='gene_name', keep='first')

#Homology
results = tid.plot.plot_homology_blast(df=df_topology, out_dir = indication + "/multi/homology/blast")
tid.plot.plot_homology_kmer(tx=df_topology.transcript_id.values, out_dir = indication + "/multi/homology/kmer")




















get_blast_db(overwrite=False)  # Set True to force re-download
df_topology = get_human_topology(genes=['PRLR'])
df_topology = df_topology[0:1]

# Then run the homology analysis
results = plot_homology_blast(
    df=df_topology,
    output_dir="homology_plots",
    verbose=True
)
plot_topology_kmer(tx=['ENST00000618457.4'])


import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# Activate automatic conversion
pandas2ri.activate()

# Import data.table in R
base = importr('base')
dt = importr('data.table')

# Load your R object (if it's saved as .rds)
r_data = base.readRDS("df_uniprot_names.rds")

# Convert to pandas
df = pandas2ri.rpy2py(r_data)

# Reset index to remove the numbered index column
df = df.reset_index(drop=True)

# Convert object columns to string to be safe
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)

# Save as Parquet
df.to_parquet("df_uniprot_names.parquet", compression="zstd", index=False)



library(data.table)
a <- readRDS("Cyno_ECD_8mer_DT.rds")
write_parquet(a, "Cyno_ECD_8mer_DT.parquet")



slides = Slides()
slides.start("custom.pptx")
slides.add_slide_title("My Title", "My Subtitle")
# Add slides with different ratios
slides.add_slide_text_plot(
    ratio="50:50", 
    plot="multi/multi_axial/ADAM8_ITGB6.png", 
    text="This is a 50:50 layout.\n\nThe text takes up half the slide width, and the plot takes up the other half."
)

slides.add_slide_text_plot(
    ratio="25:75", 
    plot="multi/multi_axial/ADAM8_ITGB6.png", 
    text="This is a 25:75 layout.\n\nText is narrower here, giving more space to the plot."
)

slides.add_slide_section("Section Title", "Optional subtitle text")

slides.add_slide_plot_subtitle(
    plot="multi/multi_axial/ADAM8_ITGB6.png", 
    title="My Title",
    subtitle="My subtitle text here"
)

slides.close()

















 














def biaxial_plot_v4(
    multis: List[str],
    malig_adata,  # Malignant AnnData (cells × genes)
    malig_med_adata,  # Malignant median AnnData (patients × genes) with positivity layer
    ref_adata,  # Healthy atlas AnnData
    gtex_adata=None,  # GTEx data
    tcga_adata=None,  # TCGA data
    out_dir: str = "multi/multi_biaxial_v4",
    show: int = 15,
    width: float = 28,
    height: float = 8,
    dpi: int = 600,
    titles: Optional[List[str]] = None
):









tid.infra.read_h5(manifest.Local_h5map[0])


import h5py

def h5ls(filename):
    """Python equivalent of R's h5ls() with more details"""
    def print_attrs(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name:50s} Dataset {str(obj.shape):20s} {obj.dtype}")
        else:
            print(f"{name:50s} Group")
    
    with h5py.File(filename, 'r') as f:
        f.visititems(print_attrs)

# Usage
h5ls(manifest.Local_h5map[0])













plot_multi_dot_plot_parallel(
    multis=["ADAM8_ITGB6", "ADAM8_MSLN", "C4BPB_CDH3"],
    ref_adata=ref_adata,
    ref_med_adata=ref_med_adata,
    out_dir="multi_plots",
    n_jobs=8  # or -1 for all cores
)


def plot_multi_dot_plot(
    multis: list,
    ref_adata,  # AnnData or VirtualAnnData: cells × genes with obs['CellType']
    ref_med_adata,  # AnnData with metadata
    out: str = "plot.pdf",
    max_log2: float = 3.0,
    width: float = 24,
    height: float = 16,
    dpi: int = 600
) -> None:


#Lets Try TCGA GTEX
tcga = get_tcga_h5()
gtex = get_gtex_h5()























