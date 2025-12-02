import py_target_id as tid
import anndata as ad
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csc_matrix
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
from importlib.resources import files
from py_target_id import utils
from tqdm import tqdm
import re

#Load Cohort
IND = os.path.basename(os.getcwd())

#Load
manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata = utils.load_cohort(IND)


def load_h5_to_andata(h5_path: str) -> ad.AnnData:
    """
    Load HDF5 file with CSC sparse matrix format into AnnData object.
    
    Parameters
    ----------
    h5_path : str
        Path to HDF5 file containing 'assays/counts.counts' sparse matrix
        and 'coldata' cell metadata.
    
    Returns
    -------
    ad.AnnData
        AnnData object with shape (n_cells, n_genes).
    """
    with h5py.File(h5_path, 'r') as f:
        counts_group = f['assays/counts.counts']
        
        data = counts_group['data'][:]
        indices = counts_group['indices'][:]
        indptr = counts_group['indptr'][:]
        shape = tuple(counts_group['shape'][:])
        
        X_csc = csc_matrix((data, indices, indptr), shape=shape)
        X = X_csc.T.tocsr()
        
        barcodes = [b.decode('utf-8') if isinstance(b, bytes) else b 
                    for b in counts_group['barcodes'][:]]
        
        genes = [g.decode('utf-8') if isinstance(g, bytes) else g 
                 for g in counts_group['genes'][:]]
        
        coldata = {}
        for key in f['coldata'].keys():
            dataset = f['coldata'][key][:]
            if dataset.dtype.kind == 'S':
                coldata[key] = [v.decode('utf-8') if isinstance(v, bytes) else v 
                               for v in dataset]
            else:
                coldata[key] = dataset
        
        obs = pd.DataFrame(coldata, index=barcodes)
    
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = genes
    
    return adata


# Find all H5 files in cell_line directory
cell_line_dir = Path("cell_line")
h5_files = sorted(cell_line_dir.glob("*.h5"))

print(f"Found {len(h5_files)} H5 files:")
for f in h5_files:
    print(f"  {f.name}")

# Load and normalize each file
adatas = []
for h5_file in h5_files:
    print(f"\nLoading {h5_file.name}...")
    adata = load_h5_to_andata(str(h5_file))
    
    # Normalize
    adata.X = tid.utils.cp10k(adata.X)
    
    # Add sample ID
    #adata.obs['sample'] = h5_file.stem.replace(".C009.ArchRCells|.C009.DecoX35.ArchRCells", "").strip(".")
    adata.obs['sample'] = re.sub(r"\.C009\.(?:DecoX35\.)?ArchRCells", "", h5_file.stem)
    print(f"  Shape: {adata.shape}")
    adatas.append(adata)

# Concatenate all samples
cell_adata = ad.concat(adatas, axis=0, label="sample", keys=[a.obs['sample'].iloc[0] for a in adatas])

#Pull Data
df_malig = pd.DataFrame({
	"type" : "Patient",
	"id" : malig_adata.obs["Patient"].str.replace(r"_FFPE|Breast_", "", regex=True).values,
	"log2exp": np.log2(malig_adata[:, "LY6K"].to_memory().X.toarray().flatten() + 1)
})

df_cell = pd.DataFrame({
	"type" : "CellLine",
	"id" : cell_adata.obs["sample"].values,
	"log2exp": np.log2(cell_adata[:, "LY6K"].X.toarray().flatten() + 1)
})

df_cell = df_cell[~df_cell["id"].str.contains("PCDHB2|HeLa_Ly6K_KO")]
df_cell["id"] = df_cell["id"].cat.add_categories(["HeLa"])
df_cell.loc[df_cell["id"] == "HeLa_Ly6K_WT", "id"] = "HeLa"

df_all = pd.concat([df_malig, df_cell], axis = 0).reset_index()

tid.plot.pd2r("df", df_all)

r('''
ids_o <- split(df$log2exp, df$id) %>% lapply(median) %>% unlist %>% sort(decreasing=TRUE) %>% names
df$id <- factor(df$id, levels = unique(ids_o))

# Get colors for x-axis labels based on type
axis_colors <- df %>% distinct(id, type) %>% arrange(match(id, ids_o)) %>% pull(type) %>% 
  {ifelse(. == "CellLine", "firebrick3", "black")}

p <- ggplot(df, aes(id, log2exp, fill = type)) +
	geom_jitter(height = 0, pch = 21, width = 0.2) +
	geom_boxplot(outlier.shape = NA, color = "black", alpha = 0.8) +
	theme_jg(xText90=TRUE) +
	theme(axis.text.x = element_text(color = axis_colors)) +
	scale_fill_manual(values = c("CellLine"="firebrick3", "Patient"="#2E95D2")) +
	ylab("Log2 (CP10k + 1)") +
	ggtitle("LY6K Expression")

pdf("LY6K-Cell-Line-Patient-Exp.pdf", width = 11, height = 6)
print(p)
dev.off()
''')



#Pull Data
df_malig = pd.DataFrame({
	"type" : "Patient",
	"id" : malig_adata.obs["Patient"].str.replace(r"_FFPE|Breast_", "", regex=True).values,
	"log2exp": np.log2(malig_adata[:, "PCDHB2"].to_memory().X.toarray().flatten() + 1)
})

df_cell = pd.DataFrame({
	"type" : "CellLine",
	"id" : cell_adata.obs["sample"].values,
	"log2exp": np.log2(cell_adata[:, "PCDHB2"].X.toarray().flatten() + 1)
})

df_cell = df_cell[~df_cell["id"].str.contains("Ly6K")]
df_cell["id"] = df_cell["id"].cat.add_categories(["HeLa"])
df_cell.loc[df_cell["id"] == "HeLa_Ly6K_WT", "id"] = "HeLa"

df_all = pd.concat([df_malig, df_cell], axis = 0).reset_index()

tid.plot.pd2r("df", df_all)

r('''
ids_o <- split(df$log2exp, df$id) %>% lapply(median) %>% unlist %>% sort(decreasing=TRUE) %>% names
df$id <- factor(df$id, levels = unique(ids_o))

# Get colors for x-axis labels based on type
axis_colors <- df %>% distinct(id, type) %>% arrange(match(id, ids_o)) %>% pull(type) %>% 
  {ifelse(. == "CellLine", "firebrick3", "black")}

p <- ggplot(df, aes(id, log2exp, fill = type)) +
	geom_jitter(height = 0, pch = 21, width = 0.2) +
	geom_boxplot(outlier.shape = NA, color = "black", alpha = 0.8) +
	theme_jg(xText90=TRUE) +
	theme(axis.text.x = element_text(color = axis_colors)) +
	scale_fill_manual(values = c("CellLine"="firebrick3", "Patient"="#2E95D2")) +
	ylab("Log2 (CP10k + 1)") +
	ggtitle("PCDHB2 Expression")

pdf("PCDHB2-Cell-Line-Patient-Exp.pdf", width = 11, height = 6)
print(p)
dev.off()
''')



#Pull Data
df_malig = pd.DataFrame({
	"type" : "Patient",
	"id" : malig_adata.obs["Patient"].str.replace(r"_FFPE|Breast_", "", regex=True).values,
	"log2exp": np.log2(malig_adata[:, "PODXL2"].to_memory().X.toarray().flatten() + 1)
})

df_cell = pd.DataFrame({
	"type" : "CellLine",
	"id" : cell_adata.obs["sample"].values,
	"log2exp": np.log2(cell_adata[:, "PODXL2"].X.toarray().flatten() + 1)
})

#df_cell = df_cell[~df_cell["id"].str.contains("Ly6K")]

df_all = pd.concat([df_malig, df_cell], axis = 0).reset_index()

tid.plot.pd2r("df", df_all)

r('''
ids_o <- split(df$log2exp, df$id) %>% lapply(median) %>% unlist %>% sort(decreasing=TRUE) %>% names
df$id <- factor(df$id, levels = unique(ids_o))

# Get colors for x-axis labels based on type
axis_colors <- df %>% distinct(id, type) %>% arrange(match(id, ids_o)) %>% pull(type) %>% 
  {ifelse(. == "CellLine", "firebrick3", "black")}

p <- ggplot(df, aes(id, log2exp, fill = type)) +
	geom_jitter(height = 0, pch = 21, width = 0.2) +
	geom_boxplot(outlier.shape = NA, color = "black", alpha = 0.8) +
	theme_jg(xText90=TRUE) +
	theme(axis.text.x = element_text(color = axis_colors)) +
	scale_fill_manual(values = c("CellLine"="firebrick3", "Patient"="#2E95D2")) +
	ylab("Log2 (CP10k + 1)") +
	ggtitle("PODXL2 Expression")

pdf("PODXL2-Cell-Line-Patient-Exp.pdf", width = 11, height = 6)
print(p)
dev.off()
''')




#Pull Data
df_malig = pd.DataFrame({
	"type" : "Patient",
	"id" : malig_adata.obs["Patient"].str.replace(r"_FFPE|Breast_", "", regex=True).values,
	"log2exp": np.log2(np.min(malig_adata[:, ["NECTIN4", "PODXL2"]].to_memory().X.toarray(), axis=1) + 1)
})

df_cell = pd.DataFrame({
	"type" : "CellLine",
	"id" : cell_adata.obs["sample"].values,
	"log2exp": np.log2(np.min(cell_adata[:, ["NECTIN4", "PODXL2"]].X.toarray(), axis=1) + 1)
})

#df_cell = df_cell[~df_cell["id"].str.contains("Ly6K")]

df_all = pd.concat([df_malig, df_cell], axis = 0).reset_index()

tid.plot.pd2r("df", df_all)

r('''
ids_o <- split(df$log2exp, df$id) %>% lapply(median) %>% unlist %>% sort(decreasing=TRUE) %>% names
df$id <- factor(df$id, levels = unique(ids_o))

# Get colors for x-axis labels based on type
axis_colors <- df %>% distinct(id, type) %>% arrange(match(id, ids_o)) %>% pull(type) %>% 
  {ifelse(. == "CellLine", "firebrick3", "black")}

p <- ggplot(df, aes(id, log2exp, fill = type)) +
	geom_jitter(height = 0, pch = 21, width = 0.2) +
	geom_boxplot(outlier.shape = NA, color = "black", alpha = 0.8) +
	theme_jg(xText90=TRUE) +
	theme(axis.text.x = element_text(color = axis_colors)) +
	scale_fill_manual(values = c("CellLine"="firebrick3", "Patient"="#2E95D2")) +
	ylab("Log2 (CP10k + 1)") +
	ggtitle("NECTIN4_PODXL2 Expression")

pdf("NECTIN4_PODXL2-Cell-Line-Patient-Exp.pdf", width = 11, height = 6)
print(p)
dev.off()
''')




