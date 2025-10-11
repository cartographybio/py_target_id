library(recount3)
library(HDF5Array)
library(TargetFinder)

#Valid Genes
all_genes <- valid_genes()

#NCPU
NCPU <- 32

# Set temporary directory to avoid default tmp space issues
tmpdir <- "TEMP"
unixtools::set.tempdir(tmpdir)

# Retrieve available human GTEx projects
projects <- recount3::available_projects(organism = "human")
gtex <- projects[projects[,3] == "gtex",]
gtex <- gtex[gtex[,1] != "STUDY_NA",]

# Parallel processing to download and prepare GTEx gene expression data
# THIS WILL BE THE SLOWEST STEP
dir.create("recount3")
dir.create("recount3/gtex")
rseL <- parallel::mclapply(seq_len(nrow(gtex)), function(x){

	message(x, " of ", nrow(gtex))
	rse <- create_rse(gtex[x,,drop=FALSE], type = "gene", annotation = "gencode_v29")
	names(assays(rse)) <- "counts"
	assays(rse)$TPM <- recount::getTPM(rse)
	assays(rse)$counts <- NULL  # Remove raw counts
	rse <- rse[paste0(seqnames(rse)) %in% paste0("chr", c(1:22, "X", "Y")),]
	rse <- rse[!grepl("PAR_Y", rownames(rse)),]
	rse <- rse[!duplicated(rowData(rse)$gene_name),]
	rownames(rse) <- rowData(rse)$gene_name

	name <- gsub("[ ()-]", "_", colData(rse)$gtex.smtsd)
	clean_names <- gsub("_+", "_", name)  # Remove multiple consecutive underscores
	clean_names <- gsub("^_|_$", "", clean_names)  # Remove leading/trailing underscores

	colnames(rse) <- paste0(gtex[x,"project",drop=FALSE][[1]], "#", clean_names, "#", colnames(rse))

	rse <- as(assay(rse), "dgCMatrix")
	rse <- subset_matrix(rse, subsetRows = all_genes)
	rse@x <- round(rse@x, 3)
	rse <- as(rse, "dgCMatrix")

	rse

}, mc.cores = NCPU, mc.preschedule = FALSE)

rse_all <- Reduce("cbind", rseL)
rse_all <- rse_all[,!grepl("Cells_Leukemia_cell_line_CML|Cells_EBV_transformed_lymphocytes|Cells_Cultured_fibroblasts", colnames(rse_all))]

library(Matrix)
writeMM(rse_all, "recount3/gtex.bulk_rna.20251010.mtx")
writeLines(rownames(rse_all), "recount3/gtex.bulk_rna.20251010.genes.txt")
writeLines(colnames(rse_all), "recount3/gtex.bulk_rna.20251010.barcodes.txt")

#colnames(rse_all) <- paste0(rep(gtex[,"project"], {lapply(rseL, ncol) %>% unlist}), "#", colnames(rse_all))
#saveRDS(rse_all, "recount3/gtex.20250825.rds", compress = FALSE)

rm(rseL)
rm(rse_all)

# Load tumor TCGA dataset
projects <- available_projects(organism = "human")
tcga <- projects[projects[,3]=="tcga",]

tcgaL <- parallel::mclapply(seq_len(nrow(tcga)), function(x){

	message(x, " of ", nrow(tcga))
	rse <- create_rse(tcga[x,,drop=FALSE], type = "gene", annotation = "gencode_v29")
	names(assays(rse)) <- "counts"
	assays(rse)$TPM <- recount::getTPM(rse)
	assays(rse)$counts <- NULL  # Remove raw counts
	rse <- rse[paste0(seqnames(rse)) %in% paste0("chr", c(1:22, "X", "Y")),]
	rse <- rse[!grepl("PAR_Y", rownames(rse)),]
	rse <- rse[!duplicated(rowData(rse)$gene_name),]
	rownames(rse) <- rowData(rse)$gene_name
	colnames(rse) <- paste0(colData(rse)$tcga.xml_bcr_patient_barcode, "#", colnames(rse))
	
	print(table(paste0(colData(rse)$tcga.gdc_cases.samples.sample_type)))

	stopifnot(all(paste0(colData(rse)$tcga.gdc_cases.samples.sample_type) != "NA"))

	#return(colData(rse)$tcga.gdc_cases.samples.sample_type)

	rse <- rse[,colData(rse)$tcga.gdc_cases.samples.sample_type != "Solid Tissue Normal"]
	rse <- as(assay(rse), "dgCMatrix")
	rse <- subset_matrix(rse, subsetRows = all_genes)
	rse@x <- round(rse@x, 3)
	rse <- as(rse, "dgCMatrix")

	rse

}, mc.cores = NCPU, mc.preschedule = FALSE)

tcga_all <- Reduce("cbind", tcgaL)
colnames(tcga_all) <- paste0(rep(tcga[,"project"], {lapply(tcgaL, ncol) %>% unlist}), "#", colnames(tcga_all))

library(Matrix)
writeMM(tcga_all, "recount3/tcga.bulk_rna.20251010.mtx")
writeLines(rownames(tcga_all), "recount3/tcga.bulk_rna.20251010.genes.txt")
writeLines(colnames(tcga_all), "recount3/tcga.bulk_rna.20251010.barcodes.txt")

#PYTHONNNN

# Python
import scanpy as sc
import pandas as pd

adata = sc.read_mtx("recount3/gtex.bulk_rna.20251010.mtx").T
adata.var_names = pd.read_csv("recount3/gtex.bulk_rna.20251010.genes.txt", header=None)[0]
adata.obs_names = pd.read_csv("recount3/gtex.bulk_rna.20251010.barcodes.txt", header=None)[0]
adata.write("recount3/gtex.bulk_rna.20251010.h5ad")

adata = sc.read_mtx("recount3/tcga.bulk_rna.20251010.mtx").T
adata.var_names = pd.read_csv("recount3/tcga.bulk_rna.20251010.genes.txt", header=None)[0]
adata.obs_names = pd.read_csv("recount3/tcga.bulk_rna.20251010.barcodes.txt", header=None)[0]
adata.write("recount3/tcga.bulk_rna.20251010.h5ad")

gsutil -m cp recount3/tcga.bulk_rna.20251010.h5ad gs://cartography_target_id_package/Other_Input/TCGA/tcga.bulk_rna.20251010.h5ad
gsutil -m cp recount3/gtex.bulk_rna.20251010.h5ad gs://cartography_target_id_package/Other_Input/GTEX/gtex.bulk_rna.20251010.h5ad


