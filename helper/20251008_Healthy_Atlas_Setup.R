suppressPackageStartupMessages({
	library(HDF5Array)
	library(rhdf5)
	library(dplyr)
	library(SummarizedExperiment)
	library(data.table)
	library(TargetFinder)
	library(yaml)
})

#system("gsutil cp 20240620/Healthy_Atlas.Surface_Gene_Matrix.ArchRCells.h5 gs://cartography_target_id/atlas_version/20240620/")
#system("gsutil cp 20250201/Healthy_Atlas.Surface_Gene_Matrix.ArchRCells.h5 gs://cartography_target_id/atlas_version/20250201/")
#system("gsutil cp 20250225/Healthy_Atlas.Surface_Gene_Matrix.ArchRCells.h5 gs://cartography_target_id/atlas_version/20250225/")

version <- "20250201"

h5atlas <- sprintf("%s/Healthy_Atlas.h5atlas", version)
h5groups <- sprintf("%s/Healthy_Atlas.Custom_Groupings.lv4.csv", version)

#Out Files
se_out <- sprintf("%s/Healthy_Atlas.Lv4.se.rds", version)
ad_out <- sprintf("%s/Healthy_Atlas.Lv4.h5ad", version)
h5_out <- sprintf("%s/Healthy_Atlas.Surface_Gene_Matrix.ArchRCells.v2.h5", version)

#Gene Sets
sgenes <- unique(c(tabs_genes(),surface_genes()))
valid_genes <- unique(valid_genes())

########################################################################################
#1. Make Se Lv4
########################################################################################

#Groups
groups <- unique(h5ls(h5atlas)[,1])
groups <- groups[!grepl("assays|coldata", groups)]
groups <- groups[groups != "/"]

id <- h5read(h5atlas, file.path(groups[1], "coldata"))
if("lv4_consensus" %in% names(id)){
	lv1 <- "lv1_consensus"
	lv2 <- "lv2_consensus"
	lv3 <- "lv3_consensus"
	lv4 <- "lv4_consensus"
}else{
	lv1 <- "lv1_v4_consensus"
	lv2 <- "lv2_v4_consensus"
	lv3 <- "lv3_v4_consensus"
	lv4 <- "lv4_v4_consensus"
}

#Summarize Method Median
mat_med <- parallel::mclapply(seq_along(groups), function(x){

	if(x %% 10 == 0) message(x, " of ", length(groups))

	#Read In Matrix
	m <- as(TENxMatrix(h5atlas, file.path(groups[x], "assays", "counts.counts")), "dgCMatrix")

	#Groups
	group_by <- as.vector(h5read(h5atlas, file.path(groups[x], "coldata", lv4)))
	group_by <- paste0(basename(groups[x]), ":", group_by)
	group_by <- gsub(" |-", "_", group_by)

	#Normalize
	m <- subset_matrix(m, subsetRows = valid_genes)

	#Normalize
	m <- normalize_matrix(m, scaleTo = 10^4)
	
	#Summarize
	m <- suppressMessages(summarize_matrix(m, group_by, metric = "median"))

	#Return
	m

}, mc.cores = 16, mc.preschedule = FALSE) %>% Reduce("cbind",.)

se <- SummarizedExperiment(
	assays = SimpleList(
		median = mat_med
	)
)
colData(se)$Tissue <- stringr::str_split(colnames(mat_med), pattern = "\\:", simplify=TRUE)[,1]
colData(se)$CellType <- stringr::str_split(colnames(mat_med), pattern = "\\:", simplify=TRUE)[,2]

#Summarize Meta
mdata <- parallel::mclapply(seq_along(groups), function(x){

	if(x %% 10 == 0) message(x)

	#Data Frame
	df <- data.frame(h5read(h5atlas, file.path(groups[x], "coldata")))
	df$combo_lv1 <- paste0(basename(groups)[x], ":", df[,lv1])
	df$combo_lv2 <- paste0(basename(groups)[x], ":", df[,lv2])
	df$combo_lv3 <- paste0(basename(groups)[x], ":", df[,lv3])
	df$combo_lv4 <- paste0(basename(groups)[x], ":", df[,lv4])

	#Summarize
	df2 <- split(df$nCells, df$combo_lv4) %>% 
		{lapply(seq_along(.), function(y){
			data.frame(
				combo_lv4 = names(.)[y],
				n_meta = length(.[[y]]),
				n_cell = sum(.[[y]])
			)
		})} %>% rbindlist %>% data.frame

	#Subset
	df3 <- df[!duplicated(df$combo_lv4), ]
	rownames(df3) <- df3$combo_lv4
	df2$combo_lv1 <- df3[df2$combo_lv4, "combo_lv1"]
	df2$combo_lv2 <- df3[df2$combo_lv4, "combo_lv2"]
	df2$combo_lv3 <- df3[df2$combo_lv4, "combo_lv3"]

	#Return
	df2

}, mc.cores = 16, mc.preschedule = FALSE) %>% Reduce("rbind",.)
rownames(mdata) <- gsub(" |-", "_", mdata[,1])
colData(se)$N_ArchRCells <- mdata[colnames(se), "n_meta"]
colData(se)$N_Cells <- mdata[colnames(se), "n_cell"]
colData(se)$Combo_Lv1 <- gsub(" |-", "_", mdata[colnames(se), "combo_lv1"])
colData(se)$Combo_Lv2 <- gsub(" |-", "_", mdata[colnames(se), "combo_lv2"])
colData(se)$Combo_Lv3 <- gsub(" |-", "_", mdata[colnames(se), "combo_lv3"])
colData(se)$Combo_Lv4 <- gsub(" |-", "_", mdata[colnames(se), "combo_lv4"])

metadata(se)$version <- version
saveRDS(se, se_out, compress=FALSE)

########################################################################################
#2. H5AD for Python Easier
########################################################################################

# # Create H5 file structure manually
# if (file.exists(ad_out)) file.remove(ad_out)

# h5createFile(ad_out)

# # Write main matrix
# h5write(as.matrix(assay(se, "median")), ad_out, "X")

# # Metadata
# h5write(as.data.frame(colData(se)), ad_out, "obs")

# # Row/column names
# h5write(rownames(colData(se)), ad_out, "obs_names")
# h5write(rownames(rowData(se)), ad_out, "var_names")

# o <- h5closeAll()

########################################################################################
#3. Surface Gene H5
########################################################################################

#Healthy
groups <- unique(h5ls(h5atlas)[,1])
groups <- groups[!grepl("assays|coldata", groups)]
groups <- groups[groups != "/"]
groups <- sample(groups)
rhdf5::h5disableFileLocking()

#Read Groupings
dfGroup <- data.frame(fread(h5groups))
rownames(dfGroup) <- dfGroup$ID

#Get All Combo Groups
df <- lapply(seq_along(groups), function(x){

	if("lv4_v4_consensus" %in% h5ls(h5atlas)[,"name"]){
		use <- "lv4_v4_consensus"
	}else{
		use <- "lv4_consensus"
	}

	data.table(
		x=groups[x],
		y=h5read(h5atlas, file.path(groups[x], "assays", "counts.counts", "barcodes")),
		z=trimws(h5read(h5atlas, file.path(groups[x], "coldata", use)), "r")
	)

}) %>% rbindlist %>% data.frame

#Combos
df$combo <- paste0(gsub("/", "", df$x), ":", df$z)
df$combo <- gsub("\\-|\\_|\\ ", "_", df$combo)
combos <- sort(unique(df$combo))
rownames(dfGroup) <- ifelse(substr(rownames(dfGroup), nchar(rownames(dfGroup)), nchar(rownames(dfGroup))) != "_", 
	rownames(dfGroup), substr(rownames(dfGroup), 1, nchar(rownames(dfGroup))-1))

#Check
stopifnot(all(combos %in% rownames(dfGroup)))

#Get Surface Genes
sgenes <- unique(surface_genes())

#Summarize
healthy_mat <- parallel::mclapply(seq_along(combos), function(x){

	if(x %% 50 == 0) message(x, " of ", length(combos))

	#Subset
	dx <- df[df$combo == combos[x],]

	#Get Matrix
	m <- as(TENxMatrix(h5atlas, file.path(dx$x[1], "assays", "counts.counts"))[, dx$y, drop=FALSE], "dgCMatrix")
	m <- subset_matrix(m, subsetRows = valid_genes())
	m <- normalize_matrix(m, scaleTo = 10000)
	m <- subset_matrix(m, subsetRows = sgenes)
	m <- as.matrix(m)
	colnames(m) <- paste0(substr(dx$x[1], 2, 100), ":", dx$z[1], ":", colnames(m))
	m

}, mc.cores = 32, mc.preschedule = FALSE) 

final_matrix <- do.call(cbind, healthy_mat)

writeTENxMatrix(as(final_matrix, "dgCMatrix"), h5_out,  group = "RNA_Norm_Counts")

########################################################################################
#4. Copy To Bucket
########################################################################################

import py_target_id as tid

h_adata = tid.utils.se_rds_to_anndata("20250225/Healthy_Atlas.Lv4.se.rds")
h_adata.write("20250225/Healthy_Atlas.Lv4.h5ad")

h_adata2 = tid.utils.se_rds_to_anndata("20250201/Healthy_Atlas.Lv4.se.rds")
h_adata2.write("20250201/Healthy_Atlas.Lv4.h5ad")


system("gsutil -m cp 20250225/* gs://cartography_target_id_package/Healthy_Atlas/FFPE/20250225/")
system("gsutil -m cp 20250201/* gs://cartography_target_id_package/Healthy_Atlas/SingleCell/20250201/")

#system("gsutil -m mv -r gs://cartography_target_id_samples/Samples_v3/* gs://cartography_target_id_package/Sample_Input/20251008/")
#system("gsutil -m mv -r gs://cartography_target_id_samples/Samples_v2/* gs://cartography_target_id_package/Sample_Input/20250811/")
#system("gsutil -m mv -r gs://cartography_target_id/Homology_Blast/* gs://cartography_target_id_package/Other_Input/Homology_Blast/")
#system("gsutil cp gs://cartography_target_id/Recount3/gtex.20250825.h5 gs://cartography_target_id_package/Other_Input/GTEX/gtex.bulk_rna.20250825.h5")
#system("gsutil cp gs://cartography_target_id/Recount3/tcga.20250825.h5 gs://cartography_target_id_package/Other_Input/TCGA/tcga.bulk_rna.20250825.h5")


