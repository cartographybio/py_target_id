library(TargetFinder)
library(readr)

# #Set Paths
# df <- read_csv("20250811_Samples.csv")
# df$out_h5 <- gsub("gs://cartography_target_id_input/", "gs://cartography_target_id_samples/Samples_v3/", df$out_h5)
# df$out_ArchRCells <- gsub("gs://cartography_target_id_input/", "gs://cartography_target_id_samples/Samples_v3/", df$out_ArchRCells)
# unique(dirname(df$in_h5))

# # Generate copy commands for h5 files
# h5 <- sprintf("gsutil -m cp '%s' '%s'", df$in_h5, df$out_h5)

# # Generate copy commands for ArchRCells files
# ar <- sprintf("gsutil -m cp '%s' '%s'", df$in_ArchRCells, df$out_ArchRCells)

# #All
# cmd <- c(h5, ar)

# #Run
# o <- parallel::mclapply(cmd, function(x) system(x), mc.cores = 32)

# Fix PDAC 
# pdac <- system("gsutil ls gs://cartography_target_id_samples/Samples_v3/input/PDAC_FFPE**", intern = TRUE)
# old <- pdac[grep("ArchRCells.rds", pdac)]
# new <- gsub(".h5", "", old)
# for(i in seq_along(old)){
#   system(sprintf("gsutil mv '%s' '%s'", old[i], new[i]))
# }

# Destination bucket
gcs_bucket <- "gs://cartography_target_id_samples/Samples_v3"

#List
all_files <- system("gsutil ls gs://cartography_target_id_samples/Samples_v3/input/**", intern = TRUE)

#Split
dt <- data.table(
  in_h5map = all_files[grepl(".h5$", all_files)]
)
dt$in_ArchRCells <- gsub(".h5", ".ArchRCells.rds", dt$in_h5map)

#To Process
dt <- dt[dt$in_ArchRCells %in% all_files]

#Run

# List all files (non-recursive; change recursive=TRUE if needed)
files_to_remove <- list.files("processed", full.names = TRUE, recursive = TRUE)

# Remove files
if (length(files_to_remove) > 0) {
  o <- file.remove(files_to_remove)
}

#Download
o <- parallel::mclapply(seq(nrow(dt)), function(i){

  message(i)

  system(sprintf("gsutil cp %s input/%s", dt$in_h5map[i], file.path(basename(dirname(dt$in_h5map[i])), basename(dt$in_h5map[i]))))
  system(sprintf("gsutil cp %s input/%s", dt$in_ArchRCells[i], file.path(basename(dirname(dt$in_ArchRCells[i])), basename(dt$in_ArchRCells[i]))))

  #Input
  h5_in <- file.path("input", basename(dirname(dt$in_h5map[i])), basename(dt$in_h5map[i])) #"input/KIRC/Kidney14.h5"
  ar_in <- file.path("input", basename(dirname(dt$in_ArchRCells[i])), basename(dt$in_ArchRCells[i])) #"input/KIRC/Kidney14.ArchRCells.rds"

  o <- system(sprintf(
    "Rscript --vanilla 20251006_Process_Single_Target_ID_Samples_v3.R %s %s",
    h5_in, ar_in
  ), intern = TRUE)

  if(any(grepl("nMalig = 0", o))){
    #Clear Input
    system(sprintf("gsutil mv %s %s", dt$in_h5map[i], gsub("Samples_v3/input", "Samples_v3/arxiv_no_malig", dt$in_h5map[i])))
    system(sprintf("gsutil mv %s %s", dt$in_ArchRCells[i], gsub("Samples_v3/input", "Samples_v3/arxiv_no_malig", dt$in_ArchRCells[i])))
  }

  clean_paths <- gsub("^\\[1\\] \"|\"$", "", grep("processed", o, value=TRUE))
  if(!all(file.exists(clean_paths))) return(NULL)

  # Loop over files and copy
  for (f in clean_paths) {
    system(sprintf("gsutil cp '%s' '%s'", f, file.path(gcs_bucket, f)))
  }

  system(sprintf("gsutil mv %s %s", dt$in_h5map[i], gsub("Samples_v3/input", "Samples_v3/arxiv", dt$in_h5map[i])))
  system(sprintf("gsutil mv %s %s", dt$in_ArchRCells[i], gsub("Samples_v3/input", "Samples_v3/arxiv", dt$in_ArchRCells[i])))

}, mc.cores = 4)

























system(sprintf("gsutil cp %s input/%s", dt$in_h5map[i], file.path(basename(dirname(dt$in_h5map[i])), basename(dt$in_h5map[i]))))
system(sprintf("gsutil cp %s input/%s", dt$in_ArchRCells[i], file.path(basename(dirname(dt$in_ArchRCells[i])), basename(dt$in_ArchRCells[i]))))

#Input
h5_in <- file.path("input", basename(dirname(dt$in_h5map[i])), basename(dt$in_h5map[i])) #"input/KIRC/Kidney14.h5"
ar_in <- file.path("input", basename(dirname(dt$in_ArchRCells[i])), basename(dt$in_ArchRCells[i])) #"input/KIRC/Kidney14.ArchRCells.rds"

o <- system(sprintf(
  "Rscript --vanilla 20251006_Process_Target_ID_Samples_v3.R %s %s",
  h5_in, ar_in
), intern = TRUE)

clean_paths <- gsub("^\\[1\\] \"|\"$", "", o[5:13])
if(!all(file.exists(clean_paths))) break

# Loop over files and copy
for (f in clean_paths) {
  system(sprintf("gsutil cp '%s' '%s'", f, file.path(gcs_bucket, f)))
}

#Clear Input
system(sprintf("gsutil mv %s %s", dt$in_h5map[i], gsub("Samples_v3/input", "Samples_v3/arxiv", dt$in_h5map[i])))
system(sprintf("gsutil mv %s %s", dt$in_ArchRCells[i], gsub("Samples_v3/input", "Samples_v3/arxiv", dt$in_ArchRCells[i])))


clean_paths[!file.exists(clean_paths)]


#Info
indication <- basename(dirname(h5_in))
out_id <- paste0(indication, "._.", gsub(".h5", "._.", basename(h5_in))) #Unique Delimiter '._.'

#Output Files
h5_out <- matching_files <- list.files(path = h5map_dir, pattern = out_id, full.names = TRUE)

file.path("processed", "h5map", paste0(out_id, ".h5"))
ar_out <- file.path("processed", "ArchRCells", paste0(out_id, ".ArchRCells.rds"))
md_out <- file.path("processed", "Metadata", paste0(out_id, ".mdata.rds"))
stats_out <- file.path("processed", "Stats", paste0(out_id, ".stats.csv"))
malig_ArchR_out <- file.path("processed", "ArchRCells_Malig", paste0(out_id, ".ArchRCells.h5"))
malig_se_out <- file.path("processed", "SE_Malig", paste0(out_id, ".malig.se.rds"))
malig_h5ad_out <- file.path("processed", "AD_Malig", paste0(out_id, ".malig.se.rds"))
zarr_h5_out <- file.path("processed", "Zarr_h5map", paste0(out_id, ".zarr.zip"))
malig_zarr_malig_h5_out <- file.path("processed", "Zarr_ArchRCells_Malig", paste0(out_id, ".ArchRCells.zarr.zip"))



o <- system(sprintf(
  "Rscript --vanilla 20251006_Process_Target_ID_Samples_v3.R %s %s",
  h5_in, ar_in
), intern = TRUE)





























library(TargetFinder)
library(jgplot2)
library(HDF5Array)

h5 <- "test/LUAD._.CBP1898._.20250811_182756.h5"
m <- TENxMatrix(h5, "/assays/RNA.counts")
rownames(m)


(dirname(unique(dirname(df$in_h5))))

library(TargetFinder)
library(jgplot2)
library(HDF5Array)

h5 <- "AML._.AML_3082._.20250811_181804.h5"
m <- as(TENxMatrix(h5, "/assays/RNA.counts"), "dgCMatrix")







 [1] "gs://cartography_magellan_data/LUAD_Freeze_250527/h5map"                           
 [2] "gs://cartography_magellan_data/LUAD_Freeze_250429/h5map"                           
 [3] "gs://cartography_h5_atlas/cancer_h5map_new/240317_PDAC_PRJNA948891"                
 [4] "gs://cartography_magellan_data/target_id_new/ffpe/h5map/Breast"                    
 [5] "gs://cartography_magellan_data/target_id_new/sc/h5map/Breast"                      
 [6] "gs://cartography_temp/050923_AML/h5map"                                            
 [7] "gs://cartography_h5_atlas/cancer_h5map"                                            
 [8] "gs://cartography_h5_atlas/Dec2024_external/h5map/231021_PRJNA753861_HNSC"          
 [9] "gs://cartography_h5_atlas/Dec2024_external/h5map/241116_PRJNA1078312_PancreaticNET"
[10] "gs://cartography_h5_atlas/Dec2024_external/h5map/241116_PRJNA658541_LiverCancer"   
[11] "gs://cartography_magellan_data/target_id/sc/h5map/240613_magellan"                 
[12] "gs://cartography_magellan_data/target_id/sc/h5map/240715_Novogene_Swift"           
[13] "gs://cartography_magellan_data/target_id/sc/h5map/240827_Novogene_Swift"           
[14] "gs://cartography_h5_atlas/cancer_h5map_new/220928_Novogene"                        
[15] "gs://cartography_h5_atlas/cancer_h5map_new/240702_Swift"                           
[16] "gs://cartography_h5_atlas/cancer_h5map_new/240909_Novogene_Swift"                  
[17] "gs://cartography_h5_atlas/cancer_h5map_new/240911_PAAD_PRJNA843078"                
[18] "gs://cartography_h5_atlas/cancer_h5map_new/241011_Novogene_Swift"                  
[19] "gs://cartography_h5_atlas/cancer_h5map_new/ovary_merged" 











