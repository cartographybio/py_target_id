library(TargetFinder)
library(jgplot2)
#library(tidyverse)
TargetFinder:::load_all_functions() #Developer mode

#Manifest
manifest <- load_manifest()
manifest <- manifest[manifest$Indication=="PDAC_FFPE"]

#Target ID
manifest <- download_manifest(manifest)

#Get Data
se_malig <- get_archr_malig_se(manifest)
pos <- percent_positive(se_malig)
pos["PADI3"]

#Get Surface Genes
sgenes <- unique(c(tabs_genes(), surface_genes(tiers = 1:2)))
sgenes <- sgenes[sgenes %in% rownames(se_malig)]
sgenes <- intersect(rownames( get_lv4_healthy_surface_matrix_h5(type="ffpe", overwrite = TRUE)), sgenes)

#Get Percent Expression Per Gene
pos <- percent_positive(se_malig[sgenes,])

#Genes For Multi Analysis
multi_genes <- names(pos[pos >= 0.5])

#Get Malig H5
malig_h5 <- get_malig_h5(manifest)[multi_genes, , drop=FALSE]

#Get Healthy Atlas H5
healthy_atlas_h5 <- get_lv4_healthy_surface_matrix_h5(type="ffpe", overwrite = TRUE)[multi_genes, , drop=FALSE]
split_cn <- stringr::str_split(colnames(healthy_atlas_h5), pattern = "\\:", simplify=TRUE)[,1:2]
ha_ids <- paste0(split_cn[,1], ":", split_cn[,2])
ha_ids <- gsub(" |\\-", "_",ha_ids)

#Filter Cell Types That We Dont Value For Target ID
df_off <- get_healthy_off_target(type="ffpe")

#Check
stopifnot(all(ha_ids %in% df_off$Combo_Lv4))

#Subset V0
ha_use <- df_off$Combo_Lv4[df_off$Off_Target.V0==4]

#Chat GPT
non_essential_survival <- c(
    "Testes",                      # Reproduction
    "Ovary",                       # Reproduction
    "Prostate",                    # Reproduction
    "Uterus",                      # Reproduction
    "Cervix",                      # Reproduction
    "Fallopian_Tube",              # Reproduction
    "Appendix"                     # Vestigial organ
)
ha_use <- ha_use[!(gsub(":.*", "", ha_use) %in% non_essential_survival)]

healthy_atlas_h5_target_id <- healthy_atlas_h5[, ha_ids %in% ha_use]

df_all <- readRDS("20251008.pdac.multi_grid.All.rds")

#Filter
df_r1 <- df_all[df_all$TargetQ_Final_v1 > 90 & df_all$Positive_Final_v2 > 0.5,]
df_r1$TIS_xy <- pmax(df_r1$TIS_x, df_r1$TIS_y) + 0.1 * pmin(df_r1$TIS_x, df_r1$TIS_y)
df_r2 <- df_r1[df_r1$TIS_xy > 70, ]
df_r3 <- df_r2[df_r2$TI_Max_1_xy >= 0.25 & df_r2$TI_Max_10_xy>= 0.5, ]
df_r4 <- df_r3[df_r3$Log2_On_Med_xy >= 0.75, ]
df_r4 <- df_r4[order(df_r4$TIS_xy, decreasing=TRUE),]
pairs_split <- str_split(rownames(df_r4), "_", simplify=TRUE)
target_genes <- unique(c(tabs_genes(), surface_genes(tiers=1)))
df_r4$surface_pair <- ((pairs_split[,1] %in% target_genes) + (pairs_split[,2] %in% target_genes)) == 2
df_r4$tabs_pair <- ((pairs_split[,1] %in% tabs_genes()) + (pairs_split[,2] %in% tabs_genes())) >= 1
df_r5 <- df_r4[df_r4$surface_pair & df_r4$tabs_pair,]
df_r5 <- df_r5[order(df_r5$TargetQ_Final_v1,decreasing=TRUE),]



df_r1 <- df_all[df_all$TargetQ_Final_v1 > 60 & df_all$Positive_Final_v2 > 0.4,]
pairs_split <- str_split(rownames(df_r1), "_", simplify=TRUE)
target_genes <- unique(c(tabs_genes(), surface_genes(tiers=1)))
df_r1$surface_pair <- ((pairs_split[,1] %in% target_genes) + (pairs_split[,2] %in% target_genes)) == 2
df_r1$tabs_pair <- ((pairs_split[,1] %in% tabs_genes()) + (pairs_split[,2] %in% tabs_genes())) >= 1
df_r1 <- df_r1[df_r1$surface_pair & df_r1$tabs_pair,]

df_r1$TIS_xy <- pmax(df_r1$TIS_x, df_r1$TIS_y) + 0.1 * pmin(df_r1$TIS_x, df_r1$TIS_y)
df_r2 <- df_r1[df_r1$TIS_xy > 10, ]
df_r3 <- df_r2[df_r2$TI_Max_1_xy >= 0.3, ]
df_r3 <- df_r3[(df_r3$Log2_On_Med_xy - df_r3$Log2_Off_1_xy) >= 0.1,]
df_top <- df_r3[df_r3$Log2_On_Med_xy >= 0.5,]
df_top <- df_top[!grepl("LEMD1", rownames(df_top)),]
df_top <- df_top[order(df_top$TargetQ_Final_v1, decreasing=TRUE),]

df_top <- df_all[unique(jeff_multis), ]
jeff_multis %in% rownames(df_all)

multis_split <- str_split(rownames(df_top), pattern = "\\_", simplif=TRUE)
genes <- unique(c((as.vector(multis_split)), c("DLL3", "MET", "EGFR", "TACSTD2", "CEACAM5", "ERBB3", "MSLN"))) #With Known
gtex_0 <- as_dgCMatrix(get_gtex_h5()[genes,], threads = 1)
tcga_0 <- as_dgCMatrix(get_tcga_h5()[genes,], threads = 1)

malig_0 <- as_dgCMatrix(get_malig_h5(manifest)[genes, ])
healthy_atlas_0 <- as_dgCMatrix(get_lv4_healthy_surface_matrix_h5(type="ffpe", overwrite = TRUE)[genes, ])
target_genes <- unique(c(tabs_genes(), surface_genes(tiers=1:2)))

pdf("top_pdac.jeff.pdf", width = 24, height = 7)

top_multis <- rownames(df_top)

title <- paste(
    top_multis,
    " TQ ", round(df_all[top_multis,]$TargetQ_Final_v1,3), 
    " PP ", round(df_all[top_multis,]$Positive_Final_v2,3)
)

plot_multi_biaxial_v4(multis= top_multis, malig=malig_0, se_malig=se_malig, healthy_atlas=healthy_atlas_0, 
    show=15, titles = title, gtex=gtex_0, tcga=tcga_0[,grep("PAAD",colnames(tcga_0))])

dev.off()


jeff_multis <- c(
	"ADAM8_ITGB6", "ADAM8_MSLN", "C4BPB_CDH3", "CEACAM5_MUCL3", "CLDN18_TMPRSS4", "ADGRF1_B3GALT5", 
	"B3GALT5_CLDN18", "CDH17_NECTIN4", "CDH3_CDHR2", "CDHR2_NOTCH3", 
	"CEACAM5_CLDN18", "IL2RG_MUCL3", "ITGB6_SEMA7A", "NECTIN4_TM4SF4",
	"CLDN1_FER1L6", "NECTIN4_SCTR", "CDH3_FER1L6", "CDH3_PTPRH", 
	"CEACAM5_TMPRSS3", "IL2RG_TMPRSS3", "ITGA2_MUCL3", 
	"MSLN_TMPRSS3", "ADAM8_ERBB3", "ADAM8_MET", "ADAM8_MST1R",
	"ADAM8_SLC44A4", "CDHR2_ITGA2", "FER1L6_ITGA2", "CLDN18_KLK7",
	"PLEKHN1_SLC44A4", "MST1R_TMPRSS3", "ITGB6_TGM2", "MSLN_PTK6", "MET_PTK6", "MSLN_PKP4",
	"FUT3_TMPRSS3", "MST1R_PLAT", "CEACAM5_PKP4", "CDHR2_CEACAM6", 
	"ADAM8_ITGA2", "CEACAM5_MYRF", "KCNN4_MSLN", "MYRF_NECTIN4",
	"ITGA2_TMPRSS3", "SCEL_SLC44A4"
)

df_sng <- df_sng[df_sng$Surface == "is_surface",]


dir.create("jeff_multis_v1")
for(i in seq_along(top_multis)){

    print(i)

    path_i <- file.path("jeff_multis_v1", paste0(sprintf("%02d",i), ".", top_multis[i]))
    dir.create(path_i, showWarnings=FALSE)

    pdf(file.path(path_i, paste0(top_multis[i], ".02_Multi_TQ_vs_PP_Improvement.pdf")))
    plot_multi_tq_vs_pp_12(multis = top_multis[i], df_sng = df_sng[df_sng$gene_name %in% target_genes,], df_combo = df_all)
    dev.off()

    pdf(file.path(path_i, paste0(top_multis[i], ".03_Multi_TCGA_Bi_Axial.pdf")), width = 24, height = 12)
    plot_multi_biaxial_tcga(multis = top_multis[i], gtex = gtex_0, tcga = tcga_0)
    dev.off()

    pdf(file.path(path_i, paste0(top_multis[i], ".04_Multi_Dot_Plots.pdf")), width = 20, height = 8)
    plot_multi_dot_plot(multis = top_multis[i], healthy_atlas = healthy_atlas_0, se_healthy_atlas = se_ha)
    dev.off()

    pdf(file.path(path_i, paste0(top_multis[i], ".05_Multi_Sng_Axial_12.pdf")), width = 16, height = 10)
    plot_multi_axial_12(multis = top_multis[i], malig = malig_0, healthy_atlas = healthy_atlas_0)
    dev.off()

    pdf(file.path(path_i, paste0(top_multis[i], ".06_Multi_Sng_Axial_vs_Known.pdf")), width = 20, height = 10)
    plot_multi_axial_vs_known(multis = top_multis[i], malig = malig_0, healthy_atlas = healthy_atlas_0)
    dev.off()

    pdf(file.path(path_i, paste0(top_multis[i], ".07_Multi_UMAP.pdf")), width = 17, height = 12)
    plot_multi_umaps(multis = top_multis[i], malig=malig_0[, !grepl("CBP3074|CBP2455|CBP3067|CBP3068", colnames(malig_0))], manifest = manifest)
    dev.off()

    pdf(file.path(path_i, paste0(top_multis[i], ".08_Multi_Radar.pdf")), width = 8, height = 8)
    plot_multi_radar_plots(multis = top_multis[i], df_sng = df_sng[df_sng$gene_name %in% target_genes,], df_combo = df_all)
    dev.off()

}

















#Add TIS
df_multi$TIS_min_xy <- pmin(df_multi$TIS_x, df_multi$TIS_y)
df_multi$TIS_mean_xy <- (df_multi$TIS_x + df_multi$TIS_y) / 2

df_top <- df_multi[(
    df_multi$TargetQ_Final_v1 >= 70 & 
    df_multi$Positive_Final_v2 >= 0.35 & 
    df_multi$TIS_mean_xy >= 40 & 
    df_multi$TIS_min_xy >= 20 & 
    df_multi$TI_Max_1_xy >= 0.2 & 
    df_multi$TI_Max_10_xy >= 0.4 & 
    df_multi$Log2_On_Med_xy >= 0.5 
    ),]

top_anchors <- rownames(df_r4) %>% 
    stringr::str_split(pattern="\\_", simplify=TRUE) %>% 
        as.vector  %>% 
            table %>% sort %>% rev %>% head(25) %>% {data.frame(TAA=names(.), N=as.vector(.))}




