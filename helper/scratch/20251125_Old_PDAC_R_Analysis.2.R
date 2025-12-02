detach("package:TargetFinder", unload = TRUE)

library(TargetFinder)
library(jgplot2)
library(tidyverse)
TargetFinder:::load_all_functions() #Developer mode

#Manifest
manifest <- load_manifest()
manifest <- manifest[manifest$Indication=="PDAC_FFPE"]

#Target ID
manifest <- download_manifest(manifest)

#Get Malig
se_malig <- get_archr_malig_se(manifest)

#Recount Bulk
gtex_h5 <- get_gtex_h5()
tcga_h5 <- get_tcga_h5()

#Heatlhy
se_ha <- get_lv4_healthy_se(type = "FFPE",  version = "20250225", overwrite = TRUE)
se_ha_weight <- get_lv4_healthy_se_weigthed(type = "FFPE", se = se_ha)

#Run
df_sng <- run_target_id(se_malig = se_malig, se_healthy = se_ha_weight)
target_genes <- unique(c(tabs_genes(), surface_genes(tiers = 1:2)))
df_sng$Surface <- ifelse(df_sng$gene_name %in% target_genes, "is_surface", "not_surface")

#Plot
pdf("PDAC.TQ_vs_PP.pdf", width = 14, height = 8)
plot_tq_vs_pp(df_sng)
dev.off()

#Top Genes
top_genes <- df_sng[df_sng$Surface == "is_surface" & df_sng$TargetQ_Final_v1 > 45 & df_sng$Positive_Final_v2 > 0.1, "gene_name"]
df_top <- df_sng[top_genes, ]
tcga <- as.matrix(tcga_h5[top_genes, ])
gtex <- as.matrix(gtex_h5[top_genes, ])

#Plot Single Axis
pdf("PDAC.Single_Axis_Plots.pdf", width = 12, height = 5)

plot_single_axial(
	genes = top_genes[1:98], 
	se_malig, se_ha, show = 25)

dev.off()

#Plot Dot Plot
pdf("PDAC.Single_Dot_Plots.pdf", width = 10, height = 6)

plot_single_dot_plot(
	genes = top_genes[1:98], 
	se_healthy_atlas = se_ha)

dev.off()

pdf("PDAC.Single_TCGA_GTEX.pdf", width = 14, height = 6)

plot_single_tcga_gtex(genes = top_genes[1:98], tcga, gtex)

dev.off()

df_topology <- get_human_topology(genes = top_genes)
df_topology <- split(df_topology, df_topology$gene_name) %>% lapply(., function(x) data.frame(x[order(x$N_AA,decreasing=TRUE)[1],,drop=FALSE])) %>% rbindlist %>% data.frame
rownames(df_topology) <- df_topology$gene_name
df_topology <- df_topology[top_genes,]

pdf("PDAC.Single_Topology_Kmer.pdf", width = 10, height = 12)
plot_topology_kmer(df_topology$transcript_id[1:98])
dev.off()

pdf("PDAC.Single_Blast.pdf", width = 28, height = 14)
plot_homology_blast(df_topology[1:98,])
dev.off()



library(officer)
library(magick)

# Number of genes
n_genes <- 98

# PDF files with their original dimensions
pdf_info <- list(
  list(file = "PDAC.Single_Axis_Plots.pdf", width = 12, height = 5),
  list(file = "PDAC.Single_Dot_Plots.pdf", width = 10, height = 6),
  list(file = "PDAC.Single_TCGA_GTEX.pdf", width = 14, height = 6),
  list(file = "PDAC.Single_Topology_Kmer.pdf", width = 10, height = 12),
  list(file = "PDAC.Single_Blast.pdf", width = 28, height = 14)
)

# Create PowerPoint (10 x 7.5 inches available)
ppt <- read_pptx()

# Define layout positions
# Left column: 55% width = 5.39 inches (0.1 to 5.49)
# Right column: 45% width = 4.41 inches (5.59 to 10)
# Available height: 6.6 inches (from 0.9 to 7.5)

left_width <- 5.39
right_width <- 4.41
left_start <- 0.1
right_start <- 5.59
top_start <- 0.9
available_height <- 6.6

# Left column: divide height into 3 parts (with small gaps)
left_height_each <- (available_height - 0.2) / 3  # 2.13 each

# Right column: divide height into 2 parts (with small gap)
right_height_each <- (available_height - 0.1) / 2  # 3.25 each

positions <- list(
  # Left column - top: Single Axis
  list(left = left_start, top = top_start, width = left_width, height = left_height_each),
  
  # Left column - middle: Dot Plot
  list(left = left_start, top = top_start + left_height_each + 0.1, width = left_width, height = left_height_each),
  
  # Left column - bottom: TCGA/GTEX
  list(left = left_start, top = top_start + 2*left_height_each + 0.2, width = left_width, height = left_height_each),
  
  # Right column - top: Topology
  list(left = right_start, top = top_start, width = right_width, height = right_height_each),
  
  # Right column - bottom: BLAST
  list(left = right_start, top = top_start + right_height_each + 0.1, width = right_width, height = right_height_each)
)

# Loop through each gene
for(i in 1:n_genes) {
  
  message("Processing gene ", i, " of ", n_genes, ": ", top_genes[i])
  
  # Add a new blank slide
  ppt <- add_slide(ppt, layout = "Blank", master = "Office Theme")
  
  # Add title with larger font
  ppt <- ph_with(ppt, value = fpar(ftext(top_genes[i], 
                                         prop = fp_text(font.size = 32, bold = TRUE))), 
                 location = ph_location(left = 0.5, top = 0.1, 
                                       width = 9, height = 0.7))
  
  # Process each PDF
  for(j in 1:5) {
    
    message("  Processing PDF ", j, ": ", pdf_info[[j]]$file)
    
    # Convert PDF page to PNG
    temp_png <- tempfile(fileext = ".png")
    
    tryCatch({
      img <- image_read_pdf(pdf_info[[j]]$file, pages = i, density = 150)
      image_write(img, path = temp_png, format = "png")
      
      # Add image to slide
      ppt <- ph_with(ppt, value = external_img(temp_png),
                     location = ph_location(left = positions[[j]]$left,
                                           top = positions[[j]]$top,
                                           width = positions[[j]]$width,
                                           height = positions[[j]]$height))
      
      # Clean up immediately
      rm(img)
      gc()
      
    }, error = function(e) {
      message("Error processing ", pdf_info[[j]]$file, " page ", i, ": ", e$message)
    })
    
    # Remove temp file
    if(file.exists(temp_png)) {
      file.remove(temp_png)
    }
  }
  
  # Force garbage collection after each slide
  gc()
}

# Save the PowerPoint
print(ppt, target = "PDAC_Gene_Summary.pptx")
message("PowerPoint presentation created: PDAC_Gene_Summary.pptx")




plot_single_tcga_gtex <- function(genes, tcga, gtex){

	tcga <- as.matrix(tcga[genes, ])
	gtex <- as.matrix(gtex[genes, ])

	for(i in seq(nrow(tcga))){

		df_tcga <- data.frame(row.names=NULL, type = "1.TCGA", id = str_split(colnames(tcga), pattern="\\#", simplify=TRUE)[,1], log2tpm = log2(tcga[i, ] + 1))
		df_gtex <- data.frame(row.names=NULL, type = "2.GTEX", id = colnames(gtex), log2tpm = log2(gtex[i, ] + 1))
		df_all <- rbind(df_tcga, df_gtex)

		# Calculate percentage > log2(11) and median for each id using base R
		pct_by_id <- aggregate(log2tpm ~ type + id, data = df_all, 
		                       FUN = function(x) c(pct = round(100 * sum(x > log2(11)) / length(x), 1),
		                                          med = median(x)))
		pct_by_id <- data.frame(type = pct_by_id$type, 
		                        id = pct_by_id$id,
		                        pct = pct_by_id$log2tpm[, "pct"],
		                        med = pct_by_id$log2tpm[, "med"])
		pct_by_id$id_label <- paste0(pct_by_id$id, " (", pct_by_id$pct, "%)")

		# Order by percentage (descending), then by median (descending) within each type
		pct_by_id <- pct_by_id[order(pct_by_id$type, -pct_by_id$pct, -pct_by_id$med), ]

		# Join back to main dataframe
		df_all <- merge(df_all, pct_by_id[, c("type", "id", "id_label", "pct", "med")], by = c("type", "id"))

		# Convert id_label to factor with levels ordered by percentage then median (descending within type)
		df_all$id_label <- factor(df_all$id_label, levels = unique(pct_by_id$id_label))


		p <- ggplot(df_all, aes(id_label, log2tpm, fill = type)) +
			geom_jitter(height = 0, width = 0.2, pch = 21, alpha = 0.5) +
			facet_wrap(~type, scales = "free_x") +
			theme_jg(xText90=TRUE) +
			geom_hline(yintercept = log2(11), lty = "dashed", color = "firebrick3", linewidth = 0.5) +
			xlab("") +
			theme(axis.text.x = element_text(size = 8)) +
			scale_fill_manual(values = c("1.TCGA"=pal_cart[4], "2.GTEX"="lightgrey")) +
			ylab("Log2(TPM + 1)") +
			theme(strip.text = element_text(size = 14), plot.title = element_text(size = 14)) +
			ggtitle(genes[i]) +
			theme(legend.position = "none") +
			geom_boxplot(outlier.size = NA, outlier.stroke = NA, fill = NA, color = "black", outlier.shape = NA, linewidth = 0.5)

		print(p)

	}

}

tcga <- as.matrix(tcga_h5[top_genes, ])
gtex <- as.matrix(gtex_h5[top_genes, ])

for(i in seq(nrow(tcga))){

	df_tcga <- data.frame(row.names=NULL, type = "1.TCGA", id = str_split(colnames(tcga), pattern="\\#", simplify=TRUE)[,1], log2tpm = log2(tcga[i, ] + 1))
	df_gtex <- data.frame(row.names=NULL, type = "2.GTEX", id = colnames(gtex), log2tpm = log2(gtex[i, ] + 1))
	df_all <- rbind(df_tcga, df_gtex)

	# Calculate percentage > log2(11) for each id using base R
	pct_by_id <- aggregate(log2tpm ~ type + id, data = df_all, 
	                       FUN = function(x) round(100 * sum(x > log2(11)) / length(x), 1))
	colnames(pct_by_id)[3] <- "pct"
	pct_by_id$id_label <- paste0(pct_by_id$id, " (", pct_by_id$pct, "%)")

	# Join back to main dataframe
	df_all <- merge(df_all, pct_by_id[, c("type", "id", "id_label")], by = c("type", "id"))

	# Convert id_label to factor to control ordering
	df_all$id_label <- factor(df_all$id_label, levels = unique(df_all$id_label[order(df_all$type, df_all$id)]))

	p <- ggplot(df_all, aes(id_label, log2tpm, fill = type)) +
		geom_jitter(height = 0, width = 0.2, pch = 21, alpha = 0.8) +
		facet_wrap(~type, scales = "free_x") +
		theme_jg(xText90=TRUE) +
		geom_hline(yintercept = log2(11), lty = "dashed") +
		xlab("") +
		theme(axis.text.x = element_text(size = 8)) +
		scale_fill_manual(values = c("1.TCGA"=pal_cart[5], "2.GTEX"=pal_cart[2])) +
		ylab("Log2(TPM + 1)") +
		theme(strip.text = element_text(size = 14), plot.title = element_text(size = 14)) +
		ggtitle(genes[i]) +
		theme(legend.position = "none")

	print(p)

}






df_tcga <- data.frame(row.names=NULL, type = "1.TCGA", id = str_split(colnames(tcga), pattern="\\#", simplify=TRUE)[,1], log2tpm = log2(tcga[i, ] + 1))
df_gtex <- data.frame(row.names=NULL, type = "2.GTEX", id = colnames(gtex), log2tpm = log2(gtex[i, ] + 1))
df_all <- rbind(df_tcga, df_gtex)

# Calculate percentage > log2(11) for each id using base R
pct_by_id <- aggregate(log2tpm ~ type + id, data = df_all, 
                       FUN = function(x) round(100 * sum(x > log2(11)) / length(x), 1))
colnames(pct_by_id)[3] <- "pct"
pct_by_id$id_label <- paste0(pct_by_id$id, " (", pct_by_id$pct, "%)")

# Join back to main dataframe
df_all <- merge(df_all, pct_by_id[, c("type", "id", "id_label")], by = c("type", "id"))

# Convert id_label to factor to control ordering
df_all$id_label <- factor(df_all$id_label, levels = unique(df_all$id_label[order(df_all$type, df_all$id)]))

p <- ggplot(df_all, aes(id_label, log2tpm, fill = type)) +
	geom_jitter(height = 0, width = 0.2, pch = 21, alpha = 0.8) +
	facet_wrap(~type, scales = "free_x") +
	theme_jg(xText90=TRUE) +
	geom_hline(yintercept = log2(11), lty = "dashed") +
	xlab("") +
	theme(axis.text.x = element_text(size = 8)) +
	scale_fill_manual(values = c("1.TCGA"=pal_cart[5], "2.GTEX"=pal_cart[2])) +
	ylab("Log2(TPM + 1)") +
	theme(strip.text = element_text(size = 14), plot.title = element_text(size = 14)) +
	ggtitle(genes[i]) +
	theme(legend.position = "none")
dev.off()






df_tcga <- data.frame(row.names=NULL, type = "1.TCGA", id = str_split(colnames(tcga), pattern="\\#", simplify=TRUE)[,1], log2tpm = log2(tcga[i, ] + 1))
df_gtex <- data.frame(row.names=NULL, type = "2.GTEX", id = colnames(gtex), log2tpm = log2(gtex[i, ] + 1))
df_all <- rbind(df_tcga, df_gtex)

pdf("test.pdf", width = 16, height = 6)

ggplot(df_all, aes(id, log2tpm, fill = type)) +
	geom_jitter(height = 0, width = 0.2, pch = 21) +
	facet_wrap(~type, scales = "free_x") +
	theme_jg(xText90=TRUE) +
	geom_hline(yintercept = log2(11), lty = "dashed")

dev.off()




#' @export
plot_single_dot_plot <- function(
    genes=NULL, 
    se_healthy_atlas=NULL, 
    max_log2 = 3
    ){

	ha <- assay(se_healthy_atlas)[genes, ,drop=FALSE]
    CT <- sub("^([^:]+:[^:]+):.*", "\\1", colnames(ha))
    CT <- gsub(" |\\-", "_", CT)

    #Fix Letters
    CT <- gsub("α", "a", CT)
    CT <- gsub("β", "B", CT)
    se_healthy_atlas$Combo_Lv4 <- gsub("α", "a", se_healthy_atlas$Combo_Lv4)
    se_healthy_atlas$Combo_Lv4 <- gsub("β", "B", se_healthy_atlas$Combo_Lv4)

    #Check
    stopifnot(all(se_healthy_atlas$Combo_Lv4 %in% CT))
    stopifnot(all(CT %in% se_healthy_atlas$Combo_Lv4))

    #Summarize Groups
    df <- data.frame(colData(se_healthy_atlas))
    df$Tissue2 <- paste0(df$Tissue)
    tissues <- unique(df$Tissue2)

    groups <- rep("Other", length(tissues))
    names(groups) <- paste0(tissues)
    groups[grep("brain|phalon", names(groups), ignore.case=TRUE)] <- "Brain"
    groups[grep("testes|uterus|ovary|cervix|prostate|fallopian", names(groups), ignore.case=TRUE)] <- "Repro."
    groups[grep("blood|thymus|lymph_node|pbmc|spleen|bone_marrow", names(groups), ignore.case=TRUE)] <- "Immune"
    groups[grep("heart|kidney|liver|lung|pancreas|peripheral|skeletal|spine", names(groups), ignore.case=TRUE)] <- "High Risk\nOrgan"
    groups[grep("blood|thymus|lymph_node|pbmc|spleen|bone_marrow", names(groups), ignore.case=TRUE)] <- "Immune"

    df$Lv1_Old <- stringr::str_split(df$Combo_Lv1, pattern="\\:", simplify=TRUE)[,2]
    df$Lv1_Old[df$Lv1_Old=="Neuron/glia"] <- "Neural"
    df$Lv1_Old[df$Lv1_Old=="Glial"] <- "Neural"
    df$Lv1_Old[df$Lv1_Old=="Glia"] <- "Neural"
    df$Lv1_Old[df$Lv1_Old=="Neuron"] <- "Neural"

    df$Lv1_Old[grep("Endothelial", rownames(df), ignore.case=TRUE)] <- "Vasculature"
    df$Lv1_Old[grep("Vascular", rownames(df), ignore.case=TRUE)] <- "Vasculature"
    df$Lv1_Old[grep("Pericyte", rownames(df), ignore.case=TRUE)] <- "Vasculature"
    df$Lv1_Old[grep("Ependymal", rownames(df), ignore.case=TRUE)] <- "Neural"
    df$Lv1_Old[grep("epithelial", rownames(df), ignore.case=TRUE)] <- "Epithelial"

    df$Lv1 <- factor(paste0(df$Lv1_Old), 
        levels = c("Neural", "Epithelial", "Stromal", "Vasculature", "Immune", "Somatic"))

    df$TissueGroup <- factor(groups[paste0(df$Tissue2)], levels = c("Brain", "High Risk\nOrgan", "Other", "Immune", "Repro."))
    lvls <- rev(unique(sort(paste0(df$Tissue2))))
    lvls <- rev(unique(sort(c(lvls, gsub("Brain\\_", "", lvls)))))
    df$Tissue2 <- factor(gsub("Brain\\_","",paste0(df$Tissue2)), levels = lvls)
    df$Tissue2 <- factor(gsub("\\_", " ", df$Tissue2), levels = rev(sort(unique(gsub("\\_", " ", df$Tissue2)))))

    #Transform
    cube <- function(x) x^3
    cube_root <- function(x) x^(1/3)
    trans_cube <- scales::trans_new(name = "cube",
                            transform = cube,
                            inverse = cube_root)

    for(i in seq(nrow(ha))){

	    #Plot 
	    df$TAA_1 <- pmin(pmax(log2(ha[i, ] + 0.95), 0), max_log2)
	    df$TAA_1[df$TAA_1==0] <- NA

	    p1 <- ggplot(df, aes(CellType, Tissue2)) +
	        geom_point(aes(size = TAA_1, fill = TAA_1), color = "black", pch=21, alpha = 0.8) +
	        theme(axis.text.x = element_blank()) +
	        scale_fill_gradientn(colors=pal_cart2, limits = c(0, max_log2)) +
	        facet_grid(TissueGroup~Lv1,scales="free") +
	        force_panelsizes(
	            rows = table(unique(df[,c("Tissue2", "TissueGroup")])[,2])[levels(df$TissueGroup)],
	            cols = (table(unique(df[,c("CellType", "Lv1")])[,2])[levels(df$Lv1)] + c(0,0,0,10,0,30))
	        ) +
	        scale_size(limits = c(0, max_log2), trans = trans_cube) +
	        theme(panel.grid.minor = element_blank(), panel.grid.major.x=element_blank()) +
	        theme(axis.ticks.x = element_blank()) +
	        ggtitle(genes[i]) +
	        scale_x_discrete(expand = c(0.05, 0.05)) + 
	        theme(plot.title = element_text(hjust = 0.5, size = 16)) +
	        ylab("") +
	        xlab("Distinct Cell Types") +
	        theme(axis.title.x = element_text(hjust = 0.5, size = 12)) +
	        theme(strip.text.x = element_text(margin = margin(0.1,0,0.1,0, "cm"))) +
	        theme(strip.text.y = element_text(angle = 0, margin = margin(0, 0.1, 0, 0.1, "cm"))) +
	        theme(axis.text.y = element_text(size = 6.5)) #+
	        #theme_small_margin(0.01) #+
	        #theme(strip.text.y = element_blank() , strip.background.y = element_blank())

	    print(p1)

	}

}





































#' @export
plot_single_axial <- function(
    genes = NULL,
    se_malig = NULL,
    se_ha = NULL,
    show = 15,
    titles = genes
){

    require(scales)
    
    # Start overall timing
    overall_start <- Sys.time()

    #read in malig
    malig_sng_med <- assay(se_malig)[genes,,drop=FALSE]
    patients <- stringr::str_split(colnames(malig_sng_med), pattern="\\.\\_\\.", simplify=TRUE)[,2]

    #read in healthy_atlas
    ha_sng_med <- assay(se_ha)[genes,,drop=FALSE]
    CT <- sub("^([^:]+:[^:]+):.*", "\\1", colnames(ha_sng_med))
    CT <- gsub(" |\\-", "_", CT)
    CT <- gsub("α", "a", CT)
    CT <- gsub("β", "B", CT)

    #Positivity
    pos_mat <- compute_positivity_matrix(assays(se_malig[rownames(se_ha),])$ranks, assays(se_malig[rownames(se_ha),])$counts)
    pos_patients <- stringr::str_split(colnames(pos_mat), pattern="\\.\\_\\.", simplify=TRUE)[,2]
    stopifnot(all(pos_patients %in% patients))
    colnames(pos_mat) <- pos_patients
    pos_mat <- pos_mat[, unique(patients)]

    # Print timing after data preparation
    for(i in seq_along(genes)){

        ts_message("Processing ", i , " of ", length(genes), " : ", genes[i])
        gx <- genes[i]

        pos_1 <- names(which(pos_mat[gx, ]))

        #Malig
        df1 <- data.frame(
            pos_patient = ifelse(patients %in% pos_1, "Malig Pos.", "Malig Neg."),
            log2_exp = log2(malig_sng_med[gx,] + 1)
        )
        df1$pos_patient <- factor(df1$pos_patient, levels = c("Malig Pos.", "Malig Neg."))
        med1 <- median(df1$log2_exp[df1$pos_patient=="Malig Pos."])
        
        # Calculate N for each group
        n_pos <- sum(df1$pos_patient == "Malig Pos.")
        n_neg <- sum(df1$pos_patient == "Malig Neg.")

        #Healthy
        df1h <- data.frame(
            tissue = stringr::str_split(colnames(ha_sng_med), pattern = "\\:", simplify=TRUE)[,1],
            log2_exp = log2(ha_sng_med[gx,] + 1),
            full = sapply(colnames(ha_sng_med), add_smart_newlines) 
        )
        df1h$full <- factor(df1h$full, levels = df1h$full[order(df1h$log2_exp, decreasing=TRUE)])
        df1h <- df1h[order(df1h$full),]

        m <- max(df1$log2_exp)
        limits <- c(0, m)
        size <- 6

        per_1 <- round(length(pos_1)/ncol(pos_mat), 3) * 100

        suppressMessages({

            # Malignant boxplot
            p1_1 <- ggplot(df1, aes(pos_patient, log2_exp)) +
                theme_jg(xText90=TRUE) +
                theme_small_margin(0.01) +
                geom_boxplot(fill=NA, outlier.size = NA, outlier.stroke = NA, outlier.shape = NA, width = 0.5) +
                geom_jitter(size = 1.5, stroke = 0.35, pch = 21, aes(fill = pos_patient), color = "black", height = 0, width = 0.2) +
                theme(legend.position = "none") +
                coord_cartesian(ylim = limits) + 
                scale_x_discrete(
                    drop = FALSE,
                    labels = c(
                        "Malig Pos." = paste0("Malig Pos.\n(N = ", n_pos,")"),
                        "Malig Neg." = paste0("Malig Neg.\n(N = ", n_neg,")")
                    )
                ) + 
                xlab("") +
                geom_hline(yintercept = med1, lty = "dashed", color = "firebrick3") +
                scale_fill_manual(values = c("Malig Pos." = "#28154C", "Malig Neg." = "lightgrey")) +
                ylab(gx) + 
                ggtitle(paste0(per_1, "%"))

            # Healthy by tissue
            p1_2 <- ggplot(df1h, aes(tissue, squish(log2_exp, limits))) +
                theme_jg(xText90=TRUE) +
                theme_small_margin(0.01) +
                geom_boxplot(fill=NA, outlier.size = NA, outlier.stroke = NA, outlier.shape = NA, width = 0.5) +
                geom_jitter(size = 1.5, stroke = 0.35, pch = 21, aes(fill = tissue), color = "black", height = 0, width = 0.2) +
                theme(legend.position = "none") +
                coord_cartesian(ylim = limits) +
                geom_hline(yintercept = med1, lty = "dashed", color = "firebrick3") +
                xlab("") + ylab("") +
                theme(axis.text.x = element_text(size=size)) +
                scale_fill_manual(values = create_pal_d(df1h$tissue)) +
                ggtitle("Facet By Tissue")

            # Top off-targets
            p1_3 <- ggplot(head(df1h, show), aes(full, squish(log2_exp, limits))) +
                theme_jg(xText90=TRUE) +
                theme_small_margin(0.01) +
                geom_point(size = 1.5, stroke = 0.35, pch = 21, aes(fill = tissue), color = "black") +
                theme(legend.position = "none") +
                coord_cartesian(ylim = limits) +
                geom_hline(yintercept = med1, lty = "dashed", color = "firebrick3") +
                xlab("") + ylab("") +
                theme(axis.text.x = element_text(size=size*0.75)) +
                scale_fill_manual(values = create_pal_d(df1h$tissue)) +
                ggtitle("Top Off Targeted Cell Types")

            # Combine with widths
            final_plot <- p1_1 + p1_2 + p1_3 + plot_layout(widths = c(1, 3, 3))

            #pdf("Plot_Single_Axis.pdf", width = 12, height = 5)
            print(final_plot)
            #dev.off()

        })

    }

    # Print final timing
    overall_elapsed <- as.numeric(difftime(Sys.time(), overall_start, units = "mins"))
    message("------  All plots completed - | Total: ", round(overall_elapsed, 1), "min")

}






























plot_tq_vs_pp <- function(
    df,
    target_q = 50,
    ppos = 10,
    facet_surface = TRUE
    ){
    library(ggplot2)
    library(ggrepel)
    library(dplyr)
    
    # Add surface protein annotation if faceting requested
    if (facet_surface) {
        target_genes <- unique(c(tabs_genes(), surface_genes(tiers = 1:2)))
        df$Surface <- ifelse(df$gene_name %in% target_genes, "is_surface", "not_surface")
    }
    
    # Subset high-quality targets for labeling
    g1 <- get_top_n_per_interval(
        data.frame(df),
        x_col = 'Positive_Final_v2',
        y_col = 'TargetQ_Final_v1',
        label_col = 'gene_name',
        n = 2,
        interval = 10
    )$gene_name
    
    g2 <- df %>% 
        data.frame() %>% 
        filter(TargetQ_Final_v1 >= target_q & Positive_Final_v2 * 100 >= ppos) %>%
        pull(gene_name)
    
    genes <- unique(c(g1, g2))
    sub_df <- df[df$gene_name %in% genes, ]
    
    # Create base plot
    p <- ggplot(df, aes(Positive_Final_v2 * 100, 
                        TargetQ_Final_v1, 
                        fill = pmin(2, log2(SC_2nd_Target_Val + 1)))) +
        geom_point(pch = 21, size = 2, color = "black") +
        geom_label_repel(
            data = sub_df,
            aes(label = gene_name),
            fill = "white",
            max.overlaps = Inf,
            size = 2
        ) +
        scale_fill_gradientn(colors = pal_cart2) +
        xlab("Positive Final (v2)") +
        ylab("Target Quality (v1)") +
        labs(fill = "Log2 Exp.") +
        theme_jg()
    
    # Add faceting if requested
    if (facet_surface) {
        p <- p + facet_wrap(~Surface)
    }
    
    
    return(p)

}


#Surface
target_genes <- unique(c(tabs_genes(), surface_genes(tiers=1:2)))
df_sng$Surface <- ifelse(df_sng$gene_name %in% target_genes, "is_surface", "not_surface")

# Subset high-quality targets for labeling
g1 <- get_top_n_per_interval(
	data.frame(df_sng),
	x_col = 'Positive_Final_v2',
	y_col = 'TargetQ_Final_v1',
	label_col = 'gene_name',
	n = 2,
	interval = 10
)

g2 <- df_sng %>% data.frame %>% 
	filter(TargetQ_Final_v1 >= 50 & Positive_Final_v2 * 100 >= 10) %>%
	pull(gene_name)

genes <- unique(c(g1, g2))
sub_df <- df_sng[df_sng$gene_name %in% genes, ]

#Plot

ggplot(df_sng, aes(Positive_Final_v2 * 100, TargetQ_Final_v1, fill = pmin(2,log2(SC_2nd_Target_Val + 1)))) +
	geom_point(pch = 21) +
	geom_label_repel(
	  data = sub_df,
	  aes(label = gene_name),
	  fill = "white",
	  max.overlaps = Inf,
	  size = 2
	) +
	facet_wrap(~Surface) +
	scale_fill_gradientn(colors = pal_cart2) +
	xlab("Positive Final (v2)") +
	ylab("Target Quality (v1)") +
	labs(fill = "Log2 Exp.")


pdf("TQ_vs_PP.pdf", width = 14, height = 8)

dev.off()



#' @export
plot_single_axial <- function(
    genes = NULL,
    malig_sng_med = NULL,
    ha_sng_med = NULL,
    show = 15,
    titles = genes
){

    require(scales)
    
    # Start overall timing
    overall_start <- Sys.time()

    #read in malig
    ts_message("Reading in malig...")
    if (!is.matrix(malig)) {
        malig <- as_dgCMatrix(malig[genes,]) %>% as.matrix
    } else {
        malig <- malig[genes,]
    }
    rownames(malig) <- genes
    patients <- stringr::str_split(colnames(malig), pattern="\\.\\_\\.", simplify=TRUE)[,2]

    #read in healthy_atlas
    ts_message("Reading in healthy atlas...")
    if (!is.matrix(healthy_atlas)) {
        ha <- as_dgCMatrix(healthy_atlas[genes,]) %>% as.matrix
    } else {
        ha <- healthy_atlas[genes,]
    }
    rownames(ha) <- genes
    CT <- sub("^([^:]+:[^:]+):.*", "\\1", colnames(ha))
    CT <- gsub(" |\\-", "_", CT)
    CT <- gsub("α", "a", CT)
    CT <- gsub("β", "B", CT)

    #Single Target Median
    malig_sng_med <- suppressMessages(summarize_matrix(malig, groups = patients, metric = "median"))
    ha_sng_med <- suppressMessages(summarize_matrix(ha, groups = CT, metric = "median"))

    #Positivity
    pos_mat <- compute_positivity_matrix(assays(se_malig[rownames(ha),])$ranks, assays(se_malig[rownames(ha),])$counts)
    pos_patients <- stringr::str_split(colnames(pos_mat), pattern="\\.\\_\\.", simplify=TRUE)[,2]
    stopifnot(all(pos_patients %in% patients))
    colnames(pos_mat) <- pos_patients
    pos_mat <- pos_mat[, unique(patients)]

    # Print timing after data preparation
    ts_message("Finished Data Summaries...")

    for(i in seq_along(genes)){

        ts_message("Processing ", i , " of ", length(genes), " : ", genes[i])
        gx <- genes[i]

        pos_1 <- names(which(pos_mat[gx, ]))

        #Malig
        df1 <- data.frame(
            pos_patient = ifelse(colnames(malig_sng_med) %in% pos_1, "Pos.", "Neg."),
            log2_exp = log2(malig_sng_med[gx,] + 1)
        )
        df1$pos_patient <- factor(df1$pos_patient, levels = c("Pos.", "Neg."))
        med1 <- median(df1$log2_exp[df1$pos_patient=="Pos."])

        #Healthy
        df1h <- data.frame(
            tissue = stringr::str_split(colnames(ha_sng_med), pattern = "\\:", simplify=TRUE)[,1],
            log2_exp = log2(ha_sng_med[gx,] + 1),
            full = sapply(colnames(ha_sng_med), add_smart_newlines) 
        )
        df1h$full <- factor(df1h$full, levels = df1h$full[order(df1h$log2_exp, decreasing=TRUE)])
        df1h <- df1h[order(df1h$full),]

        m <- max(df1$log2_exp)
        limits <- c(0, m)
        size <- 6

        per_1 <- round(length(pos_1)/ncol(pos_mat), 3) * 100

        suppressMessages({

            # Malignant boxplot
            p1_1 <- ggplot(df1, aes(pos_patient, log2_exp)) +
                theme_jg(xText90=TRUE) +
                theme_small_margin(0.01) +
                geom_boxplot(fill=NA, outlier.size = NA, outlier.stroke = NA, outlier.shape = NA, width = 0.5) +
                geom_jitter(size = 1.5, stroke = 0.35, pch = 21, aes(fill = pos_patient), color = "black", height = 0, width = 0.2) +
                theme(legend.position = "none") +
                coord_cartesian(ylim = limits) + 
                scale_x_discrete(drop = FALSE) + 
                xlab("") +
                geom_hline(yintercept = med1, lty = "dashed", color = "firebrick3") +
                scale_fill_manual(values = c("Pos." = "#28154C", "Neg." = "lightgrey")) +
                ylab(gx) + 
                ggtitle(paste0(per_1, "%"))

            # Healthy by tissue
            p1_2 <- ggplot(df1h, aes(tissue, squish(log2_exp, limits))) +
                theme_jg(xText90=TRUE) +
                theme_small_margin(0.01) +
                geom_boxplot(fill=NA, outlier.size = NA, outlier.stroke = NA, outlier.shape = NA, width = 0.5) +
                geom_jitter(size = 1.5, stroke = 0.35, pch = 21, aes(fill = tissue), color = "black", height = 0, width = 0.2) +
                theme(legend.position = "none") +
                coord_cartesian(ylim = limits) +
                geom_hline(yintercept = med1, lty = "dashed", color = "firebrick3") +
                xlab("") + ylab("") +
                theme(axis.text.x = element_text(size=size)) +
                scale_fill_manual(values = create_pal_d(df1h$tissue))

            # Top off-targets
            p1_3 <- ggplot(head(df1h, show), aes(full, squish(log2_exp, limits))) +
                theme_jg(xText90=TRUE) +
                theme_small_margin(0.01) +
                geom_point(size = 1.5, stroke = 0.35, pch = 21, aes(fill = tissue), color = "black") +
                theme(legend.position = "none") +
                coord_cartesian(ylim = limits) +
                geom_hline(yintercept = med1, lty = "dashed", color = "firebrick3") +
                xlab("") + ylab("") +
                theme(axis.text.x = element_text(size=size*0.75)) +
                scale_fill_manual(values = create_pal_d(df1h$tissue)) +
                ggtitle("Top Off Targets")

            # Combine with widths
            final_plot <- p1_1 + p1_2 + p1_3 + plot_layout(widths = c(1, 5, 3))

            print(final_plot)

        })

    }

    # Print final timing
    overall_elapsed <- as.numeric(difftime(Sys.time(), overall_start, units = "mins"))
    message("------  All plots completed - | Total: ", round(overall_elapsed, 1), "min")

}






















# Create plot
p <- ggplot(df, aes(x = Positive_Final_v2 * 100, 
                  y = TargetQ_Final_v1,
                  fill = pmin(log2(SC_2nd_Target_Val + 1), 2))) +
geom_point(pch = 21, size = 2, color = "black") +
geom_label_repel(
  data = sub_df,
  aes(label = gene_name),
  fill = "white",
  max.overlaps = Inf,
  size = 2
) +
labs(
  title = "Target Quality vs Positive Percentage",
  x = "Positive Percentage",
  y = "Target Quality Final",
  fill = "log2(SC 2nd Target + 1)"
) +
theme_bw()  # Replace with theme_jg() if you have it

ggsave(out, p, width = 8, height = 8, dpi = 600)


#Plot
pdf("TQ_vs_PP.pdf")

ggplot(df_sng[df_sng$Surface,], aes(Positive_Final_v2 * 100, TargetQ_Final_v1, fill = pmin(2,log2(SC_2nd_Target_Val + 1)))) +
	geom_point(pch = 21) 

dev.off()

df_sng[df_sng$Surface & df_sng$TargetQ_Final_v1 > 80 & df_sng$Positive_Final_v2 > 0.75, "gene_name"]


p1_tq_vs_pp(df_sng[df_sng$Surface,])

get_top_n_per_interval <- function(df, x_col, y_col, label_col, n = 2, interval = 10) {
  #' Get top N points per x-axis interval for labeling
  #' 
  #' @param df data.frame - Input dataframe
  #' @param x_col character - Column name for x-axis values (e.g., 'P_Pos_Per')
  #' @param y_col character - Column name for y-axis values (e.g., 'TargetQ_Final_v1')
  #' @param label_col character - Column name for labels (e.g., 'gene_name')
  #' @param n integer - Number of top points to select per interval
  #' @param interval numeric - Interval width as percentage (e.g., 10 for 10%)
  #' 
  #' @return data.frame - Subset with top N points per interval
  
  library(dplyr)
  
  # Scale x values to 0-100 range
  x_values <- df[[x_col]]
  x_min <- min(x_values)
  x_max <- max(x_values)
  x_scaled <- (x_values - x_min) / (x_max - x_min) * 100
  
  # Create bins
  bins <- seq(0, 100, by = interval)
  df_copy <- df
  df_copy$interval <- cut(x_scaled, 
                          breaks = bins, 
                          labels = bins[-length(bins)],
                          include.lowest = TRUE)
  
  # Get top N per interval
  top_per_interval <- df_copy %>%
    group_by(interval, .drop = FALSE) %>%
    slice_max(order_by = .data[[y_col]], n = n, with_ties = FALSE) %>%
    ungroup() %>%
    select(all_of(c(x_col, y_col, label_col)))
  
  return(as.data.frame(top_per_interval))
}

# Main plotting function
p1_tq_vs_pp <- function(df, out = "plot.pdf", target_q = 75, ppos = 25) {
  library(ggplot2)
  library(ggrepel)
  library(dplyr)
  
  # Subset high-quality targets for labeling
  g1 <- get_top_n_per_interval(
    df,
    x_col = 'Positive_Final_v2',
    y_col = 'TargetQ_Final_v1',
    label_col = 'gene_name',
    n = 2,
    interval = 10
  )[[3]]  # Extract gene_name column (3rd column)
  
  g2 <- df %>%
    filter(TargetQ_Final_v1 >= target_q & Positive_Final_v2 * 100 >= ppos) %>%
    pull(gene_name)
  
  genes <- unique(c(g1, g2))
  sub_df <- df %>% filter(gene_name %in% genes)
  
  # Create plot
  p <- ggplot(df, aes(x = Positive_Final_v2 * 100, 
                      y = TargetQ_Final_v1,
                      fill = pmin(log2(SC_2nd_Target_Val + 1), 2))) +
    geom_point(pch = 21, size = 2, color = "black") +
    geom_label_repel(
      data = sub_df,
      aes(label = gene_name),
      fill = "white",
      max.overlaps = Inf,
      size = 2
    ) +
    labs(
      title = "Target Quality vs Positive Percentage",
      x = "Positive Percentage",
      y = "Target Quality Final",
      fill = "log2(SC 2nd Target + 1)"
    ) +
    theme_bw()  # Replace with theme_jg() if you have it
  
  ggsave(out, p, width = 8, height = 8, dpi = 600)
  
  return(p)
}