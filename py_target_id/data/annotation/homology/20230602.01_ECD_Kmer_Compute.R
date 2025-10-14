library(Biostrings)
library(IRanges)
library(data.table)
library(rhdf5)
library(stringdist)
library(dplyr)
library(Matrix)
library(Rcpp)
library(RcppArmadillo)
library(stringr)
library(jgplot2)
sourceCpp("string.cpp")

get_ecd_kmers <- function(AA = NULL, P = NULL, K = 8){
	idx <- which(str_split(P, pattern = "")[[1]] %in% c("S", "O"))
	windows <- IRanges(idx, width=1L) %>% reduce
	slid_windows <- slidingWindows(windows, width = K, step = 1L) %>% unlist
	slid_windows <- slid_windows[width(slid_windows)==K]
	data.table(
		Width = nchar(AA),
		Start = start(slid_windows),
		End = end(slid_windows),
		ECD = paste0(Views(AA, slid_windows))
	)
}

gpi <- c(
	"IZUMO1R", "GFRA2", "TNFRSF10C", "TREH", "PSCA", "XPNPEP2",
	"EFNA2", "GFRA3", "FCGR3B", "SEMA7A", "TECTA", "GPC4", "CNTN5",
	"LYPD3", "VNN1", "VNN2", "CD160", "RECK", "PRNP", "THY1", "ALPL",
	"ALPP", "CEACAM5", "LPL", "UMOD", "CD55", "CD14", "MELTF", "CD48",
	"ALPI", "CFC1", "TFPI", "ALPG", "TDGF1", "NCAM1", "CD59", "FOLR2",
	"CPM", "FOLR1", "DPEP1", "CD58", "EFNA1", "NT5E", "ACHE", "CA4",
	"OMG", "CNTFR", "CD52", "CEACAM8", "GPC1", "SPAM1", "CEACAM6",
	"GPC3", "EFNA3", "EFNA4", "EFNA5", "ART1", "GAS1", "GP2", "CDH13",
	"GFRA1", "GPC5", "CNTN2", "PLAUR", "BST1", "BST2", "CNTN1", "HYAL2",
	"MSLN", "LSAMP", "ART3", "CEACAM7", "LY6D", "OPCML", "LY6K",
	"NRN1L", "SPRN", "RAET1L", "RAET1G", "RGMB", "PLET1", "LYPD4",
	"LYPD5", "ENPP6", "LYPD2", "CD109", "HJV", "OTOA", "NEGR1", "MDGA2",
	"RTN4RL1", "RTN4RL2", "GPIHBP1", "CNTN4", "GPC2", "LYPD1", "CD177",
	"MDGA1", "LYPD6B", "SPACA4", "ITLN1", "ART4", "RGMA", "BCAN",
	"TECTB", "GML", "TEX101", "LYNX1", "ULBP3", "ULBP2", "ULBP1",
	"RTN4R", "GFRA4", "DPEP2", "DPEP3", "MMP25", "NRN1", "VNN3",
	"NTM", "CNTN3", "PRND", "MMP17", "CNTN6", "NTNG1", "GPC6", "PRSS21"
)

#Get Surface Genes
surface <- rownames(readRDS("~/data/Pipelines/Target_ID_Single_v2/Inputs/Annotations/Surface_Gene_Combo_Table.0809.rds"))
surface <- c(surface, gpi) %>% unique

###########
# Human
###########

df <- readRDS("../human/Transcript_Topology_Summary.Human_v29.rds")

#Subset
df <- df[df$gene_name %in% surface,]
idx <- letterFrequency(BStringSet(df$Predict), letters = LETTERS)[, "O"] >= 1
df <- df[idx, ]

#Split
sdf <- split(df, seq(nrow(df)))

#Compute
dt <- parallel::mclapply(sdf, function(x){
	dtx <- get_ecd_kmers(
		AA = x$AA[1],
		P = x$Predict[1],
		K = 8
	)
	dtx$gene_name <- paste0(x$gene_name)
	dtx$transcript_id <- x$transcript_id
	dtx
}, mc.cores = 64, mc.preschedule = TRUE) %>% rbindlist

saveRDS(dt, "Human_ECD_8mer_DT.rds")

###########
# Mouse
###########

df <- readRDS("../mouse/Transcript_Topology_Summary.Mouse_vM32.rds")

#Subset
df <- df[toupper(df$gene_name) %in% surface | grepl("M", df$Predict),]
idx <- letterFrequency(BStringSet(df$Predict), letters = LETTERS)[, "O"] >= 1
df <- df[idx, ]

#Split
sdf <- split(df, seq(nrow(df)))

#Compute
dt <- parallel::mclapply(sdf, function(x){
	dtx <- get_ecd_kmers(
		AA = x$AA[1],
		P = x$Predict[1],
		K = 8
	)
	dtx$gene_name <- paste0(x$gene_name)
	dtx$transcript_id <- x$transcript_id
	dtx
}, mc.cores = 64, mc.preschedule = TRUE) %>% rbindlist

saveRDS(dt, "Mouse_ECD_8mer_DT.rds")

###########
# Cyno
###########

df <- readRDS("../cyno/Transcript_Topology_Summary.Cyno_v6.0.rds")

#Subset
df <- df[toupper(df$gene_name) %in% surface | grepl("M", df$Predict),]
idx <- letterFrequency(BStringSet(df$Predict), letters = LETTERS)[, "O"] >= 1
df <- df[idx, ]

#Split
sdf <- split(df, seq(nrow(df)))

#Compute
dt <- parallel::mclapply(sdf, function(x){
	dtx <- get_ecd_kmers(
		AA = x$AA[1],
		P = x$Predict[1],
		K = 8
	)
	dtx$gene_name <- paste0(x$gene_name)
	dtx$transcript_id <- x$transcript_id
	dtx
}, mc.cores = 64, mc.preschedule = TRUE) %>% rbindlist

saveRDS(dt, "Cyno_ECD_8mer_DT.rds")






















#Known Targets
targets <- read.table("~/data/Pipelines/Target_ID_Single/Custom/CB.txt")[1,]
targets <- as.vector(unlist(targets))
ix <- which(dt$gene_name %in% targets)

#Compute
dtx <- parallel::mclapply(seq_along(ix), function(j){

	message(j, " of ", length(ix))

	#Set Value
	i <- ix[j]

	#Compute Distances
	dist <- pairwise_lv_distances(
		strings1 = dt$ECD, 
		strings2 = dt$ECD[i], 
		max_distance = 1L
	)

	#Set Value To 0 For Self
	dist[i, 1] <- 0

	#Identify
	ii <- dist@i[dist@x > 0] + 1L
	gi <- unique(dt$gene_name[ii])
	gi <- gi[!(gi %in% dt$gene_name[i])]

	if(is.na(dt$ECD[i])){
		gi <- c()
	}

	#Summary
	data.table(
		GX = dt$gene_name[i],
		TX = dt$transcript_id[i],
		Width = dt$Width[i],
		Start = dt$Start[i],
		ECD = dt$ECD[i],
		N_Genes = length(gi),
		Genes = paste0(head(gi, 25), collapse = ";")
	)

}, mc.cores = 64, mc.preschedule = FALSE) %>% rbindlist

#OutDir
outDir <- "Outputs_CB_8mer_1mis_All_Outer_v2"
dir.create(outDir)
saveRDS(dtx, file.path(outDir, "CB_8mer_1mis_All_Outer_v2.rds"))

#Plot
tx <- unique(dtx$TX)

for(i in seq_along(tx)){

	message(i)

	#TX
	txi <- tx[i]
	txi2 <- str_split(txi, pattern="\\.")[[1]][[1]]
	gi <- paste0(df[df$transcript_id==txi,"gene_name"])

	#Subset
	dtxi <- dtx[dtx$TX == txi]

	#Topology
	p <- str_split(df[df$transcript_id==txi,"Predict"], pattern="")[[1]]
	top <- data.table(
		x = seq_along(p),
		y = match(p, c("I", "M", "O", "S"))
	)

	#Subset
	split_pos <- str_split(dtxi$Genes, pattern = ";")
	tab <- sort(table(unlist(split_pos)), decreasing=TRUE)
	tab <- tab[names(tab) != ""]
	tab <- sort(tab, decreasing=TRUE)

	if(length(tab)==0){
		tab <- data.frame(
			Var1 = "No_Genes",
			Freq = 0
		)
		m <- data.frame(
			Var2 = "No_Genes",
			Var1 = 1,
			value = 1
		)
	}else{
		tab <- data.frame(
			Var1 = factor(names(tab), levels = names(tab)),
			Freq = as.vector(tab)
		)
		tab <- head(tab, 50)
		#Matrix
		m0 <- lapply(split_pos, function(j){
			1*as.vector(paste0(tab$Var1) %in% j)
		}) %>% Reduce("rbind",.)
		colnames(m0) <- tab$Var1
		rownames(m0) <- dtxi$Start #seq_len(nrow(m))
		m <- reshape2::melt(m0)
		m$Var2 <- factor(m$Var2, rev(colnames(m0)))
	}


	#Plot
	pdf(file.path(outDir, paste0(gi, "_", txi2,"_8mer_1mis.pdf")), width = 10, height = 8)

	p1 <- ggplot() +
		geom_line(
			data = top, aes(x, y), 
			color = "#6BC291",
			size = 1
		) +
		scale_x_continuous(limits = c(0, dtxi$Width[1]), breaks = scales::pretty_breaks(n = 10)) +
		scale_y_continuous(breaks = seq(1, 4), limits = c(1, 4), labels = c("Inner", "Membrane", "Outer", "Signal")) +
		theme(panel.grid.minor = element_blank()) +
		xlab("AA Position") +
		theme(plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm")) +
		theme(axis.text.x = element_blank(), axis.title = element_blank()) +
		ggtitle(paste0(gi, " ", txi))

	p2 <- ggplot() +
		geom_point(
			data = dtxi, aes(Start, pmin(N_Genes, 10)), 
			color = "#2E95D2",
			#fill = "firebrick3", 
			size = 1, 
			#pch = 21,
			alpha = 0.6
		) +
		scale_x_continuous(limits = c(0, dtxi$Width[1]), breaks = scales::pretty_breaks(n = 10)) +
		scale_y_continuous(breaks = seq(0, 10), limits = c(0, 10), labels = c(0:9,"10+")) +
		theme(panel.grid.minor = element_blank()) +
		ylab("Number of Genes\nLV Distance = 0,1") +
		xlab("AA Position") +
		theme(plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm")) +
		theme(axis.text.x = element_blank(), axis.title.x = element_blank()) 

	text <- 6
	if(nrow(tab) > 15){
		text <- 6
	}else if(nrow(tab) > 8){
		text <- 8
	}else{
		text <- 10
	}

	p3 <- ggplot() +
		geom_point(data = m[m$value > 0,], aes(Var1, Var2), color = "#28154C", size = 1, alpha = 0.6) +
		xlab("AA Position") +
		ylab("Genes With Overlapping Kmers") +
		theme(plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm")) +
		theme(axis.text.y = element_text(size = text)) +
		scale_x_continuous(limits = c(0, dtxi$Width[1]), breaks = scales::pretty_breaks(n = 10)) +
		theme(panel.grid.minor = element_blank()) 

	suppressWarnings(print(p1 / p2 / p3 + plot_layout(heights = c(4, 4, 8))))

	dev.off()	

}