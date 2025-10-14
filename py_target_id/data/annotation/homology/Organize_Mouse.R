library(data.table)
library(Rsamtools)
library(seqinr)
library(dplyr)

splitEvery <- function (x = NULL, n = NULL){
    if (is.atomic(x)) {
        split(x, ceiling(seq_along(x)/n))
    }
    else {
        split(x, ceiling(seq_len(nrow(x))/n))
    }
}

#Load
fa <- readAAStringSet("gencode.vM32.pc_translations.fa.gz")

#3line
line3 <- list.files("../results", pattern = "mouse", full.names = TRUE)

#Read
dt_all <- lapply(seq_along(line3), function(x){
    message(x, ' of ', length(line3))
    o <- readr::read_lines(line3[x])
    lapply(splitEvery(o, 3), function(y){
        data.table(y[1], y[2], y[3])
    }) %>% rbindlist
}) %>% rbindlist

#Check
stopifnot(all(paste0(fa) %in% dt_all$V2))

# #Filter
# fa2 <- fa[!(paste0(fa) %in% dt_all$V2)]
# fa2 <- unique(fa2)

# #Write
# split <- splitEvery(seq_along(fa2), 120)
# o <- lapply(seq_along(split), function(x){

#     fa2[split[[x]]] %>%
#         {write.fasta(
#             sequences = stringr::str_split(paste0(.), pattern = ""),
#             names = names(.),
#             file.out = file.path("seqs", paste0("mouse_", x, ".fa"))
#         )}

# })

#Create Table
idx <- match(paste0(fa), dt_all$V2)
dt_m <- dt_all[idx]
stopifnot(all(dt_m$V2==paste0(fa)))
split <- stringr::str_split(gsub(">| ","",dt_m$V1), pattern = "\\|", simplify=TRUE)
df <- DataFrame(
    transcript_id = split[,2],
    gene_id = split[,3],
    gene_name = split[,7],
    type = split[,9],
    AA = paste0(fa),
    Predict = dt_m$V3
)
df$Predict_Summary <- lapply(stringr::str_split(df$Predict, pattern= ""), function(x){
    r <- Rle(x)
    paste0(r@lengths, r@values,collapse=".")
}) %>% unlist
df$n_membrane <- letterFrequency(BStringSet(df$Predict_Summary), letters = LETTERS)[,"M"]
df$N_Membrane <- letterFrequency(BStringSet(df$Predict_Summary), letters = LETTERS)[,"M"]
df$N_ECD <- letterFrequency(BStringSet(df$Predict_Summary), letters = LETTERS)[,"O"]
#df$N_ECD[df$N_Membrane==0] <- 0
split <- stringr::str_split(df$Predict_Summary, pattern = "\\.")
df$Min_ECD <- lapply(split, function(x){
    min(as.numeric(gsub("O", "", grep("O", x, value=TRUE))))
}) %>% unlist
df$Max_ECD <- lapply(split, function(x){
    max(as.numeric(gsub("O", "", grep("O", x, value=TRUE))))
}) %>% unlist
df$Sum_ECD <- lapply(split, function(x){
    sum(as.numeric(gsub("O", "", grep("O", x, value=TRUE))))
}) %>% unlist
df$Min_ECD[is.infinite(df$Min_ECD)] <- NA
df$Max_ECD[is.infinite(df$Max_ECD)] <- NA
df$Sum_ECD[is.infinite(df$Sum_ECD)] <- NA

saveRDS(df, "Transcript_Topology_Summary.Mouse_vM32.rds")






