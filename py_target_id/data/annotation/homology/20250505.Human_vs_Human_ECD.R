library(rhdf5)
library(parallel)
library(Rcpp)
library(data.table)
library(fastmatch)
library(dplyr)

# Rcpp variant generator
Rcpp::cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
CharacterVector generate_edit_distance_1_kmers_cpp(std::string kmer, CharacterVector amino_acids) {
  std::vector<std::string> variants;
  int kmer_length = kmer.length();
  
  // Substitutions
  for (int i = 0; i < kmer_length; i++) {
    for (int j = 0; j < amino_acids.size(); j++) {
      char aa = as<std::string>(amino_acids[j])[0];
      if (kmer[i] != aa) {
        std::string variant = kmer;
        variant[i] = aa;
        variants.push_back(variant);
      }
    }
  }
  
  // Insertions
  for (int i = 0; i <= kmer_length; i++) {
    for (int j = 0; j < amino_acids.size(); j++) {
      char aa = as<std::string>(amino_acids[j])[0];
      std::string variant = kmer.substr(0, i) + aa + kmer.substr(i);
      if (variant.length() >= 8) {
        variants.push_back(variant.substr(0, 8));
        variants.push_back(variant.substr(1, 8));
      }
    }
  }

  // Deletions
  for (int i = 0; i < kmer_length; i++) {
    std::string core = kmer.substr(0, i) + kmer.substr(i + 1);
    if (core.length() == 7) {
      for (int j = 0; j < amino_acids.size(); j++) {
        char aa = as<std::string>(amino_acids[j])[0];
        variants.push_back(std::string(1, aa) + core);
        variants.push_back(core + std::string(1, aa));
      }
    }
  }

  std::set<std::string> unique_variants(variants.begin(), variants.end());
  return CharacterVector(unique_variants.begin(), unique_variants.end());
}
')

# Load & Prepare
dth <- readRDS("Human_ECD_8mer_DT.rds")

# Get Info
dtq <- dth
dtlib <- dth
ecds_query <- unique(dtq$ECD)
ecds_library <- unique(dtlib$ECD)  # Assuming this is your lookup space
amino_acids <- strsplit("ACDEFGHIKLMNPQRSTVWXY", "")[[1]]

#Filter
ecds_query <- ecds_query[!is.na(ecds_query)]
ecds_library <- ecds_library[!is.na(ecds_library)]

# Split query into chunks
split_query <- split(ecds_query, ceiling(seq_along(ecds_query)/10000))

# Pre-index library for fast lookup
ecds_library_hash <- fastmatch::fmatch

# Process Function (Per Chunk)
process_chunk <- function(chunk, amino_acids, library_set, chunk_id) {
  message("Processing chunk ", chunk_id, " of ", length(split_query))
  
  var_list <- mclapply(chunk, function(kmer) {
    vars <- generate_edit_distance_1_kmers_cpp(kmer, amino_acids)
    valid_idx <- fmatch(vars, library_set)
    valid_vars <- vars[!is.na(valid_idx)]
    if (length(valid_vars)) {
      data.table(kmer = kmer, vars = valid_vars)
    } else {
      NULL
    }
  }, mc.cores = 64)

  result <- rbindlist(var_list, use.names = TRUE, fill = TRUE)
  
  # Optional: save each chunk
  #saveRDS(result, sprintf("results_chunk_%02d.rds", chunk_id))
  
  return(result)
}

# Run all chunks
results_list <- mapply(process_chunk, 
                       chunk = split_query, 
                       chunk_id = seq_along(split_query),
                       MoreArgs = list(amino_acids = amino_acids, library_set = ecds_library),
                       SIMPLIFY = FALSE)

# Combine
final_result <- rbindlist(results_list, use.names = TRUE, fill = TRUE)
final_result <- final_result[order(kmer)]

#Check
final_result <- final_result[stringdist::stringdist(final_result$kmer, final_result$vars, method = "lv") <= 1]

#Let's map back then
lib_ecd_map <- dtlib[, .(gene_names = paste(unique(gene_name), collapse = ";")), by = ECD]
colnames(lib_ecd_map)[1] <- "vars"
final_result2 <- left_join(final_result, lib_ecd_map, by = "vars")
final_result2$vars <- NULL

final_result3 <- final_result2[, .(gene_names = paste(unique(gene_names), collapse = ";")), by = kmer]
final_result3 <- final_result3[, .(
  gene_names = paste(sort(unique(unlist(strsplit(gene_names, ";")))), collapse = ";")
), by = kmer]
colnames(final_result3)[1] <- "ECD"
final_result4 <- left_join(dtq, final_result3)

final_result4[, filtered_gene_names := sapply(1L:.N, function(i) {
  all_genes <- unique(unlist(strsplit(gene_names[i], ";")))
  filtered <- setdiff(all_genes, gene_name[i])
  if (length(filtered) > 0) paste(filtered, collapse = ";") else ""
})]

#Summarize
out <- data.table(
  GX = final_result4$gene_name,
  TX = final_result4$transcript_id,
  Width = final_result4$Width,
  Start = final_result4$Start,
  ECD = final_result4$ECD,
  N_Genes = stringr::str_count(final_result4$filtered_gene_names, ";") + 1,
  Genes = final_result4$filtered_gene_names
)
out$N_Genes[out$Genes==""] <- 0

saveRDS(out, "human_vs_human.8mer.20250505.rds")
