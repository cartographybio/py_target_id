library(dplyr)

o1 = system("gsutil ls gs://cartography_target_id_samples/Samples_v3/processed", intern = TRUE)
o2 = system("gsutil ls gs://cartography_target_id_package/Sample_Input/20251008/processed", intern = TRUE)# Copy files from o1 to o2

for (i in seq_along(o1)) {
  src <- o1[i]
  dst <- o2[i]
  
  dir_name <- basename(gsub("/$", "", src))
  cat(sprintf("Copying files from %s to %s...\n", dir_name, basename(gsub("/$", "", dst))))
  
  cmd <- sprintf("gsutil -m cp '%s*' '%s'", src, dst)
  exit_code <- system(cmd)
  
  if (exit_code == 0) {
    cat(sprintf("✓ Successfully copied files from %s\n", dir_name))
  } else {
    cat(sprintf("✗ Error copying from %s (exit code: %d)\n", dir_name, exit_code))
  }
}

cat("\nSync complete!\n")


library(dplyr)

o1 = system("gsutil ls gs://cartography_target_id_samples/Samples_v3", intern = TRUE)
o2 = system("gsutil ls gs://cartography_target_id_package/Sample_Input/20251008", intern = TRUE)# Copy files from o1 to o2

o1 = o1[grep("arxiv", o1)]
o2 = o2[grep("arxiv", o2)]

# Copy files from o1 to o2
for (i in seq_along(o1)) {
  src <- o1[i]
  dst <- o2[i]
  
  dir_name <- basename(gsub("/$", "", src))
  cat(sprintf("Copying files from %s to %s...\n", dir_name, basename(gsub("/$", "", dst))))
  
  cmd <- sprintf("gsutil -m cp -r '%s*' '%s'", src, dst)
  exit_code <- system(cmd)
  
  if (exit_code == 0) {
    cat(sprintf("✓ Successfully copied files from %s\n", dir_name))
  } else {
    cat(sprintf("✗ Error copying from %s (exit code: %d)\n", dir_name, exit_code))
  }
}

cat("\nSync complete!\n")