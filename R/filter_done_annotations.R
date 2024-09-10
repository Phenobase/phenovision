library(tidyverse)
library(data.table)

inf_files <- list.files("/blue/guralnick/share/phenobase_inat_data/annotations/9cf8ad8", full.names = TRUE)
melissa_anno <- read_csv("output/9cf8ad8/melissa_inference_results_9cf8ad8.csv")

done <- melissa_anno$observed_image_guid
#file <- inf_files[1]
remove_done <- function(file_name, done_list) {
  dat <- read_csv(file_name)
  dat <- dat |>
    filter(!observed_image_guid %chin% done_list)
  new_file <- gsub("annotations/9cf8ad8", "annotations/9cf8ad8/reduced", file_name)
  write_csv(dat, new_file)

}

walk(inf_files, ~ remove_done(.x, done), .progress = TRUE)
