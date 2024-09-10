library(tidyverse)
source("R/concatenate_annotations_targets.R")

path <- "/blue/guralnick/share/phenobase_inat_data/annotations/9cf8ad8/reduced"

csvs <- list.files(path, full.names = TRUE)
big_file <- file.path(path, "annotations_all.csv")

concatenate_annotations(csvs, big_file)
