## DEPRECATED: Use prepare_rob_bb_annotations() in R/extract_leaf_annotations_download.R
## and the rob_annot2_file target in _targets_download_annots.R instead.
## This script is kept for reference only.

library(tidyverse)
library(unglue)
library(arrow)

source("R/ml_annotation_postprocess_targets.R")

files <- list.files("output/leaves/phenovision-init_model_02_12_2025/image_annotation_test", full.names = TRUE, recursive = TRUE)
rob_annot2 <- read_csv("data/leaves/bbPresenceScoringRescoreAllStates.csv")

file_data <- unglue_data(files,
                         "output/leaves/phenovision-init_model_02_12_2025/image_annotation_test/{part}/{detect}/{File}.{extension}",
                         convert = TRUE)

file_data <- file_data |>
  left_join(rob_annot2) |>
  drop_na(Detect_Agree) |>
  filter(part == "bb") |>
  mutate(bb = Detect_Agree, photo_id = as.character(File))

photo_meta <- open_dataset("data/phenobase_inat_data/metadata/angio_photos")
photo_meta <- photo_meta |>
  select(observation_uuid, photo_uuid, photo_id, batch_j, extension, taxon_id) |>
  filter(photo_id %in% unique(file_data$photo_id)) |>
  collect()

file_data <- file_data |>
  left_join(photo_meta)

taxa_meta <- open_dataset("data/phenobase_inat_data/metadata/taxa/part-0.parquet")

file_data <- add_taxonomy(file_data, taxa_meta)
file_data <- file_data |>
  mutate(file_name = file.path("data/phenobase_inat_data/images/medium", paste0("batch_", batch_j), paste0(photo_id, ".webp")))

write_csv(file_data, "data/leaves/rob_new_annotations_bb.csv")

