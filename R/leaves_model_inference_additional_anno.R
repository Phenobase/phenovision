library(tidyverse)
library(unglue)
library(arrow)

files <- list.files("output/leaves/phenovision-init_model_02_12_2025/image_annotation_test", full.names = TRUE, recursive = TRUE)
rob_annot2 <- read_csv("data/leaves/Rob_phenovision_bbPresenceScoring2.csv")

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
  select(observation_uuid, photo_uuid, photo_id, batch_j, extension) |>
  filter(photo_id %in% unique(file_data$photo_id)) |>
  collect()

file_data <- file_data |>
  left_join(photo_meta)

write_csv(file_data, "data/leaves/rob_new_annotations_bb.csv")

