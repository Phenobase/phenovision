library(tidyverse)
library(data.table)

source("R/ml_annotation_postprocess_targets.R")

inf_files <- list.files("/blue/guralnick/share/phenobase_inat_data/annotations/9cf8ad8", full.names = TRUE,
                        pattern = ".csv")
test_dat <- read_rds("output/epoch_4_testing_data.rds")

counts <- test_dat |>
  group_by(partition) |>
  summarize(count = n())

inat_train <- read_csv("data/inat/train.csv")
inat_val <- read_csv("data/inat/validation.csv")
inat_test <- read_csv("data/inat/test.csv")

inat_dat <- bind_rows(inat_train, inat_val, inat_test) |>
  select(photo_id, family, genus, flowering, fruiting) |>
  pivot_longer(c(flowering, fruiting), names_to = "trait", values_to = "class")

inat_ids <- unique(inat_dat$photo_id)

inat_dat_pres <- inat_dat |>
  filter(class == 1)

meta_images <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/angio_photos")
meta_obs <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/observations/part-0.parquet")

inat_dat_pres <- inat_dat_pres |>
  mutate(photo_id = as.character(photo_id)) |>
  left_join(meta_images |>
              select(photo_id, observation_uuid) |>
              filter(photo_id %in% inat_dat_pres$photo_id),
            by = "photo_id",
            copy = TRUE)

inat_dat_pres <- inat_dat_pres |>
  left_join(meta_obs |>
              select(observation_uuid, latitude, longitude, observed_on) |>
              filter(observation_uuid %in% inat_dat_pres$observation_uuid),
            by = "observation_uuid",
            copy = TRUE)

#file_name <- inf_files[1]
get_coverage_data <- function(file_name, inat_ids) {
  dat <- read_csv(file_name)
  dat <- dat |>
    mutate(photo_id = str_remove(observed_image_guid, "https://inaturalist-open-data.s3.amazonaws.com/photos/") |>
             str_remove("/small") |>
             fs::path_ext_remove())

  dat <- dat |>
    filter(!photo_id %chin% as.character(inat_ids)) |>
    select(photo_id, family, genus, latitude, longitude, observed_on = verbatim_date, trait)

  dat

}

ml_dat <- map(inf_files, ~ get_coverage_data(.x, inat_ids), .progress = TRUE) |>
  list_rbind()

all_dat <- bind_rows(
  ml_dat |>
    mutate(post_ml_only = TRUE),
  inat_dat_pres |>
    select(photo_id, family, genus, latitude, longitude, observed_on, trait) |>
    mutate(post_ml_only = FALSE)
)

write_csv(all_dat, "output/9cf8ad8/data_for_coverage_analysis_9cf8ad8.csv")
