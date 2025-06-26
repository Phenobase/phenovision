library(tidyverse)
library(arrow)

test_dat <- read_rds("output/model_04_13_2024/epoch_4_testing_data.rds")

img_meta <- open_dataset("data/phenobase_inat_data/metadata/angio_photos")

test_dat <- test_dat |>
  filter(partition == "testing")

meta <- img_meta |>
  filter(photo_id %in% test_dat$photo_id) |>
  collect()

test_dat <- test_dat |>
  left_join(meta |>
              select(photo_id, longitude, latitude) |>
              mutate(photo_id = as.numeric(photo_id)))

write_rds(test_dat, "output/model_04_13_2024/epoch_4_testing_data_only_w_coords.rds")
