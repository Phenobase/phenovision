library(tidyverse)
library(arrow)

image_dir <- "/blue/guralnick/share/phenobase_inat_data/images/medium"
test_dir <- "/blue/guralnick/share/phenobase_inat_data/inference_test"

inf_dat <- read_csv("output/model_04_13_2024/inference_results.csv")
train_dat <- read_csv("data/inat/train.csv")

inf_ids <- fs::path_ext_remove(inf_dat$file_name)
sum(inf_ids %in% train_dat$photo_id)
inf_dat <- inf_dat |>
  filter(!inf_ids %in% train_dat$photo_id)

angio_meta <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/angio_photos")

#metadat_csvs <- list.files("/blue/guralnick/share/phenobase_inat_data/metadata/angio_photos_csv", full.names = TRUE)

#photo_meta <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/photos/part-0.parquet")

set.seed(45897)

inf_fl <- inf_dat |>
  group_by(.class_flower, .equivocal_flower) |>
  slice_sample(n = 500) |>
  mutate(full_file = file.path(image_dir, file_name),
         save_file = str_remove_all(file.path(test_dir, "flower", .class_flower, .equivocal_flower, file_name), "[:space:]"))

inf_fr <- inf_dat |>
  group_by(.class_fruit, .equivocal_fruit) |>
  slice_sample(n = 500) |>
  mutate(full_file = file.path(image_dir, file_name),
         save_file = str_remove_all(file.path(test_dir, "fruit", .class_fruit, .equivocal_fruit, file_name), "[:space:]"))

map2(inf_fl$full_file, inf_fl$save_file,
     ~ file.copy(.x, .y),
     .progress = TRUE)

map2(inf_fr$full_file, inf_fr$save_file,
     ~ file.copy(.x, .y),
     .progress = TRUE)

sample_dataset <- bind_rows(inf_fl, inf_fr) |>
  distinct(file_name, .keep_all = TRUE) |>
  mutate(photo_id = as.integer(fs::path_ext_remove(file_name)))

sample_dataset <- sample_dataset |>
  left_join(angio_meta |>
              select(photo_id, observation_uuid) |>
              filter(photo_id %in% sample_dataset$photo_id),
            by = "photo_id",
            copy = TRUE)

sample_dataset <- sample_dataset |>
  mutate(inat_URL = paste0("https://www.inaturalist.org/observations/", observation_uuid))

metadat <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/observations/part-0.parquet")

sample_dataset <- sample_dataset |>
  left_join(metadat |>
              select(observation_uuid, latitude, longitude, positional_accuracy, taxon_id, quality_grade, observed_on) |>
              filter(observation_uuid %in% sample_dataset$observation_uuid),
            by = "observation_uuid",
            copy = TRUE)

taxa_meta <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/taxa/part-0.parquet")

sample_dataset <- sample_dataset |>
  left_join(taxa_meta |>
              select(taxon_id, name, ancestry, rank_level, rank) |>
              filter(taxon_id %in% sample_dataset$taxon_id),
            by = "taxon_id",
            copy = TRUE)

write_csv(sample_dataset, file.path(test_dir, "sample_inference_dataset.csv"))

sample_dataset |>
  group_by(observation_uuid) |>
  summarise(count = n()) |>
  filter(count > 1)

sample_dataset |>
  group_by(photo_id) |>
  summarise(count = n()) |>
  filter(count > 1)

sample_dataset |>
  filter(observation_uuid == "2346d329-d97e-4469-8b69-cbfc14071ab0")

tt <- sample_dataset |>
  filter(photo_id == 2315170)

