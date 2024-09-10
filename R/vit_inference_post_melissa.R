library(tidyverse)
library(arrow)
library(tidymodels)
library(probably)

source("R/ml_annotation_postprocess.R")
source("R/model_loading.R")

vers <- model_version()
inf_dat <- read_csv(file.path("output", vers, "melissa_inference_results.csv"))

corrupted <- inf_dat |>
  filter(corrupt)

inf_dat <- inf_dat |>
  filter(!corrupt)

inf_dat <- threshold_model_output(inf_dat)

meta_images <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/angio_photos")
meta_obs <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/observations/part-0.parquet")
meta_taxa <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/taxa/part-0.parquet")

fam_stats <- read_csv("output/model_04_13_2024/family_stats.csv")

inf_dat <- convert_to_ingestion(inf_dat, meta_images = meta_images, meta_obs = meta_obs,
                                meta_taxa = meta_taxa, fam_stats = fam_stats)

inf_dat <- inf_dat |>
    drop_na(latitude, longitude)

model_vers <- model_version()

vers_path <- file.path("output", model_vers)
if(!dir.exists(vers_path)) {
  dir.create(vers_path)
}

csv_path <- file.path(vers_path, paste0("melissa_inference_results_", model_vers, ".csv"))

write_csv(inf_dat, csv_path)

corrupted_photos <- corrupted |>
  mutate(photo_id = fs::path_ext_remove(file_name),
         extension = fs::path_ext(file_name))
         
corrupted_photos <- corrupted_photos |>
  left_join(meta_images |>
    select(photo_id, batch_j) |>
    filter(photo_id %in% corrupted_photos$photo_id),
  copy = TRUE)

corrupted_photos <- corrupted_photos |>
  drop_na(batch_j) |>
  mutate(path = file.path("/blue/guralnick/share/phenobase_inat_data/images/medium",
  paste0("batch_", batch_j),
  paste0(photo_id, ".", extension)))

write_csv(corrupted_photos |> select(path),
          file.path(vers_path, "corrupted_photos.csv"), append = TRUE)