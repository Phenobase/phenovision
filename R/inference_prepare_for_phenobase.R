## this script takes raw output from phenovision model, creates prediction by thresholding,
## adds metadata,

library(tidyverse)
library(arrow)
library(tidymodels)
library(probably)

image_dir <- "/blue/guralnick/share/phenobase_inat_data/images/medium"
test_dir <- "/blue/guralnick/share/phenobase_inat_data/inference_test"

inf_dat <- read_csv("output/model_04_13_2024/inference_results_raw.csv")
train_dat <- read_csv("data/inat/train.csv")

inf_ids <- fs::path_ext_remove(inf_dat$file_name)
sum(inf_ids %in% train_dat$photo_id)
inf_dat <- inf_dat |>
  filter(!inf_ids %in% train_dat$photo_id)

angio_meta <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/angio_photos")

##### use new thresholds
inf_dat <- inf_dat |>
  mutate(.class_flower = make_two_class_pred(.pred_flower, c("Detected", "Not Detected"),
                                             threshold = 0.84,
                                             buffer = c(0.56, 0.01)),
         .class_fruit = make_two_class_pred(.pred_fruit, c("Detected", "Not Detected"),
                                            threshold = 0.53,
                                            buffer = c(0.3, 0.22)),
         .equivocal_flower = ifelse(is_equivocal(.class_flower), "Equivocal", "Unequivocal"),
         .equivocal_fruit = ifelse(is_equivocal(.class_fruit), "Equivocal", "Unequivocal")) |>
  mutate(.class_flower = make_two_class_pred(.pred_flower, c("Detected", "Not Detected"),
                                             threshold = 0.84),
         .class_fruit = make_two_class_pred(.pred_fruit, c("Detected", "Not Detected"),
                                            threshold = 0.53))

set.seed(45897)

inf_dat_samp <- inf_dat |>
  slice_sample(n = 1e+6)

sample_dataset <- inf_dat_samp |>
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

families <- taxa_meta |>
  filter(rank == "family") |>
  collect()

## looks like family is (always?) position 7
sample_dataset <- sample_dataset |>
  mutate(family_id = as.integer(str_split_i(ancestry, "/", 7)))

sample_dataset <- sample_dataset |>
  left_join(families |> select(family_id = taxon_id, family = name))

### add family level stats
fam_stats <- read_csv("output/model_04_13_2024/family_stats.csv")

sample_dataset <- sample_dataset |>
  left_join(fam_stats |>
              filter(test == "test") |>
              select(family, .equivpropfamily_flower = equiv_prop_fl,
                     .equivpropfamily_fruit = equiv_prop_fr, .count_family = count,
                     .accuracyfamily_flower = .accuracy_family_flower,
                     .accuracyfamily_fruit = .accuracy_family_fruit,
                     .accuracyfamilyinclequiv_flower = .accuracy_family_flower_incl_equiv,
                     .accuracyfamilyinclequiv_fruit = .accuracy_family_fruit_incl_equiv))

sample_dataset2 <- sample_dataset |>
  select(-.logit_flower, -.logit_fruit) |>
  pivot_longer(cols = c(.pred_flower, .pred_fruit,
                        .class_flower, .class_fruit,
                        .equivocal_flower, .equivocal_fruit,
                        .equivpropfamily_flower, .equivpropfamily_fruit,
                        .accuracyfamily_flower, .accuracyfamily_fruit,
                        .accuracyfamilyinclequiv_flower, .accuracyfamilyinclequiv_fruit),
               names_to = c(".value", ".trait"),
               names_sep = "_"
               )

sample_dataset2 <- sample_dataset2 |>
  rename(count_family = .count_family,
         trait = .trait,
         prediction_prob = .pred,
         prediction_class = .class,
         equivocal = .equivocal,
         proportion_equivocal_family = .equivpropfamily,
         accuracy_excluding_equivocal_family = .accuracyfamily,
         accuracy_family = .accuracyfamilyinclequiv)

write_csv(sample_dataset2, file.path(test_dir, "sample_inference_dataset_1million.csv"))
