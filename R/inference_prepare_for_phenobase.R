## this script takes raw output from phenovision model, creates prediction by thresholding,
## adds metadata. Metadata columns are described here: https://github.com/Phenobase/phenobase_data/blob/main/data/columns.csv

set.seed(4578835)

library(tidyverse)
library(arrow)
library(tidymodels)
library(probably)
library(uuid)

image_dir <- "/blue/guralnick/share/phenobase_inat_data/images/medium"
test_dir <- "/blue/guralnick/share/phenobase_inat_data/inference_test"

inf_dat <- read_csv("output/model_04_13_2024/inference_results_raw.csv")
train_dat <- read_csv("data/inat/train.csv")

columns <- read_csv("https://raw.githubusercontent.com/Phenobase/phenobase_data/main/data/columns.csv")

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
  slice_sample(n = 20000)

sample_dataset <- inf_dat_samp |>
  distinct(file_name, .keep_all = TRUE) |>
  mutate(photo_id = fs::path_ext_remove(file_name),
  extension = fs::path_ext(file_name))

sample_dataset <- sample_dataset |>
  left_join(angio_meta |>
              select(photo_id, observation_uuid) |>
              filter(photo_id %in% sample_dataset$photo_id),
            by = "photo_id",
            copy = TRUE)

sample_dataset <- sample_dataset |>
  mutate(inat_URL = paste0("https://www.inaturalist.org/observations/", observation_uuid),
         observed_image_url = paste0("https://www.inaturalist.org/photos/", photo_id))

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

genera <- taxa_meta |>
  filter(rank == "genus") |>
  collect()

taxonomy <- sample_dataset |>
  select(file_name, ancestry) |>
  mutate(taxa_ids = str_split(ancestry, "/")) |>
  select(-ancestry) |>
  unnest_longer(taxa_ids, transform = as.integer)

fams <- taxonomy |>
  left_join(families |> select(taxon_id, family = name), by = c(taxa_ids = "taxon_id")) |>
  drop_na()

gens <- taxonomy |>
  left_join(genera |> select(taxon_id, genus = name), by = c(taxa_ids = "taxon_id")) |>
  drop_na()

## looks like family is (always?) position 7
# sample_dataset <- sample_dataset |>
#   mutate(family_id = as.integer(str_split_i(ancestry, "/", 7)))
#
# sample_dataset <- sample_dataset |>
#   left_join(families |> select(family_id = taxon_id, family = name))

sample_dataset <- sample_dataset |>
  left_join(fams |> select(file_name, family)) |>
  left_join(gens |> select(file_name, genus))

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

## filter out low certainty and 'not detected' records
sample_dataset2 <- sample_dataset2 |>
  filter(as.character(.class) == "Detected" & .equivocal == "Unequivocal")

## sample again if necessary
sample_dataset2 <- sample_dataset2 |>
  slice_sample(n = 10000)

sample_dataset2 <- sample_dataset2 |>
  mutate(machine_learning_annotation_id = UUIDgenerate(n = n()),
         datasource = "iNaturalist",
         day_of_year = yday(observed_on),
         year = year(observed_on),
         certainty = ifelse(.equivocal == "Equivocal", "Low", "High"),
         model_uri = "10.57967/hf/2763",
         basis_of_record = "MachineObservation") |>
  select(machine_learning_annotation_id,
         datasource,
         verbatim_date = observed_on,
         day_of_year,
         year,
         latitude,
         longitude,
         coordinate_uncertainty_meters = positional_accuracy,
         family,
         count_family = .count_family,
         genus,
         scientific_name = name,
         taxon_rank = rank,
         basis_of_record,
         trait = .trait,
         observed_image_guid = photo_id,
         observed_image_url,
         observed_metadata_url = inat_URL,
         certainty,
         model_uri,
         prediction_probability = .pred,
         prediction_class = .class,
         proportion_low_certainty_family = .equivpropfamily,
         accuracy_excluding_low_certainty_family = .accuracyfamily,
         accuracy_family = .accuracyfamilyinclequiv)

write_csv(sample_dataset2, file.path(test_dir, "sample_inference_dataset_10000.csv"))
# write_csv(sample_dataset |> slice_head(n = 500),
#           file.path(test_dir, "sample_inference_dataset_500_for_Erin.csv"))
