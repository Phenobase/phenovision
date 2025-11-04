## still need to calculate family-level stats and get a model uri
library(tidyverse)
library(probably)
library(arrow)
library(uuid)

inf_df <- read_rds("output/leaves/phenovision-init_model2_04_11_2025/all_inference_results_round2_epoch_1.rds")
fam_dat <- read_csv("output/leaves/phenovision-init_model2_04_11_2025/family_stats.csv")
columns <- read_csv("https://raw.githubusercontent.com/Phenobase/phenobase_data/meeting_refactor/data/columns.csv")

obs_meta <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/observations/part-0.parquet")
observers <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/observers/part-0.parquet")

model_doi <- "10.57967/hf/5785"

## pivot data
fam_equiv <- fam_dat |>
  select(family, starts_with("equiv_prop_")) |>
  pivot_longer(-family, names_to = "trait", names_prefix = "equiv_prop_", values_to = "proportion_certainty_family") |>
  mutate(proportion_certainty_family = 1 - proportion_certainty_family)

fam_acc <- fam_dat |>
  select(family, ends_with("_incl_equiv")) |>
  pivot_longer(-family, names_to = "trait", names_prefix = ".accuracy_family_leaves_", values_to = "accuracy_family") |>
  mutate(trait = case_match(trait,
                                      "green_incl_equiv" ~ "gr",
                                      "colored_incl_equiv" ~ "cl",
                                      "breaking_buds_incl_equiv" ~ "bb",
                                      .default = ""))

fam_acc_cert <- fam_dat |>
  select(family, starts_with(".accuracy_family_leaves"), -ends_with("_incl_equiv")) |>
  pivot_longer(-family, names_to = "trait", names_prefix = ".accuracy_family_leaves_", values_to = "accuracy_excluding_certainty_family") |>
  mutate(trait = case_match(trait,
                            "green" ~ "gr",
                            "colored" ~ "cl",
                            "breaking_buds" ~ "bb",
                            .default = ""))

inf_class <- inf_df |>
  select(file_name, starts_with(".class_")) |>
  mutate(.class_gr = as.character(as.factor(.class_gr)),
         .class_cl = as.character(as.factor(.class_cl)),
         .class_bb = as.character(as.factor(.class_bb))) |>
  pivot_longer(-file_name, names_to = "trait", names_prefix = ".class_", values_to = "detected")

inf_equiv <- inf_df |>
  select(file_name, starts_with(".equivocal_")) |>
  pivot_longer(-file_name, names_to = "trait", names_prefix = ".equivocal_", values_to = "equivocal")

inf_pred <- inf_df |>
  select(file_name, starts_with(".pred_leaves_")) |>
  pivot_longer(-file_name, names_to = "trait", names_prefix = ".pred_leaves_", values_to = "preds") |>
  mutate(trait = case_match(trait,
                            "green" ~ "gr",
                            "colored" ~ "cl",
                            "breaking_buds" ~ "bb",
                            .default = ""))

## filter data
inf_all <- inf_class |>
  left_join(inf_equiv) |>
  filter(detected == "Detected" & equivocal == "Unequivocal")
inf_all <- inf_all |>
  left_join(inf_df |> select(file_name, observation_uuid)) |>
  distinct(observation_uuid, .keep_all = TRUE) |>
  left_join(inf_pred)

inf_all <- inf_all |>
  left_join(inf_df |>
              select(file_name, verbatim_date = observed_on,
                     latitude,
                     longitude,
                     coordinate_uncertainty_meters = positional_accuracy,
                     family,
                     genus,
                     scientific_name = name,
                     observed_image_guid = photo_id)) |>
  mutate(datasource = "iNaturalist",
         day_of_year = yday(verbatim_date),
         year = year(verbatim_date),
         certainty = ifelse(equivocal == "Equivocal", "Low", "High"),
         model_uri = model_doi,
         basis_of_record = "MachineObservation",
         observed_metadata_url = paste0("https://www.inaturalist.org/observations/", observation_uuid),
         observed_image_url = paste0("https://www.inaturalist.org/photos/", observed_image_guid),
         prediction_probability = preds,
         prediction_class = detected)

inf_all <- inf_all |>
  left_join(fam_acc) |>
  left_join(fam_acc_cert) |>
  left_join(fam_equiv)

inf_all <- inf_all |>
  filter(trait != "no") |>
  mutate(trait = case_match(trait,
                            "gr" ~ "green leaves",
                            "cl" ~ "colored leaves",
                            "bb" ~ "breaking leaf buds"))

inf_all <- inf_all |>
  mutate(uniquer = paste0(observation_uuid, model_uri, trait),
         machine_learning_annotation_id = UUIDfromName(UUIDgenerate(1), uniquer))

cols <- columns$field[columns$machine_annotation_inat_relevance == "REQUIRED"]
cols[!cols %in% colnames(inf_all)]

#inf_all <- inf_all[ , cols[cols %in% colnames(inf_all)]]

inf_samp <- inf_all |>
  slice_sample(n = 10000)

obs_dat <- obs_meta |>
  select(observation_uuid, observer_id) |>
  filter(observation_uuid %in% inf_samp$observation_uuid) |>
  collect()
# obs_dat <- obs_dat |>
#   left_join(observers |>
#               select(observer_id, name) |>
#               filter(observer_id %in% obs_dat$observer_id),
#             copy = TRUE)

inf_samp <- inf_samp |>
  left_join(obs_dat) |>
  rename(recorded_by = observer_id)


inf_samp <- inf_samp[ , c(columns$field[columns$field %in% colnames(inf_samp)],
                          "observation_uuid")]

write_csv(inf_samp, "output/leaves/phenovision-init_model2_04_11_2025/leaves_sample_inferences_ingest_format.csv")





