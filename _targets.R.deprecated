## Load your packages, e.g. library(targets).
source("./packages.R")

## Load your R files
lapply(list.files("./R", full.names = TRUE, pattern = "_targets"), source)

conflicts_prefer(dplyr::filter)
conflicts_prefer(lubridate::yday)
conflicts_prefer(lubridate::year)

## Triggers
trigger_update_images <- FALSE
## Global parameters
num_workers <- 6L
batch_size_inference <- 1280L

tar_option_set(
  error = "null",
  memory = "auto",
  garbage_collection = TRUE
)

## tar_plan supports drake-style targets and also tar_target()
tar_plan(

# target = function_to_make(arg), ## drake style

  tar_target(model_update_date, "07/16/2025"),

  tar_target(model_doi_repro, "10.57967/hf/2763"),

  tar_target(model_doi_leaves, "10.57967/hf/5785"),

  tar_target(model_doi_path_leaves, gsub("\\/", "_", model_doi_leaves)),

  tar_target(results_dir_leaves, file.path("output", "leaves", model_doi_path_leaves)),

  tar_target(model_vers_repro, model_version(model_doi)),

  tar_target(inat_folder, "/blue/guralnick/share/phenobase_inat_data"),

  tar_target(meta_images, "/blue/guralnick/share/phenobase_inat_data/metadata/angio_photos"),

  tar_target(meta_obs, "/blue/guralnick/share/phenobase_inat_data/metadata/observations/part-0.parquet"),

  tar_target(meta_taxa, "/blue/guralnick/share/phenobase_inat_data/metadata/taxa/part-0.parquet"),

  tar_target(meta_observer, "/blue/guralnick/share/phenobase_inat_data/metadata/observers/part-0.parquet"),

  tar_target(fam_stats, "output/model_04_13_2024/family_stats.csv"),

  tar_target(images, open_dataset(meta_images) |>
                     select(photo_id, batch_j, extension) |>
                     collect() |>
                     drop_na(batch_j) |>
                     mutate(path = file.path("data/phenobase_inat_data/images/medium",
                            paste0("batch_", batch_j),
                            paste0(photo_id, ".", extension))),
             cue = tar_cue_force(trigger_update_images)),

  tar_target(batches, unique(images$batch_j)),

  tar_target(batch_df, tibble(batch_j = batches, run_name = santoku::chop_n(batch_j, 10, labels = lbl_seq(start = "1")))),

  tar_target(images_df, images |>
                          left_join(batch_df, by = "batch_j")),

  tar_target(images_batch, images_df |>
                              group_by(run_name) |>
                              tar_group(),
             iteration = "group"),

  tar_target(taxonomy, open_dataset(meta_images) |>
               distinct(taxon_id) |>
               left_join(open_dataset(meta_taxa) |>
                           select(taxon_id, ancestry),
                         by = "taxon_id",
                         copy = TRUE) |>
               collect() |>
               select(taxon_id, ancestry) |>
               mutate(taxa_ids = str_split(ancestry, "/")) |>
               select(-ancestry) |>
               unnest_longer(taxa_ids, transform = as.integer)),

  tar_target(families, open_dataset(meta_taxa) |>
               filter(rank == "family") |>
               collect()),

  tar_target(genera, open_dataset(meta_taxa) |>
               filter(rank == "genus") |>
               collect()),

  tar_target(species, open_dataset(meta_taxa) |>
               filter(rank == "genus") |>
               collect()),

  tar_target(fields, read_csv("data/Phenobase_fields.csv")),

  tar_target(field_map, tribble(~new_field, ~old_field,
                                  "dataSource", "datasource",
                                  "scientificName", "scientific_name",
                                  "trait", "trait",
                                  "family", "family",
                                  "year", "year",
                                  "dayOfYear", "day_of_year",
                                  "latitude", "latitude",
                                  "longitude", "longitude",
                                  "observedMetadataUrl", "observed_metadata_url",
                                  "annotationID", "machine_learning_annotation_id",
                                  "annotationMethod", "annotation_method",
                                  "occurrenceID", "observation_uuid",
                                  "basisOfRecord", "basis_of_record",
                                  "genus", "genus",
                                  "taxonRank", "taxon_rank",
                                  "date", "verbatim_date",
                                  "recordedBy", "recorded_by",
                                  "coordinateUncertaintyInMeters", "coordinate_uncertainty_meters",
                                  "verbatimTrait", "trait",
                                  "modelUri", "model_uri",
                                  "accuracyExcludingUncertainFamily", "accuracy_excluding_certainty_family",
                                  "ObservedImageUrl", "observed_image_url",
                                  "mappedTraits", NA,
                                  "predictionClass", "detected",
                                  "countImages", "count_images",
                                  "countFamily", "count_family",
                                  "certainty", "certainty",
                                  "predictionProbability", "pred_med",
                                  "proportionCertaintyFamily", "proportion_certainty_family",
                                  "accuracyFamily", "accuracy_family")),

  #tar_target(images_batch_split, images_batch, pattern = map(images_batch), iteration = "list"),

  ############## PhenoVisionR (reproductive structures) #################

  tar_target(annotation_outputs, annotate_images(images_batch_split, model_doi, model_vers, inat_folder,
                                                 meta_images, meta_obs, meta_taxa, fam_stats),
             pattern = map(images_batch_split), iteration = "list",
             format = "file",
             cue = tar_cue(command = FALSE,
                           depend = FALSE)),

  tar_target(annotations_big_file, concatenate_annotations(annotation_outputs),
             file.path(inat_folder, model_vers, "all_annotations.csv"),
             format = "file"),

  tar_target(bad_images, "data/inat/bad_images.rds", format = "file"),

  tar_target(file_train, "data/inat/train.csv", format = "file"),

  tar_target(file_val, "data/inat/validation.csv", format = "file"),

  tar_target(file_test, "data/inat/test.csv", format = "file"),

  tar_target(images_train, read_csv(file_train)),

  tar_target(anno_train, annotate_batch(images_batch_split$path, model_doi = model_doi)),

  ############## PhenoVisionL (leaves) ####################

  tar_target(rob_annot, read_csv("data/leaves/phenobase_dwca_annotation/rob_leaf_breaking_buds_annotation.csv") |>
               mutate(genus = word(taxon.name))),

  tar_target(fam_dat_leaves, read_csv("output/leaves/phenovision-init_model2_04_11_2025/family_stats.csv")),

  tar_target(genera_exclude_leaves, c("Logfia", "Oxalis", "Viola")),

  tar_target(genera_include_leaves, setdiff(unique(rob_annot$genus), genera_exclude_leaves)),

  tar_target(genus_taxon_ids, open_dataset(meta_taxa) |>
               filter(rank == "genus") |>
               filter(name %in% genera_include_leaves) |>
               collect() |>
               filter(active)),

  tar_target(taxonomy_leaves, taxonomy |>
               filter(taxa_ids %in% genus_taxon_ids$taxon_id)),

  ## filter for genera of interest through their taxa_ids
  tar_target(images_batch_leaves, images_batch |>
               left_join(open_dataset(meta_images) |>
                           select(photo_id, taxon_id) |>
                           filter(photo_id %in% images_batch$photo_id),
                         copy = TRUE) |>
               left_join(taxonomy_leaves) |>
               drop_na(taxa_ids),
             pattern = map(images_batch)),

  tar_target(annotations_leaves, annotate_batch(images_batch_leaves,
                                                model_doi_leaves,
                                                trait = "leaves",
                                                num_workers),
             iteration = "list", pattern = map(images_batch_leaves)),

  ## need to find a way to link thresholds with model versions, probably by
  ## uploading them to the huggingface repo along with the model...
  tar_target(thresholds_leaves, read_rds("output/leaves/phenovision-init_model2_04_11_2025/epoch_1_threshold_buffers.csv")),

  tar_target(annotations_thresholded_leaves, threshold_annotations(annotations_leaves,
                                                                   thresholds_leaves,
                                                                   trait = "leaves",
                                                                   meta_images),
             pattern = map(annotations_leaves), iteration = "list"),

  tar_target(annotations_long_leaves, convert_to_long(annotations_thresholded_leaves,
                                               trait = "leaves"),
             pattern = map(annotations_thresholded_leaves), iteration = "list"),

  tar_target(fam_dat_long_leaves, convert_fam_to_long(fam_dat_leaves, trait = "leaves")),

  tar_target(annotations_by_obs_leaves, aggregate_by_obs(annotations_long_leaves,
                                                         taxonomy, families, genera,
                                                         fam_dat_long_leaves,
                                                         meta_images,
                                                         meta_taxa),
             pattern = map(annotations_long_leaves), iteration = "list"),

  tar_target(annotations_by_obs_final_leaves, finalize_annotations_by_obs(annotations_by_obs_leaves,
                                                                          fields,
                                                                          field_map),
             pattern = map(annotations_by_obs_leaves), iteration = "list"),

  tar_target(annotations_by_obs_ingest_leaves, annotations_by_obs_final_leaves |>
               filter(certainty == "High" & predictionClass == "Detected",
                      verbatimTrait != "no live leaves") |>
               select(-proportionCertaintyFamily,
                      -countFamily,
                      -countImages,
                      -certainty,
                      -predictionProbability,
                      -predictionClass,
                      -accuracyFamily),
             pattern = map(annotations_by_obs_final_leaves),
             iteration = "list"),

  tar_target(annotations_internal, {
    write_csv(annotations_by_obs_final_leaves,
              path <- file.path(results_dir_leaves,
                                "final_internal",
                                paste0(tar_name(),
                                       ".csv")))
    path
  },
  pattern = map(annotations_by_obs_final_leaves),
  format = "file"),

  tar_target(annotations_ingest, {
    write_csv(annotations_by_obs_ingest_leaves,
              path <- file.path(results_dir_leaves,
                                "final_ingest",
                                paste0(tar_name(),
                                       ".csv")))
    path
    },
    pattern = map(annotations_by_obs_ingest_leaves),
    format = "file"),

  tar_target(annotations_internal_all_csv, concatenate_csvs(annotations_by_obs_final_leaves,
                                                            file.path(results_dir_leaves,
                                                                      "annotations_internal_all.csv")),
             pattern = map(annotations_by_obs_final_leaves),
             format = "file"),

  tar_target(annotations_ingest_all_csv, concatenate_csvs(annotations_by_obs_ingest_leaves,
                                                          file.path(results_dir_leaves,
                                                                    "annotations_ingest_all.csv")),
             pattern = map(annotations_by_obs_ingest_leaves),
             format = "file")

)
