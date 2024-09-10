## Load your packages, e.g. library(targets).
source("./packages.R")

## Load your R files
lapply(list.files("./R", full.names = TRUE, pattern = "_targets"), source)

conflicts_prefer(dplyr::filter)
conflicts_prefer(lubridate::yday)
conflicts_prefer(lubridate::year)

tar_option_set(
  error = "null",
  memory = "transient",
  garbage_collection = TRUE
)

## tar_plan supports drake-style targets and also tar_target()
tar_plan(

# target = function_to_make(arg), ## drake style

  tar_target(model_doi, "10.57967/hf/2763"),

  tar_target(model_vers, model_version(model_doi)),

  tar_target(inat_folder, "/blue/guralnick/share/phenobase_inat_data"),

  tar_target(meta_images, "/blue/guralnick/share/phenobase_inat_data/metadata/angio_photos"),

  tar_target(meta_obs, "/blue/guralnick/share/phenobase_inat_data/metadata/observations/part-0.parquet"),

  tar_target(meta_taxa, "/blue/guralnick/share/phenobase_inat_data/metadata/taxa/part-0.parquet"),

  tar_target(fam_stats, "output/model_04_13_2024/family_stats.csv"),

  tar_target(images, open_dataset(meta_images) |>
                     select(photo_id, batch_j, extension) |>
                     collect() |>
                     drop_na(batch_j) |>
                     mutate(path = file.path("/blue/guralnick/share/phenobase_inat_data/images/medium",
                            paste0("batch_", batch_j),
                            paste0(photo_id, ".", extension)))),

  tar_target(batches, unique(images$batch_j)),

  tar_target(batch_df, tibble(batch_j = batches, run_name = santoku::chop_n(batch_j, 10, labels = lbl_seq(start = "1")))),

  tar_target(images_df, images |>
                          left_join(batch_df, by = "batch_j")),

  tar_target(images_batch, images_df |>
                              group_by(run_name) |>
                              group_split()),

  tar_target(images_batch_split, images_batch, pattern = map(images_batch), iteration = "list"),

  tar_target(annotation_outputs, annotate_images(images_batch_split, model_doi, model_vers, inat_folder,
                                                 meta_images, meta_obs, meta_taxa, fam_stats),
             pattern = map(images_batch_split), iteration = "list",
             format = "file",
             cue = tar_cue(command = FALSE,
                           depend = FALSE)), ## targets style

  tar_target(annotations_big_file, concatenate_annotations(annotation_outputs),
             file.path(inat_folder, model_vers, "all_annotations.csv"),
             format = "file"),

  tar_target(bad_images, "data/inat/bad_images.rds", format = "file"),

  tar_target(file_train, "data/inat/train.csv", format = "file"),

  tar_target(file_val, "data/inat/validation.csv", format = "file"),

  tar_target(file_test, "data/inat/test.csv", format = "file"),

  tar_target(images_train, read_csv(file_train)),

  tar_target(anno_train, annotate_batch(images_batch_split$path, model_doi = model_doi))

)
