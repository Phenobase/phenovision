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

  ############################################################
  # Common configuration
  ############################################################
  
  tar_target(inat_folder, "/blue/guralnick/share/phenobase_inat_data"),
  
  tar_target(meta_images_path, "/blue/guralnick/share/phenobase_inat_data/metadata/angio_photos"),
  
  tar_target(meta_obs_path, "/blue/guralnick/share/phenobase_inat_data/metadata/observations/part-0.parquet"),
  
  tar_target(meta_taxa_path, "/blue/guralnick/share/phenobase_inat_data/metadata/taxa/part-0.parquet"),
  
  # Load metadata
  tar_target(meta_images, open_dataset(meta_images_path)),
  
  tar_target(meta_obs, open_dataset(meta_obs_path)),
  
  tar_target(meta_taxa, open_dataset(meta_taxa_path)),
  
  ############################################################
  # Flower/Fruit Pipeline
  ############################################################
  
  tar_target(ff_model_doi, "10.57967/hf/2763"),
  
  tar_target(ff_model_vers, model_version(ff_model_doi)),
  
  tar_target(ff_fam_stats, "output/model_04_13_2024/family_stats.csv"),
  
  tar_target(ff_thresholds, get_ff_default_thresholds()),
  
  # Prepare images
  tar_target(ff_images, open_dataset(meta_images_path) |>
                     select(photo_id, batch_j, extension, observation_uuid) |>
                     collect() |>
                     drop_na(batch_j) |>
                     mutate(path = file.path("/blue/guralnick/share/phenobase_inat_data/images/medium",
                            paste0("batch_", batch_j),
                            paste0(photo_id, ".", extension)))),
  
  tar_target(ff_batches, unique(ff_images$batch_j)),
  
  tar_target(ff_batch_df, tibble(batch_j = ff_batches, 
                                 run_name = santoku::chop_n(batch_j, 10, labels = lbl_seq(start = "1")))),
  
  tar_target(ff_images_df, ff_images |>
                          left_join(ff_batch_df, by = "batch_j")),
  
  tar_target(ff_images_batch, ff_images_df |>
                              group_by(run_name) |>
                              group_split()),
  
  tar_target(ff_images_batch_split, ff_images_batch, 
             pattern = map(ff_images_batch), 
             iteration = "list"),
  
  # Process batches with 4-output structure
  tar_target(ff_batch_outputs, 
             annotate_images_flowers_fruits(
               ff_images_batch_split, 
               ff_model_doi, 
               ff_model_vers, 
               inat_folder,
               meta_images, 
               meta_obs, 
               meta_taxa, 
               read_csv(ff_fam_stats),
               flower_thresholds = ff_thresholds$flower,
               fruit_thresholds = ff_thresholds$fruit,
               batch_id = ff_images_batch_split$run_name[1]
             ),
             pattern = map(ff_images_batch_split), 
             iteration = "list",
             cue = tar_cue(command = FALSE, depend = FALSE)),
  
  # Concatenate image-level results
  tar_target(ff_all_images_file, 
             concatenate_ff_image_level(
               ff_batch_outputs,
               file.path(inat_folder, ff_model_vers, "ff_all_images.csv")
             ),
             format = "file"),
  
  # Re-aggregate observations and create final outputs
  tar_target(ff_final_outputs,
             reaggregate_ff_observations(
               ff_all_images_file,
               meta_images,
               meta_obs,
               meta_taxa,
               read_csv(ff_fam_stats),
               ff_model_doi,
               ff_model_vers,
               flower_thresholds = ff_thresholds$flower,
               fruit_thresholds = ff_thresholds$fruit
             )),
  
  # Export for Zenodo
  tar_target(ff_zenodo_export,
             export_for_zenodo(
               read_csv(ff_final_outputs$zenodo_filtered),
               file.path(inat_folder, ff_model_vers, "exports"),
               pipeline_type = "flower_fruit"
             )),
  
  # Export for Phenobase
  tar_target(ff_phenobase_export,
             export_for_phenobase(
               read_csv(ff_final_outputs$phenobase_ingestion),
               file.path(inat_folder, ff_model_vers, "exports"),
               pipeline_type = "flower_fruit"
             )),
  
  ############################################################
  # Leaf Pipeline
  ############################################################
  
  tar_target(leaf_model_doi, "10.57967/hf/5785"),
  
  tar_target(leaf_model_vers, model_version(leaf_model_doi)),
  
  tar_target(leaf_model_path, 
             file.path("output/leaves/phenovision-init_model2_04_11_2025/checkpoints/vit_finetuned_round2_epoch1.pt")),
  
  tar_target(leaf_fam_stats, "output/leaves/phenovision-init_model2_04_11_2025/family_stats.csv"),
  
  tar_target(leaf_thresholds_file, "output/leaves/phenovision-init_model2_04_11_2025/epoch_1_threshold_buffers.csv"),
  
  tar_target(leaf_thresholds, load_leaf_thresholds(leaf_thresholds_file)),
  
  # Load Rob's annotations for genus filtering
  tar_target(rob_annotations, load_rob_annotations()),
  
  # Prepare leaf images with taxonomy info
  tar_target(leaf_inference_meta_raw, 
             read_csv("data/leaves/inference_metadata_03-06-2025.csv")),
  
  tar_target(leaf_images, 
             leaf_inference_meta_raw |>
               mutate(
                 path = file_name,  # Assuming file_name contains full path
                 observation_uuid = observation_uuid,
                 run_name = santoku::chop_n(1:n(), 1e6, labels = lbl_seq(start = "1"))
               )),
  
  tar_target(leaf_images_batch,
             leaf_images |>
               group_by(run_name) |>
               group_split()),
  
  tar_target(leaf_images_batch_split, 
             leaf_images_batch,
             pattern = map(leaf_images_batch),
             iteration = "list"),
  
  # Process leaf batches with 4-output structure
  tar_target(leaf_batch_outputs,
             annotate_images_leaves(
               leaf_images_batch_split,
               leaf_model_path,
               leaf_model_doi,
               leaf_model_vers,
               rob_annotations,
               inat_folder,
               meta_images,
               meta_obs,
               meta_taxa,
               read_csv(leaf_fam_stats),
               leaf_thresholds = leaf_thresholds,
               batch_id = leaf_images_batch_split$run_name[1]
             ),
             pattern = map(leaf_images_batch_split),
             iteration = "list",
             cue = tar_cue(command = FALSE, depend = FALSE)),
  
  # Concatenate leaf image-level results
  tar_target(leaf_all_images_file,
             concatenate_leaf_image_level(
               leaf_batch_outputs,
               file.path(inat_folder, leaf_model_vers, "leaf_all_images.csv")
             ),
             format = "file"),
  
  # Re-aggregate leaf observations and create final outputs
  tar_target(leaf_final_outputs,
             reaggregate_leaf_observations(
               leaf_all_images_file,
               meta_images,
               meta_obs,
               meta_taxa,
               read_csv(leaf_fam_stats),
               leaf_model_doi,
               leaf_model_vers,
               leaf_thresholds = leaf_thresholds
             )),
  
  # Export leaf data for Zenodo
  tar_target(leaf_zenodo_export,
             export_for_zenodo(
               read_csv(leaf_final_outputs$zenodo_filtered),
               file.path(inat_folder, leaf_model_vers, "exports"),
               pipeline_type = "leaf"
             )),
  
  # Export leaf data for Phenobase
  tar_target(leaf_phenobase_export,
             export_for_phenobase(
               read_csv(leaf_final_outputs$phenobase_ingestion),
               file.path(inat_folder, leaf_model_vers, "exports"),
               pipeline_type = "leaf"
             )),
  
  ############################################################
  # Validation targets
  ############################################################
  
  tar_target(validate_ff_outputs,
             validate_ff_outputs(ff_final_outputs)),
  
  tar_target(validate_leaf_outputs,
             validate_leaf_outputs(leaf_final_outputs)),
  
  tar_target(validate_ff_zenodo,
             validate_export(ff_zenodo_export)),
  
  tar_target(validate_leaf_zenodo,
             validate_export(leaf_zenodo_export))

)