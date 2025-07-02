# Enhanced flower/fruit pipeline functions for targets workflow
require(tidyverse)
require(arrow)
require(cli)
source("R/pipeline_common_targets.R")
source("R/postprocessing_targets.R")
source("R/model_loading_targets.R")
source("R/ml_annotation_targets.r")

# Main function to annotate images with flower/fruit predictions
# This replaces the original annotate_images function with 4-output structure
annotate_images_flowers_fruits <- function(images_batch, model_doi, model_vers, inat_folder,
                                         meta_images, meta_obs, meta_taxa, fam_stats,
                                         flower_thresholds = c(0.28, 0.84, 0.85),
                                         fruit_thresholds = c(0.23, 0.53, 0.75),
                                         batch_id = NULL) {
  
  cli_alert_info("Processing flower/fruit batch with {nrow(images_batch)} images")
  
  # Extract file paths
  file_paths <- images_batch$path
  
  # Run model inference
  cli_progress_step("Running model inference...")
  predictions <- annotate_batch(file_paths, model_doi)
  
  # Handle corrupted images
  corrupt_images <- handle_corrupted_images(predictions, inat_folder, model_vers)
  
  # Remove corrupted images from predictions
  predictions_clean <- predictions |>
    filter(corrupt == FALSE)
  
  # Create all four output types
  thresholds <- list(
    flower = flower_thresholds,
    fruit = fruit_thresholds
  )
  
  outputs <- create_all_output_types(
    predictions_clean,
    meta_images,
    meta_obs,
    meta_taxa,
    fam_stats,
    model_doi,
    pipeline_type = "flower_fruit",
    thresholds = thresholds,
    output_dir = file.path(inat_folder, model_vers)
  )
  
  # Save outputs with batch ID
  output_paths <- save_all_outputs(
    outputs,
    output_dir = file.path(inat_folder, model_vers),
    pipeline_type = "flower_fruit",
    batch_id = batch_id,
    version = model_vers
  )
  
  # Return paths to saved files for targets tracking
  output_paths
}

# Function to concatenate image-level results from multiple batches
concatenate_ff_image_level <- function(batch_output_paths, output_file) {
  cli_alert_info("Concatenating image-level results from {length(batch_output_paths)} batches")
  
  # Read all image-level files
  image_data <- map_dfr(batch_output_paths, ~{
    read_csv(.x$image_level, show_col_types = FALSE)
  })
  
  # Save concatenated file
  write_csv(image_data, output_file)
  cli_alert_success("Concatenated {nrow(image_data)} image records to {output_file}")
  
  output_file
}

# Function to re-aggregate after combining batches
reaggregate_ff_observations <- function(concatenated_image_file, meta_images, meta_obs, 
                                       meta_taxa, fam_stats, model_doi, model_vers,
                                       flower_thresholds = c(0.28, 0.84, 0.85),
                                       fruit_thresholds = c(0.23, 0.53, 0.75)) {
  
  cli_alert_info("Re-aggregating observations from combined batches")
  
  # Read concatenated image data
  all_images <- read_csv(concatenated_image_file, show_col_types = FALSE)
  
  # Re-aggregate to observation level
  obs_aggregated <- reaggregate_after_batch_combination(
    all_images,
    pipeline_type = "flower_fruit"
  )
  
  # Create filtered and ingestion formats from re-aggregated data
  cli_h2("Creating final filtered dataset for Zenodo")
  
  # For flower/fruit, handle each trait separately
  flower_filtered <- select_observation_representative(
    obs_aggregated,
    pipeline_type = "flower_fruit",
    trait = "flower",
    category = "detected_unequivocal"
  )
  
  fruit_filtered <- select_observation_representative(
    obs_aggregated,
    pipeline_type = "flower_fruit",
    trait = "fruit", 
    category = "detected_unequivocal"
  )
  
  zenodo_filtered <- bind_rows(
    flower_filtered |> mutate(trait = "flower"),
    fruit_filtered |> mutate(trait = "fruit")
  ) |>
    distinct() |>
    add_version_metadata(
      pipeline_type = "flower_fruit",
      description = "Final filtered dataset for Zenodo (all batches combined)"
    )
  
  # Create ingestion format
  cli_h2("Creating final ingestion format for Phenobase")
  
  # Get prediction values from image data for selected observations
  ingestion_base <- all_images |>
    inner_join(
      zenodo_filtered |> select(observation_uuid, file_name),
      by = c("observation_uuid", "file_name")
    )
  
  ingestion <- create_ingestion_format(
    ingestion_base,
    fam_stats,
    pipeline_type = "flower_fruit",
    filter_detected = TRUE,
    filter_low_certainty = TRUE
  ) |>
    add_version_metadata(
      pipeline_type = "flower_fruit",
      description = "Final formatted dataset for Phenobase database ingestion (all batches)"
    )
  
  # Save final outputs
  output_dir <- file.path(dirname(concatenated_image_file), "final")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  final_outputs <- list(
    observation_aggregated = obs_aggregated,
    zenodo_filtered = zenodo_filtered,
    phenobase_ingestion = ingestion
  )
  
  paths <- list(
    observation_aggregated = save_dataset(
      final_outputs$observation_aggregated,
      file.path(output_dir, paste0("flower_fruit_observation_aggregated_final_", model_vers, ".csv")),
      format = "csv"
    ),
    zenodo_filtered = save_dataset(
      final_outputs$zenodo_filtered,
      file.path(output_dir, paste0("flower_fruit_filtered_zenodo_final_", model_vers, ".csv")),
      format = "csv"
    ),
    phenobase_ingestion = save_dataset(
      final_outputs$phenobase_ingestion,
      file.path(output_dir, paste0("flower_fruit_phenobase_ingestion_final_", model_vers, ".csv")),
      format = "csv"
    )
  )
  
  # Generate final documentation
  generate_dataset_documentation(
    final_outputs$zenodo_filtered,
    file.path(output_dir, paste0("flower_fruit_dataset_documentation_final_", model_vers, ".md"))
  )
  
  paths
}

# Convenience function to get default thresholds
get_ff_default_thresholds <- function() {
  list(
    flower = c(0.28, 0.84, 0.85),
    fruit = c(0.23, 0.53, 0.75)
  )
}

# Function to validate outputs
validate_ff_outputs <- function(output_paths) {
  cli_h2("Validating flower/fruit outputs")
  
  # Check that all files exist
  all_exist <- all(map_lgl(output_paths, file.exists))
  
  if (!all_exist) {
    missing <- names(output_paths)[!map_lgl(output_paths, file.exists)]
    cli_alert_danger("Missing output files: {paste(missing, collapse = ', ')}")
    return(FALSE)
  }
  
  # Check file sizes
  sizes <- map_dbl(output_paths, file.size)
  
  if (any(sizes == 0)) {
    empty <- names(output_paths)[sizes == 0]
    cli_alert_warning("Empty output files: {paste(empty, collapse = ', ')}")
  }
  
  # Quick content check
  tryCatch({
    img_level <- read_csv(output_paths$image_level, n_max = 5, show_col_types = FALSE)
    obs_agg <- read_csv(output_paths$observation_aggregated, n_max = 5, show_col_types = FALSE)
    
    # Check for required columns
    required_img_cols <- c(".pred_flower", ".pred_fruit", ".class_flower", ".class_fruit", 
                          "observation_uuid", "file_name")
    required_obs_cols <- c("observation_uuid", "n_images_total", 
                          "n_flower_detected_unequivocal", "n_fruit_detected_unequivocal")
    
    missing_img <- setdiff(required_img_cols, names(img_level))
    missing_obs <- setdiff(required_obs_cols, names(obs_agg))
    
    if (length(missing_img) > 0) {
      cli_alert_warning("Missing columns in image-level data: {paste(missing_img, collapse = ', ')}")
    }
    
    if (length(missing_obs) > 0) {
      cli_alert_warning("Missing columns in observation data: {paste(missing_obs, collapse = ', ')}")
    }
    
    cli_alert_success("Flower/fruit outputs validated successfully")
    return(TRUE)
    
  }, error = function(e) {
    cli_alert_danger("Error validating outputs: {e$message}")
    return(FALSE)
  })
}