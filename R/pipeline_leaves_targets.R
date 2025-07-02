# Leaf phenology pipeline functions for targets workflow
require(tidyverse)
require(arrow)
require(cli)
require(reticulate)
source("R/pipeline_common_targets.R")
source("R/postprocessing_targets.R")
source("R/model_loading_targets.R")

# Initialize Python dependencies for leaf model
init_leaf_python_deps <- function() {
  torch <- import("torch")
  timm <- import("timm")
  PIL <- import("PIL")
  ds <- import_from_path("custom_dataset", "py")
  eval <- import_from_path("evaluate", "py")
  models_vit <- import_from_path("models_vit", "PlantCLEF2022")
  
  list(
    torch = torch,
    timm = timm,
    PIL = PIL,
    ds = ds,
    eval = eval,
    models_vit = models_vit
  )
}

# Load Rob's annotation data for genus filtering
load_rob_annotations <- function(rob_annot_path = "data/leaves/phenobase_dwca_annotation/rob_leaf_breaking_buds_annotation.csv") {
  cli_progress_step("Loading Rob's leaf breaking buds annotations...")
  
  rob_annot <- read_csv(rob_annot_path, show_col_types = FALSE) |>
    mutate(genus = word(taxon.name))
  
  # Get unique genera that have breaking bud annotations
  valid_genera <- unique(rob_annot$genus)
  
  cli_alert_success("Loaded {nrow(rob_annot)} annotations covering {length(valid_genera)} genera")
  
  list(
    annotations = rob_annot,
    valid_genera = valid_genera
  )
}

# Filter images to valid genera for leaf phenology
filter_leaf_genera <- function(images_batch, valid_genera, exclude_genera = c("Logfia")) {
  cli_progress_step("Filtering images to valid genera for leaf phenology...")
  
  # Add genus information if not present
  if (!"genus" %in% names(images_batch)) {
    cli_alert_warning("Genus information not found in images_batch, this filtering step requires genus metadata")
    return(images_batch)
  }
  
  # Filter to valid genera and exclude problematic ones
  filtered <- images_batch |>
    filter(genus %in% valid_genera) |>
    filter(!genus %in% exclude_genera)
  
  cli_alert_info("Filtered from {nrow(images_batch)} to {nrow(filtered)} images ({length(unique(filtered$genus))} genera)")
  
  filtered
}

# Function to run leaf model inference
annotate_batch_leaves <- function(images_batch, model_path, model_doi = "10.57967/hf/5785",
                                 batch_size = 2560L, device = "cuda:0") {
  
  cli_alert_info("Running leaf phenology inference on {nrow(images_batch)} images")
  
  # Initialize Python dependencies
  py_deps <- init_leaf_python_deps()
  
  # Load model
  cli_progress_step("Loading leaf model from {model_path}...")
  vit <- py_deps$torch$load(model_path)
  vit <- vit$to(device)
  vit$eval()
  
  # Create model for config
  vit2 <- py_deps$models_vit$vit_large_patch16(num_classes = 3L)
  config <- py_deps$timm$data$resolve_data_config(model = vit2)
  transform <- py_deps$timm$data$create_transform(!!!config)
  
  # Prepare data
  file_paths <- images_batch$path
  if (is.null(file_paths)) {
    file_paths <- images_batch$file_name
  }
  
  inf_img <- r_to_py(file_paths)
  
  # Create dummy labels for inference (3 classes for leaves)
  inf_leaves <- tibble(
    leaves_green = rep(0.5, nrow(images_batch)),
    leaves_colored = rep(0.5, nrow(images_batch)),
    leaves_breaking_buds = rep(0.5, nrow(images_batch))
  ) |>
    as.matrix()
  
  # Create dataset and dataloader
  inf_ds <- py_deps$ds$PhenoDatasetInf(inf_img, inf_leaves, transform = transform, inference_mode = TRUE)
  inf_dl <- py_deps$timm$data$create_loader(
    inf_ds, 
    c(3L, 224L, 224L), 
    batch_size, 
    num_workers = 6L,
    is_training = FALSE
  )
  
  # Run inference
  cli_progress_step("Running model inference...")
  inf_dat <- py_deps$eval$evaluate(inf_dl, vit, device)
  
  # Process outputs
  inf_logits <- py_deps$torch$cat(inf_dat[[1]])
  inf_preds <- py_deps$torch$nn$functional$sigmoid(inf_logits)
  inf_truth <- py_deps$torch$cat(inf_dat[[2]])
  
  # Filter out skipped images
  noskip <- inf_truth[, 0] != -999999999
  inf_preds <- as.matrix(inf_preds[noskip, ]$numpy())
  noskip <- as.vector(noskip$numpy())
  
  # Create results dataframe
  results <- as.data.frame(inf_preds) |>
    rename(
      .pred_leaves_green = V1, 
      .pred_leaves_colored = V2,
      .pred_leaves_breaking_buds = V3
    ) |>
    bind_cols(images_batch[noskip, ]) |>
    mutate(
      corrupt = FALSE,
      model_version = model_doi,
      file_name = basename(file_paths[noskip])
    )
  
  # Add corrupted images back
  if (sum(!noskip) > 0) {
    corrupt_imgs <- images_batch[!noskip, ] |>
      mutate(
        .pred_leaves_green = NA_real_,
        .pred_leaves_colored = NA_real_,
        .pred_leaves_breaking_buds = NA_real_,
        corrupt = TRUE,
        model_version = model_doi,
        file_name = basename(file_paths[!noskip])
      )
    
    results <- bind_rows(results, corrupt_imgs)
  }
  
  cli_alert_success("Completed leaf inference with {sum(!results$corrupt)} successful predictions")
  
  results
}

# Main function to process leaf images with 4-output structure
annotate_images_leaves <- function(images_batch, model_path, model_doi, model_vers, 
                                  rob_annotations, inat_folder, meta_images, meta_obs, 
                                  meta_taxa, fam_stats, leaf_thresholds = NULL,
                                  batch_id = NULL) {
  
  cli_alert_info("Processing leaf batch with {nrow(images_batch)} images")
  
  # Filter to valid genera
  valid_genera <- rob_annotations$valid_genera
  images_filtered <- filter_leaf_genera(images_batch, valid_genera)
  
  if (nrow(images_filtered) == 0) {
    cli_alert_warning("No images left after genus filtering")
    return(NULL)
  }
  
  # Run model inference
  predictions <- annotate_batch_leaves(images_filtered, model_path, model_doi)
  
  # Handle corrupted images
  corrupt_images <- handle_corrupted_images(predictions, inat_folder, model_vers)
  
  # Remove corrupted images from predictions
  predictions_clean <- predictions |>
    filter(corrupt == FALSE)
  
  # Load default thresholds if not provided
  if (is.null(leaf_thresholds)) {
    # These would typically be loaded from a saved file
    leaf_thresholds <- list(
      gr = c(0.5, 0.385, 0.115),  # threshold, lower_buffer, upper_buffer
      cl = c(0.5, 0.005, 0.495),
      bb = c(0.5, 0.165, 0.335)
    )
  }
  
  # Create all four output types
  outputs <- create_all_output_types(
    predictions_clean,
    meta_images,
    meta_obs,
    meta_taxa,
    fam_stats,
    model_doi,
    pipeline_type = "leaf",
    thresholds = leaf_thresholds,
    output_dir = file.path(inat_folder, model_vers)
  )
  
  # Save outputs with batch ID
  output_paths <- save_all_outputs(
    outputs,
    output_dir = file.path(inat_folder, model_vers),
    pipeline_type = "leaf",
    batch_id = batch_id,
    version = model_vers
  )
  
  # Return paths to saved files for targets tracking
  output_paths
}

# Function to concatenate leaf image-level results from multiple batches
concatenate_leaf_image_level <- function(batch_output_paths, output_file) {
  cli_alert_info("Concatenating leaf image-level results from {length(batch_output_paths)} batches")
  
  # Read all image-level files
  image_data <- map_dfr(batch_output_paths, ~{
    read_csv(.x$image_level, show_col_types = FALSE)
  })
  
  # Save concatenated file
  write_csv(image_data, output_file)
  cli_alert_success("Concatenated {nrow(image_data)} leaf image records to {output_file}")
  
  output_file
}

# Function to re-aggregate leaf observations after combining batches
reaggregate_leaf_observations <- function(concatenated_image_file, meta_images, meta_obs,
                                        meta_taxa, fam_stats, model_doi, model_vers,
                                        leaf_thresholds = NULL) {
  
  cli_alert_info("Re-aggregating leaf observations from combined batches")
  
  # Read concatenated image data
  all_images <- read_csv(concatenated_image_file, show_col_types = FALSE)
  
  # Re-aggregate to observation level
  obs_aggregated <- reaggregate_after_batch_combination(
    all_images,
    pipeline_type = "leaf"
  )
  
  # Create filtered dataset for Zenodo
  cli_h2("Creating final filtered dataset for Zenodo")
  
  zenodo_filtered <- select_observation_representative(
    obs_aggregated,
    pipeline_type = "leaf",
    category = "detected_unequivocal"
  ) |>
    add_version_metadata(
      pipeline_type = "leaf",
      description = "Final filtered leaf dataset for Zenodo (all batches combined)"
    )
  
  # Create ingestion format
  cli_h2("Creating final ingestion format for Phenobase")
  
  # Get prediction values from image data for selected observations
  ingestion_base <- all_images |>
    inner_join(
      zenodo_filtered |> select(observation_uuid, file_name = representative_image),
      by = c("observation_uuid", "file_name")
    )
  
  ingestion <- create_ingestion_format(
    ingestion_base,
    fam_stats,
    pipeline_type = "leaf",
    filter_detected = TRUE,
    filter_low_certainty = TRUE
  ) |>
    add_version_metadata(
      pipeline_type = "leaf",
      description = "Final formatted leaf dataset for Phenobase database ingestion (all batches)"
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
      file.path(output_dir, paste0("leaf_observation_aggregated_final_", model_vers, ".csv")),
      format = "csv"
    ),
    zenodo_filtered = save_dataset(
      final_outputs$zenodo_filtered,
      file.path(output_dir, paste0("leaf_filtered_zenodo_final_", model_vers, ".csv")),
      format = "csv"
    ),
    phenobase_ingestion = save_dataset(
      final_outputs$phenobase_ingestion,
      file.path(output_dir, paste0("leaf_phenobase_ingestion_final_", model_vers, ".csv")),
      format = "csv"
    )
  )
  
  # Generate final documentation
  generate_dataset_documentation(
    final_outputs$zenodo_filtered,
    file.path(output_dir, paste0("leaf_dataset_documentation_final_", model_vers, ".md"))
  )
  
  paths
}

# Function to load leaf model thresholds
load_leaf_thresholds <- function(threshold_file = NULL, epoch = 1) {
  if (!is.null(threshold_file) && file.exists(threshold_file)) {
    cli_progress_step("Loading leaf thresholds from {threshold_file}")
    thresholds <- read_csv(threshold_file, show_col_types = FALSE)
    
    # Convert to expected format
    list(
      gr = as.numeric(thresholds$gr),
      cl = as.numeric(thresholds$cl),
      bb = as.numeric(thresholds$bb)
    )
  } else {
    # Default thresholds based on the leaf model analysis
    cli_alert_info("Using default leaf thresholds")
    list(
      gr = c(0.5, 0.385, 0.115),  # threshold, lower_buffer, upper_buffer
      cl = c(0.5, 0.005, 0.495),
      bb = c(0.5, 0.165, 0.335)
    )
  }
}

# Function to validate leaf outputs
validate_leaf_outputs <- function(output_paths) {
  cli_h2("Validating leaf outputs")
  
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
    required_img_cols <- c(".pred_leaves_green", ".pred_leaves_colored", ".pred_leaves_breaking_buds",
                          ".class_gr", ".class_cl", ".class_bb", "observation_uuid", "file_name")
    required_obs_cols <- c("observation_uuid", "n_images_total",
                          "n_gr_detected_unequivocal", "n_cl_detected_unequivocal", 
                          "n_bb_detected_unequivocal")
    
    missing_img <- setdiff(required_img_cols, names(img_level))
    missing_obs <- setdiff(required_obs_cols, names(obs_agg))
    
    if (length(missing_img) > 0) {
      cli_alert_warning("Missing columns in image-level data: {paste(missing_img, collapse = ', ')}")
    }
    
    if (length(missing_obs) > 0) {
      cli_alert_warning("Missing columns in observation data: {paste(missing_obs, collapse = ', ')}")
    }
    
    cli_alert_success("Leaf outputs validated successfully")
    return(TRUE)
    
  }, error = function(e) {
    cli_alert_danger("Error validating outputs: {e$message}")
    return(FALSE)
  })
}