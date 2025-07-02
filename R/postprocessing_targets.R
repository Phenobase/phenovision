# Post-processing functions for creating four output types with observation-level aggregation
require(tidyverse)
require(tidymodels)
require(probably)
require(arrow)
require(uuid)
require(cli)

# Enhanced threshold application for both flower/fruit and leaf models
apply_thresholds <- function(results, pipeline_type = c("flower_fruit", "leaf"), 
                           flower_thresholds = c(0.28, 0.84, 0.85), 
                           fruit_thresholds = c(0.23, 0.53, 0.75),
                           leaf_thresholds = NULL) {
  
  pipeline_type <- match.arg(pipeline_type)
  
  if (pipeline_type == "flower_fruit") {
    # Calculate buffer ranges for equivocal zones
    fruit_range <- c(fruit_thresholds[2] - fruit_thresholds[1],
                    fruit_thresholds[3] - fruit_thresholds[2])
    
    flower_range <- c(flower_thresholds[2] - flower_thresholds[1],
                     flower_thresholds[3] - flower_thresholds[2])
    
    results <- results |>
      mutate(
        .class_flower = make_two_class_pred(.pred_flower, c("Detected", "Not Detected"),
                                          threshold = flower_thresholds[2],
                                          buffer = flower_range),
        .class_fruit = make_two_class_pred(.pred_fruit, c("Detected", "Not Detected"),
                                         threshold = fruit_thresholds[2],
                                         buffer = fruit_range),
        .equivocal_flower = ifelse(is_equivocal(.class_flower), "Equivocal", "Unequivocal"),
        .equivocal_fruit = ifelse(is_equivocal(.class_fruit), "Equivocal", "Unequivocal")
      ) |>
      mutate(
        .class_flower = make_two_class_pred(.pred_flower, c("Detected", "Not Detected"),
                                          threshold = flower_thresholds[2]),
        .class_fruit = make_two_class_pred(.pred_fruit, c("Detected", "Not Detected"),
                                         threshold = fruit_thresholds[2])
      )
  } else if (pipeline_type == "leaf") {
    # Leaf pipeline uses asymmetric thresholds
    if (is.null(leaf_thresholds)) {
      stop("Leaf thresholds must be provided for leaf pipeline")
    }
    
    results <- results |>
      mutate(
        .class_gr = make_two_class_pred(
          estimate = .pred_leaves_green,
          levels = c("Detected", "Not Detected"),
          threshold = leaf_thresholds$gr[1],
          buffer = leaf_thresholds$gr[-1]
        ),
        .class_cl = make_two_class_pred(
          estimate = .pred_leaves_colored,
          levels = c("Detected", "Not Detected"),
          threshold = leaf_thresholds$cl[1],
          buffer = leaf_thresholds$cl[-1]
        ),
        .class_bb = make_two_class_pred(
          estimate = .pred_leaves_breaking_buds,
          levels = c("Detected", "Not Detected"),
          threshold = leaf_thresholds$bb[1],
          buffer = leaf_thresholds$bb[-1]
        ),
        .equivocal_gr = ifelse(is_equivocal(.class_gr), "Equivocal", "Unequivocal"),
        .equivocal_cl = ifelse(is_equivocal(.class_cl), "Equivocal", "Unequivocal"),
        .equivocal_bb = ifelse(is_equivocal(.class_bb), "Equivocal", "Unequivocal")
      ) |>
      mutate(
        .class_gr = make_two_class_pred(
          estimate = .pred_leaves_green,
          levels = c("Detected", "Not Detected"),
          threshold = leaf_thresholds$gr[1]
        ),
        .class_cl = make_two_class_pred(
          estimate = .pred_leaves_colored,
          levels = c("Detected", "Not Detected"),
          threshold = leaf_thresholds$cl[1]
        ),
        .class_bb = make_two_class_pred(
          estimate = .pred_leaves_breaking_buds,
          levels = c("Detected", "Not Detected"),
          threshold = leaf_thresholds$bb[1]
        )
      )
    
    # Add derived "no leaves" class
    results <- results |>
      mutate(
        .class_no = case_when(
          .class_gr == "Not Detected" & 
          .class_cl == "Not Detected" & 
          .class_bb == "Not Detected" ~ "Detected",
          TRUE ~ "Not Detected"
        ),
        .equivocal_no = case_when(
          .equivocal_gr == "Equivocal" | 
          .equivocal_cl == "Equivocal" | 
          .equivocal_bb == "Equivocal" ~ "Equivocal",
          TRUE ~ "Unequivocal"
        ),
        # Add prediction probability for no leaves (inverse of max leaf probability)
        .pred_leaves_no = 1 - pmax(.pred_leaves_green, .pred_leaves_colored, .pred_leaves_breaking_buds)
      )
  }
  
  results
}

# Aggregate predictions to observation level with counts
aggregate_to_observation_level <- function(dataset, pipeline_type = c("flower_fruit", "leaf"), 
                                         seed = 123) {
  cli_progress_step("Aggregating to observation level...")
  
  pipeline_type <- match.arg(pipeline_type)
  set.seed(seed)  # For reproducible representative image selection
  
  if (pipeline_type == "flower_fruit") {
    # Process flower and fruit separately then combine
    flower_agg <- dataset |>
      group_by(observation_uuid) |>
      summarise(
        n_images_total = n(),
        n_flower_detected_unequivocal = sum(.class_flower == "Detected" & .equivocal_flower == "Unequivocal"),
        n_flower_detected_equivocal = sum(.class_flower == "Detected" & .equivocal_flower == "Equivocal"),
        n_flower_notdetected_unequivocal = sum(.class_flower == "Not Detected" & .equivocal_flower == "Unequivocal"),
        n_flower_notdetected_equivocal = sum(.class_flower == "Not Detected" & .equivocal_flower == "Equivocal"),
        # Select representative image for each category
        repr_flower_detected_unequivocal = list(file_name[.class_flower == "Detected" & .equivocal_flower == "Unequivocal"]),
        repr_flower_detected_equivocal = list(file_name[.class_flower == "Detected" & .equivocal_flower == "Equivocal"]),
        repr_flower_notdetected_unequivocal = list(file_name[.class_flower == "Not Detected" & .equivocal_flower == "Unequivocal"]),
        repr_flower_notdetected_equivocal = list(file_name[.class_flower == "Not Detected" & .equivocal_flower == "Equivocal"]),
        .groups = "drop"
      ) |>
      mutate(
        # Randomly select one representative image from each category
        repr_flower_detected_unequivocal = map_chr(repr_flower_detected_unequivocal, 
                                                  ~if(length(.x) > 0) sample(.x, 1) else NA_character_),
        repr_flower_detected_equivocal = map_chr(repr_flower_detected_equivocal, 
                                                ~if(length(.x) > 0) sample(.x, 1) else NA_character_),
        repr_flower_notdetected_unequivocal = map_chr(repr_flower_notdetected_unequivocal, 
                                                     ~if(length(.x) > 0) sample(.x, 1) else NA_character_),
        repr_flower_notdetected_equivocal = map_chr(repr_flower_notdetected_equivocal, 
                                                   ~if(length(.x) > 0) sample(.x, 1) else NA_character_)
      )
    
    fruit_agg <- dataset |>
      group_by(observation_uuid) |>
      summarise(
        n_fruit_detected_unequivocal = sum(.class_fruit == "Detected" & .equivocal_fruit == "Unequivocal"),
        n_fruit_detected_equivocal = sum(.class_fruit == "Detected" & .equivocal_fruit == "Equivocal"),
        n_fruit_notdetected_unequivocal = sum(.class_fruit == "Not Detected" & .equivocal_fruit == "Unequivocal"),
        n_fruit_notdetected_equivocal = sum(.class_fruit == "Not Detected" & .equivocal_fruit == "Equivocal"),
        # Select representative image for each category
        repr_fruit_detected_unequivocal = list(file_name[.class_fruit == "Detected" & .equivocal_fruit == "Unequivocal"]),
        repr_fruit_detected_equivocal = list(file_name[.class_fruit == "Detected" & .equivocal_fruit == "Equivocal"]),
        repr_fruit_notdetected_unequivocal = list(file_name[.class_fruit == "Not Detected" & .equivocal_fruit == "Unequivocal"]),
        repr_fruit_notdetected_equivocal = list(file_name[.class_fruit == "Not Detected" & .equivocal_fruit == "Equivocal"]),
        .groups = "drop"
      ) |>
      mutate(
        # Randomly select one representative image from each category
        repr_fruit_detected_unequivocal = map_chr(repr_fruit_detected_unequivocal, 
                                                 ~if(length(.x) > 0) sample(.x, 1) else NA_character_),
        repr_fruit_detected_equivocal = map_chr(repr_fruit_detected_equivocal, 
                                               ~if(length(.x) > 0) sample(.x, 1) else NA_character_),
        repr_fruit_notdetected_unequivocal = map_chr(repr_fruit_notdetected_unequivocal, 
                                                    ~if(length(.x) > 0) sample(.x, 1) else NA_character_),
        repr_fruit_notdetected_equivocal = map_chr(repr_fruit_notdetected_equivocal, 
                                                  ~if(length(.x) > 0) sample(.x, 1) else NA_character_)
      )
    
    # Combine flower and fruit aggregations
    obs_aggregated <- flower_agg |>
      left_join(fruit_agg, by = "observation_uuid")
    
  } else if (pipeline_type == "leaf") {
    # Process all leaf traits
    obs_aggregated <- dataset |>
      group_by(observation_uuid) |>
      summarise(
        n_images_total = n(),
        # Green leaves
        n_gr_detected_unequivocal = sum(.class_gr == "Detected" & .equivocal_gr == "Unequivocal"),
        n_gr_detected_equivocal = sum(.class_gr == "Detected" & .equivocal_gr == "Equivocal"),
        n_gr_notdetected_unequivocal = sum(.class_gr == "Not Detected" & .equivocal_gr == "Unequivocal"),
        n_gr_notdetected_equivocal = sum(.class_gr == "Not Detected" & .equivocal_gr == "Equivocal"),
        # Colored leaves
        n_cl_detected_unequivocal = sum(.class_cl == "Detected" & .equivocal_cl == "Unequivocal"),
        n_cl_detected_equivocal = sum(.class_cl == "Detected" & .equivocal_cl == "Equivocal"),
        n_cl_notdetected_unequivocal = sum(.class_cl == "Not Detected" & .equivocal_cl == "Unequivocal"),
        n_cl_notdetected_equivocal = sum(.class_cl == "Not Detected" & .equivocal_cl == "Equivocal"),
        # Breaking buds
        n_bb_detected_unequivocal = sum(.class_bb == "Detected" & .equivocal_bb == "Unequivocal"),
        n_bb_detected_equivocal = sum(.class_bb == "Detected" & .equivocal_bb == "Equivocal"),
        n_bb_notdetected_unequivocal = sum(.class_bb == "Not Detected" & .equivocal_bb == "Unequivocal"),
        n_bb_notdetected_equivocal = sum(.class_bb == "Not Detected" & .equivocal_bb == "Equivocal"),
        # No leaves
        n_no_detected_unequivocal = sum(.class_no == "Detected" & .equivocal_no == "Unequivocal"),
        n_no_detected_equivocal = sum(.class_no == "Detected" & .equivocal_no == "Equivocal"),
        n_no_notdetected_unequivocal = sum(.class_no == "Not Detected" & .equivocal_no == "Unequivocal"),
        n_no_notdetected_equivocal = sum(.class_no == "Not Detected" & .equivocal_no == "Equivocal"),
        .groups = "drop"
      )
    
    # Add representative images (simplified for leaf pipeline - can be expanded if needed)
    # For now, just select one representative image per observation
    repr_images <- dataset |>
      group_by(observation_uuid) |>
      slice_sample(n = 1) |>
      select(observation_uuid, representative_image = file_name)
    
    obs_aggregated <- obs_aggregated |>
      left_join(repr_images, by = "observation_uuid")
  }
  
  # Add back metadata from first image of each observation
  metadata_cols <- dataset |>
    group_by(observation_uuid) |>
    slice(1) |>
    select(-contains(".pred"), -contains(".class"), -contains(".equivocal"), 
           -contains(".logit"), -file_name, -path)
  
  obs_aggregated |>
    left_join(metadata_cols, by = "observation_uuid")
}

# Select representative observation for filtered datasets
select_observation_representative <- function(obs_aggregated, pipeline_type = c("flower_fruit", "leaf"),
                                            trait = NULL, category = "detected_unequivocal") {
  pipeline_type <- match.arg(pipeline_type)
  
  if (pipeline_type == "flower_fruit") {
    if (is.null(trait) || !trait %in% c("flower", "fruit")) {
      stop("Trait must be 'flower' or 'fruit' for flower_fruit pipeline")
    }
    
    # Select observations with the specified category
    col_name <- paste0("n_", trait, "_", category)
    repr_col <- paste0("repr_", trait, "_", category)
    
    obs_aggregated |>
      filter(!!sym(col_name) > 0) |>
      mutate(file_name = !!sym(repr_col)) |>
      select(-starts_with("n_"), -starts_with("repr_"))
    
  } else if (pipeline_type == "leaf") {
    # For leaf pipeline, handle multiple traits
    if (!is.null(trait)) {
      col_name <- paste0("n_", trait, "_", category)
      obs_aggregated |>
        filter(!!sym(col_name) > 0) |>
        select(-starts_with("n_"))
    } else {
      # Return all observations with any detected unequivocal
      obs_aggregated |>
        filter(n_gr_detected_unequivocal > 0 | 
               n_cl_detected_unequivocal > 0 | 
               n_bb_detected_unequivocal > 0) |>
        select(-starts_with("n_"))
    }
  }
}

# Create ingestion format for Phenobase database
create_ingestion_format <- function(dataset, fam_stats, pipeline_type = c("flower_fruit", "leaf"),
                                  filter_detected = TRUE, filter_low_certainty = TRUE) {
  
  pipeline_type <- match.arg(pipeline_type)
  
  cli_progress_step("Creating ingestion format for {pipeline_type}...")
  
  # Add family-level statistics
  if (!is.null(fam_stats)) {
    dataset <- dataset |>
      left_join(fam_stats, by = "family")
  }
  
  if (pipeline_type == "flower_fruit") {
    # For flower/fruit, we need to reshape the data
    # First get the prediction values from the original image-level data
    # This function should receive observation-level data with selected images
    
    dataset_long <- dataset |>
      pivot_longer(
        cols = c(contains("flower"), contains("fruit")),
        names_to = c(".value", "trait"),
        names_pattern = "(.+)_(flower|fruit)$"
      ) |>
      filter(!is.na(trait))
    
  } else if (pipeline_type == "leaf") {
    # Similar reshaping for leaf traits
    dataset_long <- dataset |>
      pivot_longer(
        cols = c(contains("_gr"), contains("_cl"), contains("_bb"), contains("_no")),
        names_to = c(".value", "trait_code"),
        names_pattern = "(.+)_(gr|cl|bb|no)$"
      ) |>
      mutate(
        trait = case_match(
          trait_code,
          "gr" ~ "green leaves",
          "cl" ~ "colored leaves",
          "bb" ~ "breaking leaf buds",
          "no" ~ "no leaves"
        )
      ) |>
      filter(!is.na(trait))
  }
  
  # Apply filters if requested
  if (filter_detected && filter_low_certainty) {
    # For ingestion, we want only detected & unequivocal
    dataset_long <- dataset_long |>
      filter(str_detect(names(dataset_long), "detected_unequivocal"))
  }
  
  # Format for ingestion
  dataset_long |>
    mutate(
      machine_learning_annotation_id = UUIDgenerate(n = n()),
      certainty = "High",  # Since we filtered for unequivocal
      prediction_class = "Detected"  # Since we filtered for detected
    ) |>
    select(
      machine_learning_annotation_id,
      datasource,
      verbatim_date = observed_on,
      day_of_year,
      year,
      latitude,
      longitude,
      coordinate_uncertainty_meters = positional_accuracy,
      family,
      genus,
      scientific_name = name,
      taxon_rank = rank,
      basis_of_record,
      trait,
      observed_image_guid,
      observed_image_url,
      observed_metadata_url = inat_URL,
      certainty,
      model_uri,
      prediction_class,
      everything()
    )
}

# Process predictions to create all four output types
create_all_output_types <- function(predictions, meta_images, meta_obs, meta_taxa, fam_stats,
                                  model_doi, pipeline_type = c("flower_fruit", "leaf"),
                                  thresholds = NULL, output_dir = NULL, seed = 123) {
  
  pipeline_type <- match.arg(pipeline_type)
  
  cli_alert_info("Creating all four output types for {pipeline_type} pipeline")
  
  # Apply thresholds based on pipeline type
  if (pipeline_type == "flower_fruit") {
    predictions_with_class <- apply_thresholds(
      predictions, 
      pipeline_type = "flower_fruit",
      flower_thresholds = thresholds$flower,
      fruit_thresholds = thresholds$fruit
    )
  } else {
    predictions_with_class <- apply_thresholds(
      predictions,
      pipeline_type = "leaf",
      leaf_thresholds = thresholds
    )
  }
  
  # 1. Create image-level dataset with all metadata
  cli_h2("Creating image-level dataset")
  image_level <- prepare_unfiltered_dataset(
    predictions_with_class,
    meta_images,
    meta_obs,
    meta_taxa,
    model_doi
  ) |>
    add_version_metadata(
      pipeline_type = pipeline_type,
      description = "Image-level predictions with all detection and certainty combinations"
    )
  
  # 2. Create observation-level aggregated dataset
  cli_h2("Creating observation-level aggregated dataset")
  obs_aggregated <- aggregate_to_observation_level(
    image_level,
    pipeline_type = pipeline_type,
    seed = seed
  ) |>
    add_version_metadata(
      pipeline_type = pipeline_type,
      description = "Observation-level data with counts of images in each detection/certainty category"
    )
  
  # 3. Create filtered dataset for Zenodo (observation-level, detected & unequivocal only)
  cli_h2("Creating filtered dataset for Zenodo")
  if (pipeline_type == "flower_fruit") {
    # Create separate datasets for flower and fruit, then combine
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
    
    # Combine and remove duplicate observations
    zenodo_filtered <- bind_rows(
      flower_filtered |> mutate(trait = "flower"),
      fruit_filtered |> mutate(trait = "fruit")
    ) |>
      distinct() |>
      add_version_metadata(
        pipeline_type = pipeline_type,
        description = "Filtered dataset for Zenodo (detected & unequivocal only)"
      )
  } else {
    zenodo_filtered <- select_observation_representative(
      obs_aggregated,
      pipeline_type = "leaf",
      category = "detected_unequivocal"
    ) |>
      add_version_metadata(
        pipeline_type = pipeline_type,
        description = "Filtered dataset for Zenodo (detected & unequivocal only)"
      )
  }
  
  # 4. Create ingestion format for Phenobase
  cli_h2("Creating ingestion format for Phenobase")
  # For ingestion, we need to get back the prediction probabilities
  # This requires joining back with image-level data
  ingestion_base <- image_level |>
    inner_join(
      zenodo_filtered |> select(observation_uuid, file_name),
      by = c("observation_uuid", "file_name")
    )
  
  ingestion <- create_ingestion_format(
    ingestion_base,
    fam_stats,
    pipeline_type = pipeline_type,
    filter_detected = TRUE,
    filter_low_certainty = TRUE
  ) |>
    add_version_metadata(
      pipeline_type = pipeline_type,
      description = "Formatted dataset for Phenobase database ingestion"
    )
  
  # Return list of all four datasets
  list(
    image_level = image_level,
    observation_aggregated = obs_aggregated,
    zenodo_filtered = zenodo_filtered,
    phenobase_ingestion = ingestion
  )
}

# Re-aggregate observations after combining batches
reaggregate_after_batch_combination <- function(combined_image_data, pipeline_type = c("flower_fruit", "leaf"),
                                               seed = 123) {
  cli_alert_info("Re-aggregating observations after batch combination...")
  
  # Simply call the aggregation function again on the combined data
  aggregate_to_observation_level(combined_image_data, pipeline_type, seed)
}

# Helper function to save all output types
save_all_outputs <- function(outputs, output_dir, pipeline_type, batch_id = NULL, version = NULL) {
  if (is.null(version)) {
    version <- format(Sys.Date(), "%Y%m%d")
  }
  
  # Include batch ID in path if provided
  if (!is.null(batch_id)) {
    base_dir <- file.path(output_dir, pipeline_type, version, paste0("batch_", batch_id))
  } else {
    base_dir <- file.path(output_dir, pipeline_type, version)
  }
  
  dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)
  
  cli_alert_info("Saving all outputs to {base_dir}")
  
  # Save each output type
  paths <- list(
    image_level = save_dataset(
      outputs$image_level,
      file.path(base_dir, paste0(pipeline_type, "_image_level_", version, ".csv")),
      format = "csv"
    ),
    observation_aggregated = save_dataset(
      outputs$observation_aggregated,
      file.path(base_dir, paste0(pipeline_type, "_observation_aggregated_", version, ".csv")),
      format = "csv"
    ),
    zenodo_filtered = save_dataset(
      outputs$zenodo_filtered,
      file.path(base_dir, paste0(pipeline_type, "_filtered_zenodo_", version, ".csv")),
      format = "csv"
    ),
    phenobase_ingestion = save_dataset(
      outputs$phenobase_ingestion,
      file.path(base_dir, paste0(pipeline_type, "_phenobase_ingestion_", version, ".csv")),
      format = "csv"
    )
  )
  
  # Generate documentation for Zenodo
  generate_dataset_documentation(
    outputs$zenodo_filtered,
    file.path(base_dir, paste0(pipeline_type, "_dataset_documentation_", version, ".md"))
  )
  
  paths
}