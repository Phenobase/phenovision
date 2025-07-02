# Common functions for both flower/fruit and leaf pipelines
require(tidyverse)
require(arrow)
require(uuid)
require(cli)

# Add taxonomy information (family and genus) to dataset
add_taxonomy_metadata <- function(dataset, meta_taxa) {
  cli_progress_step("Adding taxonomic metadata...")
  
  families <- meta_taxa |>
    filter(rank == "family") |>
    collect()
  
  genera <- meta_taxa |>
    filter(rank == "genus") |>
    collect()
  
  taxonomy <- dataset |>
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
  
  dataset |>
    left_join(fams |> select(file_name, family)) |>
    left_join(gens |> select(file_name, genus))
}

# Add observation metadata to dataset
add_observation_metadata <- function(dataset, meta_images, meta_obs) {
  cli_progress_step("Adding observation metadata...")
  
  dataset <- dataset |>
    mutate(photo_id = fs::path_ext_remove(file_name),
           extension = fs::path_ext(file_name))
  
  dataset <- dataset |>
    left_join(meta_images |>
                select(photo_id, observation_uuid) |>
                filter(photo_id %in% dataset$photo_id),
              by = "photo_id",
              copy = TRUE)
  
  dataset <- dataset |>
    mutate(inat_URL = paste0("https://www.inaturalist.org/observations/", observation_uuid),
           observed_image_url = paste0("https://www.inaturalist.org/photos/", photo_id))
  
  dataset |>
    left_join(meta_obs |>
                select(observation_uuid, latitude, longitude, positional_accuracy, 
                       taxon_id, quality_grade, observed_on) |>
                filter(observation_uuid %in% dataset$observation_uuid),
              by = "observation_uuid",
              copy = TRUE) |>
    mutate(observed_image_guid = file.path("https://inaturalist-open-data.s3.amazonaws.com/photos", 
                                          photo_id, paste0("small.", extension)))
}

# Create unfiltered dataset with all predictions
prepare_unfiltered_dataset <- function(predictions, meta_images, meta_obs, meta_taxa, 
                                     model_doi, datasource = "iNaturalist") {
  cli_alert_info("Preparing unfiltered dataset with all predictions")
  
  # Add metadata
  dataset <- predictions |>
    distinct(file_name, .keep_all = TRUE) |>
    add_observation_metadata(meta_images, meta_obs) |>
    add_taxonomy_metadata(meta_taxa)
  
  # Add temporal information
  dataset |>
    mutate(datasource = datasource,
           day_of_year = yday(observed_on),
           year = year(observed_on),
           model_uri = model_doi,
           basis_of_record = "MachineObservation")
}

# Create filtered dataset suitable for public release (e.g., Zenodo)
prepare_filtered_dataset <- function(unfiltered_dataset, remove_undetected = TRUE, 
                                   remove_equivocal = TRUE) {
  cli_alert_info("Preparing filtered dataset for public release")
  
  dataset <- unfiltered_dataset
  
  if (remove_undetected) {
    cli_progress_step("Removing 'Not Detected' records...")
    dataset <- dataset |>
      filter(if_any(contains(".class"), ~ . == "Detected"))
  }
  
  if (remove_equivocal) {
    cli_progress_step("Removing equivocal records...")
    dataset <- dataset |>
      filter(if_all(contains(".equivocal"), ~ . == "Unequivocal"))
  }
  
  dataset
}

# Add version metadata for dataset tracking
add_version_metadata <- function(dataset, pipeline_type = c("flower_fruit", "leaf"), 
                               version = NULL, description = NULL) {
  if (is.null(version)) {
    version <- format(Sys.Date(), "%Y%m%d")
  }
  
  attr(dataset, "pipeline_type") <- match.arg(pipeline_type)
  attr(dataset, "version") <- version
  attr(dataset, "created_date") <- Sys.Date()
  attr(dataset, "description") <- description
  attr(dataset, "row_count") <- nrow(dataset)
  
  dataset
}

# Generate dataset documentation for Zenodo
generate_dataset_documentation <- function(dataset, output_path) {
  cli_progress_step("Generating dataset documentation...")
  
  pipeline_type <- attr(dataset, "pipeline_type")
  version <- attr(dataset, "version")
  created_date <- attr(dataset, "created_date")
  row_count <- attr(dataset, "row_count")
  
  doc_content <- glue::glue("
# PhenoVision {str_to_title(str_replace(pipeline_type, '_', ' '))} Predictions Dataset

## Overview
This dataset contains machine learning predictions of plant phenological stages from iNaturalist images.

## Dataset Information
- Pipeline Type: {pipeline_type}
- Version: {version}
- Creation Date: {created_date}
- Number of Records: {format(row_count, big.mark = ',')}

## Model Information
- Model: Vision Transformer (ViT-Large/16)
- Training: Fine-tuned on iNaturalist plant phenology annotations

## Data Fields
{paste('- ', names(dataset), collapse = '\n')}

## Citation
Please cite this dataset as:
[Citation information to be added]

## License
[License information to be added]
")
  
  writeLines(doc_content, output_path)
  cli_alert_success("Documentation saved to {output_path}")
}

# Save dataset with appropriate format and compression
save_dataset <- function(dataset, output_path, format = c("csv", "parquet", "rds")) {
  format <- match.arg(format)
  
  cli_progress_step("Saving dataset as {format}...")
  
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  
  switch(format,
    csv = {
      write_csv(dataset, output_path)
      # Also create compressed version for large files
      if (file.size(output_path) > 100 * 1024 * 1024) {  # > 100MB
        cli_progress_step("Creating compressed version...")
        zip_path <- paste0(output_path, ".zip")
        zip(zip_path, output_path, flags = "-9j")
      }
    },
    parquet = write_parquet(dataset, output_path),
    rds = saveRDS(dataset, output_path)
  )
  
  cli_alert_success("Dataset saved to {output_path}")
  output_path
}

# Check for corrupted images and save list
handle_corrupted_images <- function(predictions, output_dir, model_version) {
  corrupt_images <- predictions |>
    filter(corrupt == TRUE) |>
    select(file_name, path)
  
  if (nrow(corrupt_images) > 0) {
    cli_alert_warning("Found {nrow(corrupt_images)} corrupted images")
    
    corrupt_path <- file.path(output_dir, model_version, "corrupted_image_list.csv")
    write_csv(corrupt_images, corrupt_path)
    cli_alert_info("Corrupted image list saved to {corrupt_path}")
  }
  
  corrupt_images
}