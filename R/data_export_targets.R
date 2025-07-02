# Data export functions for Zenodo and Phenobase
require(tidyverse)
require(arrow)
require(jsonlite)
require(cli)

# Create Zenodo metadata JSON
create_zenodo_metadata <- function(dataset, pipeline_type = c("flower_fruit", "leaf"),
                                 title = NULL, description = NULL, creators = NULL,
                                 version = NULL) {
  
  pipeline_type <- match.arg(pipeline_type)
  
  if (is.null(version)) {
    version <- attr(dataset, "version") %||% format(Sys.Date(), "%Y%m%d")
  }
  
  if (is.null(title)) {
    title <- paste0("PhenoVision ", 
                   str_to_title(str_replace(pipeline_type, "_", " ")), 
                   " Phenology Predictions from iNaturalist Images")
  }
  
  if (is.null(description)) {
    description <- paste0(
      "Machine learning predictions of plant phenological stages from iNaturalist images ",
      "using Vision Transformer (ViT-Large/16) models fine-tuned on expert annotations. ",
      "This dataset contains ", format(nrow(dataset), big.mark = ","), " observation-level records ",
      "with high-confidence predictions (detected and unequivocal only). ",
      ifelse(pipeline_type == "flower_fruit",
             "Predictions include flowering and fruiting stages.",
             "Predictions include green leaves, colored leaves, and breaking leaf buds.")
    )
  }
  
  if (is.null(creators)) {
    creators <- list(
      list(name = "PhenoVision Team", affiliation = "University of Florida")
    )
  }
  
  # Create metadata structure
  metadata <- list(
    title = title,
    upload_type = "dataset",
    description = description,
    creators = creators,
    version = version,
    keywords = c("phenology", "plant phenology", "machine learning", "computer vision",
                "iNaturalist", "citizen science", pipeline_type),
    access_right = "open",
    license = "cc-by-4.0",
    related_identifiers = list(
      list(
        identifier = ifelse(pipeline_type == "flower_fruit", "10.57967/hf/2763", "10.57967/hf/5785"),
        relation = "isCompiledBy",
        scheme = "doi"
      )
    ),
    communities = list(
      list(identifier = "phenology"),
      list(identifier = "inaturalist")
    ),
    grants = list(),
    subjects = list(
      list(term = "Plant Phenology", scheme = "gnd"),
      list(term = "Computer Vision", scheme = "gnd"),
      list(term = "Biodiversity Informatics", scheme = "gnd")
    )
  )
  
  metadata
}

# Export dataset for Zenodo with all required files
export_for_zenodo <- function(dataset, output_dir, pipeline_type = c("flower_fruit", "leaf"),
                            metadata = NULL, create_readme = TRUE) {
  
  pipeline_type <- match.arg(pipeline_type)
  version <- attr(dataset, "version") %||% format(Sys.Date(), "%Y%m%d")
  
  # Create output directory
  zenodo_dir <- file.path(output_dir, paste0("zenodo_", pipeline_type, "_", version))
  dir.create(zenodo_dir, recursive = TRUE, showWarnings = FALSE)
  
  cli_alert_info("Preparing Zenodo export in {zenodo_dir}")
  
  # Save main dataset
  data_file <- file.path(zenodo_dir, paste0("phenovision_", pipeline_type, "_", version, ".csv"))
  write_csv(dataset, data_file)
  cli_alert_success("Saved main dataset: {basename(data_file)}")
  
  # Create and save metadata
  if (is.null(metadata)) {
    metadata <- create_zenodo_metadata(dataset, pipeline_type)
  }
  
  metadata_file <- file.path(zenodo_dir, ".zenodo.json")
  write_json(metadata, metadata_file, pretty = TRUE, auto_unbox = TRUE)
  cli_alert_success("Created Zenodo metadata: {basename(metadata_file)}")
  
  # Create README if requested
  if (create_readme) {
    readme_content <- create_zenodo_readme(dataset, pipeline_type, version)
    readme_file <- file.path(zenodo_dir, "README.md")
    writeLines(readme_content, readme_file)
    cli_alert_success("Created README: {basename(readme_file)}")
  }
  
  # Create data dictionary
  dict_file <- file.path(zenodo_dir, "data_dictionary.csv")
  data_dict <- create_data_dictionary(dataset, pipeline_type)
  write_csv(data_dict, dict_file)
  cli_alert_success("Created data dictionary: {basename(dict_file)}")
  
  # Create citation file
  citation_file <- file.path(zenodo_dir, "CITATION.cff")
  citation_content <- create_citation_file(metadata, pipeline_type)
  writeLines(citation_content, citation_file)
  cli_alert_success("Created citation file: {basename(citation_file)}")
  
  # Compress if large
  if (file.size(data_file) > 100 * 1024 * 1024) {  # > 100MB
    cli_progress_step("Creating compressed archive...")
    zip_file <- paste0(zenodo_dir, ".zip")
    zip(zip_file, zenodo_dir, flags = "-r9")
    cli_alert_success("Created compressed archive: {basename(zip_file)}")
  }
  
  zenodo_dir
}

# Create README for Zenodo
create_zenodo_readme <- function(dataset, pipeline_type, version) {
  
  n_obs <- length(unique(dataset$observation_uuid))
  n_images <- nrow(dataset)
  n_species <- length(unique(dataset$scientific_name))
  n_genera <- length(unique(dataset$genus))
  n_families <- length(unique(dataset$family))
  
  if (pipeline_type == "flower_fruit") {
    traits <- c("flowering", "fruiting")
    model_doi <- "10.57967/hf/2763"
  } else {
    traits <- c("green leaves", "colored leaves", "breaking leaf buds")
    model_doi <- "10.57967/hf/5785"
  }
  
  readme <- glue::glue("
# PhenoVision {str_to_title(str_replace(pipeline_type, '_', ' '))} Phenology Dataset

Version: {version}

## Overview

This dataset contains machine learning predictions of plant phenological stages from iNaturalist images. 
The predictions were generated using Vision Transformer (ViT-Large/16) models fine-tuned on expert annotations.

## Dataset Statistics

- **Observations**: {format(n_obs, big.mark = ',')}
- **Images**: {format(n_images, big.mark = ',')}
- **Species**: {format(n_species, big.mark = ',')}
- **Genera**: {format(n_genera, big.mark = ',')}
- **Families**: {format(n_families, big.mark = ',')}
- **Phenological traits**: {paste(traits, collapse = ', ')}

## Model Information

- **Architecture**: Vision Transformer (ViT-Large/16)
- **Model DOI**: [{model_doi}](https://doi.org/{model_doi})
- **Training**: Fine-tuned on expert-annotated iNaturalist images
- **Predictions**: Binary classification for each phenological trait

## Data Processing

This dataset includes only high-confidence predictions:
- Detection status: Detected
- Certainty: Unequivocal (high confidence)

Each observation represents one iNaturalist observation with a representative image selected 
from potentially multiple images per observation.

## File Descriptions

- `phenovision_{pipeline_type}_{version}.csv`: Main dataset with predictions
- `data_dictionary.csv`: Description of all columns
- `README.md`: This file
- `CITATION.cff`: Citation information
- `.zenodo.json`: Zenodo metadata

## Data Fields

See `data_dictionary.csv` for detailed descriptions of all fields.

Key fields include:
- `observation_uuid`: Unique identifier for iNaturalist observation
- `scientific_name`: Species name
- `trait`: Phenological trait predicted
- `latitude`, `longitude`: Geographic coordinates
- `day_of_year`, `year`: Temporal information
- `family`, `genus`: Taxonomic classification

## Citation

Please cite this dataset using the DOI provided by Zenodo upon publication.

## License

This dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

## Contact

For questions about this dataset, please contact the PhenoVision team.
")
  
  readme
}

# Create data dictionary
create_data_dictionary <- function(dataset, pipeline_type) {
  
  # Define column descriptions
  descriptions <- tibble::tribble(
    ~column_name, ~description, ~data_type,
    "machine_learning_annotation_id", "Unique identifier for this prediction", "character",
    "observation_uuid", "iNaturalist observation identifier", "character",
    "datasource", "Source of the data (iNaturalist)", "character",
    "verbatim_date", "Date of observation", "date",
    "day_of_year", "Day of year (1-366)", "integer",
    "year", "Year of observation", "integer",
    "latitude", "Latitude in decimal degrees", "numeric",
    "longitude", "Longitude in decimal degrees", "numeric",
    "coordinate_uncertainty_meters", "Uncertainty radius in meters", "numeric",
    "family", "Taxonomic family", "character",
    "genus", "Taxonomic genus", "character",
    "scientific_name", "Scientific name of the species", "character",
    "taxon_rank", "Taxonomic rank", "character",
    "basis_of_record", "Type of observation (MachineObservation)", "character",
    "trait", "Phenological trait predicted", "character",
    "observed_image_guid", "Image identifier", "character",
    "observed_image_url", "URL to image on iNaturalist", "character",
    "observed_metadata_url", "URL to observation on iNaturalist", "character",
    "certainty", "Prediction certainty (High/Low)", "character",
    "model_uri", "DOI of the model used", "character",
    "prediction_class", "Prediction result (Detected/Not Detected)", "character",
    "prediction_probability", "Model confidence score (0-1)", "numeric",
    "file_name", "Image filename", "character",
    "quality_grade", "iNaturalist quality grade", "character"
  )
  
  # Add pipeline-specific columns
  if (pipeline_type == "flower_fruit") {
    descriptions <- descriptions |>
      bind_rows(tibble::tribble(
        ~column_name, ~description, ~data_type,
        "n_images_total", "Total images for this observation", "integer",
        "n_flower_detected_unequivocal", "Count of flower detected with high certainty", "integer",
        "n_fruit_detected_unequivocal", "Count of fruit detected with high certainty", "integer"
      ))
  } else {
    descriptions <- descriptions |>
      bind_rows(tibble::tribble(
        ~column_name, ~description, ~data_type,
        "n_images_total", "Total images for this observation", "integer",
        "n_gr_detected_unequivocal", "Count of green leaves detected with high certainty", "integer",
        "n_cl_detected_unequivocal", "Count of colored leaves detected with high certainty", "integer",
        "n_bb_detected_unequivocal", "Count of breaking buds detected with high certainty", "integer"
      ))
  }
  
  # Filter to only columns present in the dataset
  descriptions |>
    filter(column_name %in% names(dataset))
}

# Create CITATION.cff file
create_citation_file <- function(metadata, pipeline_type) {
  
  authors <- map_chr(metadata$creators, ~.x$name)
  
  cff_content <- glue::glue('
cff-version: 1.2.0
message: "If you use this dataset, please cite it as below."
title: "{metadata$title}"
version: {metadata$version}
date-released: {Sys.Date()}
authors:
{paste0("  - name: ", authors, collapse = "\n")}
type: dataset
keywords:
{paste0("  - ", metadata$keywords, collapse = "\n")}
license: CC-BY-4.0
')
  
  cff_content
}

# Export for Phenobase ingestion
export_for_phenobase <- function(dataset, output_dir, pipeline_type = c("flower_fruit", "leaf"),
                               batch_size = 50000) {
  
  pipeline_type <- match.arg(pipeline_type)
  version <- attr(dataset, "version") %||% format(Sys.Date(), "%Y%m%d")
  
  # Create output directory
  phenobase_dir <- file.path(output_dir, paste0("phenobase_", pipeline_type, "_", version))
  dir.create(phenobase_dir, recursive = TRUE, showWarnings = FALSE)
  
  cli_alert_info("Preparing Phenobase export in {phenobase_dir}")
  
  # Check if we need to split into batches
  if (nrow(dataset) > batch_size) {
    n_batches <- ceiling(nrow(dataset) / batch_size)
    cli_progress_step("Splitting into {n_batches} batches of up to {batch_size} records")
    
    for (i in seq_len(n_batches)) {
      start_idx <- (i - 1) * batch_size + 1
      end_idx <- min(i * batch_size, nrow(dataset))
      
      batch_data <- dataset[start_idx:end_idx, ]
      batch_file <- file.path(phenobase_dir, 
                             paste0("phenobase_", pipeline_type, "_", version, "_batch", i, ".csv"))
      
      write_csv(batch_data, batch_file)
      cli_alert_success("Created batch {i}: {basename(batch_file)} ({nrow(batch_data)} records)")
    }
  } else {
    # Single file
    output_file <- file.path(phenobase_dir, paste0("phenobase_", pipeline_type, "_", version, ".csv"))
    write_csv(dataset, output_file)
    cli_alert_success("Created single file: {basename(output_file)} ({nrow(dataset)} records)")
  }
  
  # Create ingestion manifest
  manifest <- list(
    dataset_type = pipeline_type,
    version = version,
    created_date = Sys.Date(),
    total_records = nrow(dataset),
    n_observations = length(unique(dataset$observation_uuid)),
    traits = unique(dataset$trait),
    files = list.files(phenobase_dir, pattern = "\\.csv$", full.names = FALSE)
  )
  
  manifest_file <- file.path(phenobase_dir, "manifest.json")
  write_json(manifest, manifest_file, pretty = TRUE, auto_unbox = TRUE)
  cli_alert_success("Created manifest: {basename(manifest_file)}")
  
  phenobase_dir
}

# Validate export completeness
validate_export <- function(export_dir, expected_files = NULL) {
  cli_h2("Validating export in {export_dir}")
  
  if (!dir.exists(export_dir)) {
    cli_alert_danger("Export directory does not exist: {export_dir}")
    return(FALSE)
  }
  
  files <- list.files(export_dir, full.names = TRUE)
  
  if (length(files) == 0) {
    cli_alert_danger("No files found in export directory")
    return(FALSE)
  }
  
  # Check for expected files
  if (!is.null(expected_files)) {
    missing <- setdiff(expected_files, basename(files))
    if (length(missing) > 0) {
      cli_alert_warning("Missing expected files: {paste(missing, collapse = ', ')}")
    }
  }
  
  # Check file sizes
  sizes <- file.size(files)
  if (any(sizes == 0)) {
    empty_files <- basename(files)[sizes == 0]
    cli_alert_danger("Empty files found: {paste(empty_files, collapse = ', ')}")
    return(FALSE)
  }
  
  # Report summary
  cli_alert_success("Export contains {length(files)} files")
  cli_alert_info("Total size: {format(sum(sizes), big.mark = ',')} bytes")
  
  TRUE
}