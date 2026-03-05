# PhenoVision Inference Pipeline
#
# This targets pipeline runs inference on plant images using both reproductive
# structures and leaf phenology models. The pipeline has independent branches
# for each model type, enabling smart re-processing:
# - Changing reproductive model DOI triggers only reproductive re-inference
# - Changing leaf model DOI triggers only leaf re-inference
# - New images trigger both
#
# Usage:
#   targets::tar_make(script = "_targets_inference.R")
#   Rscript run_pipeline.R --pipeline=inference

# =============================================================================
# Setup
# =============================================================================

# Load common configuration
source("_targets_common.R")

# Load functions
source_common()     # Common functions (model loading, etc.)
source_inference()  # Inference-specific functions

# Additional packages
library(targets)
library(tarchetypes)  # Modern targets syntax
library(conflicted)
conflicts_prefer(dplyr::filter)
conflicts_prefer(lubridate::yday)
conflicts_prefer(lubridate::year)

# Configure targets for parallel execution (reads TARGETS_WORKERS or SLURM_CPUS_PER_TASK env var)
setup_targets_parallel()

# =============================================================================
# Pipeline (Modern tar_plan() Syntax)
# =============================================================================

tar_plan(

  # =========================================================================
  # Configuration (Separate Targets for Granular Dependencies)
  # =========================================================================

  # Model DOIs (changing these triggers re-inference)
  model_doi_repro = config$model_doi_repro,
  model_doi_leaves = config$model_doi_leaves,

  # Worker configuration
  num_targets_workers = config$num_targets_workers,
  num_workers_data = config$num_workers_data,

  # Paths from common config
  metadata_photos = paths$metadata_photos,
  metadata_root = paths$metadata_root,
  images_root = paths$images_root,
  output_repro = paths$output_repro,
  output_leaves = paths$output_leaves,

  # Leaf-specific configuration
  rob_annot_path = "data/leaves/phenobase_dwca_annotation/rob_leaf_breaking_buds_annotation.csv",
  genera_exclude_leaves = c("Logfia", "Oxalis", "Viola"),

  # Threshold and family stats paths (TODO: link these to model DOI automatically)
  thresholds_leaves_path = "output/leaves/phenovision-init_model2_04_11_2025/epoch_1_threshold_buffers.csv",
  fam_dat_leaves_path = "output/leaves/phenovision-init_model2_04_11_2025/family_stats.csv",

  # Reproductive threshold and family stats paths
  thresholds_repro_path = "output/reproductive/v1.1.0/final_buffer_params.csv",
  fam_dat_repro_path = "output/reproductive/v1.1.0/family_stats.csv",

  # =========================================================================
  # Model Configuration
  # =========================================================================

  # Create output directories
  tar_target(model_doi_path_repro, gsub("\\/", "_", model_doi_repro)),
  tar_target(model_doi_path_leaves, gsub("\\/", "_", model_doi_leaves)),
  tar_target(results_dir_repro, file.path(output_repro, model_doi_path_repro)),
  tar_target(results_dir_leaves, file.path(output_leaves, model_doi_path_leaves)),

  # Load models
  tar_target(model_repro, load_phenovision(model_doi_repro, type = "classifier")),
  tar_target(model_leaves, load_phenovision(model_doi_leaves, type = "classifier")),

  # Model versions
  tar_target(model_vers_repro, model_version(model_doi_repro)),
  tar_target(model_vers_leaves, model_version(model_doi_leaves)),

  # =========================================================================
  # Metadata and Paths
  # =========================================================================

  tar_target(meta_images_path, metadata_photos),
  tar_target(meta_obs_path, file.path(metadata_root, "observations/part-0.parquet")),
  tar_target(meta_taxa_path, file.path(metadata_root, "taxa/part-0.parquet")),
  tar_target(meta_observer_path, file.path(metadata_root, "observers/part-0.parquet")),

  # =========================================================================
  # Image Loading and Batching
  # =========================================================================

  # Load all images (changing this triggers processing new images)
  tar_target(
    images,
    {
      arrow::open_dataset(meta_images_path) |>
        dplyr::select(photo_id, batch_j, extension, taxon_id) |>
        dplyr::collect() |>
        tidyr::drop_na(batch_j) |>
        dplyr::mutate(
          path = file.path(
            images_root,
            paste0("batch_", batch_j),
            paste0(photo_id, ".", extension)
          )
        )
    }
  ),

  # Create batch groups for parallelization
  tar_target(batches, unique(images$batch_j)),

  tar_target(
    batch_df,
    tibble::tibble(
      batch_j = batches,
      run_name = santoku::chop_n(batch_j, 10, labels = santoku::lbl_seq(start = "1"))
    )
  ),

  tar_target(
    images_df,
    images |> dplyr::left_join(batch_df, by = "batch_j")
  ),

  # Group images by run_name for pattern mapping
  tar_target(
    images_batch,
    images_df |>
      dplyr::group_by(run_name) |>
      targets::tar_group(),
    iteration = "group"
  ),

  # =========================================================================
  # Taxonomy
  # =========================================================================

  tar_target(
    taxonomy,
    {
      arrow::open_dataset(meta_images_path) |>
        dplyr::distinct(taxon_id) |>
        dplyr::left_join(
          arrow::open_dataset(meta_taxa_path) |>
            dplyr::select(taxon_id, ancestry),
          by = "taxon_id",
          copy = TRUE
        ) |>
        dplyr::collect() |>
        dplyr::select(taxon_id, ancestry) |>
        dplyr::mutate(taxa_ids = stringr::str_split(ancestry, "/")) |>
        dplyr::select(-ancestry) |>
        tidyr::unnest_longer(taxa_ids, transform = as.integer)
    }
  ),

  tar_target(
    families,
    arrow::open_dataset(meta_taxa_path) |>
      dplyr::filter(rank == "family") |>
      dplyr::collect()
  ),

  tar_target(
    genera,
    arrow::open_dataset(meta_taxa_path) |>
      dplyr::filter(rank == "genus") |>
      dplyr::collect()
  ),

  # =========================================================================
  # Field Mapping for Phenobase
  # =========================================================================

  tar_target(
    field_map,
    tibble::tribble(
      ~new_field, ~old_field,
      "dataSource", "datasource",
      "scientificName", "scientific_name",
      "trait", "trait",
      "family", "family",
      "year", "year",
      "dayOfYear", "day_of_year",
      "latitude", "latitude",
      "longitude", "longitude",
      "observedMetadataUrl", "observed_metadata_url",
      "annotationMethod", "annotation_method",
      "occurrenceID", "observation_uuid",
      "genus", "genus",
      "date", "verbatim_date",
      "recordedBy", "recorded_by",
      "coordinateUncertaintyInMeters", "coordinate_uncertainty_meters",
      "verbatimTrait", "trait",
      "modelUri", "model_uri",
      "accuracyExcludingUncertainFamily", "accuracy_excluding_certainty_family",
      "ObservedImageUrl", "observed_image_url",
      "predictionClass", "detected",
      "countImages", "count_images",
      "countFamily", "count_family",
      "certainty", "certainty",
      "predictionProbability", "pred_med",
      "proportionCertaintyFamily", "proportion_certainty_family",
      "accuracyFamily", "accuracy_family"
    )
  ),

  # =========================================================================
  # REPRODUCTIVE STRUCTURES BRANCH
  # =========================================================================

  # --- Threshold Loading ---
  # Load and convert threshold buffer params to expected list format

  tar_target(
    thresholds_repro_raw,
    readr::read_csv(thresholds_repro_path)
  ),

  tar_target(
    thresholds_repro,
    {
      # Convert CSV to list format expected by threshold_annotations()
      fl_row <- thresholds_repro_raw |> dplyr::filter(class == "flower")
      fr_row <- thresholds_repro_raw |> dplyr::filter(class == "fruit")

      list(
        fl = c(fl_row$threshold, fl_row$buffer_lower, fl_row$buffer_upper),
        fr = c(fr_row$threshold, fr_row$buffer_lower, fr_row$buffer_upper)
      )
    }
  ),

  # --- Family Statistics ---

  tar_target(
    fam_dat_repro_raw,
    readr::read_csv(fam_dat_repro_path)
  ),

  tar_target(
    fam_dat_long_repro,
    {
      # The reproductive family stats file has a different format
      # (family, n, n_positive, accuracy, class)
      # Convert to the expected long format for aggregate_by_obs

      # Note: This assumes equiv_prop columns exist, otherwise we may need to
      # generate placeholder values or modify the approach
      fam_dat_repro_raw |>
        dplyr::rename(trait = class) |>
        dplyr::mutate(
          trait = dplyr::case_match(
            trait,
            "flower" ~ "fl",
            "fruit" ~ "fr",
            .default = trait
          ),
          # Placeholder values - update when full family stats available
          proportion_certainty_family = NA_real_,
          accuracy_family = accuracy,
          accuracy_excluding_certainty_family = accuracy,
          count_family = n
        ) |>
        dplyr::select(
          family, trait,
          proportion_certainty_family,
          accuracy_family,
          accuracy_excluding_certainty_family,
          count_family
        )
    }
  ),

  # --- Inference (runs on ALL images, not filtered like leaves) ---

  tar_target(
    annotations_repro,
    annotate_batch(
      images_batch,
      model_doi_repro,
      trait = "flower/fruit",
      num_workers = num_workers_data
    ),
    iteration = "list",
    pattern = map(images_batch)
  ),

  # --- Threshold Application ---

  tar_target(
    annotations_thresholded_repro,
    threshold_annotations(
      annotations_repro,
      thresholds_repro,
      trait = "flower/fruit",
      meta_images = meta_images_path
    ),
    pattern = map(annotations_repro),
    iteration = "list"
  ),

  # --- Convert to Long Format ---

  tar_target(
    annotations_long_repro,
    convert_to_long(annotations_thresholded_repro, trait = "flower/fruit"),
    pattern = map(annotations_thresholded_repro),
    iteration = "list"
  ),

  # --- Aggregate by Observation ---

  tar_target(
    annotations_by_obs_repro,
    aggregate_by_obs(
      annotations_long_repro,
      taxonomy,
      families,
      genera,
      fam_dat_long_repro,
      meta_images = meta_images_path,
      meta_taxa = meta_taxa_path
    ),
    pattern = map(annotations_long_repro),
    iteration = "list"
  ),

  # --- Final Formatting ---

  tar_target(
    annotations_by_obs_final_repro,
    finalize_annotations_by_obs(
      annotations_by_obs_repro,
      fields = NULL,
      field_map = field_map
    ),
    pattern = map(annotations_by_obs_repro),
    iteration = "list"
  ),

  # --- Filter for Ingestion (High Certainty Detections Only) ---

  tar_target(
    annotations_by_obs_ingest_repro,
    {
      annotations_by_obs_final_repro |>
        dplyr::filter(
          certainty == "High",
          predictionClass == "Detected"
        ) |>
        dplyr::select(
          -proportionCertaintyFamily,
          -countFamily,
          -countImages,
          -certainty,
          -predictionProbability,
          -predictionClass,
          -accuracyFamily
        )
    },
    pattern = map(annotations_by_obs_final_repro),
    iteration = "list"
  ),

  # --- Output: Reproductive CSVs ---

  tar_target(
    annotations_internal_repro,
    {
      dir.create(file.path(results_dir_repro, "final_internal"),
                 recursive = TRUE, showWarnings = FALSE)
      path <- file.path(results_dir_repro, "final_internal",
                        paste0(tar_name(), ".csv"))
      readr::write_csv(annotations_by_obs_final_repro, path)
      path
    },
    pattern = map(annotations_by_obs_final_repro),
    format = "file"
  ),

  tar_target(
    annotations_ingest_repro,
    {
      dir.create(file.path(results_dir_repro, "final_ingest"),
                 recursive = TRUE, showWarnings = FALSE)
      path <- file.path(results_dir_repro, "final_ingest",
                        paste0(tar_name(), ".csv"))
      readr::write_csv(annotations_by_obs_ingest_repro, path)
      path
    },
    pattern = map(annotations_by_obs_ingest_repro),
    format = "file"
  ),

  # Concatenate all reproductive internal format CSVs
  tar_target(
    annotations_internal_repro_all_csv,
    {
      path <- file.path(results_dir_repro, "annotations_internal_all.csv")
      if (file.exists(path)) file.remove(path)
      concatenate_csvs(annotations_by_obs_final_repro, path)
    },
    pattern = map(annotations_by_obs_final_repro),
    format = "file"
  ),

  # Concatenate all reproductive ingestion format CSVs
  tar_target(
    annotations_ingest_repro_all_csv,
    {
      path <- file.path(results_dir_repro, "annotations_ingest_all.csv")
      if (file.exists(path)) file.remove(path)
      concatenate_csvs(annotations_by_obs_ingest_repro, path)
    },
    pattern = map(annotations_by_obs_ingest_repro),
    format = "file"
  ),

  # =========================================================================
  # LEAF PHENOLOGY BRANCH
  # =========================================================================

  # --- Taxonomy Filtering for Leaves ---

  tar_target(
    rob_annot,
    readr::read_csv(rob_annot_path) |>
      dplyr::mutate(genus = stringr::word(taxon.name))
  ),

  tar_target(genera_include_leaves, setdiff(unique(rob_annot$genus), genera_exclude_leaves)),

  tar_target(
    genus_taxon_ids,
    arrow::open_dataset(meta_taxa_path) |>
      dplyr::filter(rank == "genus") |>
      dplyr::filter(name %in% genera_include_leaves) |>
      dplyr::collect() |>
      dplyr::filter(active)
  ),

  tar_target(
    taxonomy_leaves,
    taxonomy |> dplyr::filter(taxa_ids %in% genus_taxon_ids$taxon_id)
  ),

  # Filter image batches to only target genera
  tar_target(
    images_batch_leaves,
    {
      images_batch |>
        dplyr::left_join(
          arrow::open_dataset(meta_images_path) |>
            dplyr::select(photo_id, taxon_id) |>
            dplyr::filter(photo_id %in% images_batch$photo_id),
          copy = TRUE
        ) |>
        dplyr::left_join(taxonomy_leaves, by = "taxon_id") |>
        tidyr::drop_na(taxa_ids)
    },
    pattern = map(images_batch)
  ),

  # --- Inference ---

  tar_target(
    annotations_leaves,
    annotate_batch(
      images_batch_leaves,
      model_doi_leaves,
      trait = "leaves",
      num_workers = num_workers_data
    ),
    iteration = "list",
    pattern = map(images_batch_leaves)
  ),

  # --- Threshold Loading ---

  tar_target(
    thresholds_leaves,
    readr::read_csv(thresholds_leaves_path)
  ),

  # --- Threshold Application ---

  tar_target(
    annotations_thresholded_leaves,
    threshold_annotations(
      annotations_leaves,
      thresholds_leaves,
      trait = "leaves",
      meta_images = meta_images_path
    ),
    pattern = map(annotations_leaves),
    iteration = "list"
  ),

  # --- Convert to Long Format ---

  tar_target(
    annotations_long_leaves,
    convert_to_long(annotations_thresholded_leaves, trait = "leaves"),
    pattern = map(annotations_thresholded_leaves),
    iteration = "list"
  ),

  # --- Family Statistics ---

  tar_target(
    fam_dat_leaves,
    readr::read_csv(fam_dat_leaves_path)
  ),

  tar_target(
    fam_dat_long_leaves,
    convert_fam_to_long(fam_dat_leaves, trait = "leaves")
  ),

  # --- Aggregate by Observation ---

  tar_target(
    annotations_by_obs_leaves,
    aggregate_by_obs(
      annotations_long_leaves,
      taxonomy,
      families,
      genera,
      fam_dat_long_leaves,
      meta_images = meta_images_path,
      meta_taxa = meta_taxa_path
    ),
    pattern = map(annotations_long_leaves),
    iteration = "list"
  ),

  # --- Final Formatting ---

  tar_target(
    annotations_by_obs_final_leaves,
    finalize_annotations_by_obs(
      annotations_by_obs_leaves,
      fields = NULL,
      field_map = field_map
    ),
    pattern = map(annotations_by_obs_leaves),
    iteration = "list"
  ),

  # --- Filter for Ingestion (High Certainty Detections Only) ---

  tar_target(
    annotations_by_obs_ingest_leaves,
    {
      annotations_by_obs_final_leaves |>
        dplyr::filter(
          certainty == "High",
          predictionClass == "Detected",
          verbatimTrait != "no live leaves"
        ) |>
        dplyr::select(
          -proportionCertaintyFamily,
          -countFamily,
          -countImages,
          -certainty,
          -predictionProbability,
          -predictionClass,
          -accuracyFamily
        )
    },
    pattern = map(annotations_by_obs_final_leaves),
    iteration = "list"
  ),

  # =========================================================================
  # OUTPUT: Individual Batch Files
  # =========================================================================

  # Write internal format (one CSV per batch)
  tar_target(
    annotations_internal,
    {
      # Create output directory if needed
      dir.create(file.path(results_dir_leaves, "final_internal"),
                 recursive = TRUE,
                 showWarnings = FALSE)

      # Write CSV
      path <- file.path(
        results_dir_leaves,
        "final_internal",
        paste0(tar_name(), ".csv")
      )
      readr::write_csv(annotations_by_obs_final_leaves, path)
      path
    },
    pattern = map(annotations_by_obs_final_leaves),
    format = "file"
  ),

  # Write ingestion format (one CSV per batch)
  tar_target(
    annotations_ingest,
    {
      # Create output directory if needed
      dir.create(file.path(results_dir_leaves, "final_ingest"),
                 recursive = TRUE,
                 showWarnings = FALSE)

      # Write CSV
      path <- file.path(
        results_dir_leaves,
        "final_ingest",
        paste0(tar_name(), ".csv")
      )
      readr::write_csv(annotations_by_obs_ingest_leaves, path)
      path
    },
    pattern = map(annotations_by_obs_ingest_leaves),
    format = "file"
  ),

  # =========================================================================
  # OUTPUT: Combined Files
  # =========================================================================

  # Concatenate all internal format CSVs into one file
  tar_target(
    annotations_internal_all_csv,
    {
      path <- file.path(results_dir_leaves, "annotations_internal_all.csv")
      # Remove existing file to start fresh
      if (file.exists(path)) file.remove(path)
      concatenate_csvs(annotations_by_obs_final_leaves, path)
    },
    pattern = map(annotations_by_obs_final_leaves),
    format = "file"
  ),

  # Concatenate all ingestion format CSVs into one file
  tar_target(
    annotations_ingest_all_csv,
    {
      path <- file.path(results_dir_leaves, "annotations_ingest_all.csv")
      # Remove existing file to start fresh
      if (file.exists(path)) file.remove(path)
      concatenate_csvs(annotations_by_obs_ingest_leaves, path)
    },
    pattern = map(annotations_by_obs_ingest_leaves),
    format = "file"
  )
)
