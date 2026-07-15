# PhenoVision Inference Pipeline
#
# This targets pipeline runs inference on plant images using both reproductive
# structures and leaf phenology models. The pipeline has independent branches
# for each model type, enabling smart re-processing:
# - Changing model version triggers only that branch's re-inference
# - New images trigger both branches
#
# Model versions are resolved via model_registry.yaml, which maps
# version -> DOI, thresholds, family stats, and output paths.
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
source_common()     # Common functions (model loading, versioning, registry)
source_inference()  # Inference-specific functions

# Additional packages
library(targets)
library(tarchetypes)  # Modern targets syntax
library(conflicted)
conflicts_prefer(dplyr::filter)
conflicts_prefer(lubridate::yday)
conflicts_prefer(lubridate::year)

# Configure targets for parallel execution (reads TARGETS_WORKERS or SLURM_CPUS_PER_TASK env var)
# gpu_workers creates a separate "gpu" controller for inference targets,
# preventing multiple workers from loading models onto the same GPU simultaneously.
# With 3 L4 GPUs, 3 inference branches run in parallel on separate GPUs.
setup_targets_parallel(gpu_workers = 3)

# =============================================================================
# Pipeline (Modern tar_plan() Syntax)
# =============================================================================

tar_plan(

  # =========================================================================
  # Configuration (Separate Targets for Granular Dependencies)
  # =========================================================================

  # --- Model Versions ---
  # Changing a version triggers re-inference for that branch only.
  # Versions are resolved via model_registry.yaml -> DOI, thresholds, etc.
  model_version_repro = "v1.1.0",
  model_version_leaves = "v1.0.0",

  # --- Worker Configuration ---
  num_workers_data = 4L,
  batch_size_inference = 1280L,

  # --- Data Paths ---
  metadata_photos = "/blue/guralnick/share/phenobase_inat_data/metadata/angio_photos",
  metadata_root = "/blue/guralnick/share/phenobase_inat_data/metadata",
  images_root = "/blue/guralnick/share/phenobase_inat_data/images/medium",

  # --- Leaf-Specific Filtering ---
  rob_annot_path = "data/leaves/phenobase_dwca_annotation/rob_leaf_breaking_buds_annotation.csv",
  genera_exclude_leaves = c("Logfia", "Oxalis", "Viola"),

  # =========================================================================
  # Model Registry Lookup (resolves version -> DOI, paths, thresholds)
  # =========================================================================

  # Registry info for each model type
  tar_target(model_info_repro, get_model_info(model_version_repro, "reproductive")),
  tar_target(model_info_leaves, get_model_info(model_version_leaves, "leaves")),

  # DOIs (derived from registry)
  tar_target(model_doi_repro, model_info_repro$doi),
  tar_target(model_doi_leaves, model_info_leaves$doi),

  # Results directories (inference output organized by version)
  tar_target(results_dir_repro, file.path(model_info_repro$output_dir, "inference")),
  tar_target(results_dir_leaves, file.path(model_info_leaves$output_dir, "inference")),

  # Data dates (from training data snapshot, used in output filenames only).
  # Read directly from registry to avoid coupling with model_info targets,
  # which would cascade invalidation through model loading and GPU inference.
  tar_target(data_date_repro, {
    reg <- yaml::read_yaml("model_registry.yaml")
    reg$reproductive$versions[[model_version_repro]]$data_date %||% "unknown"
  }),
  tar_target(data_date_leaves, {
    reg <- yaml::read_yaml("model_registry.yaml")
    reg$leaves$versions[[model_version_leaves]]$data_date %||% "unknown"
  }),

  # Load models from HuggingFace via DOI
  tar_target(model_repro, load_phenovision(model_doi_repro, type = "classifier")),
  tar_target(model_leaves, load_phenovision(model_doi_leaves, type = "classifier")),

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
          # WebP store: on-disk files are <photo_id>.webp (the `extension` column is kept only for
          # the iNat S3 fetch URL, not the on-disk filename). See R/webp_paths_common.R.
          path = webp_image_path(images_root, batch_j, photo_id)
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
  # memory = "persistent" keeps this in the controller's memory so it doesn't
  # re-read 836 MB from store for each of the ~188 downstream pattern branches
  tar_target(
    images_batch,
    images_df |>
      dplyr::group_by(run_name) |>
      targets::tar_group(),
    iteration = "group",
    memory = "persistent"
  ),

  # =========================================================================
  # Taxonomy
  # =========================================================================

  # memory = "persistent" on taxonomy/families/genera because they are
  # dependencies of ~188 pattern branches (aggregate_by_obs for both models)
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
    },
    memory = "persistent"
  ),

  tar_target(
    families,
    arrow::open_dataset(meta_taxa_path) |>
      dplyr::filter(rank == "family") |>
      dplyr::collect(),
    memory = "persistent"
  ),

  tar_target(
    genera,
    arrow::open_dataset(meta_taxa_path) |>
      dplyr::filter(rank == "genus") |>
      dplyr::collect(),
    memory = "persistent"
  ),

  # =========================================================================
  # Field Mapping for Phenobase
  # =========================================================================

  tar_target(
    field_map,
    tibble::tribble(
      ~new_field, ~old_field,
      # Field names match Phenobase schema (fields/Phenobase_fields - MasterSheet.csv)
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
      "modelUri", "model_uri",
      "accuracyExcludingUncertainFamily", "accuracy_excluding_certainty_family",
      "observedImageUrl", "observed_image_url",
      "predictionClass", "detected",
      "countImages", "count_images",
      "countFamily", "count_family",
      "certainty", "certainty",
      "predictionProbability", "pred_med",
      "proportionCertaintyFamily", "proportion_certainty_family",
      "accuracyFamily", "accuracy_family",
      "observedImageGuid", "observed_image_guid",
      "basisOfRecord", "basis_of_record",
      "annotationID", "machine_learning_annotation_id"
    ),
    memory = "persistent"
  ),

  # =========================================================================
  # REPRODUCTIVE STRUCTURES BRANCH
  # =========================================================================

  # --- Threshold Loading (via model registry) ---

  # memory = "persistent" on thresholds and family stats — tiny objects
  # used by 94 pattern branches each
  tar_target(
    thresholds_repro,
    load_model_thresholds(model_version_repro, "reproductive"),
    memory = "persistent"
  ),

  # --- Family Statistics ---

  tar_target(
    fam_dat_repro_raw,
    readr::read_csv(model_info_repro$family_stats_path, show_col_types = FALSE)
  ),

  tar_target(
    fam_dat_long_repro,
    convert_fam_to_long(fam_dat_repro_raw, trait = "flower/fruit"),
    memory = "persistent"
  ),

  # --- Inference (runs on ALL images, not filtered like leaves) ---

  tar_target(
    annotations_repro,
    annotate_batch(
      images_batch,
      model_doi_repro,
      trait = "flower/fruit",
      num_workers = num_workers_data,
      batch_size = batch_size_inference
    ),
    iteration = "list",
    pattern = map(images_batch),
    resources = tar_resources(crew = tar_resources_crew(controller = "gpu"))
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
                        paste0(targets::tar_name(), "_",
                               model_version_repro, "_",
                               data_date_repro, ".csv"))
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
                        paste0(targets::tar_name(), "_",
                               model_version_repro, "_",
                               data_date_repro, ".csv"))
      readr::write_csv(annotations_by_obs_ingest_repro, path)
      path
    },
    pattern = map(annotations_by_obs_ingest_repro),
    format = "file"
  ),

  # Concatenate all reproductive internal format CSVs
  # Reads per-batch CSV files one at a time to avoid loading all into memory.
  # Depends on annotations_internal_repro (format = "file", character vector of paths).
  tar_target(
    annotations_internal_repro_all_csv,
    {
      dir.create(results_dir_repro, recursive = TRUE, showWarnings = FALSE)
      fname <- paste0("annotations_internal_repro_",
                       model_version_repro, "_", data_date_repro, ".csv")
      path <- file.path(results_dir_repro, fname)
      if (file.exists(path)) file.remove(path)
      for (f in annotations_internal_repro) {
        concatenate_csvs(readr::read_csv(f, show_col_types = FALSE), path)
      }
      path
    },
    format = "file"
  ),

  # Concatenate all reproductive ingestion format CSVs
  tar_target(
    annotations_ingest_repro_all_csv,
    {
      dir.create(results_dir_repro, recursive = TRUE, showWarnings = FALSE)
      fname <- paste0("annotations_ingest_repro_",
                       model_version_repro, "_", data_date_repro, ".csv")
      path <- file.path(results_dir_repro, fname)
      if (file.exists(path)) file.remove(path)
      for (f in annotations_ingest_repro) {
        concatenate_csvs(readr::read_csv(f, show_col_types = FALSE), path)
      }
      path
    },
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
    taxonomy |> dplyr::filter(taxa_ids %in% genus_taxon_ids$taxon_id),
    memory = "persistent"
  ),

  # Filter image batches to only target genera
  # Note: images_batch already has taxon_id from the images target (parquet select).
  # No need to re-scan the 26.6 GB parquet per branch.
  tar_target(
    images_batch_leaves,
    images_batch |>
      dplyr::left_join(taxonomy_leaves, by = "taxon_id") |>
      tidyr::drop_na(taxa_ids),
    pattern = map(images_batch)
  ),

  # --- Inference ---

  tar_target(
    annotations_leaves,
    annotate_batch(
      images_batch_leaves,
      model_doi_leaves,
      trait = "leaves",
      num_workers = num_workers_data,
      batch_size = batch_size_inference
    ),
    iteration = "list",
    pattern = map(images_batch_leaves),
    resources = tar_resources(crew = tar_resources_crew(controller = "gpu"))
  ),

  # --- Threshold Loading (via model registry) ---

  tar_target(
    thresholds_leaves,
    load_model_thresholds(model_version_leaves, "leaves"),
    memory = "persistent"
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
    readr::read_csv(model_info_leaves$family_stats_path, show_col_types = FALSE)
  ),

  tar_target(
    fam_dat_long_leaves,
    convert_fam_to_long(fam_dat_leaves, trait = "leaves"),
    memory = "persistent"
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
      dir.create(file.path(results_dir_leaves, "final_internal"),
                 recursive = TRUE, showWarnings = FALSE)
      path <- file.path(
        results_dir_leaves,
        "final_internal",
        paste0(targets::tar_name(), "_",
               model_version_leaves, "_",
               data_date_leaves, ".csv")
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
      dir.create(file.path(results_dir_leaves, "final_ingest"),
                 recursive = TRUE, showWarnings = FALSE)
      path <- file.path(
        results_dir_leaves,
        "final_ingest",
        paste0(targets::tar_name(), "_",
               model_version_leaves, "_",
               data_date_leaves, ".csv")
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

  # Concatenate all leaves internal format CSVs
  # Reads per-batch CSV files one at a time to avoid loading all into memory.
  tar_target(
    annotations_internal_all_csv,
    {
      dir.create(results_dir_leaves, recursive = TRUE, showWarnings = FALSE)
      fname <- paste0("annotations_internal_leaves_",
                       model_version_leaves, "_", data_date_leaves, ".csv")
      path <- file.path(results_dir_leaves, fname)
      if (file.exists(path)) file.remove(path)
      for (f in annotations_internal) {
        concatenate_csvs(readr::read_csv(f, show_col_types = FALSE), path)
      }
      path
    },
    format = "file"
  ),

  # Concatenate all leaves ingestion format CSVs
  tar_target(
    annotations_ingest_all_csv,
    {
      dir.create(results_dir_leaves, recursive = TRUE, showWarnings = FALSE)
      fname <- paste0("annotations_ingest_leaves_",
                       model_version_leaves, "_", data_date_leaves, ".csv")
      path <- file.path(results_dir_leaves, fname)
      if (file.exists(path)) file.remove(path)
      for (f in annotations_ingest) {
        concatenate_csvs(readr::read_csv(f, show_col_types = FALSE), path)
      }
      path
    },
    format = "file"
  ),

  # =========================================================================
  # PRODUCTION DATASETS: Combined Repro + Leaves
  # =========================================================================
  # Merges reproductive and leaf annotations into unified files.
  # Output to output/production_datasets/{date}/ with versioned filenames.
  # Uses streaming concatenation to avoid loading all data into memory.

  tar_target(
    production_date,
    format(Sys.Date(), "%Y-%m-%d")
  ),

  tar_target(
    production_dir,
    {
      d <- file.path("output", "production_datasets", production_date)
      dir.create(d, recursive = TRUE, showWarnings = FALSE)
      d
    }
  ),

  # Combined internal format (all predictions, both models)
  tar_target(
    production_internal_csv,
    {
      fname <- paste0("annotations_internal_all_",
                       "repro-", model_version_repro, "_",
                       "leaves-", model_version_leaves, "_",
                       production_date, ".csv")
      path <- file.path(production_dir, fname)
      if (file.exists(path)) file.remove(path)
      # Stream repro batches
      for (f in annotations_internal_repro) {
        concatenate_csvs(readr::read_csv(f, show_col_types = FALSE), path)
      }
      # Stream leaves batches
      for (f in annotations_internal) {
        concatenate_csvs(readr::read_csv(f, show_col_types = FALSE), path)
      }
      path
    },
    format = "file"
  ),

  # Combined ingest format (high-certainty detections, both models)
  tar_target(
    production_ingest_csv,
    {
      fname <- paste0("annotations_ingest_all_",
                       "repro-", model_version_repro, "_",
                       "leaves-", model_version_leaves, "_",
                       production_date, ".csv")
      path <- file.path(production_dir, fname)
      if (file.exists(path)) file.remove(path)
      # Stream repro batches
      for (f in annotations_ingest_repro) {
        concatenate_csvs(readr::read_csv(f, show_col_types = FALSE), path)
      }
      # Stream leaves batches
      for (f in annotations_ingest) {
        concatenate_csvs(readr::read_csv(f, show_col_types = FALSE), path)
      }
      path
    },
    format = "file"
  ),

  # =========================================================================
  # VERIFICATION SAMPLES: Random subset for human review
  # =========================================================================
  # Samples ~1000 rows from each model type (ceiling(1000/n_branches) per batch)
  # for human verification of annotation quality.
  # Output nested under the dated production folder: {production_dir}/test/
  # Uses cached production_date to match production dataset filenames.

  # --- Internal format (all columns) ---

  tar_target(
    verification_sample_repro_internal,
    {
      n_per_batch <- ceiling(1000 / length(annotations_internal_repro))
      test_dir <- file.path(production_dir, "test")
      dir.create(test_dir, recursive = TRUE, showWarnings = FALSE)
      path <- file.path(test_dir,
                         paste0("verification_sample_repro_internal_",
                                model_version_repro, "_", production_date, ".csv"))
      if (file.exists(path)) file.remove(path)
      set.seed(42)
      for (f in annotations_internal_repro) {
        batch <- readr::read_csv(f, show_col_types = FALSE)
        samp <- batch |> dplyr::slice_sample(n = min(n_per_batch, nrow(batch)))
        concatenate_csvs(samp, path)
      }
      path
    },
    format = "file"
  ),

  tar_target(
    verification_sample_leaves_internal,
    {
      n_per_batch <- ceiling(1000 / length(annotations_internal))
      test_dir <- file.path(production_dir, "test")
      dir.create(test_dir, recursive = TRUE, showWarnings = FALSE)
      path <- file.path(test_dir,
                         paste0("verification_sample_leaves_internal_",
                                model_version_leaves, "_", production_date, ".csv"))
      if (file.exists(path)) file.remove(path)
      set.seed(42)
      for (f in annotations_internal) {
        batch <- readr::read_csv(f, show_col_types = FALSE)
        samp <- batch |> dplyr::slice_sample(n = min(n_per_batch, nrow(batch)))
        concatenate_csvs(samp, path)
      }
      path
    },
    format = "file"
  ),

  tar_target(
    verification_sample_combined_internal,
    {
      test_dir <- file.path(production_dir, "test")
      dir.create(test_dir, recursive = TRUE, showWarnings = FALSE)
      path <- file.path(test_dir,
                         paste0("verification_sample_combined_internal_",
                                production_date, ".csv"))
      if (file.exists(path)) file.remove(path)
      concatenate_csvs(readr::read_csv(verification_sample_repro_internal, show_col_types = FALSE), path)
      concatenate_csvs(readr::read_csv(verification_sample_leaves_internal, show_col_types = FALSE), path)
      path
    },
    format = "file"
  ),

  # --- Ingest format (high-certainty detections only) ---

  tar_target(
    verification_sample_repro_ingest,
    {
      n_per_batch <- ceiling(1000 / length(annotations_ingest_repro))
      test_dir <- file.path(production_dir, "test")
      dir.create(test_dir, recursive = TRUE, showWarnings = FALSE)
      path <- file.path(test_dir,
                         paste0("verification_sample_repro_ingest_",
                                model_version_repro, "_", production_date, ".csv"))
      if (file.exists(path)) file.remove(path)
      set.seed(42)
      for (f in annotations_ingest_repro) {
        batch <- readr::read_csv(f, show_col_types = FALSE)
        samp <- batch |> dplyr::slice_sample(n = min(n_per_batch, nrow(batch)))
        concatenate_csvs(samp, path)
      }
      path
    },
    format = "file"
  ),

  tar_target(
    verification_sample_leaves_ingest,
    {
      n_per_batch <- ceiling(1000 / length(annotations_ingest))
      test_dir <- file.path(production_dir, "test")
      dir.create(test_dir, recursive = TRUE, showWarnings = FALSE)
      path <- file.path(test_dir,
                         paste0("verification_sample_leaves_ingest_",
                                model_version_leaves, "_", production_date, ".csv"))
      if (file.exists(path)) file.remove(path)
      set.seed(42)
      for (f in annotations_ingest) {
        batch <- readr::read_csv(f, show_col_types = FALSE)
        samp <- batch |> dplyr::slice_sample(n = min(n_per_batch, nrow(batch)))
        concatenate_csvs(samp, path)
      }
      path
    },
    format = "file"
  ),

  tar_target(
    verification_sample_combined_ingest,
    {
      test_dir <- file.path(production_dir, "test")
      dir.create(test_dir, recursive = TRUE, showWarnings = FALSE)
      path <- file.path(test_dir,
                         paste0("verification_sample_combined_ingest_",
                                production_date, ".csv"))
      if (file.exists(path)) file.remove(path)
      concatenate_csvs(readr::read_csv(verification_sample_repro_ingest, show_col_types = FALSE), path)
      concatenate_csvs(readr::read_csv(verification_sample_leaves_ingest, show_col_types = FALSE), path)
      path
    },
    format = "file"
  )
)
