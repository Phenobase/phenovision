# PhenoVision Annotation Download Pipeline (Parquet-Based)
#
# This targets pipeline handles downloading and processing plant phenology
# annotations from iNaturalist using the collaborator's parquet-based workflow:
# 1. Update iNaturalist metadata (creates/updates angio_photos parquet)
# 2. Update phenology annotations (creates/updates inat_annotation parquet)
# 3. Extract reproductive and leaf annotations from parquets
# 4. Split data into train/validation/test sets using tidymodels
# 5. Download images for needed batches
# 6. Export to CSV files for training pipelines
#
# Key features:
# - Uses tar_plan() for modern targets syntax
# - Separate config targets for granular dependency tracking
# - Pure R implementation (no Python dependencies)
# - Parquet-based workflow from collaborator's scripts
# - Efficient incremental updates (only downloads NEW data)
# - Batch-based image organization (100k images per batch)
# - Includes both reproductive and leaf annotations
# - Stratified splits using rsample
#
# Based on collaborator scripts:
# - 01_get_new_inat_metadata2.R (metadata updates)
# - 02_download_images_batch.R (image downloads)
# - 03_get_latest_iNat_annotation.R (annotation extraction)
#
# Usage:
#   targets::tar_make(script = "_targets_download_annots.R")
#   # Crew controller automatically parallelizes with 6 workers
#
#   # Or via CLI wrapper:
#   Rscript run_pipeline.R --pipeline=download_annots

# =============================================================================
# Setup
# =============================================================================

library(targets)
library(tarchetypes)
library(crew)

# Load common configuration
source("_targets_common.R")

# Load functions
source_common()     # Common functions
source_download()   # Download-specific functions

# Additional packages
library(conflicted)
conflicts_prefer(dplyr::filter)

# Determine worker count from environment
# Priority: SLURM_CPUS_PER_TASK > TARGETS_WORKERS > default (6)
.download_workers <- {
  slurm_cpus <- Sys.getenv("SLURM_CPUS_PER_TASK", unset = "")
  targets_workers <- Sys.getenv("TARGETS_WORKERS", unset = "")
  if (nchar(slurm_cpus) > 0) {
    as.integer(slurm_cpus)
  } else if (nchar(targets_workers) > 0) {
    as.integer(targets_workers)
  } else {
    6L  # Default for this pipeline
  }
}
message("Download pipeline using ", .download_workers, " workers")

# Set targets options with crew controller
tar_option_set(
  packages = c("tidyverse", "arrow", "jsonlite", "data.table", "rsample", "curl"),
  format = "rds",  # Standard R format (qs would be faster but requires qs2 package)
  controller = crew_controller_local(
    workers = .download_workers,
    seconds_idle = 60  # Keep workers alive for 60 seconds after finishing
  )
)

# Configure parallel processing
# Using crew for parallelization (modern targets approach)
# - Runs batch downloads in parallel via dynamic branching (workers from env var)
# - Each worker runs download_batch_from_list() which uses curl::multi_download()
#   with built-in concurrency for efficient I/O (100 files per micro-batch)
# - Simply use: tar_make(script = "_targets_download_annots.R")

# =============================================================================
# Pipeline
# =============================================================================

tar_plan(

  # ===========================================================================
  # Configuration (Separate Targets for Granular Dependencies)
  # ===========================================================================

  # Paths to parquet datasets (managed by collaborator scripts).
  # Sourced from the single `paths` list in _targets_common.R -- see the data_root note
  # there. The derived entries below stay as file.path(<target>, ...) on purpose, so
  # `targets` keeps the dependency edge from the parent path target.
  metadata_dir = paths$metadata_root,
  annotation_dir = paths$metadata_annotation,
  photos_parquet = file.path(metadata_dir, "angio_photos"),
  annotations_parquet = file.path(annotation_dir, "inat_annotation"),

  # Output directories
  output_dir_repro = paths$data_inat,
  output_dir_leaves = paths$data_leaves,

  # Split parameters (separate targets so changing one doesn't invalidate all)
  train_prop = 0.6,
  val_prop = 0.2,
  test_prop = 0.2,
  split_seed = 234987,
  split_pool = 0.025,  # Min 2.5% of data per stratum

  # Rob's manual leaf annotation files
  rob_annot_csv = file.path(output_dir_leaves, "phenobase_dwca_annotation/rob_leaf_breaking_buds_annotation.csv"),
  bb_scoring_csv = file.path(output_dir_leaves, "bbPresenceScoringRescoreAllStates.csv"),

  # Target genera for leaf annotations (from collaborator workflow)
  leaf_target_genera = {
    # Load Rob's annotations to get target genera
    rob <- read_csv(rob_annot_csv, show_col_types = FALSE) %>%
      mutate(genus = word(`taxon.name`))
    genera <- unique(rob$genus)
    # Exclude problematic genera
    setdiff(genera, c("Logfia", "Oxalis", "Viola"))
  },

  # Image root for file paths
  images_root = paths$images_root,

  # Image download settings
  image_batch_size = 100000,  # Images per batch
  image_download_cores = 6,  # Parallel download workers (matches available cores)
  image_download_status_dir = file.path(metadata_dir, "img_download_status"),  # Failed download logs

  # Data update trigger: change this date to force a full data update.
  # When this value changes, all download + extraction + processing targets re-run.
  # Set to the date you're triggering the update. No need to reset afterwards.
  data_update_trigger = "2026-07-14",

  # ===========================================================================
  # Step 1: Update iNaturalist Metadata (Granular Targets)
  # ===========================================================================

  # 1a. Download metadata tar.gz (26+ GB, ~30 min)
  # Returns list(path, mtime, size) — no format="file" to avoid hashing 27GB
  tar_target(
    metadata_tarfile,
    {
      data_update_trigger  # Dependency: changing trigger forces re-download
      download_inat_metadata(metadata_dir = metadata_dir)
    }
  ),

  # 1b. Extract metadata (70+ GB extracted, ~10 min)
  tar_target(
    metadata_extracted,
    extract_inat_metadata(
      tarfile = metadata_tarfile$path,
      metadata_dir = metadata_dir
    ),
    format = "file"
  ),

  # 1c. Filter taxa to angiosperms (uses awk, ~30 sec)
  tar_target(
    angio_taxa_ids,
    filter_angio_taxa(metadata_extracted)
  ),

  # 1d. Filter observations to research-grade angiosperms (uses awk, ~15 min)
  tar_target(
    angio_obs_uuids,
    filter_angio_observations(metadata_extracted, angio_taxa_ids)
  ),

  # 1e. Filter photos from angiosperm observations (uses awk, ~20 min)
  tar_target(
    angio_photos_raw,
    filter_angio_photos(metadata_extracted, angio_obs_uuids)
  ),

  # 1f. Identify NEW photos compared to existing parquet (~5 min)
  tar_target(
    angio_photos_new,
    identify_new_photos(
      angio_photos_raw,
      parquet_path = photos_parquet,
      metadata_dir = metadata_dir
    )
  ),

  # 1f2. Enrich ONLY NEW photos with observation data (~1 min for 1.7M photos)
  tar_target(
    angio_photos_new_enriched,
    enrich_photos_with_observations(angio_photos_new$new_photos, metadata_extracted)
  ),

  # 1g. Assign batch numbers to new photos (~1 min)
  tar_target(
    angio_photos_batched,
    assign_batch_numbers(
      list(old_batch_max = angio_photos_new$old_batch_max,
           new_photos = angio_photos_new_enriched),
      batch_size = image_batch_size,
      parquet_path = photos_parquet
    )
  ),

  # 1h. Write updated parquet dataset (~10-30 min depending on size)
  tar_target(
    photos_parquet_updated,
    write_photos_parquet(
      angio_photos_batched,
      parquet_path = photos_parquet,
      metadata_dir = metadata_dir
    ),
    format = "file"
  ),

  # ===========================================================================
  # Step 2: Update Phenology Annotations (Granular Targets)
  # ===========================================================================

  # 2a. Download DwC archive (10+ GB, ~2 hours)
  # Returns list(path, mtime, size) — no format="file" to avoid hashing 10GB
  tar_target(
    dwc_zipfile,
    {
      data_update_trigger  # Same trigger controls both download branches
      download_dwc_archive(annotation_dir = annotation_dir)
    }
  ),

  # 2b. Extract DwC archive (~30 min)
  tar_target(
    dwc_extracted,
    extract_dwc_archive(
      zipfile = dwc_zipfile$path,
      annotation_dir = annotation_dir
    ),
    format = "file"
  ),

  # 2c. Parse phenology annotations from DwC (~10 min)
  tar_target(
    phenology_annots_raw,
    parse_phenology_dwc(dwc_extracted, annotation_dir)
  ),

  # 2d. Write updated annotations parquet (~5 min)
  tar_target(
    annotations_parquet_updated,
    write_annotations_parquet(
      phenology_annots_raw,
      parquet_path = annotations_parquet,
      annotation_dir = annotation_dir
    ),
    format = "file"
  ),

  # ===========================================================================
  # Step 3: Extract Reproductive Annotations from Parquets
  # ===========================================================================

  # Extract reproductive annotations by joining parquets
  repro_annotations = extract_reproductive_from_parquet(
    annotation_parquet = annotations_parquet_updated,
    photos_parquet = photos_parquet_updated,
    images_root = images_root
  ),

  # Write full reproductive annotations to CSV (for training pipeline)
  # Filename uses data_update_trigger date to track which data snapshot was used
  tar_target(
    repro_annotations_full_csv,
    {
      output_file <- file.path(output_dir_repro,
                               paste0("repro_annotations_full_", data_update_trigger, ".csv"))
      write_csv(repro_annotations, output_file)
      message(sprintf("Wrote %s annotations to %s",
                      format(nrow(repro_annotations), big.mark = ","),
                      output_file))
      output_file
    },
    format = "file"
  ),

  # ===========================================================================
  # Step 4: Extract Leaf Annotations from Parquets
  # ===========================================================================

  # Track Rob's manual annotation files
  tar_target(rob_annot_file, rob_annot_csv, format = "file"),
  tar_target(bb_scoring_file, bb_scoring_csv, format = "file"),

  # Generate rob_annot2 data from scoring CSV + metadata
  # (replaces the lost rob_new_annotations_bb.csv static file)
  tar_target(
    rob_annot2_file,
    {
      out_path <- "data/leaves/rob_new_annotations_bb.csv"
      result <- prepare_rob_bb_annotations(
        scoring_csv = bb_scoring_file,
        photos_parquet = photos_parquet_updated,
        taxa_parquet = file.path(metadata_dir, "taxa/part-0.parquet"),
        images_root = images_root
      )
      readr::write_csv(result, out_path)
      out_path
    },
    format = "file"
  ),

  # Extract leaf annotations by joining parquets and Rob's annotations
  leaf_annotations = extract_leaf_from_parquet(
    annotation_parquet = annotations_parquet_updated,
    photos_parquet = photos_parquet_updated,
    rob_annot_csv = rob_annot_file,
    rob_annot2_csv = rob_annot2_file,
    images_root = images_root,
    target_genera = leaf_target_genera
  ),

  # Split leaf data using tidymodels
  leaf_splits = split_leaf_data(
    leaf_annotations,
    train_prop = train_prop,
    val_prop = val_prop,
    test_prop = test_prop,
    seed = split_seed,
    pool = split_pool
  ),

  # Write leaf splits to CSV
  tar_target(
    leaf_train_csv,
    write_csv_split(leaf_splits, "train", output_dir_leaves, images_root),
    format = "file"
  ),

  tar_target(
    leaf_val_csv,
    write_csv_split(leaf_splits, "val", output_dir_leaves, images_root),
    format = "file"
  ),

  tar_target(
    leaf_test_csv,
    write_csv_split(leaf_splits, "test", output_dir_leaves, images_root),
    format = "file"
  ),

  tar_target(
    leaf_seconds_csv,
    write_csv_split(leaf_splits, "seconds", output_dir_leaves, images_root),
    format = "file"
  ),

  # ===========================================================================
  # Step 5: Download Images for ALL Batches (Not Just Annotated)
  # ===========================================================================
  #
  # New architecture based on collaborator's script:
  # 1. Get ALL unique batch_j values from parquet (single query)
  # 2. Compare against existing batch folders to find missing batches
  # 3. Pre-extract ALL photo metadata for missing batches (single query)
  # 4. Prepare download lists with pre-computed URLs/paths (no parquet access)
  # 5. Dynamic branching: Map over download lists (each worker gets pre-extracted data)
  # 6. Aggregate results
  #
  # This ensures we download ALL images for inference, not just annotated ones.
  # Pre-extraction avoids parallel parquet access bottleneck.
  #
  # ===========================================================================

  # Step 5.1: Get all unique batch IDs from parquet (SINGLE QUERY)
  all_batches_in_parquet = identify_all_batches(
    parquet_path = photos_parquet_updated
  ),

  # Step 5.2: Compare against existing folders to find missing batches
  batches_to_download = identify_missing_batches_v2(
    all_batches = all_batches_in_parquet,
    images_root = images_root
  ),

  # Step 5.3: Pre-extract photo metadata for ALL missing batches (SINGLE QUERY)
  # This avoids multiple workers querying the parquet in parallel
  batch_metadata_extracted = extract_batch_metadata(
    parquet_path = photos_parquet_updated,
    batches_to_download = batches_to_download
  ),

  # Step 5.4: Prepare download lists with pre-computed URLs and destination paths
  # Converts metadata to list-of-lists structure for efficient distribution
  # IMPORTANT: iteration = "list" ensures pattern=map() indexes as [[i]] not [i]
  tar_target(
    batch_download_lists,
    prepare_batch_download_lists(
      batch_metadata = batch_metadata_extracted,
      images_root = images_root,
      size = "medium"
    ),
    iteration = "list"
  ),

  # Step 5.5: Download batches using dynamic branching
  # Each worker receives pre-extracted data (NO parquet access!)
  # Crew controller parallelizes across 6 workers automatically
  tar_target(
    batch_download_status,
    download_batch_from_list(
      batch_data = batch_download_lists,
      status_dir = image_download_status_dir
    ),
    pattern = map(batch_download_lists),
    iteration = "list"
  ),

  # Step 5.6: Aggregate download results across all batches
  download_summary_batches = summarize_batch_downloads(
    batch_download_status = batch_download_status,
    all_batches = all_batches_in_parquet,
    batches_to_download = batches_to_download
  ),

  # ===========================================================================
  # Step 6: Summary Statistics
  # ===========================================================================

  download_summary = {
    # Dependencies
    repro_annotations_full_csv  # Full annotations for training pipeline
    leaf_train_csv
    leaf_val_csv
    leaf_test_csv
    leaf_seconds_csv
    download_summary_batches  # Changed from image_batches_downloaded

    # Compute summary
    list(
      reproductive = list(
        total = nrow(repro_annotations),
        full_csv = repro_annotations_full_csv,
        note = "Splitting now handled by training pipeline with versioning"
      ),
      leaves = list(
        train = nrow(leaf_splits$train),
        val = nrow(leaf_splits$val),
        test = nrow(leaf_splits$test),
        seconds = nrow(leaf_splits$seconds),
        total = nrow(leaf_annotations),
        files = list(
          train = leaf_train_csv,
          val = leaf_val_csv,
          test = leaf_test_csv,
          seconds = leaf_seconds_csv
        )
      ),
      images = list(
        batches_available = length(all_batches_in_parquet),
        batches_to_download = length(batches_to_download),
        download_summary = download_summary_batches
      ),
      message = paste0(
        "\n",
        paste(rep("=", 70), collapse = ""), "\n",
        "PARQUET-BASED DOWNLOAD COMPLETE\n",
        paste(rep("=", 70), collapse = ""), "\n",
        "Reproductive Annotations:\n",
        sprintf("  Total:      %d\n", nrow(repro_annotations)),
        sprintf("  Full CSV:   %s\n", repro_annotations_full_csv),
        sprintf("  Note:       Splitting handled by training pipeline\n\n"),
        "Leaf Annotations:\n",
        sprintf("  Train:      %d (%.1f%%)\n", nrow(leaf_splits$train),
                100 * nrow(leaf_splits$train) / nrow(leaf_annotations)),
        sprintf("  Validation: %d (%.1f%%)\n", nrow(leaf_splits$val),
                100 * nrow(leaf_splits$val) / nrow(leaf_annotations)),
        sprintf("  Test:       %d (%.1f%%)\n", nrow(leaf_splits$test),
                100 * nrow(leaf_splits$test) / nrow(leaf_annotations)),
        sprintf("  Seconds:    %d (for round 2 training)\n", nrow(leaf_splits$seconds)),
        sprintf("  Total:      %d\n\n", nrow(leaf_annotations)),
        "Image Downloads:\n",
        sprintf("  Batches available:   %d\n", length(all_batches_in_parquet)),
        sprintf("  Batches to download: %d\n", length(batches_to_download)),
        sprintf("  Batch IDs (first 10): %s\n\n", paste(head(batches_to_download, 10), collapse = ", ")),
        "Output Files:\n",
        "  Reproductive:\n",
        sprintf("    Full annotations: %s\n", repro_annotations_full_csv),
        "\n  Leaves:\n",
        sprintf("    Train:      %s\n", leaf_train_csv),
        sprintf("    Validation: %s\n", leaf_val_csv),
        sprintf("    Test:       %s\n", leaf_test_csv),
        sprintf("    Seconds:    %s\n", leaf_seconds_csv),
        "\n  Parquet Datasets:\n",
        sprintf("    Photos:      %s\n", photos_parquet_updated),
        sprintf("    Annotations: %s\n", annotations_parquet_updated),
        "\n",
        paste(rep("=", 70), collapse = "")
      )
    )
  }
)
