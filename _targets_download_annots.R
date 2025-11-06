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
#   Rscript run_pipeline.R --pipeline=download_annots

# =============================================================================
# Setup
# =============================================================================

library(targets)
library(tarchetypes)

# Load common configuration
source("_targets_common.R")

# Load functions
source_common()     # Common functions
source_download()   # Download-specific functions

# Additional packages
library(conflicted)
conflicts_prefer(dplyr::filter)

# Set targets options
tar_option_set(
  packages = c("tidyverse", "arrow", "jsonlite", "data.table", "rsample"),
  format = "qs"  # Faster than rds for large data
)

# =============================================================================
# Pipeline
# =============================================================================

tar_plan(

  # ===========================================================================
  # Configuration (Separate Targets for Granular Dependencies)
  # ===========================================================================

  # Paths to parquet datasets (managed by collaborator scripts)
  metadata_dir = "data/phenobase_inat_data/metadata",
  annotation_dir = "data/phenobase_inat_data/metadata/phenobase_dwca_annotation",
  photos_parquet = file.path(metadata_dir, "angio_photos"),
  annotations_parquet = file.path(annotation_dir, "inat_annotation"),

  # Output directories
  output_dir_repro = "data/inat",
  output_dir_leaves = "data/leaves",

  # Split parameters (separate targets so changing one doesn't invalidate all)
  train_prop = 0.6,
  val_prop = 0.2,
  test_prop = 0.2,
  split_seed = 234987,
  split_pool = 0.025,  # Min 2.5% of data per stratum

  # Rob's manual leaf annotation files
  rob_annot_csv = "data/leaves/phenobase_dwca_annotation/rob_leaf_breaking_buds_annotation.csv",
  rob_annot2_csv = "data/leaves/rob_new_annotations_bb.csv",

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
  images_root = "/blue/guralnick/share/phenobase_inat_data/images/medium",

  # Image download settings
  image_batch_size = 100000,  # Images per batch
  image_download_cores = 10,  # Parallel download workers

  # Update frequency flags
  force_metadata_update = FALSE,  # Set to TRUE to force re-download of metadata
  force_annotation_update = FALSE,  # Set to TRUE to force re-download of DwC archive

  # ===========================================================================
  # Step 1: Update iNaturalist Metadata (Parquet-Based)
  # ===========================================================================

  # Update angio_photos parquet with latest iNaturalist data
  # This downloads metadata tar.gz, filters to angiosperms, assigns batches
  tar_target(
    photos_parquet_updated,
    update_inat_metadata(
      metadata_dir = metadata_dir,
      batch_size = image_batch_size,
      force_download = force_metadata_update
    ),
    format = "file"
  ),

  # ===========================================================================
  # Step 2: Update Phenology Annotations (Parquet-Based)
  # ===========================================================================

  # Update inat_annotation parquet with latest DwC archive
  # This downloads DwC zip, extracts observations, filters for annotations
  tar_target(
    annotations_parquet_updated,
    update_phenology_annotations(
      annotation_dir = annotation_dir,
      force_download = force_annotation_update
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

  # Split reproductive data using tidymodels
  repro_splits = split_repro_data(
    repro_annotations,
    train_prop = train_prop,
    val_prop = val_prop,
    test_prop = test_prop,
    seed = split_seed
  ),

  # Write reproductive splits to CSV
  tar_target(
    repro_train_csv,
    write_csv_split(repro_splits, "train", output_dir_repro, images_root),
    format = "file"
  ),

  tar_target(
    repro_val_csv,
    write_csv_split(repro_splits, "val", output_dir_repro, images_root),
    format = "file"
  ),

  tar_target(
    repro_test_csv,
    write_csv_split(repro_splits, "test", output_dir_repro, images_root),
    format = "file"
  ),

  # ===========================================================================
  # Step 4: Extract Leaf Annotations from Parquets
  # ===========================================================================

  # Track Rob's manual annotation files
  tar_target(rob_annot_file, rob_annot_csv, format = "file"),
  tar_target(rob_annot2_file, rob_annot2_csv, format = "file"),

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
  # Step 5: Download Images for Needed Batches
  # ===========================================================================

  # Identify which batches are needed for training
  needed_batches = {
    # Get batch IDs from all CSVs
    repro_batches <- bind_rows(
      read_csv(repro_train_csv, show_col_types = FALSE),
      read_csv(repro_val_csv, show_col_types = FALSE),
      read_csv(repro_test_csv, show_col_types = FALSE)
    ) %>%
      pull(file_name) %>%
      str_extract("batch_[0-9]+") %>%
      str_remove("batch_") %>%
      as.integer() %>%
      unique()

    leaf_batches <- bind_rows(
      read_csv(leaf_train_csv, show_col_types = FALSE),
      read_csv(leaf_val_csv, show_col_types = FALSE),
      read_csv(leaf_test_csv, show_col_types = FALSE),
      read_csv(leaf_seconds_csv, show_col_types = FALSE)
    ) %>%
      pull(file_name) %>%
      str_extract("batch_[0-9]+") %>%
      str_remove("batch_") %>%
      as.integer() %>%
      unique()

    sort(unique(c(repro_batches, leaf_batches)))
  },

  # Download images for needed batches
  # NOTE: This may take a long time! Consider running separately.
  tar_target(
    image_batches_downloaded,
    download_images_by_batch(
      parquet_path = photos_parquet_updated,
      image_dir = dirname(images_root),
      batch_ids = needed_batches,
      n_cores = image_download_cores
    ),
    format = "file"
  ),

  # ===========================================================================
  # Step 6: Summary Statistics
  # ===========================================================================

  download_summary = {
    # Dependencies
    repro_train_csv
    repro_val_csv
    repro_test_csv
    leaf_train_csv
    leaf_val_csv
    leaf_test_csv
    leaf_seconds_csv
    image_batches_downloaded

    # Compute summary
    list(
      reproductive = list(
        train = nrow(repro_splits$train),
        val = nrow(repro_splits$val),
        test = nrow(repro_splits$test),
        total = nrow(repro_annotations),
        files = list(
          train = repro_train_csv,
          val = repro_val_csv,
          test = repro_test_csv
        )
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
        batches_needed = length(needed_batches),
        batches_downloaded = length(image_batches_downloaded),
        batch_ids = needed_batches
      ),
      message = paste0(
        "\n",
        paste(rep("=", 70), collapse = ""), "\n",
        "PARQUET-BASED DOWNLOAD AND SPLIT COMPLETE\n",
        paste(rep("=", 70), collapse = ""), "\n",
        "Reproductive Annotations:\n",
        sprintf("  Train:      %d (%.1f%%)\n", nrow(repro_splits$train),
                100 * nrow(repro_splits$train) / nrow(repro_annotations)),
        sprintf("  Validation: %d (%.1f%%)\n", nrow(repro_splits$val),
                100 * nrow(repro_splits$val) / nrow(repro_annotations)),
        sprintf("  Test:       %d (%.1f%%)\n", nrow(repro_splits$test),
                100 * nrow(repro_splits$test) / nrow(repro_annotations)),
        sprintf("  Total:      %d\n\n", nrow(repro_annotations)),
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
        sprintf("  Batches needed:     %d\n", length(needed_batches)),
        sprintf("  Batches downloaded: %d\n", length(image_batches_downloaded)),
        sprintf("  Batch IDs: %s\n\n", paste(head(needed_batches, 10), collapse = ", ")),
        "Output Files:\n",
        "  Reproductive:\n",
        sprintf("    Train:      %s\n", repro_train_csv),
        sprintf("    Validation: %s\n", repro_val_csv),
        sprintf("    Test:       %s\n", repro_test_csv),
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
