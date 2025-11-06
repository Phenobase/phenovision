# PhenoVision Annotation Download Pipeline
#
# This targets pipeline handles downloading and processing plant phenology
# annotations from iNaturalist:
# 1. Download DwC archive of annotated observations
# 2. Download iNaturalist open data metadata (photos.csv)
# 3. Extract and process reproductive annotations
# 4. Extract and process leaf annotations
# 5. Split data into train/validation/test sets using tidymodels
# 6. Export to CSV files for training pipelines
#
# Key features:
# - Uses tar_plan() for modern targets syntax
# - Separate config targets for granular dependency tracking
# - Pure R implementation (no Python dependencies)
# - Includes both reproductive and leaf annotations
# - Stratified splits using rsample
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

  # URLs
  dwc_url = "https://www.inaturalist.org/observations/phenobase-observations-dwca.zip",
  metadata_url = "https://inaturalist-open-data.s3.amazonaws.com/metadata/inaturalist-open-data-latest.tar.gz",

  # Output directories
  output_dir_repro = "data/inat",
  output_dir_leaves = "data/leaves",

  # Split parameters (separate targets so changing one doesn't invalidate all)
  train_prop = 0.6,
  val_prop = 0.2,
  test_prop = 0.2,
  split_seed = 234987,
  split_pool = 0.025,  # Min 2.5% of data per stratum

  # Leaf annotation paths
  leaf_parquet_path = "data/leaves/phenobase_dwca_annotation/inat_annotation/part-0.parquet",
  rob_annot_csv = "data/leaves/phenobase_dwca_annotation/rob_leaf_breaking_buds_annotation.csv",
  rob_annot2_csv = "data/leaves/rob_new_annotations_bb.csv",
  photo_metadata_path = "data/phenobase_inat_data/metadata/angio_photos",

  # Image root for file paths
  images_root = "/blue/guralnick/share/phenobase_inat_data/images/medium",

  # ===========================================================================
  # Step 1: Download Raw Data (R Functions, No Python!)
  # ===========================================================================

  # Download DwC archive with annotated observations
  tar_target(
    dwc_archive,
    download_file(dwc_url, file.path(output_dir_repro, basename(dwc_url))),
    format = "file"
  ),

  # Download iNaturalist metadata (contains photo URLs and metadata)
  tar_target(
    metadata_archive,
    download_file(metadata_url, file.path(output_dir_repro, basename(metadata_url))),
    format = "file"
  ),

  # ===========================================================================
  # Step 2: Extract Reproductive Annotations
  # ===========================================================================

  repro_annotations = extract_reproductive_annotations(dwc_archive, metadata_archive),

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
  # Step 3: Extract Leaf Annotations
  # ===========================================================================

  # Track leaf annotation files
  tar_target(leaf_parquet_file, leaf_parquet_path, format = "file"),
  tar_target(rob_annot_file, rob_annot_csv, format = "file"),
  tar_target(rob_annot2_file, rob_annot2_csv, format = "file"),

  # Extract leaf annotations
  leaf_annotations = extract_leaf_annotations(
    leaf_parquet_file,
    rob_annot_file,
    rob_annot2_file,
    photo_metadata_path
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
  # Step 4: Image Downloads (Placeholder - Waiting for Collaborator Script)
  # ===========================================================================

  # TODO: Collaborator is providing script for downloading images from parquet
  # This will download actual image files to local storage
  # For now, we assume images are already downloaded to /blue/guralnick/share/phenobase_inat_data/
  image_download_status = {
    message("Image downloads: Using existing images in /blue/guralnick/share/phenobase_inat_data/")
    message("TODO: Integrate collaborator's image download script when available")
    "pending_collaborator_script"
  },

  # ===========================================================================
  # Step 5: Summary Statistics
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
      image_downloads = image_download_status,
      message = paste0(
        "\n",
        paste(rep("=", 70), collapse = ""), "\n",
        "DOWNLOAD AND SPLIT COMPLETE\n",
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
        "\n",
        paste(rep("=", 70), collapse = "")
      )
    )
  }
)
