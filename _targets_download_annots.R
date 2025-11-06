# PhenoVision Annotation Download Pipeline
#
# This targets pipeline handles downloading and processing plant phenology
# annotations from iNaturalist:
# 1. Download DwC archive of annotated observations
# 2. Download iNaturalist open data metadata (photos.csv)
# 3. Extract and merge annotations
# 4. Split data into train/validation/test sets
# 5. Export to CSV files for training pipelines
#
# Usage:
#   targets::tar_make(script = "_targets_download_annots.R")
#   Rscript run_pipeline.R --pipeline=download_annots

# =============================================================================
# Setup
# =============================================================================

# Load common configuration
source("_targets_common.R")

# Load functions
source_common()     # Common functions
source_download()   # Download-specific functions (if any)

# Additional packages
library(conflicted)
conflicts_prefer(dplyr::filter)

# Configure targets for sequential execution (downloads are sequential)
setup_targets_parallel(workers = 1)

# =============================================================================
# Pipeline
# =============================================================================

list(

  # ===========================================================================
  # Download Configuration
  # ===========================================================================

  # Download parameters
  tar_target(
    download_config,
    list(
      # URLs
      dwc_url = "https://www.inaturalist.org/observations/phenobase-observations-dwca.zip",
      metadata_url = "https://inaturalist-open-data.s3.amazonaws.com/metadata/inaturalist-open-data-latest.tar.gz",

      # Output paths
      output_dir = "data/inat",
      dwc_file = "data/inat/phenobase-observations-dwca.zip",
      metadata_file = "data/inat/inaturalist-open-data-latest.tar.gz",
      merged_parquet = "data/inat/inaturalist-phenobase-data.parquet",

      # Split parameters
      train_split = 0.6,
      val_split = 0.2,
      test_split = 0.2,
      seed = 234987,

      # Date stamping
      append_date = FALSE  # Set TRUE to append date to downloaded files
    )
  ),

  # Create output directory
  tar_target(
    output_dir,
    {
      dir.create(download_config$output_dir, recursive = TRUE, showWarnings = FALSE)
      download_config$output_dir
    },
    format = "file"
  ),

  # ===========================================================================
  # Step 1: Download Raw Data
  # ===========================================================================

  # Download DwC archive with annotated observations
  tar_target(
    dwc_archive,
    {
      output_dir  # Dependency

      # Run Python download script
      cmd <- sprintf(
        "python phenobase/download_inat_data.py --output-dir %s --dwc-url %s%s",
        download_config$output_dir,
        download_config$dwc_url,
        if (download_config$append_date) " --today" else ""
      )

      system(cmd, intern = FALSE)

      # Return path to downloaded file
      download_config$dwc_file
    },
    format = "file"
  ),

  # Download iNaturalist metadata (contains photo URLs)
  tar_target(
    metadata_archive,
    {
      output_dir  # Dependency

      # Run Python download script
      cmd <- sprintf(
        "python phenobase/download_inat_data.py --output-dir %s --metadata-url %s%s",
        download_config$output_dir,
        download_config$metadata_url,
        if (download_config$append_date) " --today" else ""
      )

      system(cmd, intern = FALSE)

      # Return path to downloaded file
      download_config$metadata_file
    },
    format = "file"
  ),

  # ===========================================================================
  # Step 2: Extract and Merge Annotations
  # ===========================================================================

  tar_target(
    merged_data,
    {
      # Dependencies
      dwc_archive
      metadata_archive

      # Run Python extraction script
      cmd <- sprintf(
        "python phenobase/extract_inat_data.py --dwc-file %s --metadata-file %s --out-parquet %s",
        download_config$dwc_file,
        download_config$metadata_file,
        download_config$merged_parquet
      )

      system(cmd, intern = FALSE)

      # Return path to merged parquet
      download_config$merged_parquet
    },
    format = "file"
  ),

  # ===========================================================================
  # Step 3: Split Data into Train/Val/Test
  # ===========================================================================

  tar_target(
    split_data,
    {
      # Dependency
      merged_data

      # Run Python split script
      cmd <- sprintf(
        paste0(
          "python phenobase/split_data.py ",
          "--observations %s ",
          "--train-split %.2f ",
          "--val-split %.2f ",
          "--test-split %.2f ",
          "--seed %d"
        ),
        download_config$merged_parquet,
        download_config$train_split,
        download_config$val_split,
        download_config$test_split,
        download_config$seed
      )

      system(cmd, intern = FALSE)

      # Return path to parquet (modified in place)
      download_config$merged_parquet
    },
    format = "file",
    # Force re-run if split parameters change
    cue = tar_cue(mode = "always")
  ),

  # ===========================================================================
  # Step 4: Export to CSV Files for Training
  # ===========================================================================

  # Load the split parquet data
  tar_target(
    annotations_df,
    {
      split_data  # Dependency
      arrow::read_parquet(download_config$merged_parquet)
    }
  ),

  # Export reproductive structures data
  tar_target(
    train_csv,
    {
      path <- file.path(download_config$output_dir, "train.csv")

      annotations_df |>
        dplyr::filter(split == "train") |>
        dplyr::select(photo_id, extension, flowering, fruiting, reproductiveCondition,
                      scientificName, order, family, genus) |>
        dplyr::mutate(
          file_name = file.path(
            "/blue/guralnick/share/phenobase_inat_data/images/medium",
            paste0(photo_id, ".", extension)
          )
        ) |>
        readr::write_csv(path)

      path
    },
    format = "file"
  ),

  tar_target(
    validation_csv,
    {
      path <- file.path(download_config$output_dir, "validation.csv")

      annotations_df |>
        dplyr::filter(split == "val") |>
        dplyr::select(photo_id, extension, flowering, fruiting, reproductiveCondition,
                      scientificName, order, family, genus) |>
        dplyr::mutate(
          file_name = file.path(
            "/blue/guralnick/share/phenobase_inat_data/images/medium",
            paste0(photo_id, ".", extension)
          )
        ) |>
        readr::write_csv(path)

      path
    },
    format = "file"
  ),

  tar_target(
    test_csv,
    {
      path <- file.path(download_config$output_dir, "test.csv")

      annotations_df |>
        dplyr::filter(split == "test") |>
        dplyr::select(photo_id, extension, flowering, fruiting, reproductiveCondition,
                      scientificName, order, family, genus) |>
        dplyr::mutate(
          file_name = file.path(
            "/blue/guralnick/share/phenobase_inat_data/images/medium",
            paste0(photo_id, ".", extension)
          )
        ) |>
        readr::write_csv(path)

      path
    },
    format = "file"
  ),

  # ===========================================================================
  # Step 5: Summary Statistics
  # ===========================================================================

  tar_target(
    download_summary,
    {
      # Dependencies
      train_csv
      validation_csv
      test_csv

      # Compute summary statistics
      list(
        total_observations = nrow(annotations_df),
        train_n = sum(annotations_df$split == "train"),
        val_n = sum(annotations_df$split == "val"),
        test_n = sum(annotations_df$split == "test"),

        flowering_n = sum(annotations_df$flowering == 1),
        fruiting_n = sum(annotations_df$fruiting == 1),
        both_n = sum(annotations_df$flowering == 1 & annotations_df$fruiting == 1),

        n_orders = length(unique(annotations_df$order)),
        n_families = length(unique(annotations_df$family)),
        n_genera = length(unique(annotations_df$genus)),
        n_species = length(unique(annotations_df$scientificName)),

        output_files = list(
          train = train_csv,
          validation = validation_csv,
          test = test_csv
        ),

        message = paste0(
          "\n",
          paste(rep("=", 70), collapse = ""), "\n",
          "DOWNLOAD AND SPLIT COMPLETE\n",
          paste(rep("=", 70), collapse = ""), "\n",
          sprintf("Total observations: %d\n", nrow(annotations_df)),
          sprintf("  Train: %d (%.1f%%)\n", sum(annotations_df$split == "train"),
                  100 * mean(annotations_df$split == "train")),
          sprintf("  Val:   %d (%.1f%%)\n", sum(annotations_df$split == "val"),
                  100 * mean(annotations_df$split == "val")),
          sprintf("  Test:  %d (%.1f%%)\n\n", sum(annotations_df$split == "test"),
                  100 * mean(annotations_df$split == "test")),
          sprintf("Phenology annotations:\n"),
          sprintf("  Flowering: %d (%.1f%%)\n", sum(annotations_df$flowering == 1),
                  100 * mean(annotations_df$flowering == 1)),
          sprintf("  Fruiting:  %d (%.1f%%)\n", sum(annotations_df$fruiting == 1),
                  100 * mean(annotations_df$fruiting == 1)),
          sprintf("  Both:      %d (%.1f%%)\n\n",
                  sum(annotations_df$flowering == 1 & annotations_df$fruiting == 1),
                  100 * mean(annotations_df$flowering == 1 & annotations_df$fruiting == 1)),
          sprintf("Taxonomic coverage:\n"),
          sprintf("  Orders:  %d\n", length(unique(annotations_df$order))),
          sprintf("  Families: %d\n", length(unique(annotations_df$family))),
          sprintf("  Genera:   %d\n", length(unique(annotations_df$genus))),
          sprintf("  Species:  %d\n\n", length(unique(annotations_df$scientificName))),
          sprintf("Output files:\n"),
          sprintf("  Train:      %s\n", train_csv),
          sprintf("  Validation: %s\n", validation_csv),
          sprintf("  Test:       %s\n", test_csv),
          paste(rep("=", 70), collapse = "")
        )
      )
    }
  )
)
