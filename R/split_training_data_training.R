#' Split and Version Training Data
#'
#' Splits annotations into train/val/test with version-tagged filenames.
#' Creates a metadata file documenting the split parameters.
#'
#' @param annotations Data frame with annotations (must have "good" column for filtering)
#' @param version Character. Version string (e.g., "v1.1.0")
#' @param output_dir Character. Directory to write CSV files
#' @param split_type Character. Type of annotations: "repro" or "leaf" (default: "repro")
#' @param train_prop Numeric. Training proportion (default: 0.6)
#' @param val_prop Numeric. Validation proportion (default: 0.2)
#' @param test_prop Numeric. Test proportion (default: 0.2)
#' @param seed Integer. Random seed (default: 234987)
#' @param pool Numeric. For leaf data only, min stratum size (default: 0.025)
#'
#' @return List with paths to versioned CSV files and metadata
#'
#' @details
#' This function:
#' 1. Filters to good images only (good == TRUE)
#' 2. Performs stratified splitting using rsample
#' 3. Writes CSVs with versioned filenames: train_v1.1.0.csv, val_v1.1.0.csv, test_v1.1.0.csv
#' 4. Creates metadata file: split_metadata_v1.1.0.json
#'
#' Stratification:
#' - Reproductive: order|flowering|fruiting
#' - Leaf: family|leaves_green|leaves_colored|leaves_breaking_buds
#'
#' @examples
#' \dontrun{
#' annotations <- read_parquet("data/inat_annotations_full.parquet")
#' annotations_good <- filter(annotations, good == TRUE)
#' splits <- split_and_version_training_data(
#'   annotations_good,
#'   version = "v1.1.0",
#'   output_dir = "data/inat"
#' )
#' }
#'
#' @export
split_and_version_training_data <- function(
    annotations,
    version,
    output_dir,
    split_type = c("repro", "leaf"),
    train_prop = 0.6,
    val_prop = 0.2,
    test_prop = 0.2,
    seed = 234987,
    pool = 0.025
) {

  library(rsample)
  library(dplyr)
  library(readr)
  library(jsonlite)

  split_type <- match.arg(split_type)

  cat("=====================================\n")
  cat("Versioned Data Splitting\n")
  cat("=====================================\n")
  cat("Version:", version, "\n")
  cat("Type:", split_type, "\n")
  cat("Output dir:", output_dir, "\n")
  cat("Split proportions: Train", train_prop, "/ Val", val_prop, "/ Test", test_prop, "\n")
  cat("Seed:", seed, "\n\n")

  # Check that we have good column
  if (!"good" %in% names(annotations)) {
    stop("annotations must contain a 'good' column from detect_bad_images()")
  }

  # Filter to good images only
  total_images <- nrow(annotations)
  good_annotations <- annotations %>% filter(good == TRUE)
  num_good <- nrow(good_annotations)
  num_bad <- total_images - num_good

  cat("Filtering to good images:\n")
  cat("  Total images:", total_images, "\n")
  cat("  Good images:", num_good, sprintf("(%.2f%%)\n", 100 * num_good / total_images))
  cat("  Removed (bad):", num_bad, sprintf("(%.2f%%)\n", 100 * num_bad / total_images))
  cat("\n")

  # Check proportions sum to 1
  if (abs(train_prop + val_prop + test_prop - 1.0) > 0.001) {
    stop("Proportions must sum to 1.0")
  }

  # Create stratification column based on split type
  if (split_type == "repro") {
    # Reproductive: stratify by order, flowering, fruiting
    required_cols <- c("order", "flowering", "fruiting")
    missing <- setdiff(required_cols, names(good_annotations))
    if (length(missing) > 0) {
      stop(paste0("Missing required columns for reproductive split: ", paste(missing, collapse = ", ")))
    }

    good_annotations <- good_annotations %>%
      mutate(strata = paste(order, flowering, fruiting, sep = "|"))

  } else if (split_type == "leaf") {
    # Leaf: stratify by family and leaf states
    required_cols <- c("family", "leaves_green", "leaves_colored", "leaves_breaking_buds")
    missing <- setdiff(required_cols, names(good_annotations))
    if (length(missing) > 0) {
      stop(paste0("Missing required columns for leaf split: ", paste(missing, collapse = ", ")))
    }

    good_annotations <- good_annotations %>%
      mutate(strata = paste(family, leaves_green, leaves_colored, leaves_breaking_buds, sep = "|"))
  }

  # Set seed for reproducibility
  set.seed(seed)

  # Perform split using rsample
  cat("Performing stratified split...\n")

  if (split_type == "leaf") {
    # Leaf split with pool parameter
    splits <- initial_validation_split(
      good_annotations,
      prop = c(train_prop, val_prop),
      strata = strata,
      pool = pool
    )
  } else {
    # Reproductive split (no pool parameter)
    splits <- initial_validation_split(
      good_annotations,
      prop = c(train_prop, val_prop),
      strata = strata
    )
  }

  # Extract splits and remove strata column
  train_data <- training(splits) %>% select(-strata, -good)
  val_data <- validation(splits) %>% select(-strata, -good)
  test_data <- testing(splits) %>% select(-strata, -good)

  # Summary
  cat("\nSplit results:\n")
  cat(sprintf("  Train:      %d observations (%.1f%%)\n", nrow(train_data), 100 * nrow(train_data) / num_good))
  cat(sprintf("  Validation: %d observations (%.1f%%)\n", nrow(val_data), 100 * nrow(val_data) / num_good))
  cat(sprintf("  Test:       %d observations (%.1f%%)\n", nrow(test_data), 100 * nrow(test_data) / num_good))
  cat("\n")

  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # Write versioned CSV files
  cat("Writing versioned CSV files...\n")

  train_path <- file.path(output_dir, paste0("train_", version, ".csv"))
  val_path <- file.path(output_dir, paste0("val_", version, ".csv"))
  test_path <- file.path(output_dir, paste0("test_", version, ".csv"))

  write_csv(train_data, train_path)
  write_csv(val_data, val_path)
  write_csv(test_data, test_path)

  cat(sprintf("  Train: %s (%d rows)\n", train_path, nrow(train_data)))
  cat(sprintf("  Val:   %s (%d rows)\n", val_path, nrow(val_data)))
  cat(sprintf("  Test:  %s (%d rows)\n", test_path, nrow(test_data)))
  cat("\n")

  # Create metadata file documenting the split
  metadata <- list(
    version = version,
    split_type = split_type,
    split_timestamp = as.character(Sys.time()),
    split_params = list(
      train_prop = train_prop,
      val_prop = val_prop,
      test_prop = test_prop,
      seed = seed,
      pool = if (split_type == "leaf") pool else NULL
    ),
    data_stats = list(
      total_images = total_images,
      good_images = num_good,
      bad_images = num_bad,
      train_n = nrow(train_data),
      val_n = nrow(val_data),
      test_n = nrow(test_data)
    ),
    file_paths = list(
      train = train_path,
      val = val_path,
      test = test_path
    )
  )

  metadata_path <- file.path(output_dir, paste0("split_metadata_", version, ".json"))
  write_json(metadata, metadata_path, pretty = TRUE, auto_unbox = TRUE)

  cat("Metadata written to:", metadata_path, "\n")
  cat("=====================================\n\n")

  # Return paths and metadata
  return(list(
    train_path = train_path,
    val_path = val_path,
    test_path = test_path,
    metadata_path = metadata_path,
    metadata = metadata,
    n_train = nrow(train_data),
    n_val = nrow(val_data),
    n_test = nrow(test_data)
  ))
}
