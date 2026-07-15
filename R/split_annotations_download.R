#' Split Reproductive Annotations into Train/Val/Test
#'
#' Splits reproductive phenology data using stratified sampling.
#'
#' @param df Data frame with reproductive annotations
#' @param train_prop Numeric. Proportion for training (default: 0.6)
#' @param val_prop Numeric. Proportion for validation (default: 0.2)
#' @param test_prop Numeric. Proportion for test (default: 0.2)
#' @param seed Integer. Random seed for reproducibility (default: 234987)
#'
#' @return List with three data frames: train, val, test
#'
#' @details
#' Uses rsample::initial_validation_split() with stratification by:
#' - order (taxonomic order)
#' - flowering (0/1)
#' - fruiting (0/1)
#'
#' This ensures balanced representation of taxonomy and phenological states
#' across all splits.
#'
#' The stratification follows the approach from phenobase/split_data.py but
#' using tidymodels/rsample for cleaner R implementation.
#'
#' @examples
#' \dontrun{
#' splits <- split_repro_data(repro_df, train_prop = 0.6, val_prop = 0.2, test_prop = 0.2)
#' train_data <- splits$train
#' val_data <- splits$val
#' test_data <- splits$test
#' }
#'
#' @export
split_repro_data <- function(df, train_prop = 0.6, val_prop = 0.2, test_prop = 0.2, seed = 234987) {

  library(rsample)
  library(dplyr)

  message("Splitting reproductive annotations...")

  # Check proportions sum to 1
  if (abs(train_prop + val_prop + test_prop - 1.0) > 0.001) {
    stop("Proportions must sum to 1.0")
  }

  # Create stratification column
  df <- df %>%
    mutate(strata = paste(order, flowering, fruiting, sep = "|"))

  # Set seed
  set.seed(seed)

  # Create initial split (training vs testing+validation)
  # initial_validation_split wants c(train_prop, val_prop) where test = 1 - train - val
  splits <- initial_validation_split(
    df,
    prop = c(train_prop, val_prop),
    strata = strata
  )

  # Extract splits
  train_data <- training(splits) %>% select(-strata)
  val_data <- validation(splits) %>% select(-strata)
  test_data <- testing(splits) %>% select(-strata)

  # Summary
  message(sprintf("  Train:      %d observations (%.1f%%)", nrow(train_data), 100 * nrow(train_data) / nrow(df)))
  message(sprintf("  Validation: %d observations (%.1f%%)", nrow(val_data), 100 * nrow(val_data) / nrow(df)))
  message(sprintf("  Test:       %d observations (%.1f%%)", nrow(test_data), 100 * nrow(test_data) / nrow(df)))

  return(list(
    train = train_data,
    val = val_data,
    test = test_data
  ))
}


#' Split Leaf Annotations into Train/Val/Test/Seconds
#'
#' Splits leaf phenology data using stratified sampling, with special handling
#' for multiple photos per observation.
#'
#' @param df Data frame with leaf annotations (must have 'count' column)
#' @param train_prop Numeric. Proportion for training (default: 0.6)
#' @param val_prop Numeric. Proportion for validation (default: 0.2)
#' @param test_prop Numeric. Proportion for test (default: 0.2)
#' @param seed Integer. Random seed for reproducibility (default: 234987)
#' @param pool Numeric. Minimum proportion of data in each stratum (default: 0.025)
#'
#' @return List with four data frames: train, val, test, seconds
#'
#' @details
#' Uses rsample::initial_validation_split() with stratification by:
#' - family
#' - leaves_green (0/1)
#' - leaves_colored (0/1)
#' - leaves_breaking_buds (0/1)
#'
#' Observations with multiple photos (count > 1) are separated into 'seconds'
#' for use in second-round training. Only single-photo observations are split.
#'
#' The pool parameter (default 0.025 = 2.5%) ensures minimum data per stratum.
#'
#' @examples
#' \dontrun{
#' splits <- split_leaf_data(leaf_df, train_prop = 0.6, val_prop = 0.2, test_prop = 0.2)
#' train_data <- splits$train
#' val_data <- splits$val
#' test_data <- splits$test
#' seconds_data <- splits$seconds  # For second-round training
#' }
#'
#' @export
split_leaf_data <- function(df, train_prop = 0.6, val_prop = 0.2, test_prop = 0.2, seed = 234987, pool = 0.025) {

  library(rsample)
  library(dplyr)

  message("Splitting leaf annotations...")

  # Check proportions sum to 1
  if (abs(train_prop + val_prop + test_prop - 1.0) > 0.001) {
    stop("Proportions must sum to 1.0")
  }

  # Separate single vs multiple photo observations
  singles <- df %>% filter(count == 1)
  multiples <- df %>% filter(count > 1)

  message(sprintf("  Single-photo observations: %d", nrow(singles)))
  message(sprintf("  Multi-photo observations:  %d (saved as 'seconds')", nrow(multiples)))

  # Set seed
  set.seed(seed)

  # Split only single-photo observations
  splits <- initial_validation_split(
    singles,
    prop = c(train_prop, val_prop),
    strata = strata,
    pool = pool
  )

  # Extract splits
  train_data <- training(splits) %>% select(-strata, -count)
  val_data <- validation(splits) %>% select(-strata, -count)
  test_data <- testing(splits) %>% select(-strata, -count)
  seconds_data <- multiples %>% select(-strata, -count)

  # Summary
  message(sprintf("  Train:      %d observations (%.1f%%)", nrow(train_data), 100 * nrow(train_data) / nrow(singles)))
  message(sprintf("  Validation: %d observations (%.1f%%)", nrow(val_data), 100 * nrow(val_data) / nrow(singles)))
  message(sprintf("  Test:       %d observations (%.1f%%)", nrow(test_data), 100 * nrow(test_data) / nrow(singles)))
  message(sprintf("  Seconds:    %d observations (for round 2 training)", nrow(seconds_data)))

  return(list(
    train = train_data,
    val = val_data,
    test = test_data,
    seconds = seconds_data
  ))
}


#' Write Split Data to CSV
#'
#' Helper function to write a split to CSV with file path formatting.
#'
#' @param split_list List with train/val/test splits
#' @param split_name Character. Which split to write ("train", "val", "test", "seconds")
#' @param output_dir Character. Output directory
#' @param images_root Character. Root path for images (default: "data/phenobase_inat_data/images/medium")
#'
#' @return Character. Path to written CSV file (for targets format = "file")
#'
#' @details
#' Adds file_name column if not present, using photo_id and extension.
#' Writes CSV to output_dir/{split_name}.csv.
#'
#' @examples
#' \dontrun{
#' train_path <- write_csv_split(splits, "train", "data/inat")
#' }
#'
#' @export
write_csv_split <- function(split_list, split_name, output_dir,
                             images_root = "data/phenobase_inat_data/images/medium") {

  library(readr)
  library(dplyr)

  # Get the split
  split_data <- split_list[[split_name]]

  if (is.null(split_data)) {
    stop(sprintf("Split '%s' not found in split_list", split_name))
  }

  # Add file_name if not present
  if (!"file_name" %in% names(split_data)) {
    split_data <- split_data %>%
      mutate(file_name = file.path(images_root, paste0(photo_id, ".webp")))
  }

  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # Write CSV
  output_file <- file.path(output_dir, paste0(split_name, ".csv"))
  write_csv(split_data, output_file)

  message(sprintf("  Wrote %s: %d rows to %s", split_name, nrow(split_data), output_file))

  return(output_file)
}
