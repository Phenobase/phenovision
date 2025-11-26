#' Detect Bad Images in Training Data
#'
#' Tests each image in the annotations dataframe to determine if it can be
#' loaded successfully. Adds a "good" column to the dataframe.
#'
#' @param annotations_df Data frame with a `file_name` column containing image paths
#' @param batch_size Batch size for DataLoader (default 1024)
#' @param num_workers Number of parallel workers for DataLoader (default 16)
#' @param dummy_target_cols Columns to use as dummy targets (not used for detection,
#'   but required by PhenoDatasetDeleter). Default c("flowering", "fruiting") for repro.
#'
#' @return annotations_df with additional "good" column (TRUE/FALSE)
#'
#' @details
#' Uses the PhenoDatasetDeleter class from py/remove_bad_images.py to test each
#' image. An image is marked as bad (good = FALSE) if it fails to:
#' - Load with PIL.Image.open()
#' - Convert to RGB mode
#'
#' Bad images are logged to console for investigation.
#'
#' @examples
#' \dontrun{
#' annotations <- read_csv("data/inat_annotations_full.csv")
#' annotations_with_quality <- detect_bad_images(annotations)
#' good_annotations <- filter(annotations_with_quality, good == TRUE)
#' }
#'
#' @export
detect_bad_images <- function(
    annotations_df,
    batch_size = 1024L,
    num_workers = 16L,
    dummy_target_cols = c("flowering", "fruiting")
) {

  # Check required column exists
  if (!"file_name" %in% names(annotations_df)) {
    stop("annotations_df must contain a 'file_name' column")
  }

  # Check dummy target columns exist
  missing_cols <- setdiff(dummy_target_cols, names(annotations_df))
  if (length(missing_cols) > 0) {
    stop(paste0("annotations_df missing required columns: ",
                paste(missing_cols, collapse = ", ")))
  }

  cat("=====================================\n")
  cat("Detecting Bad Images\n")
  cat("=====================================\n")
  cat("Total images to test:", nrow(annotations_df), "\n")
  cat("Batch size:", batch_size, "\n")
  cat("Workers:", num_workers, "\n\n")

  # Import Python modules
  torch <- reticulate::import("torch")
  del <- reticulate::import_from_path("remove_bad_images", "py")

  # Prepare image paths and dummy targets
  img_paths <- reticulate::r_to_py(annotations_df$file_name)
  dummy_targets <- annotations_df |>
    dplyr::select(dplyr::all_of(dummy_target_cols)) |>
    as.matrix()

  # Create PhenoDatasetDeleter and DataLoader
  cat("Creating DataLoader...\n")
  ds <- del$PhenoDatasetDeleter(img_paths, dummy_targets)
  dl <- torch$utils$data$DataLoader(ds, batch_size, num_workers = num_workers)

  # Iterate through batches to test all images
  cat("Testing images...\n")
  batch_results <- reticulate::iterate(dl)

  # Extract filenames and bad flags
  all_filenames <- purrr::map(batch_results, 1) |>
    purrr::list_c()

  bad_flags <- purrr::map(batch_results, ~ as.vector(.x[[2]]$numpy())) |>
    purrr::list_c()

  # Convert to character for matching (in case of encoding differences)
  all_filenames <- as.character(unlist(all_filenames))

  # Create lookup dataframe: filename -> is_good
  image_quality <- tibble::tibble(
    file_name = all_filenames,
    good = bad_flags == 0  # bad_flag = 0 means good, 1 means bad
  )

  # Count bad images
  num_bad <- sum(!image_quality$good)
  num_good <- sum(image_quality$good)

  cat("\n=====================================\n")
  cat("Detection Results\n")
  cat("=====================================\n")
  cat("Good images:", num_good, sprintf("(%.2f%%)\n", 100 * num_good / nrow(annotations_df)))
  cat("Bad images:", num_bad, sprintf("(%.2f%%)\n", 100 * num_bad / nrow(annotations_df)))

  # Log bad images for investigation
  if (num_bad > 0) {
    cat("\nBad images detected:\n")
    bad_images <- image_quality |>
      dplyr::filter(!good) |>
      dplyr::pull(file_name)

    # Print first 20 bad images
    n_to_print <- min(20, length(bad_images))
    for (i in 1:n_to_print) {
      cat(sprintf("  %d. %s\n", i, bad_images[i]))
    }
    if (num_bad > 20) {
      cat(sprintf("  ... and %d more\n", num_bad - 20))
    }
  }

  cat("\n")

  # Join quality back to original dataframe
  # Use left_join to preserve original row order
  result <- annotations_df |>
    dplyr::left_join(image_quality, by = "file_name")

  # Sanity check: ensure all rows got a quality assessment
  if (any(is.na(result$good))) {
    warning("Some images did not receive a quality assessment!")
  }

  return(result)
}
