#' Download Images by Batch from iNaturalist Open Data
#'
#' Downloads images for batches defined in the angio_photos parquet dataset.
#' Handles parallel downloads, retries failed downloads, and tracks progress.
#'
#' This function wraps the logic from:
#' /blue/guralnick/share/phenobase_inat_data/Rcode/02_download_images_batch.R
#'
#' @param parquet_path Character. Path to angio_photos parquet dataset
#' @param image_dir Character. Root directory for images (default: "data/phenobase_inat_data/images")
#' @param size Character. Image size to download (default: "medium")
#' @param batch_ids Integer vector. Specific batch IDs to download (default: NULL = download all missing)
#' @param n_cores Integer. Number of parallel download workers (default: 10)
#' @param retry_failed Logical. Retry previously failed downloads (default: TRUE)
#'
#' @return Character vector. Paths to downloaded batch directories
#'
#' @details
#' This function:
#' 1. Reads the angio_photos parquet dataset to get batch assignments
#' 2. Identifies which batches need downloading (compares with existing folders)
#' 3. Downloads images for each batch in parallel using curl::multi_download
#' 4. Stores images in images/medium/batch_{j}/ folders
#' 5. Handles retries for failed downloads (0-byte or NA files)
#' 6. Tracks download status in metadata/img_download_status/
#'
#' The iNaturalist open data URLs follow this pattern:
#' https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/medium.{extension}
#'
#' Images are organized by batch to enable efficient parallel processing:
#' - images/medium/batch_1/ (first 100k images)
#' - images/medium/batch_2/ (next 100k images)
#' - etc.
#'
#' @examples
#' \dontrun{
#' # Download all missing batches
#' batch_dirs <- download_images_by_batch(
#'   parquet_path = "data/phenobase_inat_data/metadata/angio_photos",
#'   n_cores = 10
#' )
#'
#' # Download specific batches
#' batch_dirs <- download_images_by_batch(
#'   parquet_path = "data/phenobase_inat_data/metadata/angio_photos",
#'   batch_ids = 1:5,
#'   n_cores = 5
#' )
#' }
#'
#' @export
download_images_by_batch <- function(parquet_path,
                                      image_dir = "data/phenobase_inat_data/images",
                                      size = "medium",
                                      batch_ids = NULL,
                                      n_cores = 10,
                                      retry_failed = TRUE) {

  library(tidyverse)
  library(arrow)
  library(curl)
  library(parallel)

  message("Preparing to download images by batch...")

  # Create directories
  images_size_dir <- file.path(image_dir, size)
  dir.create(images_size_dir, recursive = TRUE, showWarnings = FALSE)

  status_dir <- file.path(dirname(parquet_path), "img_download_status")
  dir.create(status_dir, recursive = TRUE, showWarnings = FALSE)

  # =========================================================================
  # Step 1: Identify batches to download
  # =========================================================================

  message("  1. Identifying batches to download...")

  # Get all batch IDs from parquet
  angio_photos <- open_dataset(parquet_path)

  all_batches <- angio_photos %>%
    select(batch_j) %>%
    distinct() %>%
    collect() %>%
    pull(batch_j) %>%
    sort()

  # Get already downloaded batches
  batch_done <- list.files(images_size_dir, pattern = "^batch_", full.names = TRUE)
  batch_done_int <- as.integer(str_extract(batch_done, "[0-9]+$"))

  # Determine which batches to download
  if (!is.null(batch_ids)) {
    batches_to_download <- intersect(batch_ids, all_batches)
    message(sprintf("    User specified %d batches", length(batches_to_download)))
  } else {
    batches_to_download <- setdiff(all_batches, batch_done_int)
  }

  batches_to_download <- sort(batches_to_download)

  if (length(batches_to_download) == 0) {
    message("    No batches need downloading - all complete!")
    return(batch_done)
  }

  # Estimate number of images
  max_batch <- max(batches_to_download)
  n_last <- angio_photos %>%
    filter(batch_j == max_batch) %>%
    count() %>%
    collect() %>%
    pull(n)

  n_imgs_to_download <- 100000 * (length(batches_to_download) - 1) + n_last

  message(sprintf("    Batches to download: %d (batches %d to %d)",
                  length(batches_to_download),
                  min(batches_to_download),
                  max(batches_to_download)))
  message(sprintf("    Estimated images: %s", format(n_imgs_to_download, big.mark = ",")))

  # =========================================================================
  # Step 2: Download images batch by batch (parallelized)
  # =========================================================================

  message(sprintf("  2. Downloading images using %d parallel workers...", n_cores))

  # Helper function to download one batch
  download_batch <- function(batch_id, size, parquet_path, images_size_dir, status_dir, retry_failed) {

    f_target <- file.path(images_size_dir, paste0("batch_", batch_id))

    # Load photos for this batch
    photos_i <- open_dataset(parquet_path) %>%
      filter(batch_j == batch_id) %>%
      collect() %>%
      mutate(img_id = paste(photo_id, extension, sep = "."))

    # Check if batch already exists - only download missing images
    if (dir.exists(f_target)) {
      img_done <- list.files(f_target)
      photos_i <- photos_i %>%
        filter(!img_id %in% img_done)

      # All downloaded already?
      if (nrow(photos_i) == 0) {
        return(f_target)
      }
    }

    # Create batch directory
    dir.create(f_target, recursive = TRUE, showWarnings = FALSE)

    # Construct URLs and destination paths
    photos_i <- photos_i %>%
      mutate(
        f_url = sprintf("https://inaturalist-open-data.s3.amazonaws.com/photos/%s/%s.%s",
                        photo_id, size, extension),
        des_f = file.path(f_target, paste0(photo_id, ".", extension))
      )

    # Download images
    xi <- curl::multi_download(
      urls = photos_i$f_url,
      destfiles = photos_i$des_f,
      resume = TRUE,
      progress = FALSE
    )

    # Check for failed downloads (0-byte or NA files)
    fs <- file.size(photos_i$des_f)
    fs_0 <- (fs == 0 | is.na(fs))

    if (any(fs_0) && retry_failed) {
      # Retry failed downloads once
      xi2 <- curl::multi_download(
        urls = photos_i$f_url[fs_0],
        destfiles = photos_i$des_f[fs_0],
        resume = TRUE,
        progress = FALSE
      )

      # Check again
      fs2 <- file.size(photos_i$des_f[fs_0])
      fs2_0 <- (fs2 == 0 | is.na(fs2))

      if (any(fs2_0)) {
        # Save status of failed images
        write_csv(xi2, file.path(status_dir, paste0("status_", batch_id, ".csv")))
        warning(sprintf("Batch %d: %d images failed to download", batch_id, sum(fs2_0)))
      }
    }

    return(f_target)
  }

  # Split batches into chunks for parallel processing
  batch_chunks <- split(
    batches_to_download,
    ceiling(seq_along(batches_to_download) / n_cores)
  )

  # Process chunks sequentially, batches within chunks in parallel
  downloaded_dirs <- character()

  for (i in seq_along(batch_chunks)) {
    message(sprintf("    Processing chunk %d of %d (%d batches in parallel)...",
                    i, length(batch_chunks), length(batch_chunks[[i]])))

    chunk_dirs <- mclapply(
      batch_chunks[[i]],
      download_batch,
      size = size,
      parquet_path = parquet_path,
      images_size_dir = images_size_dir,
      status_dir = status_dir,
      retry_failed = retry_failed,
      mc.cores = min(length(batch_chunks[[i]]), n_cores)
    )

    downloaded_dirs <- c(downloaded_dirs, unlist(chunk_dirs))
  }

  # =========================================================================
  # Step 3: Update metadata date file
  # =========================================================================

  metadata_date_file <- file.path(dirname(parquet_path), "metadata_date.txt")
  cat(
    sprintf("    About %s images were downloaded into folders from batch %d to %d on %s\n",
            format(n_imgs_to_download, big.mark = ","),
            min(batches_to_download),
            max(batches_to_download),
            Sys.Date()),
    file = metadata_date_file,
    append = TRUE
  )

  message(sprintf("Download complete! %d batches downloaded to %s",
                  length(batches_to_download), images_size_dir))

  return(downloaded_dirs)
}
