# ==============================================================================
# Image Download Functions - Targets Dynamic Branching with Pre-Extraction
# ==============================================================================
#
# This file contains download functions that use targets dynamic branching
# with pre-extracted metadata to avoid parallel parquet access bottlenecks.
#
# Key improvements over original approach:
# - Downloads ALL batches from parquet (not just annotated ones)
# - Pre-extracts ALL metadata in single parquet query before branching
# - Each worker receives pre-computed URLs/paths (no parquet access)
# - Maintains retry logic and resume support from collaborator's script
#
# Architecture:
#   1. identify_all_batches() - Get unique batch_j from parquet
#   2. identify_missing_batches_v2() - Compare vs existing folders
#   3. extract_batch_metadata() - Single query for all needed batches
#   4. prepare_batch_download_lists() - Convert to list-of-lists with URLs
#   5. download_batch_from_list() - Download using pre-extracted data
#   6. summarize_batch_downloads() - Aggregate results
#
# Based on collaborator's script:
#   /blue/guralnick/share/phenobase_inat_data/Rcode/02_download_images_batch.R
#
# Created: 2025-01-08
# ==============================================================================

#' Get All Unique Batch IDs from Parquet
#'
#' Queries the angio_photos parquet to get all unique batch_j values.
#' This is done ONCE as an upstream target before identifying which batches
#' need downloading.
#'
#' @param parquet_path Path to angio_photos parquet dataset
#' @return Integer vector of all batch IDs in the parquet
#' @export
identify_all_batches <- function(parquet_path) {
  library(arrow)
  library(dplyr)

  message("Querying parquet for all unique batch IDs...")

  all_batches <- open_dataset(parquet_path) %>%
    select(batch_j) %>%
    distinct() %>%
    collect() %>%
    pull(batch_j) %>%
    sort()

  message(sprintf("Found %d unique batches in parquet (range: %d to %d)",
                  length(all_batches), min(all_batches), max(all_batches)))

  return(all_batches)
}


#' Identify Batches That Need Downloading (All Images, Not Just Annotated)
#'
#' Compares ALL batches in parquet against existing batch folders to find
#' missing batches. This ensures we download all images for inference, not
#' just annotated images.
#'
#' @param all_batches Integer vector of all batch IDs from parquet
#' @param images_root Path to images directory (e.g., "data/.../images/medium")
#' @return Integer vector of batch IDs that need downloading
#' @export
identify_missing_batches_v2 <- function(all_batches, images_root) {

  # Get already downloaded batches
  batch_folders <- list.files(images_root, pattern = "^batch_", full.names = FALSE)

  if (length(batch_folders) == 0) {
    message(sprintf("No existing batches found - will download all %d batches",
                    length(all_batches)))
    return(sort(all_batches))
  }

  batch_done_int <- as.integer(str_extract(batch_folders, "[0-9]+$"))

  # Find missing batches
  missing <- setdiff(all_batches, batch_done_int)

  if (length(missing) == 0) {
    message("All batches already downloaded!")
    return(integer(0))
  }

  message(sprintf(
    "Found %d batches to download (out of %d total, %d already exist)",
    length(missing), length(all_batches), length(batch_done_int)
  ))

  return(sort(missing))
}


#' Extract Photo Metadata for All Batches Needing Download
#'
#' Performs a SINGLE parquet query to extract ALL photo metadata for batches
#' that need downloading. This avoids multiple workers querying the parquet
#' in parallel (which can be a bottleneck).
#'
#' Returns a tibble with all photo information needed for download.
#'
#' @param parquet_path Path to angio_photos parquet dataset
#' @param batches_to_download Integer vector of batch IDs to download
#' @return Tibble with columns: batch_j, photo_id, extension
#' @export
extract_batch_metadata <- function(parquet_path, batches_to_download) {
  library(arrow)
  library(dplyr)

  if (length(batches_to_download) == 0) {
    message("No batches to download - returning empty tibble")
    return(tibble(batch_j = integer(), photo_id = character(), extension = character()))
  }

  message(sprintf(
    "Extracting photo metadata for %d batches from parquet (SINGLE QUERY)...",
    length(batches_to_download)
  ))

  start_time <- Sys.time()

  # Single parquet query for all needed batches
  batch_metadata <- open_dataset(parquet_path) %>%
    filter(batch_j %in% batches_to_download) %>%
    select(batch_j, photo_id, extension) %>%
    collect()

  end_time <- Sys.time()
  elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))

  message(sprintf(
    "Extracted metadata for %s photos across %d batches in %.1f seconds",
    format(nrow(batch_metadata), big.mark = ","),
    length(unique(batch_metadata$batch_j)),
    elapsed
  ))

  return(batch_metadata)
}


#' Prepare Download Lists for Each Batch
#'
#' Converts extracted metadata into a list-of-lists structure where each
#' element contains pre-computed URLs and destination paths for one batch.
#' This allows workers to download without accessing the parquet.
#'
#' @param batch_metadata Tibble from extract_batch_metadata()
#' @param images_root Path to images directory
#' @param size Image size to download (default: "medium")
#' @return Named list where each element is a list with:
#'   - batch_id: Integer batch ID
#'   - photo_ids: Character vector of photo IDs
#'   - extensions: Character vector of file extensions
#'   - urls: Character vector of pre-computed download URLs
#'   - dest_files: Character vector of pre-computed destination paths
#'   - n_photos: Number of photos in this batch
#' @export
prepare_batch_download_lists <- function(batch_metadata,
                                          images_root,
                                          size = "medium") {
  library(dplyr)
  library(stringr)

  if (nrow(batch_metadata) == 0) {
    message("No metadata to prepare - returning empty list")
    return(list())
  }

  message(sprintf(
    "Preparing download lists for %d batches...",
    length(unique(batch_metadata$batch_j))
  ))

  # Split metadata by batch
  batches_split <- split(batch_metadata, batch_metadata$batch_j)

  # Convert each batch to download list with pre-computed URLs and paths
  batch_lists <- lapply(batches_split, function(batch) {
    batch_id <- unique(batch$batch_j)[1]
    batch_dir <- file.path(images_root, paste0("batch_", batch_id))

    # Deduplicate by photo_id and extension to avoid duplicate downloads
    batch_dedup <- batch %>%
      distinct(photo_id, extension, .keep_all = TRUE)

    list(
      batch_id = batch_id,
      photo_ids = batch_dedup$photo_id,
      extensions = batch_dedup$extension,
      urls = sprintf("https://inaturalist-open-data.s3.amazonaws.com/photos/%s/%s.%s",
                     batch_dedup$photo_id, size, batch_dedup$extension),
      dest_files = file.path(batch_dir, paste0(batch_dedup$photo_id, ".", batch_dedup$extension)),
      n_photos = nrow(batch_dedup)
    )
  })

  # Return UNnamed list for targets pattern=map() compatibility
  # (Named lists can cause issues with dynamic branching)
  # Note: split() creates named lists by default, so we explicitly unname
  batch_lists_unnamed <- unname(batch_lists)

  message(sprintf(
    "Prepared %d download lists with %s total photos",
    length(batch_lists_unnamed),
    format(sum(sapply(batch_lists_unnamed, `[[`, "n_photos")), big.mark = ",")
  ))

  return(batch_lists_unnamed)
}


#' Download Images for One Batch Using Pre-Extracted Data
#'
#' Downloads all images for a single batch using pre-computed URLs and paths.
#' NO parquet access - all data comes from upstream target.
#'
#' Includes:
#' - Resume support (checks existing files)
#' - Two-stage retry logic (from collaborator's script)
#' - File size validation
#' - Failed image logging to CSV
#'
#' @param batch_data List with pre-computed URLs/paths from prepare_batch_download_lists()
#' @param status_dir Path to directory for logging failed downloads
#' @return List with batch download status:
#'   - batch_id: Integer batch ID
#'   - success: Logical, whether all images downloaded successfully
#'   - n_images_attempted: Number of images attempted (after checking existing)
#'   - n_images_success: Number of images successfully downloaded
#'   - n_images_failed: Number of images that failed after retry
#'   - start_time: When batch download started
#'   - end_time: When batch download completed
#'   - elapsed_secs: Time elapsed in seconds
#' @export
download_batch_from_list <- function(batch_data, status_dir) {
  library(dplyr)
  library(curl)
  library(readr)

  batch_id <- batch_data$batch_id
  batch_start <- Sys.time()

  message(sprintf("\n[Batch %d] Starting download...", batch_id))

  # ===========================================================================
  # Step 1: Check for existing images (resume capability)
  # ===========================================================================

  batch_dir <- dirname(batch_data$dest_files[1])

  # Create batch directory if needed
  dir.create(batch_dir, recursive = TRUE, showWarnings = FALSE)

  # Check which images already exist
  img_ids <- paste(batch_data$photo_ids, batch_data$extensions, sep = ".")

  if (dir.exists(batch_dir)) {
    img_done <- list.files(batch_dir)

    if (length(img_done) > 0) {
      # Filter to only download missing images
      to_download_mask <- !img_ids %in% img_done

      if (sum(to_download_mask) == 0) {
        message(sprintf("[Batch %d] All images already downloaded - skipping", batch_id))
        return(list(
          batch_id = batch_id,
          success = TRUE,
          n_images_attempted = 0,
          n_images_success = 0,
          n_images_failed = 0,
          start_time = batch_start,
          end_time = Sys.time(),
          elapsed_secs = 0
        ))
      }

      # Update download lists to only include missing images
      urls_to_download <- batch_data$urls[to_download_mask]
      dest_to_download <- batch_data$dest_files[to_download_mask]

      message(sprintf("[Batch %d] Resuming: %s already downloaded, %s remaining",
                      batch_id,
                      format(length(img_done), big.mark = ","),
                      format(sum(to_download_mask), big.mark = ",")))
    } else {
      urls_to_download <- batch_data$urls
      dest_to_download <- batch_data$dest_files
    }
  } else {
    urls_to_download <- batch_data$urls
    dest_to_download <- batch_data$dest_files
  }

  n_to_download <- length(urls_to_download)

  # ===========================================================================
  # Step 2: ATTEMPT 1 - Initial download IN MICRO-BATCHES
  # ===========================================================================
  # CRITICAL: Must use small batches (≤100 files) with multiplex=FALSE
  # Empirical testing showed:
  #   - 5000 files: 22% success (massive failures)
  #   - 100 files with multiplex=FALSE: 100% success
  # HTTP/2 multiplexing causes connection failures with S3

  chunk_size <- 100  # CRITICAL: Do not increase above 100
  n_chunks <- ceiling(n_to_download / chunk_size)

  message(sprintf("[Batch %d] Downloading %s images in %d micro-batches of ~%d (attempt 1)...",
                  batch_id, format(n_to_download, big.mark = ","),
                  n_chunks, chunk_size))

  # Download each micro-batch
  xi_list <- list()
  for (chunk_i in seq_len(n_chunks)) {
    start_idx <- (chunk_i - 1) * chunk_size + 1
    end_idx <- min(chunk_i * chunk_size, n_to_download)
    chunk_indices <- start_idx:end_idx

    if (chunk_i %% 50 == 0) {  # Report progress every 50 chunks
      message(sprintf("[Batch %d] Chunk %d/%d: downloading images %d-%d...",
                      batch_id, chunk_i, n_chunks, start_idx, end_idx))
    }

    xi_chunk <- curl::multi_download(
      urls = urls_to_download[chunk_indices],
      destfiles = dest_to_download[chunk_indices],
      resume = TRUE,
      progress = FALSE,
      multiplex = FALSE,      # CRITICAL: Disable HTTP/2 multiplexing
      multi_timeout = 60      # CRITICAL: Longer timeout
    )

    xi_list[[chunk_i]] <- xi_chunk

    # Small delay to avoid overwhelming server
    if (chunk_i < n_chunks) Sys.sleep(0.5)
  }

  # Combine all chunk results
  xi <- do.call(rbind, xi_list)

  # ===========================================================================
  # Step 3: Check for failed downloads (0-byte or NA files)
  # ===========================================================================

  fs <- file.size(dest_to_download)
  fs_0 <- (fs == 0 | is.na(fs))
  n_failed_attempt1 <- sum(fs_0)

  # ===========================================================================
  # Step 4: ATTEMPT 2 - Retry failed downloads
  # ===========================================================================

  if (n_failed_attempt1 > 0) {
    message(sprintf("[Batch %d] Retrying %s failed downloads in micro-batches (attempt 2)...",
                    batch_id, format(n_failed_attempt1, big.mark = ",")))

    # Keep successful downloads from attempt 1
    xi_success <- xi[!fs_0, ]

    # Get indices of failed downloads
    failed_indices <- which(fs_0)

    # Micro-batch the retry downloads with same settings
    n_retry_chunks <- ceiling(length(failed_indices) / chunk_size)

    xi2_list <- list()
    for (chunk_i in seq_len(n_retry_chunks)) {
      start_idx <- (chunk_i - 1) * chunk_size + 1
      end_idx <- min(chunk_i * chunk_size, length(failed_indices))
      chunk_failed_indices <- failed_indices[start_idx:end_idx]

      if (chunk_i %% 50 == 0) {
        message(sprintf("[Batch %d] Retry chunk %d/%d: %d images...",
                        batch_id, chunk_i, n_retry_chunks, length(chunk_failed_indices)))
      }

      xi2_chunk <- curl::multi_download(
        urls = urls_to_download[chunk_failed_indices],
        destfiles = dest_to_download[chunk_failed_indices],
        resume = TRUE,
        progress = FALSE,
        multiplex = FALSE,      # CRITICAL
        multi_timeout = 60      # CRITICAL
      )

      xi2_list[[chunk_i]] <- xi2_chunk

      if (chunk_i < n_retry_chunks) Sys.sleep(0.5)
    }

    # Combine retry results
    xi2 <- do.call(rbind, xi2_list)

    # ===========================================================================
    # Step 5: Check retry results and log permanent failures
    # ===========================================================================

    # Check file sizes for the originally failed downloads
    fs2 <- file.size(dest_to_download[failed_indices])
    fs2_0 <- (fs2 == 0 | is.na(fs2))
    n_failed_attempt2 <- sum(fs2_0)

    if (n_failed_attempt2 > 0) {
      # Create status directory if needed
      dir.create(status_dir, recursive = TRUE, showWarnings = FALSE)

      # Log all download attempts for this batch (both successful and failed)
      status_csv <- file.path(status_dir, paste0("status_", batch_id, ".csv"))
      write_csv(xi2, status_csv)

      # Get indices of permanently failed images
      perm_failed_indices <- failed_indices[fs2_0]

      # Get info about permanently failed images
      failed_urls <- urls_to_download[perm_failed_indices]
      failed_files <- dest_to_download[perm_failed_indices]

      warning(sprintf(
        "[Batch %d] %s images PERMANENTLY FAILED after retry - logged to %s",
        batch_id, format(n_failed_attempt2, big.mark = ","), status_csv
      ))
    }
  } else {
    n_failed_attempt2 <- 0
  }

  # ===========================================================================
  # Step 6: Calculate summary statistics
  # ===========================================================================

  batch_end <- Sys.time()
  elapsed <- as.numeric(difftime(batch_end, batch_start, units = "secs"))

  n_success <- n_to_download - n_failed_attempt2
  success_all <- (n_failed_attempt2 == 0)

  message(sprintf(
    "[Batch %d] Complete in %.1f sec: %s/%s images successful (%.1f%%)",
    batch_id, elapsed,
    format(n_success, big.mark = ","),
    format(n_to_download, big.mark = ","),
    100 * n_success / n_to_download
  ))

  # ===========================================================================
  # Step 7: Return status object (NOT using format="file")
  # ===========================================================================

  return(list(
    batch_id = batch_id,
    success = success_all,
    n_images_attempted = n_to_download,
    n_images_success = n_success,
    n_images_failed = n_failed_attempt2,
    start_time = batch_start,
    end_time = batch_end,
    elapsed_secs = elapsed
  ))
}


#' Summarize Download Results Across All Batches
#'
#' Aggregates status from all batch downloads (via dynamic branching) to
#' produce final summary statistics.
#'
#' @param batch_download_status List of batch status objects from pattern=map()
#' @param all_batches All batch IDs in parquet (for context)
#' @param batches_to_download Batch IDs that were attempted
#' @return List with summary statistics
#' @export
summarize_batch_downloads <- function(batch_download_status,
                                       all_batches,
                                       batches_to_download) {

  # Handle case where nothing was downloaded (all batches already exist)
  if (length(batch_download_status) == 0) {
    summary_msg <- paste0(
      "\n", paste(rep("=", 70), collapse = ""), "\n",
      "IMAGE DOWNLOAD SUMMARY\n",
      paste(rep("=", 70), collapse = ""), "\n",
      "All batches already downloaded - nothing to do!\n",
      sprintf("Total batches in parquet: %d\n", length(all_batches)),
      paste(rep("=", 70), collapse = "")
    )

    message(summary_msg)

    return(list(
      message = summary_msg,
      n_total_batches = length(all_batches),
      n_attempted = 0,
      n_success = 0,
      n_failed = 0,
      total_images_attempted = 0,
      total_images_success = 0,
      total_images_failed = 0,
      total_elapsed_mins = 0
    ))
  }

  # ===========================================================================
  # Aggregate statistics across all batches
  # ===========================================================================

  n_batches_attempted <- length(batch_download_status)

  # Extract fields from status objects
  batch_ids <- sapply(batch_download_status, `[[`, "batch_id")
  success_flags <- sapply(batch_download_status, `[[`, "success")
  n_images_attempted <- sapply(batch_download_status, `[[`, "n_images_attempted")
  n_images_success <- sapply(batch_download_status, `[[`, "n_images_success")
  n_images_failed <- sapply(batch_download_status, `[[`, "n_images_failed")
  elapsed_secs <- sapply(batch_download_status, `[[`, "elapsed_secs")

  # Calculate summaries
  n_batches_success <- sum(success_flags)
  n_batches_failed <- sum(!success_flags)
  total_images_attempted <- sum(n_images_attempted)
  total_images_success <- sum(n_images_success)
  total_images_failed <- sum(n_images_failed)
  total_elapsed_secs <- sum(elapsed_secs)
  total_elapsed_mins <- total_elapsed_secs / 60

  # Success rates
  batch_success_rate <- 100 * n_batches_success / n_batches_attempted
  image_success_rate <- if (total_images_attempted > 0) {
    100 * total_images_success / total_images_attempted
  } else {
    100
  }

  # ===========================================================================
  # Build summary message
  # ===========================================================================

  summary_msg <- sprintf(
    paste0(
      "\n", paste(rep("=", 70), collapse = ""), "\n",
      "IMAGE DOWNLOAD SUMMARY\n",
      paste(rep("=", 70), collapse = ""), "\n",
      "Batches in parquet:       %s\n",
      "Batches attempted:        %s\n",
      "Batches successful:       %s (%.1f%%)\n",
      "Batches with failures:    %s\n",
      "\n",
      "Images attempted:         %s\n",
      "Images successful:        %s (%.1f%%)\n",
      "Images failed:            %s\n",
      "\n",
      "Total time:               %.1f minutes (%.1f hours)\n",
      "Average per batch:        %.1f seconds\n",
      paste(rep("=", 70), collapse = "")
    ),
    format(length(all_batches), big.mark = ","),
    format(n_batches_attempted, big.mark = ","),
    format(n_batches_success, big.mark = ","), batch_success_rate,
    format(n_batches_failed, big.mark = ","),
    format(total_images_attempted, big.mark = ","),
    format(total_images_success, big.mark = ","), image_success_rate,
    format(total_images_failed, big.mark = ","),
    total_elapsed_mins, total_elapsed_mins / 60,
    total_elapsed_secs / n_batches_attempted
  )

  message(summary_msg)

  # ===========================================================================
  # Return summary object
  # ===========================================================================

  return(list(
    message = summary_msg,
    n_total_batches = length(all_batches),
    n_to_download = length(batches_to_download),
    n_attempted = n_batches_attempted,
    n_success = n_batches_success,
    n_failed = n_batches_failed,
    total_images_attempted = total_images_attempted,
    total_images_success = total_images_success,
    total_images_failed = total_images_failed,
    total_elapsed_mins = total_elapsed_mins,
    batch_success_rate = batch_success_rate,
    image_success_rate = image_success_rate,
    success_batch_ids = batch_ids[success_flags],
    failed_batch_ids = batch_ids[!success_flags]
  ))
}
