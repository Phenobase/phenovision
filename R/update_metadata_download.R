#' Update iNaturalist Metadata from Open Data
#'
#' Downloads latest iNaturalist open data metadata, filters to angiosperms,
#' identifies new photos, assigns batch numbers, and updates the parquet dataset.
#'
#' This function wraps the logic from:
#' /blue/guralnick/share/phenobase_inat_data/Rcode/01_get_new_inat_metadata2.R
#'
#' @param metadata_dir Character. Root directory for metadata (default: "data/phenobase_inat_data/metadata")
#' @param batch_size Integer. Number of images per batch (default: 100000)
#' @param force_download Logical. Re-download metadata even if recent file exists (default: FALSE)
#'
#' @return Character. Path to updated parquet dataset directory
#'
#' @details
#' This function:
#' 1. Downloads latest iNaturalist metadata tar.gz (~120+ GB memory required!)
#' 2. Filters taxa to angiosperms (ancestry contains "47125")
#' 3. Uses awk to filter observations.csv for plant research-grade observations
#' 4. Filters photos.csv to only photos from those observations
#' 5. Compares with existing metadata to find NEW photo_ids
#' 6. Assigns new batch numbers (continuing from old_batch_max)
#' 7. Combines old and new photos and writes to parquet
#'
#' The resulting parquet dataset has columns:
#' - photo_id, observation_uuid, observer_id, extension
#' - photo_uuid, license, width, height, position
#' - batch_j (for organizing downloads)
#'
#' @examples
#' \dontrun{
#' parquet_path <- update_inat_metadata(
#'   metadata_dir = "data/phenobase_inat_data/metadata",
#'   batch_size = 100000
#' )
#' }
#'
#' @export
update_inat_metadata <- function(metadata_dir = "data/phenobase_inat_data/metadata",
                                   batch_size = 100000,
                                   force_download = FALSE) {

  library(tidyverse)
  library(arrow)
  library(xfun)

  message("Updating iNaturalist metadata...")

  # Create metadata directory if needed
  dir.create(metadata_dir, recursive = TRUE, showWarnings = FALSE)

  # Output parquet path
  parquet_path <- file.path(metadata_dir, "angio_photos")

  # =========================================================================
  # Step 1: Download latest metadata (if needed)
  # =========================================================================

  metadata_url <- "https://inaturalist-open-data.s3.amazonaws.com/metadata/inaturalist-open-data-latest.tar.gz"
  meta_file <- file.path(metadata_dir, "inaturalist-open-data-latest.tar.gz")

  # Check if recent metadata exists (within 30 days)
  download_needed <- force_download
  if (!download_needed && file.exists(meta_file)) {
    file_age_days <- as.numeric(difftime(Sys.time(), file.mtime(meta_file), units = "days"))
    if (file_age_days > 30) {
      message(sprintf("  Existing metadata is %d days old, re-downloading...", round(file_age_days)))
      download_needed <- TRUE
    } else {
      message(sprintf("  Using existing metadata (%.1f days old)", file_age_days))
    }
  } else {
    download_needed <- TRUE
  }

  if (download_needed) {
    message("  1. Downloading latest iNaturalist metadata (this requires 120+ GB memory)...")
    xfun::download_file(metadata_url, meta_file)

    message("  2. Extracting metadata archive...")
    system(paste0("tar -xvzf ", meta_file, " -C ", metadata_dir))
  } else {
    message("  Skipping download, using existing metadata")
  }

  # =========================================================================
  # Step 2: Filter taxa to angiosperms
  # =========================================================================

  message("  3. Filtering taxa to angiosperms...")

  taxa_file <- file.path(metadata_dir, "taxa.csv")
  inat_taxa <- read_tsv(taxa_file, show_col_types = FALSE)

  angio_taxa <- inat_taxa %>%
    filter(grepl("47125", ancestry)) %>%  # Angiosperm taxon ID
    filter(rank %in% c("species", "subspecies", "variety"))

  message(sprintf("    Found %s angiosperm taxa", format(nrow(angio_taxa), big.mark = ",")))

  # =========================================================================
  # Step 3: Filter observations using awk (efficient for huge files)
  # =========================================================================

  message("  4. Filtering observations to research-grade angiosperms...")

  obs_file <- file.path(metadata_dir, "observations.csv")
  obs_filtered_file <- file.path(metadata_dir, "angio_observations.csv")

  # Write taxa IDs for awk filtering
  taxa_ids_file <- tempfile(fileext = ".txt")
  writeLines(as.character(angio_taxa$taxon_id), taxa_ids_file)

  # Use awk to filter observations (much faster than R for huge files)
  # Column 6 = taxon_id, Column 7 = quality_grade
  awk_cmd <- sprintf(
    "awk 'BEGIN {FS=\"\\t\"; while(getline < \"%s\") taxa[$0]=1} NR > 1 && $6 in taxa && $7 == \"research\" {print}' %s > %s",
    taxa_ids_file,
    obs_file,
    obs_filtered_file
  )
  system(awk_cmd)

  # Read filtered observations
  angio_obs <- read_tsv(
    obs_filtered_file,
    col_names = c("observer_id", "observation_uuid", "observed_on",
                  "time_observed_at", "time_zone", "taxon_id", "quality_grade"),
    col_types = cols(.default = col_character()),
    show_col_types = FALSE
  )

  message(sprintf("    Found %s research-grade observations", format(nrow(angio_obs), big.mark = ",")))

  # =========================================================================
  # Step 4: Filter photos to those from our observations
  # =========================================================================

  message("  5. Filtering photos...")

  photos_file <- file.path(metadata_dir, "photos.csv")

  # Write observation UUIDs for awk filtering
  obs_uuids_file <- tempfile(fileext = ".txt")
  writeLines(angio_obs$observation_uuid, obs_uuids_file)

  # Use awk to filter photos (Column 2 = observation_uuid)
  photos_filtered_file <- file.path(metadata_dir, "angio_photos.csv")
  awk_cmd2 <- sprintf(
    "awk 'BEGIN {FS=\"\\t\"; while(getline < \"%s\") obs[$0]=1} NR > 1 && $2 in obs {print}' %s > %s",
    obs_uuids_file,
    photos_file,
    photos_filtered_file
  )
  system(awk_cmd2)

  # Read filtered photos
  angio_photos_new_raw <- read_tsv(
    photos_filtered_file,
    col_types = cols(.default = col_character()),
    show_col_types = FALSE
  )

  message(sprintf("    Found %s photos from angiosperm observations", format(nrow(angio_photos_new_raw), big.mark = ",")))

  # =========================================================================
  # Step 5: Compare with existing metadata to find NEW photos
  # =========================================================================

  message("  6. Identifying new photos...")

  # Check if old parquet exists
  if (dir.exists(parquet_path)) {
    angio_photos_old <- open_dataset(parquet_path) %>%
      select(photo_id, batch_j) %>%
      collect()

    old_photo_ids <- angio_photos_old$photo_id
    old_batch_max <- max(angio_photos_old$batch_j, na.rm = TRUE)

    message(sprintf("    Existing dataset: %s photos in %d batches",
                    format(length(old_photo_ids), big.mark = ","),
                    old_batch_max))

    # Filter to NEW photos only
    angio_photos_new <- angio_photos_new_raw %>%
      filter(!photo_id %in% old_photo_ids)

    message(sprintf("    New photos: %s", format(nrow(angio_photos_new), big.mark = ",")))

  } else {
    message("    No existing dataset found - this is the first run")
    angio_photos_new <- angio_photos_new_raw
    old_batch_max <- 0
    angio_photos_old <- NULL
  }

  # =========================================================================
  # Step 6: Assign batch numbers to new photos
  # =========================================================================

  if (nrow(angio_photos_new) > 0) {
    message(sprintf("  7. Assigning batch numbers (%d images per batch)...", batch_size))

    n_batches_new <- ceiling(nrow(angio_photos_new) / batch_size)
    batch_vector <- rep(1:n_batches_new, each = batch_size, length.out = nrow(angio_photos_new))
    batch_vector <- batch_vector + old_batch_max  # Continue from old max

    angio_photos_new <- angio_photos_new %>%
      mutate(batch_j = batch_vector)

    message(sprintf("    Assigned batches %d to %d (%d new batches)",
                    old_batch_max + 1,
                    max(batch_vector),
                    n_batches_new))
  }

  # =========================================================================
  # Step 7: Combine old and new, write to parquet
  # =========================================================================

  message("  8. Writing updated parquet dataset...")

  if (!is.null(angio_photos_old) && nrow(angio_photos_new) > 0) {
    # Combine old (with batch_j) and new
    angio_photos_combined <- bind_rows(
      angio_photos_old %>% left_join(
        angio_photos_new_raw %>% select(-batch_j),
        by = "photo_id"
      ),
      angio_photos_new
    )
  } else if (nrow(angio_photos_new) > 0) {
    # First run - only new photos
    angio_photos_combined <- angio_photos_new
  } else {
    # No new photos - just return existing path
    message("    No new photos to add - dataset unchanged")
    message("Metadata update complete!")
    return(parquet_path)
  }

  # Write to parquet (partitioned by batch_j for efficient reading)
  write_dataset(
    angio_photos_combined,
    path = parquet_path,
    format = "parquet",
    partitioning = NULL  # Don't partition - keeps it simple
  )

  message(sprintf("    Wrote %s total photos to %s",
                  format(nrow(angio_photos_combined), big.mark = ","),
                  parquet_path))

  # =========================================================================
  # Step 8: Update metadata date file
  # =========================================================================

  metadata_date_file <- file.path(metadata_dir, "metadata_date.txt")
  cat(
    sprintf("Metadata updated on %s: %s total photos in %d batches\n",
            Sys.Date(),
            format(nrow(angio_photos_combined), big.mark = ","),
            max(angio_photos_combined$batch_j)),
    file = metadata_date_file,
    append = TRUE
  )

  message("Metadata update complete!")
  return(parquet_path)
}
