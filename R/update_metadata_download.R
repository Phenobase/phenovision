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
    # Use --strip-components=1 to remove the dated top-level directory
    # This ensures files are extracted directly to metadata_dir regardless of archive structure
    system(paste0("tar -xvzf ", meta_file, " -C ", metadata_dir, " --strip-components=1"))
  } else {
    message("  Skipping download, using existing metadata")
  }

  # =========================================================================
  # Step 2: Filter taxa to angiosperms (MEMORY OPTIMIZED - uses awk)
  # =========================================================================

  message("  3. Filtering taxa to angiosperms (using awk to avoid loading 5GB file)...")

  taxa_file <- file.path(metadata_dir, "taxa.csv")
  taxa_ids_file <- tempfile(fileext = ".txt")

  # Use awk to extract angiosperm taxon_ids directly (avoids reading 5GB file into R)
  # Column 1 = taxon_id, Column 5 = rank, Column 7 = ancestry
  # Filter: ancestry contains "47125" AND rank in (species, subspecies, variety)
  awk_taxa_cmd <- sprintf(
    "awk 'BEGIN {FS=\"\\t\"} NR > 1 && $7 ~ /47125/ && ($5 == \"species\" || $5 == \"subspecies\" || $5 == \"variety\") {print $1}' %s > %s",
    taxa_file,
    taxa_ids_file
  )
  system(awk_taxa_cmd)

  # Count taxa IDs (for reporting)
  n_taxa <- as.integer(system(sprintf("wc -l < %s", taxa_ids_file), intern = TRUE))
  message(sprintf("    Found %s angiosperm taxa (extracted with awk)", format(n_taxa, big.mark = ",")))

  # =========================================================================
  # Step 3: Filter observations using awk (efficient for huge files)
  # =========================================================================

  message("  4. Filtering observations to research-grade angiosperms...")

  obs_file <- file.path(metadata_dir, "observations.csv")
  obs_uuids_file <- tempfile(fileext = ".txt")

  # Use awk to filter observations AND extract observation_uuids directly
  # Column 2 = observation_uuid, Column 6 = taxon_id, Column 7 = quality_grade
  # This avoids loading 30GB of observations into R - we only need the UUIDs
  awk_obs_cmd <- sprintf(
    "awk 'BEGIN {FS=\"\\t\"; while(getline < \"%s\") taxa[$0]=1} NR > 1 && $6 in taxa && $7 == \"research\" {print $2}' %s > %s",
    taxa_ids_file,
    obs_file,
    obs_uuids_file
  )
  system(awk_obs_cmd)

  # Count observations (for reporting)
  n_obs <- as.integer(system(sprintf("wc -l < %s", obs_uuids_file), intern = TRUE))
  message(sprintf("    Found %s research-grade observations (extracted UUIDs with awk)", format(n_obs, big.mark = ",")))

  # =========================================================================
  # Step 4: Filter photos to those from our observations
  # =========================================================================

  message("  5. Filtering photos...")

  photos_file <- file.path(metadata_dir, "photos.csv")
  photos_filtered_file <- file.path(metadata_dir, "angio_photos.csv")

  # Use awk to filter photos (Column 2 = observation_uuid)
  # Observation UUIDs are already in obs_uuids_file from previous step
  awk_photos_cmd <- sprintf(
    "awk 'BEGIN {FS=\"\\t\"; while(getline < \"%s\") obs[$0]=1} NR == 1 || $2 in obs {print}' %s > %s",
    obs_uuids_file,
    photos_file,
    photos_filtered_file
  )
  system(awk_photos_cmd)

  # Read filtered photos
  angio_photos_new_raw <- read_tsv(
    photos_filtered_file,
    col_types = cols(.default = col_character()),
    show_col_types = FALSE
  )

  message(sprintf("    Found %s photos from angiosperm observations", format(nrow(angio_photos_new_raw), big.mark = ",")))

  # =========================================================================
  # Step 5: Compare with existing metadata to find NEW photos (MEMORY OPTIMIZED)
  # =========================================================================

  message("  6. Identifying new photos...")

  # Check if old parquet exists
  if (dir.exists(parquet_path)) {
    # MEMORY OPTIMIZATION: Use hash set instead of full dataframe
    # Old approach: collect(photo_id, batch_j) = ~15 GB
    # New approach: hash set of photo_ids only = ~500 MB
    source("R/utils_memory_efficient_download.R")
    old_photo_ids <- create_photo_id_hashset(parquet_path)

    # Get max batch number (need this for continuing batch numbering)
    old_batch_max <- open_dataset(parquet_path) %>%
      summarise(max_batch = max(batch_j, na.rm = TRUE)) %>%
      collect() %>%
      pull(max_batch)

    message(sprintf("    Existing dataset: %s photos in %d batches",
                    format(length(old_photo_ids), big.mark = ","),
                    old_batch_max))

    # Filter to NEW photos only (using hash set comparison)
    angio_photos_new <- angio_photos_new_raw %>%
      filter(!photo_id %in% old_photo_ids)

    message(sprintf("    New photos: %s", format(nrow(angio_photos_new), big.mark = ",")))

  } else {
    message("    No existing dataset found - this is the first run")
    angio_photos_new <- angio_photos_new_raw
    old_batch_max <- 0
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
  # Step 7: Combine old and new, write to parquet (MEMORY OPTIMIZED)
  # =========================================================================

  message("  8. Writing updated parquet dataset...")

  if (dir.exists(parquet_path) && nrow(angio_photos_new) > 0) {
    # MEMORY OPTIMIZATION: Use Arrow union instead of bind_rows()
    # Old approach: bind_rows(old, new) + left_join = ~200 GB peak
    # New approach: Arrow union = ~75 GB peak (only new data in memory)

    # Write new photos to temporary parquet
    temp_new_path <- file.path(tempdir(), "temp_new_photos")
    dir.create(temp_new_path, recursive = TRUE, showWarnings = FALSE)

    write_dataset(angio_photos_new, path = temp_new_path, format = "parquet")
    message(sprintf("    Wrote %s new photos to temporary parquet",
                    format(nrow(angio_photos_new), big.mark = ",")))

    # Create temporary union output path
    temp_union_path <- file.path(tempdir(), "temp_union_photos")
    dir.create(temp_union_path, recursive = TRUE, showWarnings = FALSE)

    # Union datasets using Arrow (no memory spike)
    union_ds <- open_dataset(c(parquet_path, temp_new_path))
    write_dataset(union_ds, path = temp_union_path, format = "parquet")

    # Replace old parquet with union
    unlink(parquet_path, recursive = TRUE)
    file.rename(temp_union_path, parquet_path)

    # Clean up
    unlink(temp_new_path, recursive = TRUE)

    # Count total photos
    n_total <- open_dataset(parquet_path) %>%
      count() %>%
      collect() %>%
      pull(n)

    message(sprintf("    Wrote %s total photos to %s (using Arrow union)",
                    format(n_total, big.mark = ","),
                    parquet_path))

  } else if (nrow(angio_photos_new) > 0) {
    # First run - only new photos
    write_dataset(angio_photos_new, path = parquet_path, format = "parquet")

    message(sprintf("    Wrote %s photos to %s (first run)",
                    format(nrow(angio_photos_new), big.mark = ","),
                    parquet_path))

  } else {
    # No new photos - just return existing path
    message("    No new photos to add - dataset unchanged")
    message("Metadata update complete!")
    return(parquet_path)
  }

  # =========================================================================
  # Step 8: Update metadata date file
  # =========================================================================

  metadata_date_file <- file.path(metadata_dir, "metadata_date.txt")

  # Get stats from parquet (avoid loading entire dataset)
  stats <- open_dataset(parquet_path) %>%
    summarise(
      n_photos = n(),
      max_batch = max(batch_j, na.rm = TRUE)
    ) %>%
    collect()

  cat(
    sprintf("Metadata updated on %s: %s total photos in %d batches\n",
            Sys.Date(),
            format(stats$n_photos, big.mark = ","),
            stats$max_batch),
    file = metadata_date_file,
    append = TRUE
  )

  message("Metadata update complete!")
  return(parquet_path)
}
