#' Granular Functions for iNaturalist Metadata Update Pipeline
#'
#' These functions break down the metadata update process into individual
#' computational steps, each cacheable as a separate target.
#'
#' This design follows targets best practices:
#' - Each function is a single computational step
#' - Intermediate results are cached
#' - If a step fails, we don't re-run earlier successful steps
#' - Trade-off: More disk space for intermediate results
#'
#' Author: Russell Dinnage & Claude Code
#' Date: 2025-01-06

#' Download iNaturalist Metadata Tar.gz
#'
#' Downloads the latest iNaturalist open data metadata archive.
#' Checks if existing file is recent (< 30 days) and skips if not forced.
#'
#' @param metadata_dir Directory to store metadata
#' @param force_download Re-download even if recent file exists
#' @return Path to downloaded tar.gz file
#' @export
download_inat_metadata <- function(metadata_dir, force_download = FALSE) {
  library(xfun)

  message("Downloading iNaturalist metadata...")

  # Create metadata directory if needed
  dir.create(metadata_dir, recursive = TRUE, showWarnings = FALSE)

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
    message("  Downloading latest iNaturalist metadata (26+ GB, this may take 30+ minutes)...")
    xfun::download_file(metadata_url, meta_file)
  }

  message(sprintf("Metadata tar.gz ready: %s", meta_file))
  return(meta_file)
}


#' Extract iNaturalist Metadata Archive
#'
#' Extracts the tar.gz metadata archive. Uses --strip-components=1 to remove
#' dated top-level directory for consistent paths.
#'
#' @param tarfile Path to tar.gz file
#' @param metadata_dir Directory to extract to
#' @return Vector of extracted file paths
#' @export
extract_inat_metadata <- function(tarfile, metadata_dir) {
  message("Extracting iNaturalist metadata...")

  # Check if extraction is needed
  taxa_file <- file.path(metadata_dir, "taxa.csv")
  obs_file <- file.path(metadata_dir, "observations.csv")
  photos_file <- file.path(metadata_dir, "photos.csv")

  if (all(file.exists(c(taxa_file, obs_file, photos_file)))) {
    message("  Using existing extracted files")
  } else {
    message("  Extracting metadata archive (70+ GB, this may take 10+ minutes)...")
    # Use --strip-components=1 to remove the dated top-level directory
    system(paste0("tar -xzf ", tarfile, " -C ", metadata_dir, " --strip-components=1"))
  }

  extracted_files <- c(taxa_file, obs_file, photos_file)
  message(sprintf("Extracted %d files", length(extracted_files)))

  return(extracted_files)
}


#' Filter Taxa to Angiosperms
#'
#' Uses awk to efficiently filter taxa.csv to angiosperms only.
#' Returns a character vector of taxon IDs.
#'
#' @param metadata_extracted Vector of metadata file paths
#' @return Character vector of angiosperm taxon IDs
#' @export
filter_angio_taxa <- function(metadata_extracted) {
  message("Filtering taxa to angiosperms...")

  taxa_file <- metadata_extracted[grep("taxa.csv", metadata_extracted)]
  taxa_ids_file <- tempfile(fileext = ".txt")

  # Use awk to extract angiosperm taxon_ids
  # Column 1 = taxon_id, Column 2 = ancestry, Column 4 = rank
  awk_cmd <- sprintf(
    "awk 'BEGIN {FS=\"\\t\"} NR > 1 && $2 ~ /47125/ && ($4 == \"species\" || $4 == \"subspecies\" || $4 == \"variety\") {print $1}' %s > %s",
    taxa_file,
    taxa_ids_file
  )
  system(awk_cmd)

  # Read taxa IDs
  taxa_ids <- readLines(taxa_ids_file)
  message(sprintf("  Found %s angiosperm taxa", format(length(taxa_ids), big.mark = ",")))

  return(taxa_ids)
}


#' Filter Observations to Research-Grade Angiosperms
#'
#' Uses awk to filter observations.csv to research-grade observations
#' of angiosperm taxa. Returns a character vector of observation UUIDs.
#'
#' @param metadata_extracted Vector of metadata file paths
#' @param angio_taxa_ids Character vector of taxon IDs
#' @return Character vector of observation UUIDs
#' @export
filter_angio_observations <- function(metadata_extracted, angio_taxa_ids) {
  message("Filtering observations to research-grade angiosperms...")

  obs_file <- metadata_extracted[grep("observations.csv", metadata_extracted)]

  # Write taxa IDs to temp file for awk
  taxa_ids_file <- tempfile(fileext = ".txt")
  writeLines(angio_taxa_ids, taxa_ids_file)

  obs_uuids_file <- tempfile(fileext = ".txt")

  # Use awk to filter observations
  # Column 1 = observation_uuid, Column 6 = taxon_id, Column 7 = quality_grade
  awk_cmd <- sprintf(
    "awk 'BEGIN {FS=\"\\t\"; while(getline < \"%s\") taxa[$0]=1} NR > 1 && $6 in taxa && $7 == \"research\" {print $1}' %s > %s",
    taxa_ids_file,
    obs_file,
    obs_uuids_file
  )
  system(awk_cmd)

  # Read observation UUIDs
  obs_uuids <- readLines(obs_uuids_file)
  message(sprintf("  Found %s research-grade observations", format(length(obs_uuids), big.mark = ",")))

  return(obs_uuids)
}


#' Filter Photos from Angiosperm Observations
#'
#' Uses awk to filter photos.csv to photos from angiosperm observations.
#' Returns a tibble of photo metadata.
#'
#' @param metadata_extracted Vector of metadata file paths
#' @param angio_obs_uuids Character vector of observation UUIDs
#' @return Tibble of photo metadata
#' @export
filter_angio_photos <- function(metadata_extracted, angio_obs_uuids) {
  library(readr)

  message("Filtering photos from angiosperm observations...")

  photos_file <- metadata_extracted[grep("photos.csv", metadata_extracted)]

  # Write observation UUIDs to temp file for awk
  obs_uuids_file <- tempfile(fileext = ".txt")
  writeLines(angio_obs_uuids, obs_uuids_file)

  photos_filtered_file <- tempfile(fileext = ".csv")

  # Use awk to filter photos
  # Column 3 = observation_uuid
  awk_cmd <- sprintf(
    "awk 'BEGIN {FS=\"\\t\"; while(getline < \"%s\") obs[$0]=1} NR == 1 || $3 in obs {print}' %s > %s",
    obs_uuids_file,
    photos_file,
    photos_filtered_file
  )
  system(awk_cmd)

  # Read filtered photos with proper column types to match existing parquet schema
  angio_photos <- read_tsv(
    photos_filtered_file,
    col_types = cols(
      photo_uuid = col_character(),
      photo_id = col_character(),
      observation_uuid = col_character(),
      observer_id = col_integer(),
      extension = col_character(),
      license = col_character(),
      width = col_integer(),
      height = col_integer(),
      position = col_integer()
    ),
    show_col_types = FALSE
  )

  message(sprintf("  Found %s photos from angiosperm observations", format(nrow(angio_photos), big.mark = ",")))

  return(angio_photos)
}


#' Enrich Photos with Observation Data
#'
#' Joins photos with observations to add latitude, longitude, taxon_id, etc.
#' Uses awk for memory-efficient extraction and join.
#'
#' @param angio_photos Tibble of angiosperm photos
#' @param metadata_extracted Vector of metadata file paths
#' @return Tibble of photos enriched with observation data
#' @export
enrich_photos_with_observations <- function(angio_photos, metadata_extracted) {
  library(readr)
  library(dplyr)
  library(lubridate)

  message("Enriching photos with observation data...")
  message(sprintf("  Processing %s photos", format(nrow(angio_photos), big.mark = ",")))

  obs_file <- metadata_extracted[grep("observations.csv", metadata_extracted)]

  # Extract needed columns from observations using awk
  # Columns: 1=observation_uuid, 3=latitude, 4=longitude, 5=positional_accuracy, 6=taxon_id, 8=observed_on, 9=anomaly_score
  obs_filtered_file <- tempfile(fileext = ".csv")

  # Write header
  header_cmd <- sprintf(
    "echo 'observation_uuid,latitude,longitude,positional_accuracy,taxon_id,observed_on,anomaly_score' > %s",
    obs_filtered_file
  )
  system(header_cmd)

  # Extract columns with awk
  awk_cmd <- sprintf(
    "awk 'BEGIN {FS=\"\\t\"; OFS=\",\"} NR > 1 {print $1,$3,$4,$5,$6,$8,$9}' %s >> %s",
    obs_file,
    obs_filtered_file
  )
  system(awk_cmd)

  message("  Reading observation data...")

  # Read observation data
  obs_data <- read_csv(
    obs_filtered_file,
    col_types = cols(
      observation_uuid = col_character(),
      latitude = col_double(),
      longitude = col_double(),
      positional_accuracy = col_integer(),
      taxon_id = col_integer(),
      observed_on = col_date(),
      anomaly_score = col_double()
    ),
    show_col_types = FALSE
  )

  message("  Joining photos with observation data...")

  # Join photos with observations
  photos_enriched <- angio_photos %>%
    left_join(obs_data, by = "observation_uuid") %>%
    mutate(
      yr = lubridate::year(observed_on),
      mth = lubridate::month(observed_on)
    )

  message(sprintf("  Enriched %s photos with observation data", format(nrow(photos_enriched), big.mark = ",")))

  return(photos_enriched)
}


#' Identify New Photos Compared to Existing Parquet
#'
#' Compares new photos against existing parquet dataset to find only NEW photos.
#' Uses memory-efficient hash set for comparison.
#'
#' @param angio_photos_raw Tibble of all angiosperm photos
#' @param parquet_path Path to existing parquet dataset
#' @param metadata_dir Metadata directory
#' @return List with old_batch_max and new_photos tibble
#' @export
identify_new_photos <- function(angio_photos_raw, parquet_path, metadata_dir) {
  library(arrow)
  library(dplyr)

  message("Identifying new photos...")

  parquet_full_path <- file.path(metadata_dir, basename(parquet_path))

  if (dir.exists(parquet_full_path)) {
    # Load existing photo IDs as hash set
    source("R/utils_memory_efficient_download.R")
    old_photo_ids <- create_photo_id_hashset(parquet_full_path)

    # Get max batch number
    old_batch_max <- open_dataset(parquet_full_path) %>%
      summarise(max_batch = max(batch_j, na.rm = TRUE)) %>%
      collect() %>%
      pull(max_batch)

    message(sprintf("  Existing dataset: %s photos in %d batches",
                    format(length(old_photo_ids), big.mark = ","),
                    old_batch_max))

    # Filter to NEW photos only
    new_photos <- angio_photos_raw %>%
      filter(!photo_id %in% old_photo_ids)

    message(sprintf("  New photos: %s", format(nrow(new_photos), big.mark = ",")))

  } else {
    message("  No existing dataset found - this is the first run")
    new_photos <- angio_photos_raw
    old_batch_max <- 0
  }

  return(list(
    old_batch_max = old_batch_max,
    new_photos = new_photos
  ))
}


#' Assign Batch Numbers to New Photos
#'
#' Assigns batch numbers to new photos, continuing from old_batch_max.
#'
#' @param angio_photos_new List with old_batch_max and new_photos
#' @param batch_size Number of images per batch
#' @param parquet_path Path to parquet dataset
#' @return Tibble of new photos with batch_j column
#' @export
assign_batch_numbers <- function(angio_photos_new, batch_size, parquet_path) {
  library(dplyr)

  new_photos <- angio_photos_new$new_photos
  old_batch_max <- angio_photos_new$old_batch_max

  if (nrow(new_photos) == 0) {
    message("No new photos to assign batches")
    return(new_photos %>% mutate(batch_j = integer()))
  }

  message(sprintf("Assigning batch numbers (%d images per batch)...", batch_size))

  n_batches_new <- ceiling(nrow(new_photos) / batch_size)
  batch_vector <- rep(1:n_batches_new, each = batch_size, length.out = nrow(new_photos))
  batch_vector <- batch_vector + old_batch_max  # Continue from old max

  new_photos_batched <- new_photos %>%
    mutate(batch_j = batch_vector)

  message(sprintf("  Assigned batches %d to %d (%d new batches)",
                  old_batch_max + 1,
                  max(batch_vector),
                  n_batches_new))

  return(new_photos_batched)
}


#' Write Updated Photos Parquet Dataset
#'
#' Combines old and new photos and writes to parquet using Arrow union.
#'
#' @param angio_photos_batched Tibble of new photos with batch_j
#' @param parquet_path Relative path to parquet dataset
#' @param metadata_dir Metadata directory
#' @return Path to updated parquet dataset
#' @export
write_photos_parquet <- function(angio_photos_batched, parquet_path, metadata_dir) {
  library(arrow)
  library(dplyr)

  message("Writing updated parquet dataset...")

  parquet_full_path <- file.path(metadata_dir, basename(parquet_path))

  if (nrow(angio_photos_batched) == 0) {
    message("  No new photos to add - dataset unchanged")
    return(parquet_full_path)
  }

  if (dir.exists(parquet_full_path)) {
    # Get old schema to ensure new data matches
    old_ds <- open_dataset(parquet_full_path)
    old_schema <- schema(old_ds)

    message("  Casting new data to match existing schema...")

    # Convert new photos to Arrow Table and cast to old schema
    new_photos_table <- arrow_table(angio_photos_batched)
    new_photos_table <- new_photos_table$cast(old_schema)

    # Write new photos to temporary parquet WITH explicit schema
    temp_new_path <- file.path(tempdir(), "temp_new_photos")
    dir.create(temp_new_path, recursive = TRUE, showWarnings = FALSE)

    write_dataset(new_photos_table, path = temp_new_path, format = "parquet", schema = old_schema)
    message(sprintf("  Wrote %s new photos to temporary parquet with matching schema",
                    format(nrow(angio_photos_batched), big.mark = ",")))

    # Create temporary union output path
    temp_union_path <- file.path(tempdir(), "temp_union_photos")
    dir.create(temp_union_path, recursive = TRUE, showWarnings = FALSE)

    # Union datasets using Arrow
    new_ds <- open_dataset(temp_new_path)
    union_ds <- dplyr::union_all(old_ds, new_ds)
    write_dataset(union_ds, path = temp_union_path, format = "parquet", schema = old_schema)

    # Replace old parquet with union
    unlink(parquet_full_path, recursive = TRUE)
    file.rename(temp_union_path, parquet_full_path)

    # Clean up
    unlink(temp_new_path, recursive = TRUE)

    # Count total photos
    n_total <- open_dataset(parquet_full_path) %>%
      count() %>%
      collect() %>%
      pull(n)

    message(sprintf("  Wrote %s total photos to %s",
                    format(n_total, big.mark = ","),
                    parquet_full_path))

  } else {
    # First run - only new photos
    write_dataset(angio_photos_batched, path = parquet_full_path, format = "parquet")

    message(sprintf("  Wrote %s photos to %s (first run)",
                    format(nrow(angio_photos_batched), big.mark = ","),
                    parquet_full_path))
  }

  # Update metadata date file
  metadata_date_file <- file.path(metadata_dir, "metadata_date.txt")
  stats <- open_dataset(parquet_full_path) %>%
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
  return(parquet_full_path)
}
