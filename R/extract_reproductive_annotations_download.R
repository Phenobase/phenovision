#' Extract Reproductive Annotations from iNaturalist Data
#'
#' Extracts and merges observation and photo data to create reproductive
#' phenology annotations (flowering and fruiting).
#'
#' @param dwc_zip Character. Path to DwC archive zip file
#' @param metadata_tgz Character. Path to iNaturalist metadata tar.gz file
#'
#' @return Data frame with columns:
#'   - observation_uuid: Observation ID
#'   - photo_id: Photo ID
#'   - extension: File extension
#'   - flowering: 1/0 indicator
#'   - fruiting: 1/0 indicator
#'   - reproductiveCondition: Original annotation string
#'   - scientificName, order, family, genus: Taxonomy
#'
#' @details
#' This function:
#' 1. Extracts observations.csv from the DwC zip (from iNaturalist/Phenobase)
#' 2. Filters for research-grade observations with reproductive annotations
#' 3. Extracts photos.csv from the metadata tar.gz (from iNaturalist open data)
#' 4. Merges observations with photos (single-photo observations only)
#' 5. Parses flowering/fruiting flags from reproductiveCondition field
#'
#' The reproductiveCondition field contains pipe-separated values like:
#' - "flowering"
#' - "fruiting"
#' - "flowering|fruiting"
#' - "flowering|no evidence of flowering" (excluded)
#'
#' @examples
#' \dontrun{
#' repro_df <- extract_reproductive_annotations(
#'   "data/inat/phenobase-observations-dwca.zip",
#'   "data/inat/inaturalist-open-data-latest.tar.gz"
#' )
#' }
#'
#' @export
extract_reproductive_annotations <- function(dwc_zip, metadata_tgz) {

  library(dplyr)
  library(readr)

  message("Extracting reproductive annotations...")

  # =========================================================================
  # Step 1: Extract observations from DwC zip
  # =========================================================================

  message("  1. Extracting observations from DwC archive...")

  # List files in zip
  zip_files <- unzip(dwc_zip, list = TRUE)$Name

  # Find observations.csv
  obs_file <- grep("observations\\.csv$", zip_files, value = TRUE)
  if (length(obs_file) == 0) {
    stop("observations.csv not found in DwC archive")
  }

  # Extract to temp file
  temp_obs <- tempfile(fileext = ".csv")
  on.exit(unlink(temp_obs), add = TRUE)
  unzip(dwc_zip, files = obs_file, exdir = dirname(temp_obs))
  file.rename(file.path(dirname(temp_obs), obs_file), temp_obs)

  # Read observations
  observations <- read_csv(temp_obs, show_col_types = FALSE)

  # Filter for research-grade with reproductive annotations
  observations <- observations %>%
    filter(grepl("research-grade", datasetName, fixed = TRUE)) %>%
    filter(!is.na(reproductiveCondition)) %>%
    filter(reproductiveCondition != "") %>%
    # Exclude ambiguous annotations
    filter(reproductiveCondition != "flowering|no evidence of flowering") %>%
    select(
      observation_uuid = otherCatalogueNumbers,
      reproductiveCondition,
      scientificName,
      order,
      family,
      genus = genus  # May need to extract from scientificName if missing
    )

  message(sprintf("    Found %d research-grade observations with reproductive annotations", nrow(observations)))

  # =========================================================================
  # Step 2: Extract photos from metadata tar.gz
  # =========================================================================

  message("  2. Extracting photos from metadata archive...")

  # List files in tar.gz
  tar_files <- untar(metadata_tgz, list = TRUE)

  # Find photos.csv (or photos.csv.gz)
  photo_file <- grep("photos\\.csv", tar_files, value = TRUE)
  if (length(photo_file) == 0) {
    stop("photos.csv not found in metadata archive")
  }

  # Extract to temp file
  temp_photos_dir <- tempdir()
  untar(metadata_tgz, files = photo_file, exdir = temp_photos_dir)
  temp_photos <- file.path(temp_photos_dir, photo_file)
  on.exit(unlink(temp_photos), add = TRUE)

  # Read photos (tab-separated)
  photos <- read_tsv(
    temp_photos,
    col_select = c(photo_id, observation_uuid, extension),
    show_col_types = FALSE
  )

  message(sprintf("    Loaded %d photos", nrow(photos)))

  # Filter to only our observations
  photos <- photos %>%
    filter(observation_uuid %in% observations$observation_uuid)

  message(sprintf("    Matched %d photos to annotated observations", nrow(photos)))

  # =========================================================================
  # Step 3: Filter to single-photo observations
  # =========================================================================

  message("  3. Filtering to single-photo observations...")

  # Count photos per observation
  photo_counts <- photos %>%
    group_by(observation_uuid) %>%
    summarise(n_photos = n(), .groups = "drop")

  # Keep only observations with exactly 1 photo
  single_photo_obs <- photo_counts %>%
    filter(n_photos == 1) %>%
    pull(observation_uuid)

  photos <- photos %>%
    filter(observation_uuid %in% single_photo_obs)

  message(sprintf("    Kept %d single-photo observations", length(single_photo_obs)))

  # =========================================================================
  # Step 4: Merge observations with photos
  # =========================================================================

  message("  4. Merging observations with photos...")

  merged <- photos %>%
    inner_join(observations, by = "observation_uuid")

  message(sprintf("    Merged dataset: %d records", nrow(merged)))

  # =========================================================================
  # Step 5: Parse flowering/fruiting flags
  # =========================================================================

  message("  5. Parsing flowering/fruiting flags...")

  merged <- merged %>%
    mutate(
      # Split reproductiveCondition by pipe, clean whitespace
      rc_flags = strsplit(reproductiveCondition, "\\|"),
      rc_flags = lapply(rc_flags, function(x) trimws(x)),

      # Check for flowering/fruiting
      flowering = as.integer(sapply(rc_flags, function(x) "flowering" %in% x)),
      fruiting = as.integer(sapply(rc_flags, function(x) "fruiting" %in% x))
    ) %>%
    select(-rc_flags)

  # Summary
  n_flowering <- sum(merged$flowering == 1)
  n_fruiting <- sum(merged$fruiting == 1)
  n_both <- sum(merged$flowering == 1 & merged$fruiting == 1)

  message(sprintf("    Flowering: %d (%.1f%%)", n_flowering, 100 * n_flowering / nrow(merged)))
  message(sprintf("    Fruiting:  %d (%.1f%%)", n_fruiting, 100 * n_fruiting / nrow(merged)))
  message(sprintf("    Both:      %d (%.1f%%)", n_both, 100 * n_both / nrow(merged)))

  message("Extraction complete!")

  return(merged)
}
