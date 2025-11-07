#' Granular Functions for Phenology Annotations Update Pipeline
#'
#' These functions break down the DwC annotation update process into individual
#' computational steps, each cacheable as a separate target.
#'
#' Author: Russell Dinnage & Claude Code
#' Date: 2025-01-06

#' Download iNaturalist DwC Archive
#'
#' Downloads the Phenobase observations DwC archive from iNaturalist.
#' Checks if existing file is recent (< 7 days) and skips if not forced.
#'
#' @param annotation_dir Directory to store annotations
#' @param force_download Re-download even if recent file exists
#' @return Path to downloaded zip file
#' @export
download_dwc_archive <- function(annotation_dir, force_download = FALSE) {
  library(xfun)

  message("Downloading DwC archive...")

  # Create annotation directory
  dir.create(annotation_dir, recursive = TRUE, showWarnings = FALSE)

  dwc_url <- "https://www.inaturalist.org/observations/phenobase-observations-dwca.zip"
  dwc_zip <- file.path(annotation_dir, "phenobase_dwca.zip")

  # Check if recent DwC exists (within 7 days - iNaturalist updates weekly)
  download_needed <- force_download
  if (!download_needed && file.exists(dwc_zip)) {
    file_age_days <- as.numeric(difftime(Sys.time(), file.mtime(dwc_zip), units = "days"))
    if (file_age_days > 7) {
      message(sprintf("  Existing DwC archive is %d days old, re-downloading...", round(file_age_days)))
      download_needed <- TRUE
    } else {
      message(sprintf("  Using existing DwC archive (%.1f days old)", file_age_days))
    }
  } else {
    download_needed <- TRUE
  }

  if (download_needed) {
    message("  Downloading DwC archive from iNaturalist (10+ GB, this may take 2+ hours)...")
    xfun::download_file(dwc_url, dwc_zip, quiet = FALSE)
  }

  message(sprintf("DwC archive ready: %s", dwc_zip))
  return(dwc_zip)
}


#' Extract iNaturalist DwC Archive
#'
#' Extracts the DwC zip file to get observations.csv.
#'
#' @param zipfile Path to DwC zip file
#' @param annotation_dir Directory to extract to
#' @return Path to extracted observations.csv
#' @export
extract_dwc_archive <- function(zipfile, annotation_dir) {
  message("Extracting DwC archive...")

  obs_file <- file.path(annotation_dir, "observations.csv")

  if (file.exists(obs_file)) {
    message("  Using existing extracted observations.csv")
  } else {
    message("  Extracting DwC archive (this may take 30+ minutes)...")
    system2("unzip", args = c("-o", zipfile, "-d", annotation_dir), stdout = TRUE)
  }

  if (!file.exists(obs_file)) {
    stop("observations.csv not found - extraction may have failed")
  }

  message(sprintf("Extracted observations.csv: %s", obs_file))
  return(obs_file)
}


#' Parse Phenology Annotations from DwC
#'
#' Reads observations.csv and filters for rows with phenology annotations
#' (reproductiveCondition or dynamicProperties). Uses awk pre-filtering for
#' memory efficiency.
#'
#' @param dwc_extracted Path to observations.csv
#' @param annotation_dir Directory for temp files
#' @return Tibble of phenology annotations
#' @export
parse_phenology_dwc <- function(dwc_extracted, annotation_dir) {
  library(readr)
  library(dplyr)

  message("Parsing phenology annotations from DwC...")

  obs_file <- dwc_extracted

  # Use data.table::fread() for fast CSV reading (5-10x faster than read_csv)
  # The 'select' parameter only reads needed columns from disk
  message("  Reading and filtering annotations with data.table::fread()...")

  annots_clean <- data.table::fread(
    obs_file,
    select = c(
      "eventDate",
      "taxonID",
      "scientificName",
      "class",
      "order",
      "family",
      "genus",
      "reproductiveCondition",
      "dynamicProperties",
      "otherCatalogueNumbers"
    ),
    colClasses = "character"  # Read all as character for consistency
  ) %>%
    as_tibble() %>%
    # Filter for rows where reproductiveCondition OR dynamicProperties is not empty
    filter(!is.na(reproductiveCondition) | !is.na(dynamicProperties)) %>%
    filter(reproductiveCondition != "" | dynamicProperties != "") %>%
    rename(
      observation_uuid = otherCatalogueNumbers,
      reproductive_condition = reproductiveCondition,
      dynamic_properties = dynamicProperties,
      taxon_id = taxonID,
      scientific_name = scientificName,
      observed_on = eventDate
      # Keep class, order, family, genus as-is (already lowercase in DwC)
    )

  message(sprintf("  Found %s observations with phenology annotations",
                  format(nrow(annots_clean), big.mark = ",")))

  return(annots_clean)
}


#' Write Annotations Parquet Dataset
#'
#' Writes phenology annotations to parquet format.
#'
#' @param phenology_annots_raw Tibble of annotations
#' @param parquet_path Relative path to parquet dataset
#' @param annotation_dir Annotation directory
#' @return Path to parquet dataset
#' @export
write_annotations_parquet <- function(phenology_annots_raw, parquet_path, annotation_dir) {
  library(arrow)

  message("Writing annotations parquet...")

  parquet_full_path <- file.path(annotation_dir, basename(parquet_path))

  # Write to parquet
  write_dataset(phenology_annots_raw, path = parquet_full_path, format = "parquet")

  # Count annotations
  n_annots <- open_dataset(parquet_full_path) %>%
    count() %>%
    collect() %>%
    pull(n)

  message(sprintf("  Wrote %s annotations to %s",
                  format(n_annots, big.mark = ","),
                  parquet_full_path))

  message("Annotations update complete!")
  return(parquet_full_path)
}
