#' Update Phenology Annotations from iNaturalist DwC Archive
#'
#' Downloads the latest DwC archive from iNaturalist/Phenobase with annotated observations,
#' extracts and filters annotations, and writes to parquet dataset.
#'
#' This function wraps the logic from:
#' /blue/guralnick/share/phenobase_inat_data/Rcode/03_get_latest_iNat_annotation.R
#'
#' @param annotation_dir Character. Directory for annotation data (default: "data/phenobase_inat_data/metadata/phenobase_dwca_annotation")
#' @param force_download Logical. Re-download DwC archive even if recent file exists (default: FALSE)
#'
#' @return Character. Path to updated annotation parquet dataset
#'
#' @details
#' This function:
#' 1. Downloads DwC archive from iNaturalist (phenobase-observations-dwca.zip)
#' 2. Extracts observations.csv (contains reproductiveCondition and dynamicProperties)
#' 3. Filters for observations with either reproductive or leaf annotations
#' 4. Writes to parquet: metadata/phenobase_dwca_annotation/inat_annotation/
#'
#' The observations.csv contains these key fields:
#' - reproductiveCondition: Pipe-separated reproductive states (flowering, fruiting, etc.)
#' - dynamicProperties: JSON with leaf states (green leaves, colored leaves, etc.)
#' - otherCatalogueNumbers: observation_uuid (for joining with photos)
#'
#' This parquet dataset is then joined with angio_photos to get photo_ids.
#'
#' @examples
#' \dontrun{
#' parquet_path <- update_phenology_annotations(
#'   annotation_dir = "data/phenobase_inat_data/metadata/phenobase_dwca_annotation"
#' )
#' }
#'
#' @export
update_phenology_annotations <- function(annotation_dir = "data/phenobase_inat_data/metadata/phenobase_dwca_annotation",
                                          force_download = FALSE) {

  library(tidyverse)
  library(arrow)
  library(data.table)
  library(xfun)

  message("Updating phenology annotations...")

  # Create annotation directory
  dir.create(annotation_dir, recursive = TRUE, showWarnings = FALSE)

  # Output parquet path
  parquet_path <- file.path(annotation_dir, "inat_annotation")

  # =========================================================================
  # Step 1: Download DwC archive (if needed)
  # =========================================================================

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
    message("  1. Downloading DwC archive from iNaturalist...")
    message("     (This may be large - 40+ GB - and take some time)")
    xfun::download_file(dwc_url, dwc_zip, quiet = FALSE)

    message("  2. Extracting DwC archive...")
    # Use system unzip (more reliable for large files > 4GB)
    system2("unzip", args = c("-o", dwc_zip, "-d", annotation_dir), stdout = TRUE)
  } else {
    message("  Skipping download, using existing DwC archive")
  }

  # =========================================================================
  # Step 2: Read observations.csv and filter for annotations
  # =========================================================================

  message("  3. Reading observations and filtering for annotations...")
  message("     (This requires significant memory - ~90 GB)")

  obs_file <- file.path(annotation_dir, "observations.csv")

  if (!file.exists(obs_file)) {
    stop("observations.csv not found - extraction may have failed")
  }

  # Read observations using data.table (faster and less memory than readr)
  annotated_inat <- data.table::fread(
    obs_file,
    select = c(
      "id", "occurrenceID", "basisOfRecord", "recordedBy", "recordedByID",
      "identifiedBy", "identifiedByID", "captive", "eventDate",
      "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
      "countryCode", "identificationID", "datasetName", "dateIdentified",
      "identificationRemarks", "taxonID", "scientificName", "taxonRank",
      "family", "genus", "license", "sex", "lifeStage", "reproductiveCondition",
      "vitality", "dynamicProperties", "otherCatalogueNumbers"
    )
  )

  # Filter for records with either reproductive or leaf annotations
  annotated_inat <- annotated_inat %>%
    filter(!(reproductiveCondition == "" & dynamicProperties == "")) %>%
    rename(observation_uuid = otherCatalogueNumbers)

  message(sprintf("    Found %s annotated observations", format(nrow(annotated_inat), big.mark = ",")))

  # Summary stats
  n_repro <- sum(annotated_inat$reproductiveCondition != "")
  n_leaf <- sum(annotated_inat$dynamicProperties != "")

  message(sprintf("    - Reproductive annotations: %s", format(n_repro, big.mark = ",")))
  message(sprintf("    - Leaf annotations: %s", format(n_leaf, big.mark = ",")))

  # =========================================================================
  # Step 3: Write to parquet
  # =========================================================================

  message("  4. Writing annotation parquet dataset...")

  arrow::write_dataset(
    annotated_inat,
    path = parquet_path,
    format = "parquet"
  )

  message(sprintf("    Wrote %s annotated observations to %s",
                  format(nrow(annotated_inat), big.mark = ","),
                  parquet_path))

  # Free memory
  rm(annotated_inat)
  gc()

  # =========================================================================
  # Step 4: Update metadata date file
  # =========================================================================

  metadata_date_file <- file.path(dirname(annotation_dir), "metadata_date.txt")
  cat(
    sprintf("Annotations updated on %s from DwC archive dated %s\n",
            Sys.Date(),
            format(file.mtime(dwc_zip), "%Y-%m-%d")),
    file = metadata_date_file,
    append = TRUE
  )

  message("Annotation update complete!")
  return(parquet_path)
}
