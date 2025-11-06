#' Extract Reproductive Annotations from Parquet Datasets
#'
#' Joins annotation parquet with photo parquet to extract reproductive
#' phenology annotations (flowering and fruiting).
#'
#' @param annotation_parquet Character. Path to annotation parquet dataset
#' @param photos_parquet Character. Path to photos parquet dataset
#' @param images_root Character. Root path for image files
#'
#' @return Data frame with columns:
#'   - observation_uuid, photo_id, extension, batch_j
#'   - flowering, fruiting (0/1 indicators)
#'   - reproductiveCondition (original annotation string)
#'   - scientificName, family, genus, order
#'   - file_name (full path to image)
#'
#' @details
#' This function:
#' 1. Reads annotation parquet and filters for reproductive annotations
#' 2. Joins with photos parquet to get photo_ids and batch assignments
#' 3. Filters to single-photo observations only
#' 4. Parses flowering/fruiting flags from reproductiveCondition
#' 5. Constructs file paths using batch_j
#'
#' @examples
#' \dontrun{
#' repro_df <- extract_reproductive_from_parquet(
#'   annotation_parquet = "data/phenobase_inat_data/metadata/phenobase_dwca_annotation/inat_annotation",
#'   photos_parquet = "data/phenobase_inat_data/metadata/angio_photos"
#' )
#' }
#'
#' @export
extract_reproductive_from_parquet <- function(annotation_parquet,
                                               photos_parquet,
                                               images_root = "data/phenobase_inat_data/images/medium") {

  library(tidyverse)
  library(arrow)

  message("Extracting reproductive annotations from parquet...")

  # =========================================================================
  # Step 1: Read annotations and filter for reproductive
  # =========================================================================

  message("  1. Loading reproductive annotations...")

  annotations <- open_dataset(annotation_parquet) %>%
    filter(reproductiveCondition != "") %>%
    select(
      observation_uuid,
      reproductiveCondition,
      scientificName,
      taxonRank,
      family,
      genus,
      datasetName
    ) %>%
    collect()

  message(sprintf("    Found %s observations with reproductive annotations",
                  format(nrow(annotations), big.mark = ",")))

  # Filter for research-grade only
  annotations <- annotations %>%
    filter(grepl("research-grade", datasetName, fixed = TRUE)) %>%
    select(-datasetName)

  message(sprintf("    Filtered to %s research-grade observations",
                  format(nrow(annotations), big.mark = ",")))

  # =========================================================================
  # Step 2: Join with photos parquet
  # =========================================================================

  message("  2. Joining with photo metadata...")

  photos <- open_dataset(photos_parquet) %>%
    select(observation_uuid, photo_id, extension, batch_j) %>%
    filter(observation_uuid %in% !!annotations$observation_uuid) %>%
    collect()

  message(sprintf("    Matched %s photos", format(nrow(photos), big.mark = ",")))

  # =========================================================================
  # Step 3: Filter to single-photo observations
  # =========================================================================

  message("  3. Filtering to single-photo observations...")

  single_photo_obs <- photos %>%
    group_by(observation_uuid) %>%
    filter(n() == 1) %>%
    ungroup()

  message(sprintf("    Kept %s single-photo observations",
                  format(nrow(single_photo_obs), big.mark = ",")))

  # =========================================================================
  # Step 4: Merge and parse flags
  # =========================================================================

  message("  4. Parsing flowering/fruiting flags...")

  merged <- single_photo_obs %>%
    inner_join(annotations, by = "observation_uuid") %>%
    mutate(
      # Parse reproductiveCondition (pipe-separated)
      flowering = as.integer(grepl("flowering", reproductiveCondition, fixed = TRUE)),
      fruiting = as.integer(grepl("fruits or seeds", reproductiveCondition, fixed = TRUE))
    )

  # Exclude ambiguous annotations
  merged <- merged %>%
    filter(reproductiveCondition != "flowering|no evidence of flowering")

  # Summary
  n_flowering <- sum(merged$flowering == 1)
  n_fruiting <- sum(merged$fruiting == 1)
  n_both <- sum(merged$flowering == 1 & merged$fruiting == 1)

  message(sprintf("    Flowering: %s (%.1f%%)", format(n_flowering, big.mark = ","),
                  100 * n_flowering / nrow(merged)))
  message(sprintf("    Fruiting:  %s (%.1f%%)", format(n_fruiting, big.mark = ","),
                  100 * n_fruiting / nrow(merged)))
  message(sprintf("    Both:      %s (%.1f%%)", format(n_both, big.mark = ","),
                  100 * n_both / nrow(merged)))

  # =========================================================================
  # Step 5: Add file paths
  # =========================================================================

  merged <- merged %>%
    mutate(
      file_name = file.path(
        images_root,
        paste0("batch_", batch_j),
        paste0(photo_id, ".", extension)
      )
    ) %>%
    # Extract order from taxonomy (may need to join with taxa table for full taxonomy)
    mutate(order = NA_character_)  # Will need taxa table for this

  message("Extraction complete!")
  return(merged)
}


#' Extract Leaf Annotations from Parquet Datasets
#'
#' Joins annotation parquet with photo parquet to extract leaf
#' phenology annotations.
#'
#' @param annotation_parquet Character. Path to annotation parquet dataset
#' @param photos_parquet Character. Path to photos parquet dataset
#' @param rob_annot_csv Character. Path to Rob's breaking leaf buds annotations
#' @param rob_annot2_csv Character. Path to Rob's additional annotations
#' @param images_root Character. Root path for image files
#' @param target_genera Character vector. Genera to include (default: NULL = all)
#'
#' @return Data frame with columns:
#'   - observation_uuid, photo_id, extension, batch_j
#'   - leaves_green, leaves_colored, leaves_breaking_buds (0/1 indicators)
#'   - scientificName, family, genus
#'   - file_name (full path to image)
#'   - count (number of photos per observation)
#'   - strata (stratification column for splitting)
#'
#' @details
#' This function:
#' 1. Reads annotation parquet and filters for leaf annotations (dynamicProperties)
#' 2. Parses JSON from dynamicProperties field
#' 3. Joins with photos parquet to get photo_ids and batch assignments
#' 4. Filters to target genera (if specified)
#' 5. Merges with Rob's manual breaking_buds annotations
#' 6. Counts photos per observation
#' 7. Creates stratification column for splitting
#'
#' @examples
#' \dontrun{
#' leaf_df <- extract_leaf_from_parquet(
#'   annotation_parquet = "data/phenobase_inat_data/metadata/phenobase_dwca_annotation/inat_annotation",
#'   photos_parquet = "data/phenobase_inat_data/metadata/angio_photos",
#'   rob_annot_csv = "data/leaves/phenobase_dwca_annotation/rob_leaf_breaking_buds_annotation.csv",
#'   rob_annot2_csv = "data/leaves/rob_new_annotations_bb.csv"
#' )
#' }
#'
#' @export
extract_leaf_from_parquet <- function(annotation_parquet,
                                       photos_parquet,
                                       rob_annot_csv,
                                       rob_annot2_csv,
                                       images_root = "data/phenobase_inat_data/images/medium",
                                       target_genera = NULL) {

  library(tidyverse)
  library(arrow)
  library(jsonlite)
  library(data.table)

  message("Extracting leaf annotations from parquet...")

  # =========================================================================
  # Step 1: Read annotations and filter for leaf annotations
  # =========================================================================

  message("  1. Loading leaf annotations...")

  annotations <- open_dataset(annotation_parquet) %>%
    filter(dynamicProperties != "") %>%
    select(
      observation_uuid,
      dynamicProperties,
      scientificName,
      taxonRank,
      family,
      genus,
      datasetName
    ) %>%
    collect()

  message(sprintf("    Found %s observations with leaf annotations",
                  format(nrow(annotations), big.mark = ",")))

  # =========================================================================
  # Step 2: Parse JSON from dynamicProperties
  # =========================================================================

  message("  2. Parsing JSON from dynamicProperties...")

  # Fix double double quotes issue
  annotations <- annotations %>%
    mutate(dynamicProperties = str_replace_all(dynamicProperties, fixed('""'), '"'))

  # Parse JSON
  leaf_json <- purrr::map(
    annotations$dynamicProperties,
    purrr::possibly(fromJSON, otherwise = NULL),
    .progress = TRUE
  )

  # Extract leaf state flags
  annotations <- annotations %>%
    mutate(
      leaves = purrr::map(leaf_json, "leaves"),
      leaves_green = as.numeric(purrr::map_lgl(
        leaves,
        ~ "green leaves" %chin% .x,
        .progress = TRUE
      )),
      leaves_colored = as.numeric(purrr::map_lgl(
        leaves,
        ~ "colored leaves" %chin% .x,
        .progress = TRUE
      )),
      leaves_no_live = as.numeric(purrr::map_lgl(
        leaves,
        ~ "no live leaves" %chin% .x,
        .progress = TRUE
      )),
      leaves_breaking_buds = as.numeric(purrr::map_lgl(
        leaves,
        ~ "breaking leaf buds" %chin% .x,
        .progress = TRUE
      ))
    ) %>%
    select(-leaves, -dynamicProperties)

  # =========================================================================
  # Step 3: Filter by genera (if specified)
  # =========================================================================

  if (!is.null(target_genera)) {
    message(sprintf("  3. Filtering to %d target genera...", length(target_genera)))

    # Exclude breaking buds from iNat annotations (use Rob's instead)
    annotations <- annotations %>%
      filter(!leaves_breaking_buds == 1) %>%
      filter(genus %chin% target_genera)

    message(sprintf("    Filtered to %s observations",
                    format(nrow(annotations), big.mark = ",")))
  } else {
    # Still exclude breaking buds from iNat
    annotations <- annotations %>%
      filter(!leaves_breaking_buds == 1)
  }

  # =========================================================================
  # Step 4: Join with photos parquet
  # =========================================================================

  message("  4. Joining with photo metadata...")

  photos <- open_dataset(photos_parquet) %>%
    select(observation_uuid, photo_id, extension, batch_j) %>%
    filter(observation_uuid %in% !!annotations$observation_uuid) %>%
    collect()

  merged <- annotations %>%
    left_join(photos, by = "observation_uuid") %>%
    filter(!is.na(batch_j))

  message(sprintf("    Matched %s photos", format(nrow(merged), big.mark = ",")))

  # =========================================================================
  # Step 5: Add file paths
  # =========================================================================

  merged <- merged %>%
    mutate(
      file_name = file.path(
        images_root,
        paste0("batch_", batch_j),
        paste0(photo_id, ".", extension)
      )
    )

  # =========================================================================
  # Step 6: Process Rob's manual annotations
  # =========================================================================

  message("  5. Processing Rob's manual annotations...")

  # Get family info for Rob's annotations
  fams <- merged %>%
    select(genus, family) %>%
    distinct()

  # First set of Rob's annotations
  rob_annot <- read_csv(rob_annot_csv, show_col_types = FALSE) %>%
    mutate(genus = word(`taxon.name`))

  rob_photos <- list.files(
    "data/leaves/phenobase_dwca_annotation/images_breaking_leaf_buds",
    full.names = FALSE
  )

  rob_annot_final <- rob_annot %>%
    left_join(
      tibble(
        photo_id = rob_photos,
        file_name = file.path(
          "data/leaves/phenobase_dwca_annotation/images_breaking_leaf_buds",
          rob_photos
        )
      ),
      by = "photo_id"
    ) %>%
    drop_na(file_name) %>%
    left_join(fams, by = "genus") %>%
    mutate(
      leaves_green = as.numeric(purrr::map_lgl(
        leaves,
        ~ grepl("Green leaves", .x, fixed = TRUE)
      )),
      leaves_colored = as.numeric(purrr::map_lgl(
        leaves,
        ~ grepl("Colored leaves", .x, fixed = TRUE)
      )),
      leaves_no_live = as.numeric(purrr::map_lgl(
        leaves,
        ~ grepl("No live leaves", .x, fixed = TRUE)
      )),
      leaves_breaking_buds = as.numeric(purrr::map_lgl(
        leaves,
        ~ grepl("Breaking leaf buds", .x, fixed = TRUE)
      ))
    ) %>%
    select(
      observation_uuid,
      scientificName = `taxon.name`,
      genus,
      family,
      file_name,
      leaves_green,
      leaves_colored,
      leaves_no_live,
      leaves_breaking_buds
    )

  # Second set of Rob's annotations
  rob_annot2 <- read_csv(rob_annot2_csv, show_col_types = FALSE) %>%
    mutate(
      leaves_breaking_buds = bb,
      photo_id = as.character(photo_id)
    ) %>%
    left_join(
      merged %>%
        group_by(observation_uuid) %>%
        mutate(count = n()) %>%
        ungroup() %>%
        filter(count == 1) %>%
        select(photo_id, leaves_green, leaves_colored, leaves_no_live),
      by = "photo_id"
    ) %>%
    mutate(
      leaves_green = ifelse(is.na(leaves_green), as.numeric(`Green leaves (0/1)`), leaves_green),
      leaves_colored = ifelse(is.na(leaves_colored), 0, leaves_colored),
      leaves_no_live = ifelse(leaves_green == 0 & leaves_colored == 0 & leaves_breaking_buds == "0", 1, 0)
    ) %>%
    filter(leaves_breaking_buds != "U") %>%
    mutate(leaves_breaking_buds = as.numeric(leaves_breaking_buds))

  # Combine Rob's annotations
  rob_annot_combined <- bind_rows(
    rob_annot_final,
    rob_annot2 %>%
      select(
        observation_uuid,
        scientificName = name,
        genus,
        family,
        file_name,
        leaves_green,
        leaves_colored,
        leaves_no_live,
        leaves_breaking_buds
      )
  )

  message(sprintf("    Added %d manual annotations from Rob",
                  nrow(rob_annot_combined)))

  # =========================================================================
  # Step 7: Combine all annotations
  # =========================================================================

  message("  6. Combining all annotations...")

  all_meta <- merged %>%
    select(
      observation_uuid,
      scientificName,
      genus,
      family,
      file_name,
      leaves_green,
      leaves_colored,
      leaves_no_live,
      leaves_breaking_buds
    ) %>%
    bind_rows(rob_annot_combined)

  # Remove observations with all zeros
  all_meta <- all_meta %>%
    filter(!(leaves_green == 0 & leaves_colored == 0 &
             leaves_breaking_buds == 0 & leaves_no_live == 0)) %>%
    select(-leaves_no_live)

  # Count photos per observation
  all_meta <- all_meta %>%
    group_by(observation_uuid) %>%
    mutate(count = n()) %>%
    ungroup()

  # Add stratification column
  all_meta <- all_meta %>%
    mutate(strata = paste(family, leaves_green, leaves_colored,
                          leaves_breaking_buds, sep = "|"))

  message(sprintf("    Final dataset: %s observations",
                  format(nrow(all_meta), big.mark = ",")))

  # Summary
  summ <- all_meta %>%
    group_by(leaves_green, leaves_colored, leaves_breaking_buds) %>%
    summarise(count = n(), .groups = "drop")

  message("    Leaf state distribution:")
  print(summ)

  message("Extraction complete!")
  return(all_meta)
}
