#' Extract Leaf Annotations from iNaturalist Data
#'
#' Extracts and processes leaf phenology annotations from iNaturalist parquet
#' files and Rob's manual annotations.
#'
#' @param leaf_parquet_path Character. Path to parquet file with iNaturalist leaf annotations
#' @param rob_annot_csv Character. Path to Rob's breaking leaf buds annotations
#' @param rob_annot2_csv Character. Path to Rob's additional annotations
#' @param photo_metadata_path Character. Path to photo metadata dataset (default: from paths$metadata_photos)
#'
#' @return Data frame with columns:
#'   - observation_uuid: Observation ID
#'   - scientificName: Scientific name
#'   - genus, family: Taxonomy
#'   - file_name: Path to image file
#'   - leaves_green: 1/0 indicator
#'   - leaves_colored: 1/0 indicator
#'   - leaves_breaking_buds: 1/0 indicator
#'   - count: Number of photos per observation
#'   - strata: Stratification column for splitting
#'
#' @details
#' This function:
#' 1. Loads iNaturalist leaf annotations from parquet
#' 2. Parses JSON from dynamicProperties field
#' 3. Extracts leaf state flags (green, colored, breaking buds)
#' 4. Filters to target genera (excludes Logfia, Oxalis, Viola)
#' 5. Merges with Rob's manual annotations for breaking leaf buds
#' 6. Joins with photo metadata to get file paths
#' 7. Removes observations with no live leaves and all zeros
#' 8. Creates stratification column for splitting
#'
#' The dynamicProperties JSON contains leaf annotations like:
#' {"leaves": ["green leaves", "colored leaves"]}
#'
#' @examples
#' \dontrun{
#' leaf_df <- extract_leaf_annotations(
#'   "data/leaves/phenobase_dwca_annotation/inat_annotation/part-0.parquet",
#'   "data/leaves/phenobase_dwca_annotation/rob_leaf_breaking_buds_annotation.csv",
#'   "data/leaves/rob_new_annotations_bb.csv"
#' )
#' }
#'
#' @export
extract_leaf_annotations <- function(leaf_parquet_path,
                                     rob_annot_csv,
                                     rob_annot2_csv,
                                     photo_metadata_path = "data/phenobase_inat_data/metadata/angio_photos") {

  library(dplyr)
  library(tidyr)
  library(stringr)
  library(arrow)
  library(jsonlite)
  library(data.table)

  message("Extracting leaf annotations...")

  # =========================================================================
  # Step 1: Load iNaturalist leaf annotations
  # =========================================================================

  message("  1. Loading iNaturalist leaf annotations from parquet...")

  leaf_annot <- read_parquet(leaf_parquet_path) %>%
    filter(dynamic_properties != "")

  message(sprintf("    Loaded %d observations with leaf annotations", nrow(leaf_annot)))

  # =========================================================================
  # Step 2: Fix double quotes and parse JSON
  # =========================================================================

  message("  2. Parsing JSON from dynamic_properties...")

  # Fix weird double double quotes issue
  leaf_annot <- leaf_annot %>%
    mutate(dynamic_properties = str_replace_all(dynamic_properties, fixed('""'), '"'))

  # Parse JSON (use possibly() to handle errors gracefully)
  leaf_json <- purrr::map(
    leaf_annot$dynamic_properties,
    purrr::possibly(fromJSON, otherwise = NULL),
    .progress = TRUE
  )

  # =========================================================================
  # Step 3: Extract leaf state flags
  # =========================================================================

  message("  3. Extracting leaf state flags...")

  meta <- leaf_annot %>%
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
    )

  # =========================================================================
  # Step 4: Filter by genera
  # =========================================================================

  message("  4. Filtering by target genera...")

  # Load Rob's annotations to get target genera
  rob_annot <- read_csv(rob_annot_csv, show_col_types = FALSE) %>%
    mutate(genus = word(taxon.name))

  keep_genera <- unique(rob_annot$genus)
  # Remove problematic genera
  genera_exclude <- c("Logfia", "Oxalis", "Viola")
  keep_genera <- setdiff(keep_genera, genera_exclude)

  message(sprintf("    Target genera: %d", length(keep_genera)))
  message(sprintf("    Excluded: %s", paste(genera_exclude, collapse = ", ")))

  # Filter annotations (exclude breaking buds from iNat - only use Rob's)
  meta <- meta %>%
    filter(!leaves_breaking_buds == 1) %>%
    filter(genus %chin% keep_genera)

  message(sprintf("    Filtered to %d observations", nrow(meta)))

  # =========================================================================
  # Step 5: Join with photo metadata
  # =========================================================================

  message("  5. Joining with photo metadata...")

  photo_meta <- open_dataset(photo_metadata_path) %>%
    select(observation_uuid, photo_uuid, photo_id, batch_j, extension) %>%
    filter(observation_uuid %in% unique(c(meta$observation_uuid, rob_annot$observation_uuid))) %>%
    collect()

  meta <- meta %>%
    left_join(photo_meta, by = "observation_uuid") %>%
    drop_na(batch_j)

  message(sprintf("    Matched %d observations to photos", nrow(meta)))

  # =========================================================================
  # Step 6: Add file paths
  # =========================================================================

  meta <- meta %>%
    mutate(file_name = file.path(
      "data", "phenobase_inat_data", "images", "medium",
      paste0("batch_", batch_j),
      paste0(photo_id, ".", extension)
    ))

  # =========================================================================
  # Step 7: Process Rob's manual annotations
  # =========================================================================

  message("  6. Processing Rob's manual annotations...")

  # Get family info
  fams <- meta %>%
    select(genus, family) %>%
    distinct()

  # Process first set of Rob's annotations
  rob_photos <- list.files(
    "data/leaves/phenobase_dwca_annotation/images_breaking_leaf_buds",
    full.names = FALSE
  )
  rob_photo_ids <- fs::path_ext_remove(rob_photos)

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
        ~ grepl("Green leaves", .x, fixed = TRUE),
        .progress = TRUE
      )),
      leaves_colored = as.numeric(purrr::map_lgl(
        leaves,
        ~ grepl("Colored leaves", .x, fixed = TRUE),
        .progress = TRUE
      )),
      leaves_no_live = as.numeric(purrr::map_lgl(
        leaves,
        ~ grepl("No live leaves", .x, fixed = TRUE),
        .progress = TRUE
      )),
      leaves_breaking_buds = as.numeric(purrr::map_lgl(
        leaves,
        ~ grepl("Breaking leaf buds", .x, fixed = TRUE),
        .progress = TRUE
      ))
    ) %>%
    select(
      observation_uuid,
      scientificName = taxon.name,
      genus,
      family,
      file_name,
      leaves_green,
      leaves_colored,
      leaves_no_live,
      leaves_breaking_buds
    )

  # Process second set of Rob's annotations
  rob_annot2 <- read_csv(rob_annot2_csv, show_col_types = FALSE) %>%
    mutate(
      leaves_breaking_buds = bb,
      photo_id = as.character(photo_id)
    ) %>%
    left_join(
      meta %>%
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
  rob_annot_final <- bind_rows(
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

  message(sprintf("    Added %d manual annotations from Rob", nrow(rob_annot_final)))

  # =========================================================================
  # Step 8: Combine all annotations
  # =========================================================================

  message("  7. Combining all annotations...")

  all_meta <- meta %>%
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
    bind_rows(rob_annot_final)

  # Add stratification column
  all_meta <- all_meta %>%
    mutate(strata = paste(family, leaves_green, leaves_colored,
                          leaves_no_live, leaves_breaking_buds, sep = "|"))

  # Count photos per observation
  all_meta <- all_meta %>%
    group_by(observation_uuid) %>%
    mutate(count = n()) %>%
    ungroup()

  # Remove observations with all zeros (no live leaves and no breaking buds)
  all_meta <- all_meta %>%
    filter(!(leaves_green == 0 & leaves_colored == 0 &
             leaves_breaking_buds == 0 & leaves_no_live == 0)) %>%
    select(-leaves_no_live)  # Remove no_live column (computed from others)

  message(sprintf("    Final dataset: %d observations", nrow(all_meta)))

  # Summary
  summ <- all_meta %>%
    group_by(leaves_green, leaves_colored, leaves_breaking_buds) %>%
    summarise(count = n(), .groups = "drop")

  message("    Leaf state distribution:")
  print(summ)

  message("Extraction complete!")

  return(all_meta)
}
