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

  annotations_ds <- open_dataset(annotation_parquet) %>%
    filter(reproductive_condition != "") %>%
    select(
      observation_uuid,
      reproductive_condition,
      scientific_name,
      taxon_id
    )

  # Count for logging
  annotation_count <- annotations_ds %>% count() %>% pull(n)

  message(sprintf("    Found %s observations with reproductive annotations",
                  format(annotation_count, big.mark = ",")))

  # Note: Annotations from DwC are already filtered to research-grade in parse_phenology_dwc()
  message(sprintf("    All %s observations are research-grade",
                  format(annotation_count, big.mark = ",")))

  # =========================================================================
  # Step 2: Join with photos parquet (Arrow-to-Arrow)
  # =========================================================================

  message("  2. Joining with photo metadata (Arrow-to-Arrow join)...")

  # Open photos dataset and cast observation_uuid to match annotations (utf8/string)
  photos_ds <- open_dataset(photos_parquet) %>%
    select(observation_uuid, photo_id, extension, batch_j) %>%
    mutate(observation_uuid = cast(observation_uuid, utf8()))

  # Join two Arrow datasets, then collect
  photos_joined <- photos_ds %>%
    inner_join(annotations_ds, by = "observation_uuid") %>%
    collect()

  message(sprintf("    Matched %s photos",
                  format(nrow(photos_joined), big.mark = ",")))

  # =========================================================================
  # Step 3: Filter to single-photo observations
  # =========================================================================

  message("  3. Filtering to single-photo observations...")

  # Filter to single-photo observations (n() not supported in Arrow)
  single_photo_obs <- photos_joined %>%
    group_by(observation_uuid) %>%
    filter(n() == 1) %>%
    ungroup()

  message(sprintf("    Kept %s single-photo observations",
                  format(nrow(single_photo_obs), big.mark = ",")))

  # =========================================================================
  # Step 4: Parse flowering/fruiting flags
  # =========================================================================

  message("  4. Parsing flowering/fruiting flags...")

  merged <- single_photo_obs %>%
    mutate(
      # Parse reproductive_condition (pipe-separated)
      flowering = as.integer(grepl("flowers", reproductive_condition, fixed = TRUE) & !grepl("no flowers or fruits", reproductive_condition, fixed = TRUE)),
      fruiting = as.integer(grepl("fruits or seeds", reproductive_condition, fixed = TRUE))
    )

  # Exclude ambiguous/contradictory annotations (e.g., "flowers|no flowers or fruits")
  merged <- merged %>%
    filter(!grepl("|no flowers or fruits", reproductive_condition, fixed = TRUE))

  # Summary
  n_flowering <- sum(merged$flowering == 1)
  n_fruiting <- sum(merged$fruiting == 1)
  n_both <- sum(merged$flowering == 1 & merged$fruiting == 1)
  n_neither <- sum(merged$flowering == 0 & merged$fruiting == 0)

  message(sprintf("    Flowering: %s (%.1f%%)", format(n_flowering, big.mark = ","),
                  100 * n_flowering / nrow(merged)))
  message(sprintf("    Fruiting:  %s (%.1f%%)", format(n_fruiting, big.mark = ","),
                  100 * n_fruiting / nrow(merged)))
  message(sprintf("    Both:      %s (%.1f%%)", format(n_both, big.mark = ","),
                  100 * n_both / nrow(merged)))
  message(sprintf("    Neither:      %s (%.1f%%)", format(n_neither, big.mark = ","),
                  100 * n_neither / nrow(merged)))

  # =========================================================================
  # Step 5: Add file paths
  # =========================================================================

  merged <- merged %>%
    mutate(
      file_name = file.path(
        images_root,
        paste0("batch_", batch_j),
        paste0(photo_id, ".webp")  # WebP store; `extension` kept only for the S3 URL (R/webp_paths_common.R)
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
    filter(dynamic_properties != "") %>%
    select(
      observation_uuid,
      dynamic_properties,
      scientific_name,
      genus,
      family
    ) %>%
    collect()

  message(sprintf("    Found %s observations with leaf annotations",
                  format(nrow(annotations), big.mark = ",")))

  # =========================================================================
  # Step 2: Parse JSON from dynamicProperties (MEMORY OPTIMIZED)
  # =========================================================================

  message("  2. Parsing JSON from dynamicProperties...")
  message("     (MEMORY OPTIMIZED: Chunked processing + single-pass extraction)")

  # Fix double double quotes issue
  annotations <- annotations %>%
    mutate(dynamic_properties = str_replace_all(dynamic_properties, fixed('""'), '"'))

  # MEMORY OPTIMIZATION: Parse JSON in chunks to avoid loading all parsed structures at once
  # Old approach: parse all JSON at once = ~20 GB for 5M strings
  # New approach: parse in 100k chunks = ~2 GB peak

  source("R/utils_memory_efficient_download.R")

  # Define extraction function for chunked processing
  extract_leaf_flags_chunk <- function(parsed_json_chunk) {
    # Use single-pass extraction (avoids 4 separate map_lgl calls)
    extract_leaf_flags_single_pass(parsed_json_chunk)
  }

  # Chunk size (adjust based on available memory)
  chunk_size <- 100000

  n_rows <- nrow(annotations)
  n_chunks <- ceiling(n_rows / chunk_size)

  message(sprintf("    Processing %s JSON strings in %d chunks of %d...",
                  format(n_rows, big.mark = ","), n_chunks, chunk_size))

  # Pre-allocate results list
  leaf_flags_list <- vector("list", n_chunks)

  for (i in seq_len(n_chunks)) {
    start_idx <- (i - 1) * chunk_size + 1
    end_idx <- min(i * chunk_size, n_rows)

    # Get chunk of JSON strings
    json_chunk <- annotations$dynamic_properties[start_idx:end_idx]

    # Parse JSON
    parsed_chunk <- purrr::map(
      json_chunk,
      purrr::possibly(fromJSON, otherwise = NULL)
    )

    # Extract all flags in single pass
    leaf_flags_list[[i]] <- extract_leaf_flags_single_pass(parsed_chunk)

    if (i %% 10 == 0 || i == n_chunks) {
      message(sprintf("      Processed chunk %d/%d (%.1f%%)",
                      i, n_chunks, 100 * i / n_chunks))
    }

    # Explicit garbage collection every 20 chunks
    if (i %% 20 == 0) {
      gc()
    }
  }

  # Combine results
  leaf_flags <- bind_rows(leaf_flags_list)

  # Add flags to annotations
  annotations <- annotations %>%
    bind_cols(leaf_flags) %>%
    select(-dynamic_properties)

  message("    JSON parsing complete!")

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

  # Open photos and cast observation_uuid to string type (to match annotations)
  photos_ds <- open_dataset(photos_parquet) %>%
    select(observation_uuid, photo_id, extension, batch_j) %>%
    mutate(observation_uuid = cast(observation_uuid, utf8()))

  # Write annotations to temp Arrow table for Arrow join
  annotations_tbl <- arrow_table(annotations, schema = schema(
    observation_uuid = utf8(),
    scientific_name = utf8(),
    genus = utf8(),
    family = utf8(),
    leaves_green = int32(),
    leaves_colored = int32(),
    leaves_no_live = int32(),
    leaves_breaking_buds = int32()
  ))

  # Join in Arrow, then collect
  merged <- photos_ds %>%
    inner_join(annotations_tbl, by = "observation_uuid") %>%
    collect()

  message(sprintf("    Matched %s photos", format(nrow(merged), big.mark = ",")))

  # =========================================================================
  # Step 5: Add file paths
  # =========================================================================

  merged <- merged %>%
    mutate(
      file_name = file.path(
        images_root,
        paste0("batch_", batch_j),
        paste0(photo_id, ".webp")  # WebP store; `extension` kept only for the S3 URL (R/webp_paths_common.R)
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
      scientificName = scientific_name,
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
