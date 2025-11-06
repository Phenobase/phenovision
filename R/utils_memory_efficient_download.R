#' Memory-Efficient Utility Functions for Download Pipeline
#'
#' This file contains helper functions optimized for low memory usage
#' when processing large iNaturalist datasets.

#' Create Hash Set from Photo IDs in Parquet
#'
#' Loads only photo_id column from parquet and converts to integer vector
#' for memory-efficient %in% comparisons.
#'
#' @param parquet_path Character. Path to parquet dataset
#'
#' @return Integer vector of photo IDs (hash set)
#'
#' @details
#' Memory comparison:
#' - Full dataframe with batch_j: ~15 GB for 62M rows
#' - Integer vector only: ~500 MB for 62M IDs
#'
#' @export
create_photo_id_hashset <- function(parquet_path) {
  library(arrow)

  message("  Creating photo ID hash set for efficient comparison...")

  photo_ids <- open_dataset(parquet_path) %>%
    select(photo_id) %>%
    collect() %>%
    pull(photo_id)

  # Convert to integer if needed
  if (is.character(photo_ids)) {
    photo_ids <- as.integer(photo_ids)
  }

  message(sprintf("    Loaded %s photo IDs into memory (~%.1f MB)",
                  format(length(photo_ids), big.mark = ","),
                  object.size(photo_ids) / 1024^2))

  return(photo_ids)
}


#' Union Arrow Datasets Without Loading into Memory
#'
#' Combines old and new parquet datasets using Arrow's zero-copy union,
#' avoiding the massive memory spike from bind_rows().
#'
#' @param old_parquet_path Character. Path to existing parquet dataset
#' @param new_df Data frame. New data to append
#' @param temp_dir Character. Directory for temporary parquet (default: tempdir())
#'
#' @return Character. Path to temporary unified parquet
#'
#' @details
#' Memory comparison:
#' - bind_rows(old, new): ~200 GB peak (old + new + combined)
#' - Arrow union: ~75 GB peak (only new data in memory)
#'
#' The function:
#' 1. Writes new data to temporary parquet partition
#' 2. Opens both old and temp as datasets
#' 3. Unions them at Arrow query level
#' 4. Writes union to new output parquet
#'
#' @export
arrow_union_datasets <- function(old_parquet_path, new_df, temp_dir = tempdir()) {
  library(arrow)
  library(dplyr)

  message("  Performing memory-efficient Arrow union...")

  # Write new data to temporary parquet
  temp_new_path <- file.path(temp_dir, "temp_new_photos")
  dir.create(temp_new_path, recursive = TRUE, showWarnings = FALSE)

  write_dataset(new_df, path = temp_new_path, format = "parquet")

  message(sprintf("    Wrote %s new rows to temporary parquet",
                  format(nrow(new_df), big.mark = ",")))

  # Open both datasets
  old_ds <- open_dataset(old_parquet_path)
  new_ds <- open_dataset(temp_new_path)

  # Get schema (use old dataset's schema as reference)
  schema <- old_ds$schema

  # Ensure new dataset has same schema (add missing columns if needed)
  new_cols <- names(new_ds)
  old_cols <- names(old_ds)

  if (!all(new_cols %in% old_cols)) {
    warning("New dataset has columns not in old dataset. Union may fail.")
  }

  # Create union output path
  union_output <- file.path(temp_dir, "temp_union_photos")
  dir.create(union_output, recursive = TRUE, showWarnings = FALSE)

  # Union and write (Arrow does this without loading full data into memory)
  union_ds <- open_dataset(c(old_parquet_path, temp_new_path))

  write_dataset(union_ds, path = union_output, format = "parquet")

  message(sprintf("    Union complete: combined dataset at %s", union_output))

  # Clean up temporary new path
  unlink(temp_new_path, recursive = TRUE)

  return(union_output)
}


#' Parse JSON in Chunks to Reduce Memory
#'
#' Processes large JSON columns in chunks to avoid loading
#' all parsed structures into memory at once.
#'
#' @param json_strings Character vector. JSON strings to parse
#' @param chunk_size Integer. Rows per chunk (default: 100000)
#' @param extract_fn Function. Function to extract values from parsed JSON
#'
#' @return Vector of extracted values
#'
#' @details
#' Memory comparison:
#' - Parse all at once: ~20 GB for 5M JSON strings
#' - Parse in 100k chunks: ~2 GB peak
#'
#' @export
parse_json_chunked <- function(json_strings, chunk_size = 100000, extract_fn) {
  library(jsonlite)
  library(purrr)

  n <- length(json_strings)
  n_chunks <- ceiling(n / chunk_size)

  message(sprintf("  Parsing %s JSON strings in %d chunks...",
                  format(n, big.mark = ","), n_chunks))

  results <- vector("list", n_chunks)

  for (i in seq_len(n_chunks)) {
    start_idx <- (i - 1) * chunk_size + 1
    end_idx <- min(i * chunk_size, n)

    chunk_json <- json_strings[start_idx:end_idx]

    # Parse JSON
    parsed <- purrr::map(
      chunk_json,
      purrr::possibly(fromJSON, otherwise = NULL)
    )

    # Extract values
    results[[i]] <- extract_fn(parsed)

    if (i %% 10 == 0) {
      message(sprintf("    Processed chunk %d/%d", i, n_chunks))
    }
  }

  # Combine results
  do.call(c, results)
}


#' Extract Leaf Flags in Single Pass
#'
#' Extracts all leaf state flags from parsed JSON in one iteration,
#' avoiding 4 separate purrr::map_lgl() calls.
#'
#' @param parsed_json List. Parsed JSON objects from dynamicProperties
#'
#' @return Data frame with columns: leaves_green, leaves_colored,
#'         leaves_no_live, leaves_breaking_buds
#'
#' @details
#' Memory comparison:
#' - 4 separate map_lgl calls: ~10 GB temporary vectors
#' - Single pass: ~2.5 GB (one dataframe)
#'
#' @export
extract_leaf_flags_single_pass <- function(parsed_json) {
  library(data.table)

  n <- length(parsed_json)

  # Pre-allocate result vectors
  leaves_green <- integer(n)
  leaves_colored <- integer(n)
  leaves_no_live <- integer(n)
  leaves_breaking_buds <- integer(n)

  # Single pass through data
  for (i in seq_along(parsed_json)) {
    leaves <- parsed_json[[i]][["leaves"]]

    if (!is.null(leaves) && length(leaves) > 0) {
      leaves_green[i] <- as.integer("green leaves" %chin% leaves)
      leaves_colored[i] <- as.integer("colored leaves" %chin% leaves)
      leaves_no_live[i] <- as.integer("no live leaves" %chin% leaves)
      leaves_breaking_buds[i] <- as.integer("breaking leaf buds" %chin% leaves)
    }
  }

  data.frame(
    leaves_green = leaves_green,
    leaves_colored = leaves_colored,
    leaves_no_live = leaves_no_live,
    leaves_breaking_buds = leaves_breaking_buds
  )
}


#' Filter Large CSV with awk Before Reading into R
#'
#' Uses awk to pre-filter a large CSV file, reading only matching rows into R.
#' Much more memory-efficient than read then filter.
#'
#' @param input_file Character. Path to input CSV
#' @param output_file Character. Path for filtered output
#' @param column_num Integer. Column number to check (1-indexed)
#' @param pattern Character. awk pattern (default: check column is not empty)
#'
#' @return Character. Path to filtered file
#'
#' @details
#' For observations.csv filtering by non-empty reproductiveCondition:
#' - Read all then filter in R: ~90 GB
#' - awk filter then read: ~30 GB
#'
#' @export
awk_filter_csv <- function(input_file, output_file, column_num, pattern = NULL) {

  if (is.null(pattern)) {
    # Default: check if column is not empty
    pattern <- sprintf("$%d != \"\"", column_num)
  }

  # Build awk command
  # Skip header (NR > 1), check pattern
  awk_cmd <- sprintf(
    "awk 'BEGIN {FS=\"\\t\"} NR == 1 || (%s) {print}' %s > %s",
    pattern,
    input_file,
    output_file
  )

  message(sprintf("  Filtering %s with awk...", basename(input_file)))
  message(sprintf("    Pattern: %s", pattern))

  system(awk_cmd)

  # Check output exists
  if (!file.exists(output_file)) {
    stop("awk filtering failed - output file not created")
  }

  # Report size reduction
  input_size <- file.size(input_file) / 1024^3
  output_size <- file.size(output_file) / 1024^3
  reduction_pct <- 100 * (1 - output_size / input_size)

  message(sprintf("    Input:  %.1f GB", input_size))
  message(sprintf("    Output: %.1f GB", output_size))
  message(sprintf("    Reduction: %.1f%%", reduction_pct))

  return(output_file)
}
