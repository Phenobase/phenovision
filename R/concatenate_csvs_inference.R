#' Concatenate Data Frame to CSV File
#'
#' Appends a data frame to a CSV file, creating the file if it doesn't exist.
#' Used for accumulating batch results into a single CSV.
#'
#' @param annotations Data frame to write
#' @param csv_path Character. Path to CSV file
#'
#' @return Character. The csv_path (for use in targets pipelines)
#'
#' @details
#' This is a simple wrapper around `readr::write_csv()` with `append = TRUE`.
#' If the file doesn't exist, it creates it. If it exists, it appends without
#' writing headers again.
#'
#' Useful in targets pipelines for aggregating results from pattern-mapped
#' batch processing.
#'
#' @examples
#' \dontrun{
#' # In a targets pipeline with pattern mapping
#' tar_target(
#'   csv_outputs,
#'   concatenate_csvs(batch_annotations, "output/all_annotations.csv"),
#'   pattern = map(batch_annotations),
#'   format = "file"
#' )
#' }
#'
#' @export
concatenate_csvs <- function(annotations, csv_path) {
  readr::write_csv(annotations, csv_path, append = file.exists(csv_path))
  csv_path
}
