#' Concatenate Multiple Annotation CSV Files
#'
#' Reads multiple CSV files and concatenates them into a single output CSV.
#' Used for combining batch annotation results into one master file.
#'
#' @param annotation_outputs Character vector of paths to CSV files
#' @param file_name Character. Path to output CSV file
#'
#' @return Character. The file_name (for use in targets pipelines)
#'
#' @details
#' This function:
#' 1. Removes NULL/empty paths from annotation_outputs
#' 2. Reads each CSV file
#' 3. Appends each to the output file
#' 4. Shows progress bar during processing
#'
#' Useful for final aggregation step when multiple batch CSV files need to be
#' combined into a single file for upload or distribution.
#'
#' **Note:** This appends to existing files, so make sure to delete file_name
#' before running if you want to start fresh.
#'
#' @examples
#' \dontrun{
#' # Combine all batch CSV files
#' batch_files <- list.files("output/batches", pattern = "*.csv", full.names = TRUE)
#' final_file <- concatenate_annotations(
#'   batch_files,
#'   "output/all_annotations_combined.csv"
#' )
#' }
#'
#' @export
concatenate_annotations <- function(annotation_outputs, file_name) {

  # Remove NULL or empty paths
  annotation_outputs <- purrr::compact(annotation_outputs)

  # Read each CSV and append to output file
  purrr::walk(
    annotation_outputs,
    ~ readr::write_csv(
      readr::read_csv(.x, show_col_types = FALSE),
      file_name,
      append = TRUE
    ),
    .progress = TRUE
  )

  return(file_name)
}
