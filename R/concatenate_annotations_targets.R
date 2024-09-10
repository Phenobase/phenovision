#' .. content for \description{} (no empty lines) ..
#'
#' .. content for \details{} ..
#'
#' @title
#' @param annotation_outputs
#' @return
#' @author rdinnager
#' @export
concatenate_annotations <- function(annotation_outputs, file_name) {

  annotation_outputs <- compact(annotation_outputs)
  walk(annotation_outputs, ~ write_csv(read_csv(.x), file_name, append = TRUE), .progress = TRUE)
  return(file_name)

}
