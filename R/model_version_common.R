#' Get Model Version from DOI
#'
#' Extract the version (commit hash) of a HuggingFace model from its DOI.
#'
#' @param doi Character. DOI for the model (e.g., "10.57967/hf/2763")
#'
#' @return Character. The git commit hash (version) of the model
#'
#' @details
#' This function resolves a DOI via DataCite and extracts the version
#' attribute, which corresponds to a specific git commit hash in the
#' HuggingFace repository.
#'
#' @examples
#' \dontrun{
#' # Get version of reproductive structures model
#' ver <- model_version("10.57967/hf/2763")
#' print(ver)
#' }
#'
#' @export
model_version <- function(doi = "10.57967/hf/2763") {
  mod_dat <- rdatacite::dc_dois(doi)
  ver <- mod_dat$data$attributes$version
  ver
}
