#' Load PhenoVision Model from DOI
#'
#' Load a PhenoVision model from HuggingFace Hub using a DOI identifier.
#' The DOI is resolved via DataCite to get the HuggingFace repository and
#' version (commit hash), then the model is loaded using transformers.
#'
#' @param doi Character. DOI for the model (e.g., "10.57967/hf/2763")
#' @param type Character. Type of model to load. Currently only "classifier" is implemented.
#'
#' @return A transformers model object (AutoModelForImageClassification)
#'
#' @details
#' This function:
#' 1. Resolves the DOI using rdatacite to get the HuggingFace repo URL and version
#' 2. Extracts the repo name and commit hash
#' 3. Loads the model using transformers AutoModelForImageClassification
#'
#' The "latent_code" type is not yet implemented.
#'
#' @examples
#' \dontrun{
#' # Load reproductive structures model
#' model <- load_phenovision("10.57967/hf/2763")
#'
#' # Load leaf phenology model
#' model_leaves <- load_phenovision("10.57967/hf/5785")
#' }
#'
#' @export
load_phenovision <- function(doi = "10.57967/hf/2763",
                              type = c("classifier", "latent_code")) {

  type <- match.arg(type)

  # Import required Python modules
  tr <- reticulate::import("transformers")

  # Resolve DOI to get HuggingFace repo and version
  mod_dat <- rdatacite::dc_dois(doi)
  ver <- mod_dat$data$attributes$version
  repo <- gsub("https://huggingface.co/", "", mod_dat$data$attributes$url)

  if (type == "classifier") {
    model <- tr$AutoModelForImageClassification$from_pretrained(
      repo,
      revision = ver
    )
    return(model)
  }

  if (type == "latent_code") {
    stop("latent_code type not yet implemented")
  }
}
