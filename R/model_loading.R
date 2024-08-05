require(reticulate)
require(rdatacite)

load_phenovision <- function(doi = "10.57967/hf/2763", type = c("classifier", "latent_code")) {
  type <- match.arg(type)
  tr <- reticulate::import("transformers")
  mod_dat <- rdatacite::dc_dois(doi)
  ver <- mod_dat$data$attributes$version
  if(type == "classifier") {
    model <- tr$AutoModelForImageClassification$from_pretrained("phenobase/phenovision", revision = ver)
    return(model)
  }
  if(type == "latent_code") {

  }

}
