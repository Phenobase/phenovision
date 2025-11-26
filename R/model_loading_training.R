require(reticulate)
require(rdatacite)

#' Load PhenoVision model from HuggingFace DOI for training
#'
#' This function loads a PhenoVision model from HuggingFace Hub using its DOI,
#' converts it from HuggingFace format to timm format, and prepares it for
#' training with the timm-based training scripts.
#'
#' @param doi DOI of the model on HuggingFace Hub (e.g., "10.57967/hf/2763")
#' @param num_classes Number of output classes for the model (2 for reproductive, 3 for leaves)
#' @param reinit_head Logical. If TRUE, always reinitialize the classification head.
#'                    If FALSE, keep existing head if num_classes matches.
#' @param device Device to load model on ("cuda" or "cpu")
#'
#' @return A timm ViT model ready for training
#'
#' @examples
#' # Load current reproductive model for fine-tuning
#' vit <- load_phenovision_for_training(
#'   doi = "10.57967/hf/2763",
#'   num_classes = 2L,
#'   reinit_head = FALSE
#' )
#'
#' # Load reproductive model but reinitialize head for new task
#' vit <- load_phenovision_for_training(
#'   doi = "10.57967/hf/2763",
#'   num_classes = 3L,  # Different number of classes
#'   reinit_head = TRUE
#' )
load_phenovision_for_training <- function(doi = "10.57967/hf/2763",
                                           num_classes = 2L,
                                           reinit_head = FALSE,
                                           device = "cuda") {

  # Load transformers library
  tr <- reticulate::import("transformers")

  # Get model metadata from DOI
  mod_dat <- rdatacite::dc_dois(doi)
  ver <- mod_dat$data$attributes$version
  repo <- gsub("https://huggingface.co/", "", mod_dat$data$attributes$url)

  message(sprintf("Loading model from %s (version: %s)", repo, ver))

  # Load HuggingFace model
  hf_model <- tr$AutoModelForImageClassification$from_pretrained(
    repo,
    revision = ver,
    local_files_only = FALSE  # Use cache if available, download if not
  )

  # Get state dict
  hf_state_dict <- hf_model$state_dict()

  # Get model config to determine hidden size
  config <- hf_model$config
  hidden_size <- as.integer(config$hidden_size)

  message(sprintf("Converting HuggingFace model to timm format (hidden_size=%d)", hidden_size))

  # Import conversion function
  convert_hf_timm <- reticulate::import_from_path("convert_hf_to_timm", "py")

  # Convert state dict to timm format
  timm_state_dict <- convert_hf_timm$convert_hf_to_timm(
    hf_state_dict,
    hidden_size = hidden_size
  )

  # Import model architecture
  models_vit <- reticulate::import_from_path("models_vit", "PlantCLEF2022")

  # Create timm model with specified number of classes
  vit <- models_vit$vit_large_patch16(num_classes = as.integer(num_classes))

  # Get expected number of classes from loaded model
  hf_num_classes <- as.integer(hf_model$config$num_labels)

  # Determine if we need to reinitialize the head
  needs_reinit <- reinit_head || (num_classes != hf_num_classes)

  if (needs_reinit) {
    message(sprintf(
      "Head will be reinitialized (requested: %s, num_classes mismatch: %s)",
      reinit_head,
      num_classes != hf_num_classes
    ))

    # Remove head weights from state dict if present
    # (allows strict=TRUE loading of backbone only)
    timm_state_dict_no_head <- reticulate::py_dict(
      keys = reticulate::iterate(timm_state_dict$keys()) %>%
        purrr::keep(~!grepl("^head\\.", .x)),
      values = reticulate::iterate(timm_state_dict$keys()) %>%
        purrr::keep(~!grepl("^head\\.", .x)) %>%
        purrr::map(~timm_state_dict[[.x]])
    )

    # Load weights (strict=FALSE to allow missing head)
    msg <- vit$load_state_dict(timm_state_dict_no_head, strict = FALSE)
    message("Loaded backbone weights (head excluded)")

    # Initialize new classification head
    torch <- reticulate::import("torch")
    torch$nn$init$trunc_normal_(vit$head$weight, std = 1e-5)
    if (!is.null(vit$head$bias)) {
      torch$nn$init$zeros_(vit$head$bias)
    }
    message("Initialized new classification head")

  } else {
    message("Loading full model including classification head")

    # Load all weights including head
    msg <- vit$load_state_dict(timm_state_dict, strict = TRUE)
  }

  # Move to device
  vit <- vit[[device]]()

  message(sprintf("Model ready on %s", device))

  return(vit)
}


#' Validate HuggingFace to timm conversion
#'
#' Runs inference with both HuggingFace and timm models on test images
#' to verify that the conversion produces equivalent outputs.
#'
#' @param doi DOI of the model to validate
#' @param test_images Path to test images or torch tensor of images
#' @param num_images Number of images to test (if test_images is a path)
#' @param device Device to run on
#'
#' @return List with validation metrics
validate_hf_to_timm_conversion <- function(doi = "10.57967/hf/2763",
                                            test_images = NULL,
                                            num_images = 10,
                                            device = "cuda") {

  message("Loading HuggingFace model...")
  source("R/model_loading_targets.R")
  hf_model <- load_phenovision(doi = doi, type = "classifier")

  # Get number of classes from HF model
  num_classes <- as.integer(hf_model$config$num_labels)

  message("Converting to timm model...")
  timm_model <- load_phenovision_for_training(
    doi = doi,
    num_classes = num_classes,
    reinit_head = FALSE,  # Keep same head for fair comparison
    device = device
  )

  # If test_images not provided, create random test images
  if (is.null(test_images)) {
    torch <- reticulate::import("torch")
    message(sprintf("Creating %d random test images (224x224x3)", num_images))
    test_images <- torch$randn(as.integer(num_images), 3L, 224L, 224L)
  }

  # Import validation function
  convert_hf_timm <- reticulate::import_from_path("convert_hf_to_timm", "py")

  message("Running validation...")
  results <- convert_hf_timm$validate_conversion(
    hf_model = hf_model,
    timm_model = timm_model,
    test_images = test_images,
    device = device
  )

  # Print results
  message("\n=== Validation Results ===")
  message(sprintf("Max absolute difference: %.6e", results$max_abs_diff))
  message(sprintf("Mean absolute difference: %.6e", results$mean_abs_diff))
  message(sprintf("Max relative difference: %.6e", results$max_rel_diff))
  message(sprintf("Mean relative difference: %.6e", results$mean_rel_diff))
  message(sprintf("Outputs match (rtol=1e-4): %s", results$outputs_match))

  if (results$outputs_match) {
    message("\n✓ Conversion validated successfully!")
  } else {
    warning("\n✗ Conversion validation FAILED - outputs do not match!")
  }

  return(results)
}
