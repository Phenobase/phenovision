#!/usr/bin/env Rscript

#' Standalone script to validate HuggingFace to timm model conversion
#'
#' This script loads a PhenoVision model from HuggingFace Hub, converts it
#' to timm format, and validates that the outputs match.
#'
#' Usage:
#'   Rscript R/validate_hf_to_timm_conversion.R [doi] [num_images]
#'
#' Examples:
#'   Rscript R/validate_hf_to_timm_conversion.R
#'   Rscript R/validate_hf_to_timm_conversion.R "10.57967/hf/2763" 20

library(reticulate)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
doi <- if (length(args) >= 1) args[1] else "10.57967/hf/2763"
num_images <- if (length(args) >= 2) as.integer(args[2]) else 10L

cat("=================================================\n")
cat("HuggingFace → timm Conversion Validation\n")
cat("=================================================\n\n")
cat(sprintf("DOI: %s\n", doi))
cat(sprintf("Test images: %d\n", num_images))
cat(sprintf("Device: cuda\n\n"))

# Set up reticulate
use_condaenv("reticulate-gpu2", required = TRUE)

# Source the loading function
source("R/model_loading_training.R")

# Run validation
results <- validate_hf_to_timm_conversion(
  doi = doi,
  test_images = NULL,  # Will create random images
  num_images = num_images,
  device = "cuda"
)

# Exit with appropriate code
if (results$outputs_match) {
  cat("\n✓ VALIDATION PASSED\n")
  quit(status = 0)
} else {
  cat("\n✗ VALIDATION FAILED\n")
  quit(status = 1)
}
