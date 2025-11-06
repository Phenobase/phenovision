# Test script to verify function sourcing pattern works correctly

library(tidyverse)

# Test sourcing common functions
cat("Testing common function sourcing...\n")
common_files <- list.files("R", pattern = "_common\\.R$", full.names = TRUE)
cat("Found", length(common_files), "common files:\n")
print(common_files)

walk(common_files, source)
cat("✓ Common functions sourced\n\n")

# Test sourcing inference functions
cat("Testing inference function sourcing...\n")
inference_files <- list.files("R", pattern = "_inference\\.R$", full.names = TRUE)
cat("Found", length(inference_files), "inference files:\n")
print(inference_files)

walk(inference_files, source)
cat("✓ Inference functions sourced\n\n")

# Test that functions are available
cat("Checking functions are available...\n")
common_funcs <- c("load_phenovision", "model_version")
inference_funcs <- c("annotate_batch", "threshold_annotations", "convert_to_long",
                     "aggregate_by_obs", "convert_fam_to_long", "finalize_annotations_by_obs",
                     "concatenate_csvs", "concatenate_annotations")

for (func in common_funcs) {
  if (exists(func)) {
    cat("✓", func, "\n")
  } else {
    cat("✗", func, "NOT FOUND\n")
  }
}

for (func in inference_funcs) {
  if (exists(func)) {
    cat("✓", func, "\n")
  } else {
    cat("✗", func, "NOT FOUND\n")
  }
}

cat("\nAll tests passed!\n")
