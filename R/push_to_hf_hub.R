#!/usr/bin/env Rscript

#' Unified HuggingFace Hub Upload Script for PhenoVision Models
#'
#' Converts a timm checkpoint to HuggingFace format, validates the conversion,
#' uploads the model, and optionally uploads companion files (thresholds,
#' family stats, training report).
#'
#' Usage:
#'   Rscript R/push_to_hf_hub.R \
#'     --checkpoint output/reproductive/v1.1.0/checkpoints/phenovision_v1.1.0_epoch7.pt \
#'     --hf-repo phenobase/phenovision \
#'     --num-labels 2 \
#'     --version v1.1.0 \
#'     --thresholds output/reproductive/v1.1.0/final_buffer_params.csv \
#'     --family-stats output/reproductive/v1.1.0/family_stats.csv
#'
#' Based on: https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/convert_vit_mae_to_pytorch.py

library(argparse)

# --- Argument Parsing ---
parser <- ArgumentParser(description = "Upload a PhenoVision model to HuggingFace Hub")

parser$add_argument("--checkpoint", required = TRUE,
                    help = "Path to timm model checkpoint (.pt file)")
parser$add_argument("--hf-repo", required = TRUE,
                    help = "HuggingFace repository (e.g., 'phenobase/phenovision')")
parser$add_argument("--num-labels", required = TRUE, type = "integer",
                    help = "Number of output labels (2 for reproductive, 3 for leaves)")
parser$add_argument("--version", default = NULL,
                    help = "Model version string for commit message (e.g., 'v1.1.0')")
parser$add_argument("--thresholds", default = NULL,
                    help = "Path to thresholds file to upload as companion file")
parser$add_argument("--family-stats", default = NULL,
                    help = "Path to family stats file to upload as companion file")
parser$add_argument("--report", default = NULL,
                    help = "Path to training report to upload as companion file")
parser$add_argument("--dry-run", action = "store_true", default = FALSE,
                    help = "Show what would be done without actually pushing")

args <- parser$parse_args()

# --- Header ---
cat(strrep("=", 70), "\n")
cat("PhenoVision Model Upload to HuggingFace Hub\n")
cat(strrep("=", 70), "\n\n")
cat("Checkpoint:  ", args$checkpoint, "\n")
cat("HF Repo:     ", args$hf_repo, "\n")
cat("Num labels:  ", args$num_labels, "\n")
cat("Version:     ", if (!is.null(args$version)) args$version else "(not specified)", "\n")
cat("Dry run:     ", args$dry_run, "\n\n")

# --- Validate inputs ---
if (!file.exists(args$checkpoint)) {
  stop("Checkpoint file not found: ", args$checkpoint)
}

companion_files <- list()
if (!is.null(args$thresholds)) {
  if (!file.exists(args$thresholds)) stop("Thresholds file not found: ", args$thresholds)
  companion_files[["thresholds"]] <- args$thresholds
}
if (!is.null(args$family_stats)) {
  if (!file.exists(args$family_stats)) stop("Family stats file not found: ", args$family_stats)
  companion_files[["family_stats"]] <- args$family_stats
}
if (!is.null(args$report)) {
  if (!file.exists(args$report)) stop("Report file not found: ", args$report)
  companion_files[["report"]] <- args$report
}

if (length(companion_files) > 0) {
  cat("Companion files:\n")
  for (name in names(companion_files)) {
    cat("  ", name, ": ", companion_files[[name]], "\n")
  }
  cat("\n")
}

# --- Set up Python environment ---
cat("Setting up Python environment...\n")
library(reticulate)
use_condaenv("reticulate-gpu2", required = TRUE)

torch <- import("torch")
timm <- import("timm")
transformers <- import("transformers")
convert <- import_from_path("convert", "py")

# These imports are required before torch$load() because the checkpoint
# was saved with pickle references to PlantCLEF2022 model classes
models_vit <- import_from_path("models_vit", "PlantCLEF2022")
pos_embed <- import_from_path("pos_embed", "PlantCLEF2022/util")

# --- Load timm checkpoint ---
cat("Loading timm checkpoint: ", args$checkpoint, "\n")
vit <- torch$load(args$checkpoint)

# --- Create HuggingFace ViT config ---
cat("Creating HuggingFace ViT config (ViT-L/16, num_labels=", args$num_labels, ")...\n")
config <- transformers$models$vit$ViTConfig()
config$hidden_size <- 1024L
config$intermediate_size <- 4096L
config$num_hidden_layers <- 24L
config$num_attention_heads <- 16L
config$num_labels <- as.integer(args$num_labels)

vit_tr <- transformers$models$vit$ViTForImageClassification(config)

# --- Convert state dict ---
cat("Converting state dict from timm to HuggingFace format...\n")
state_dict <- vit$state_dict()
new_state_dict <- convert$convert_state_dict(state_dict, config)
names(new_state_dict) <- gsub("head.", "classifier.", names(new_state_dict), fixed = TRUE)

vit_tr$load_state_dict(new_state_dict)
cat("State dict loaded successfully.\n")

# --- Validate conversion with random tensors ---
cat("Validating conversion with random test images...\n")
vit$eval()
vit_tr$eval()

# Move both models to same device
device <- if (torch$cuda$is_available()) "cuda" else "cpu"
vit <- vit$to(device)
vit_tr <- vit_tr$to(device)

torch$no_grad()
test_input <- torch$randn(c(4L, 3L, 224L, 224L))$to(device)
test_orig <- vit(test_input)
test_new <- vit_tr(test_input)

orig_np <- as.matrix(test_orig$cpu()$detach()$numpy())
new_np <- as.matrix(test_new$logits$cpu()$detach()$numpy())

comparison <- all.equal(orig_np, new_np)
if (is.character(comparison)) {
  cat("Conversion validation: ", comparison, "\n")
  # Check if relative difference is acceptably small
  max_diff <- max(abs(orig_np - new_np))
  cat("Max absolute difference: ", max_diff, "\n")
  if (max_diff > 1e-3) {
    stop("Conversion validation FAILED: outputs differ too much (max diff = ", max_diff, ")")
  }
  cat("Difference is within acceptable tolerance.\n\n")
} else {
  cat("Conversion validation: EXACT match.\n\n")
}

# --- Upload model ---
if (args$dry_run) {
  cat("[DRY RUN] Would push model to: ", args$hf_repo, "\n")
} else {
  commit_msg <- if (!is.null(args$version)) {
    paste0("Upload PhenoVision ", args$version)
  } else {
    "Upload PhenoVision model"
  }
  cat("Pushing model to ", args$hf_repo, "...\n")
  vit_tr$push_to_hub(args$hf_repo, commit_message = commit_msg)
  cat("Model uploaded successfully.\n\n")
}

# --- Upload image preprocessor config ---
# Without a preprocessor_config.json on the Hub, `AutoImageProcessor.from_pretrained`
# fails with "OSError: Can't load image processor". This reproduces the timm eval
# transform used for training/inference (datasets.py build_transform, is_train=False):
# resize shortest edge to 256 (bicubic) -> center-crop 224 -> rescale 1/255 ->
# ImageNet-default normalization. Same preprocessing for the reproductive (2-label)
# and leaves (3-label) models, so this runs regardless of --num-labels.
if (args$dry_run) {
  cat("[DRY RUN] Would push preprocessor_config.json (ConvNextImageProcessor) to ", args$hf_repo, "\n")
} else {
  cat("Pushing image preprocessor config to ", args$hf_repo, "...\n")
  processor <- transformers$ConvNextImageProcessor(
    size         = reticulate::dict(shortest_edge = 224L),
    crop_pct     = 0.875,
    resample     = 3L,   # PIL bicubic
    do_rescale   = TRUE,
    do_normalize = TRUE,
    image_mean   = c(0.485, 0.456, 0.406),
    image_std    = c(0.229, 0.224, 0.225)
  )
  processor$push_to_hub(args$hf_repo, commit_message = "Add/update preprocessor_config.json")
  cat("Preprocessor config uploaded.\n\n")
}

# --- Upload companion files ---
if (length(companion_files) > 0) {
  if (args$dry_run) {
    cat("[DRY RUN] Would upload companion files:\n")
    for (name in names(companion_files)) {
      cat("  ", basename(companion_files[[name]]), " -> ", args$hf_repo, "\n")
    }
  } else {
    hf_api <- import("huggingface_hub")$HfApi()
    cat("Uploading companion files...\n")
    for (name in names(companion_files)) {
      local_path <- companion_files[[name]]
      repo_filename <- basename(local_path)
      cat("  Uploading ", repo_filename, "...\n")
      hf_api$upload_file(
        path_or_fileobj = local_path,
        path_in_repo = repo_filename,
        repo_id = args$hf_repo,
        commit_message = paste0("Add ", name, if (!is.null(args$version)) paste0(" (", args$version, ")") else "")
      )
    }
    cat("Companion files uploaded.\n\n")
  }
}

# --- Next steps ---
cat(strrep("=", 70), "\n")
cat("NEXT STEPS\n")
cat(strrep("=", 70), "\n\n")
cat("1. Go to https://huggingface.co/", args$hf_repo, "/settings\n", sep = "")
cat("2. Click 'Create DOI' to mint a DataCite DOI\n")
cat("3. Update model_registry.yaml with the new DOI:\n")
if (!is.null(args$version)) {
  cat("   - Set doi: \"<new-DOI>\" for version ", args$version, "\n")
}
cat("4. Test inference: Rscript run_pipeline.R --pipeline=inference\n\n")
cat(strrep("=", 70), "\n")
