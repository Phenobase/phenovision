#!/usr/bin/env Rscript
#
# PhenoVision Pipeline Runner
#
# A unified interface for running any of the PhenoVision targets pipelines.
# Can run locally or submit to SLURM.
#
# Usage:
#   Rscript run_pipeline.R --pipeline=inference
#   Rscript run_pipeline.R --pipeline=train_repro --submit
#   Rscript run_pipeline.R --pipeline=train_leaf --workers=8
#
# Author: Russell Dinnage & Claude Code
# Date: 2025-01-06

# =============================================================================
# Parse Command Line Arguments
# =============================================================================

library(argparse)

parser <- ArgumentParser(
  description = "Run PhenoVision targets pipelines",
  epilog = paste(
    "Examples:\n",
    "  # Run inference locally with 4 workers\n",
    "  Rscript run_pipeline.R --pipeline=inference --workers=4\n\n",
    "  # Submit training to SLURM\n",
    "  Rscript run_pipeline.R --pipeline=train_repro --submit\n\n",
    "  # Run download pipeline (sequential)\n",
    "  Rscript run_pipeline.R --pipeline=download_annots --workers=1\n",
    sep = ""
  ),
  formatter_class = "argparse.RawDescriptionHelpFormatter"
)

parser$add_argument(
  "--pipeline",
  type = "character",
  required = TRUE,
  help = "Which pipeline to run",
  choices = c("download_annots", "train_repro", "train_leaf", "inference")
)

parser$add_argument(
  "--submit",
  action = "store_true",
  default = FALSE,
  help = "Submit to SLURM instead of running locally"
)

parser$add_argument(
  "--workers",
  type = "integer",
  default = 4,
  help = "Number of parallel workers for targets (local run only)"
)

parser$add_argument(
  "--debug",
  action = "store_true",
  default = FALSE,
  help = "Run in debug mode (sequential, stop on error)"
)

parser$add_argument(
  "--dry-run",
  action = "store_true",
  default = FALSE,
  help = "Show what would be run without actually running it"
)

args <- parser$parse_args()

# =============================================================================
# Determine Targets File
# =============================================================================

targets_file <- paste0("_targets_", args$pipeline, ".R")

if (!file.exists(targets_file)) {
  stop("Targets file not found: ", targets_file, "\n",
       "Available pipelines: download_annots, train_repro, train_leaf, inference")
}

# =============================================================================
# SLURM Submission
# =============================================================================

if (args$submit) {
  # Construct SLURM script name
  slurm_script <- paste0("scripts/submit_", args$pipeline, ".sh")

  if (!file.exists(slurm_script)) {
    stop("SLURM script not found: ", slurm_script)
  }

  cat("========================================\n")
  cat("Submitting pipeline to SLURM\n")
  cat("========================================\n")
  cat("Pipeline:      ", args$pipeline, "\n")
  cat("Targets file:  ", targets_file, "\n")
  cat("SLURM script:  ", slurm_script, "\n")
  cat("========================================\n\n")

  if (args$dry_run) {
    cat("DRY RUN: Would execute:\n")
    cat("  sbatch", slurm_script, "\n")
    quit(status = 0)
  }

  # Submit the job
  cmd <- paste("sbatch", slurm_script)
  result <- system(cmd, intern = TRUE)

  cat(result, "\n")
  cat("\nJob submitted successfully!\n")
  cat("Check status with: squeue -u $USER\n")
  cat("Check logs in: logs/\n")

  quit(status = 0)
}

# =============================================================================
# Local Execution
# =============================================================================

cat("========================================\n")
cat("Running pipeline locally\n")
cat("========================================\n")
cat("Pipeline:      ", args$pipeline, "\n")
cat("Targets file:  ", targets_file, "\n")
cat("Workers:       ", args$workers, "\n")
cat("Debug mode:    ", args$debug, "\n")
cat("Dry run:       ", args$dry_run, "\n")
cat("========================================\n\n")

if (args$dry_run) {
  cat("DRY RUN: Would execute:\n")
  cat("  targets::tar_make(script = '", targets_file, "', callr_workers = ", args$workers, ")\n", sep = "")
  quit(status = 0)
}

# Load targets
library(targets)

# Set working directory to project root (where this script lives)
script_dir <- dirname(normalizePath(commandArgs(trailingOnly = FALSE)[4]))
if (script_dir != "") {
  setwd(script_dir)
}

cat("Working directory: ", getwd(), "\n\n")

# Run the pipeline
start_time <- Sys.time()

tryCatch({
  if (args$debug) {
    # Debug mode: sequential, stop on error
    cat("Running in DEBUG mode (sequential execution)\n\n")
    tar_make(
      script = targets_file,
      callr_function = NULL  # Run in current session for debugging
    )
  } else {
    # Normal mode: parallel
    tar_make(
      script = targets_file,
      callr_workers = args$workers
    )
  }

  end_time <- Sys.time()
  duration <- difftime(end_time, start_time, units = "mins")

  cat("\n========================================\n")
  cat("Pipeline completed successfully!\n")
  cat("========================================\n")
  cat("Duration: ", round(duration, 2), " minutes\n")
  cat("Completed at: ", format(end_time), "\n")
  cat("========================================\n")

}, error = function(e) {
  end_time <- Sys.time()
  duration <- difftime(end_time, start_time, units = "mins")

  cat("\n========================================\n")
  cat("Pipeline FAILED\n")
  cat("========================================\n")
  cat("Error: ", e$message, "\n")
  cat("Duration before error: ", round(duration, 2), " minutes\n")
  cat("========================================\n\n")

  if (args$debug) {
    cat("Running in debug mode. Workspace saved for inspection.\n")
    cat("Check targets metadata with:\n")
    cat("  targets::tar_meta(script = '", targets_file, "')\n", sep = "")
    cat("  targets::tar_workspace(name_of_failed_target)\n")
  }

  quit(status = 1)
})

cat("\nTo visualize the pipeline:\n")
cat("  targets::tar_visnetwork(script = '", targets_file, "')\n\n", sep = "")
cat("To check status:\n")
cat("  targets::tar_progress(script = '", targets_file, "')\n\n", sep = "")
