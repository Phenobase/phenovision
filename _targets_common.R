# Common Configuration for PhenoVision Targets Pipelines
#
# This file contains shared configuration and setup code used by all
# pipeline-specific _targets_*.R files.
#
# Usage: source("_targets_common.R") at the top of each pipeline file

# =============================================================================
# Library Loading
# =============================================================================

library(targets)
library(tarchetypes)
library(future)
library(future.callr)

# =============================================================================
# Package Conflict Resolution
# =============================================================================

# Prefer lubridate versions over data.table for date functions
library(conflicted)
conflicts_prefer(
  lubridate::year,
  lubridate::month,
  lubridate::week,
  lubridate::mday,
  lubridate::wday,
  lubridate::hour,
  lubridate::minute,
  lubridate::second,
  .quiet = TRUE
)

# =============================================================================
# Project Paths
# =============================================================================

# Root directories
project_root <- here::here()

# Where the bulk iNaturalist store lives.
#
# The default is RELATIVE to the project root. On every cluster we run on, `data/` is a
# symlink to the real store, so this resolves without any per-site configuration:
#   Vulcan      data -> /project/aip-dinnage/phenovision/data
#   HiPerGator  data -> /blue/guralnick/share/... (the historical symlink farm)
#
# Keeping the canonical value relative is deliberate: `targets` hashes these strings into
# its cache keys, so an absolute path would invalidate every downstream target merely by
# mounting the same tree somewhere else. It also removes the split this file used to have,
# where the download pipeline used relative paths and the inference pipeline hardcoded
# /blue/guralnick absolutes for the same directories.
#
# Set PHENOVISION_DATA_ROOT to an absolute path for a site that cannot symlink.
# NOTE: relative paths require the working directory to be the project root. `targets`
# guarantees that for pipeline runs; a script run by hand must `cd` there first.
data_root <- Sys.getenv("PHENOVISION_DATA_ROOT", unset = "data/phenobase_inat_data")

# Data paths
paths <- list(
  # Input data
  data_root = data_root,
  images_root = file.path(data_root, "images/medium"),
  images_shards = file.path(data_root, "images/shards"),
  metadata_root = file.path(data_root, "metadata"),
  metadata_photos = file.path(data_root, "metadata/angio_photos"),
  metadata_annotation = file.path(data_root, "metadata/phenobase_dwca_annotation"),

  # Raw iNaturalist Open Data snapshot. This is a SHARED store: one dump serves every
  # project on aip-dinnage (phenovision, iNat-leps, ...), so nobody re-downloads 36 GB.
  # It lives at /project/aip-dinnage/shared/inat-open-data/<YYYY-MM-DD>/ with a `latest`
  # symlink, and is reached here through data/phenobase_inat_data/opendata.
  opendata = file.path(data_root, "opendata"),
  image_download_status = file.path(data_root, "metadata/img_download_status"),

  # WebDataset shard manifests (Vulcan pull)
  manifests = "manifests",
  shard_lists = "manifests/shard_lists",

  # Local data
  data_inat = "data/inat",
  data_leaves = "data/leaves",

  # Models
  models_dir = "models",
  pretrained_plantclef = "models/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth",

  # Outputs
  output_root = "output",
  output_repro = "output/reproductive",
  output_leaves = "output/leaves",
  output_annotations = "output/annotations",

  # Guild AI
  guild_home = ".guild"
)

# Create output directories if they don't exist
for (path in paths[grepl("^output_", names(paths))]) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
}

# =============================================================================
# Python interpreter for reticulate
# =============================================================================
# Which Python to use is SITE configuration, not project code, so it must not be
# hardcoded in a tracked file (it used to live in a committed .Renviron pinning the
# HiPerGator conda env, which made a fresh clone anywhere else fail at the first
# reticulate call).
#
# Priority: an explicit RETICULATE_PYTHON always wins -- from the environment, from a
# local (gitignored) .Renviron, or from a submit script. Only if it is unset do we fall
# back to the first interpreter that actually exists on this machine.
if (!nzchar(Sys.getenv("RETICULATE_PYTHON"))) {
  reticulate_candidates <- c(
    path.expand("~/venvs/phenovision/bin/python"),                        # Vulcan (Alliance)
    "/blue/guralnick/r.dinnage/.conda/envs/reticulate-gpu2/bin/python"    # UF HiPerGator
  )
  reticulate_found <- reticulate_candidates[file.exists(reticulate_candidates)]
  if (length(reticulate_found) > 0) {
    Sys.setenv(RETICULATE_PYTHON = reticulate_found[1])
  } else {
    warning(
      "RETICULATE_PYTHON is unset and no known interpreter was found. Tried:\n  ",
      paste(reticulate_candidates, collapse = "\n  "),
      "\nSet RETICULATE_PYTHON (see .Renviron.example) before running anything that ",
      "calls Python.",
      call. = FALSE
    )
  }
}

# =============================================================================
# Configuration Parameters
# =============================================================================

config <- list(
  # Inference
  batch_size_inference = 1280,  # Number of images per batch for inference
  num_workers_data = 4,         # Number of data loader workers

  # Training
  batch_size_training = 384,    # Batch size for training
  num_epochs_default = 100,     # Default number of epochs

  # Parallelization
  num_targets_workers = 10,     # Number of parallel targets workers

  # Model versions (DOIs)
  model_doi_repro = "10.57967/hf/2763",    # Current reproductive model
  model_doi_leaves = "10.57967/hf/5785"     # Current leaf model
)

# =============================================================================
# Python/Reticulate Setup
# =============================================================================

# Set up Python environment
setup_python <- function() {
  library(reticulate)

  # Use specific conda environment
  use_condaenv("reticulate-gpu2", required = FALSE)

  # Import required Python modules
  py_modules <- c("torch", "torchvision", "PIL", "numpy")

  for (module in py_modules) {
    if (!py_module_available(module)) {
      warning("Python module not available: ", module)
    }
  }

  message("Python environment configured")
}

# =============================================================================
# Function Sourcing Utilities
# =============================================================================

#' Source all R files matching a pattern
#'
#' @param pattern Regular expression pattern for file names
#' @param path Directory to search (default: "R")
#' @param verbose Print file names as they're sourced
source_pattern <- function(pattern, path = "R", verbose = TRUE) {
  files <- list.files(path, pattern = pattern, full.names = TRUE)

  if (length(files) == 0) {
    warning("No files found matching pattern: ", pattern, " in ", path)
    return(invisible(NULL))
  }

  if (verbose) {
    message("Sourcing ", length(files), " files matching '", pattern, "':")
  }

  for (file in files) {
    if (verbose) {
      message("  - ", basename(file))
    }
    source(file)
  }

  invisible(files)
}

#' Source common functions (used by all pipelines)
source_common <- function(verbose = TRUE) {
  source_pattern("_common\\.R$", verbose = verbose)
}

#' Source inference-specific functions
source_inference <- function(verbose = TRUE) {
  source_pattern("_inference\\.R$", verbose = verbose)
}

#' Source training-specific functions
source_training <- function(verbose = TRUE) {
  source_pattern("_training\\.R$", verbose = verbose)
}

#' Source download-specific functions
source_download <- function(verbose = TRUE) {
  source_pattern("_download\\.R$", verbose = verbose)
}

# =============================================================================
# Targets Options
# =============================================================================

#' Configure targets for parallel execution
#'
#' Worker count priority:
#' 1. Explicit function argument (not NULL) - allows training to force workers=0
#' 2. SLURM_CPUS_PER_TASK env var - automatically set by SLURM from --cpus-per-task
#' 3. TARGETS_WORKERS env var - set by run_pipeline.R --workers=N for local runs
#' 4. config$num_targets_workers - default (10)
setup_targets_parallel <- function(workers = NULL, gpu_workers = NULL) {
  # Determine worker count based on priority
  if (!is.null(workers)) {
    # Explicit argument takes priority (allows training to force workers=0)
    message("Using explicit workers argument: ", workers, " workers")
  } else {
    slurm_cpus <- Sys.getenv("SLURM_CPUS_PER_TASK", unset = "")
    targets_workers <- Sys.getenv("TARGETS_WORKERS", unset = "")

    if (nchar(slurm_cpus) > 0) {
      workers <- as.integer(slurm_cpus)
      message("Using SLURM_CPUS_PER_TASK: ", workers, " workers")
    } else if (nchar(targets_workers) > 0) {
      workers <- as.integer(targets_workers)
      message("Using TARGETS_WORKERS env var: ", workers, " workers")
    } else {
      workers <- config$num_targets_workers
      message("Using config default: ", workers, " workers")
    }
  }

  tar_option_set(
    packages = c(
      "arrow", "dplyr", "tidyr", "purrr", "readr", "stringr",
      "reticulate", "here", "fs", "cli",
      "ggplot2", "patchwork",  # Needed for plotting functions
      "guildai",  # Needed for best epoch selection from training runs
      "filelock"  # Needed for GPU lock acquisition in annotate_batch
    ),
    format = "rds",
    error = "continue",  # Continue on errors
    memory = "transient",  # Don't keep objects in memory
    garbage_collection = TRUE,
    workspace_on_error = TRUE,  # Save workspace on error for debugging
  )
  if(workers > 0) {
    if (!is.null(gpu_workers) && gpu_workers > 0) {
      # Create controller group: separate GPU and CPU pools
      # GPU targets run sequentially to avoid CUDA OOM on single GPU
      cpu_workers <- max(1, workers - gpu_workers)
      tar_option_set(
        controller = crew::crew_controller_group(
          crew::crew_controller_local(name = "default", workers = cpu_workers),
          crew::crew_controller_local(name = "gpu", workers = gpu_workers)
        )
      )
      message("Targets configured for ", cpu_workers, " CPU + ", gpu_workers, " GPU workers")
    } else {
      tar_option_set(controller = crew::crew_controller_local(workers = workers))
      message("Targets configured for ", workers, " parallel workers")
    }
  } else {
    tar_option_set(deployment = "main")
    message("Targets configured for sequential execution (main process)")
  }
}

#' Configure targets for sequential execution (debugging)
setup_targets_sequential <- function() {
  tar_option_set(
    packages = c(
      "arrow", "dplyr", "tidyr", "purrr", "readr", "stringr",
      "reticulate", "here", "fs", "cli",
      "ggplot2", "patchwork",  # Needed for plotting functions
      "guildai",  # Needed for best epoch selection from training runs
      "filelock"  # Needed for GPU lock acquisition in annotate_batch
    ),
    format = "rds",
    error = "stop",  # Stop on first error
    memory = "persistent",  # Keep objects in memory for debugging
    garbage_collection = FALSE,
    workspace_on_error = TRUE
  )

  message("Targets configured for sequential execution (debugging mode)")
}

# =============================================================================
# Utility Functions
# =============================================================================

#' Get timestamp string for file names
timestamp_str <- function() {
  format(Sys.time(), "%Y%m%d_%H%M%S")
}

#' Get date string for file names
date_str <- function() {
  format(Sys.Date(), "%Y%m%d")
}

#' Print pipeline header
print_pipeline_header <- function(pipeline_name) {
  cli::cli_rule(left = pipeline_name, right = date_str())
  cli::cli_alert_info("Working directory: {getwd()}")
  cli::cli_alert_info("Project root: {project_root}")
  cli::cli_alert_info("Data root: {data_root}")
}

#' Run a PhenoVision Pipeline
#'
#' Convenience function to run pipelines from an R console.
#' Can run locally or submit to SLURM.
#'
#' @param pipeline Character. Pipeline name: "inference", "train_repro",
#'   "train_leaf", or "download_annots"
#' @param submit Logical. If TRUE, submit to SLURM instead of running locally.
#'   Default FALSE.
#' @param workers Integer. Number of parallel workers (ignored if submit=TRUE).
#'   Default uses config value (10).
#' @param debug Logical. If TRUE, run sequentially in current session for
#'   debugging. Default FALSE.
#' @param dry_run Logical. If TRUE, show what would be run without executing.
#'   Default FALSE.
#'
#' @examples
#' \dontrun{
#' # Run inference locally with default workers
#' run_pipeline("inference")
#'
#' # Submit training to SLURM
#' run_pipeline("train_repro", submit = TRUE)
#'
#' # Run with specific worker count
#' run_pipeline("inference", workers = 4)
#'
#' # Debug mode (sequential, in current session)
#' run_pipeline("train_repro", debug = TRUE)
#' }
run_pipeline <- function(pipeline,
                         submit = FALSE,
                         workers = NULL,
                         debug = FALSE,
                         dry_run = FALSE) {

  # Validate pipeline name

  valid_pipelines <- c("inference", "train_repro", "train_leaf", "download_annots")
  if (!pipeline %in% valid_pipelines) {
    stop("Invalid pipeline: '", pipeline, "'\n",
         "Valid options: ", paste(valid_pipelines, collapse = ", "))
  }

  targets_file <- paste0("_targets_", pipeline, ".R")
  slurm_script <- paste0("scripts/submit_", pipeline, ".sh")

  # Resolve the targets project name and store directory from _targets.yaml
  project_names <- c(
    inference = "main",
    train_repro = "train_repro",
    train_leaf = "train_leaf",
    download_annots = "download_annots"
  )
  project_name <- project_names[[pipeline]]
  Sys.setenv(TAR_PROJECT = project_name)
  store_dir <- targets::tar_config_get("store", project = project_name)

  # Check files exist

if (!file.exists(targets_file)) {
    stop("Targets file not found: ", targets_file)
  }

  if (submit) {
    # =========================================================================
    # SLURM Submission
    # =========================================================================
    if (!file.exists(slurm_script)) {
      stop("SLURM script not found: ", slurm_script)
    }

    cat("========================================\n")
    cat("Submitting pipeline to SLURM\n")
    cat("========================================\n")
    cat("Pipeline:      ", pipeline, "\n")
    cat("Targets file:  ", targets_file, "\n")
    cat("SLURM script:  ", slurm_script, "\n")
    cat("========================================\n\n")

    if (dry_run) {
      cat("DRY RUN: Would execute:\n")
      cat("  system('sbatch", slurm_script, "')\n")
      return(invisible(NULL))
    }

    # Submit the job
    result <- system(paste("sbatch", slurm_script), intern = TRUE)
    cat(result, "\n")
    cat("\nJob submitted successfully!\n")
    cat("Check status with: squeue -u $USER\n")
    cat("Check logs in: logs/\n")

    return(invisible(result))

  } else {
    # =========================================================================
    # Local Execution
    # =========================================================================

    # Set workers
    if (is.null(workers)) {
      workers <- config$num_targets_workers
    }

    cat("========================================\n")
    cat("Running pipeline locally\n")
    cat("========================================\n")
    cat("Pipeline:      ", pipeline, "\n")
    cat("Targets file:  ", targets_file, "\n")
    cat("Store:         ", store_dir, "\n")
    cat("Workers:       ", workers, "\n")
    cat("Debug mode:    ", debug, "\n")
    cat("========================================\n\n")

    if (dry_run) {
      cat("DRY RUN: Would execute:\n")
      cat("  Sys.setenv(TARGETS_WORKERS = ", workers, ")\n", sep = "")
      cat("  targets::tar_make(script = '", targets_file, "', store = '", store_dir, "')\n", sep = "")
      return(invisible(NULL))
    }

    # Set TARGETS_WORKERS env var
    Sys.setenv(TARGETS_WORKERS = workers)

    if (debug) {
      cat("Running in DEBUG mode (sequential execution)\n\n")
      Sys.setenv(TARGETS_WORKERS = 0)
      targets::tar_make(
        script = targets_file,
        store = store_dir,
        callr_function = NULL  # Run in current session for debugging
      )
    } else {
      targets::tar_make(script = targets_file, store = store_dir)
    }

    cat("\nTo visualize the pipeline:\n")
    cat("  targets::tar_visnetwork(script = '", targets_file, "', store = '", store_dir, "')\n\n", sep = "")

    return(invisible(TRUE))
  }
}

# =============================================================================
# Initialization Message
# =============================================================================

message("================================")
message("PhenoVision Targets Common Setup")
message("================================")
message("Loaded at: ", Sys.time())
message("Project root: ", project_root)
message("Data root: ", data_root)
message("")
message("Available functions:")
message("  - run_pipeline(name, submit=FALSE)  # Run or submit pipeline")
message("  - source_common()")
message("  - source_inference()")
message("  - source_training()")
message("  - source_download()")
message("  - setup_targets_parallel()")
message("  - setup_targets_sequential()")
message("  - setup_python()")
message("================================")
