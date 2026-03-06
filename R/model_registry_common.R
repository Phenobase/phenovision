#' Model Version Registry Functions
#'
#' Functions for looking up model metadata by version. Uses model_registry.yaml
#' to map version -> DOI, thresholds path, family stats path, etc.
#'
#' @examples
#' \dontrun{
#' # Get all info for a model version
#' info <- get_model_info("v1.1.0", "reproductive")
#' info$doi
#' info$thresholds_path
#' info$family_stats_path
#'
#' # Load thresholds in the format expected by threshold_annotations()
#' thresholds <- load_model_thresholds("v1.1.0", "reproductive")
#' # Returns list(fl = c(threshold, buffer_lower, buffer_upper), ...)
#' }

#' Get Model Information from Registry
#'
#' Looks up a model version in model_registry.yaml and returns paths and metadata.
#'
#' @param version Character. Model version (e.g., "v1.1.0")
#' @param model_type Character. Either "reproductive" or "leaves"
#' @param registry_path Character. Path to registry YAML file.
#'   Default: "model_registry.yaml" (project root)
#'
#' @return A list with:
#'   - `version`: The version string
#'   - `model_type`: The model type
#'   - `doi`: DOI string (or NULL if not yet uploaded)
#'   - `hf_repo`: HuggingFace repository name
#'   - `num_labels`: Number of output labels (2 for reproductive, 3 for leaves)
#'   - `output_dir`: Local output directory path
#'   - `thresholds_path`: Full path to thresholds file
#'   - `thresholds_format`: Format of thresholds file ("csv" or "rds")
#'   - `family_stats_path`: Full path to family stats file
#'
#' @export
get_model_info <- function(version,
                           model_type = c("reproductive", "leaves"),
                           registry_path = "model_registry.yaml") {
  model_type <- match.arg(model_type)

  if (!file.exists(registry_path)) {
    stop("Model registry not found: ", registry_path,
         "\nExpected at project root: model_registry.yaml")
  }

  registry <- yaml::read_yaml(registry_path)

  if (!model_type %in% names(registry)) {
    stop("Unknown model type: ", model_type,
         "\nAvailable types: ", paste(names(registry), collapse = ", "))
  }

  type_registry <- registry[[model_type]]
  versions <- type_registry$versions

  if (!version %in% names(versions)) {
    stop("Unknown version '", version, "' for model type '", model_type, "'",
         "\nAvailable versions: ", paste(names(versions), collapse = ", "))
  }

  ver_info <- versions[[version]]
  output_dir <- ver_info$output_dir

  # num_labels: version-level overrides type-level
  num_labels <- if (!is.null(ver_info$num_labels)) ver_info$num_labels else type_registry$num_labels

  # Build full paths for artifacts
  thresholds_path <- if (!is.null(ver_info$thresholds_file)) {
    file.path(output_dir, ver_info$thresholds_file)
  } else {
    NULL
  }

  family_stats_path <- if (!is.null(ver_info$family_stats_file)) {
    file.path(output_dir, ver_info$family_stats_file)
  } else {
    NULL
  }

  list(
    version = version,
    model_type = model_type,
    doi = ver_info$doi,
    hf_repo = ver_info$hf_repo,
    data_date = ver_info$data_date,
    num_labels = num_labels,
    output_dir = output_dir,
    thresholds_path = thresholds_path,
    thresholds_format = ver_info$thresholds_format,
    family_stats_path = family_stats_path
  )
}


#' Load Model Thresholds
#'
#' Loads threshold parameters from disk in the format expected by
#' threshold_annotations(). Handles both CSV and RDS formats.
#'
#' @param version Character. Model version (e.g., "v1.1.0")
#' @param model_type Character. Either "reproductive" or "leaves"
#' @param registry_path Character. Path to registry YAML file.
#'
#' @return A named list of threshold vectors:
#'   - For reproductive: list(fl = c(threshold, buffer_lower, buffer_upper),
#'                            fr = c(threshold, buffer_lower, buffer_upper))
#'   - For leaves: list(gr = c(...), cl = c(...), bb = c(...))
#'
#' @export
load_model_thresholds <- function(version,
                                  model_type = c("reproductive", "leaves"),
                                  registry_path = "model_registry.yaml") {
  model_type <- match.arg(model_type)
  info <- get_model_info(version, model_type, registry_path)

  if (is.null(info$thresholds_path)) {
    stop("No thresholds file defined for ", model_type, " ", version)
  }

  if (!file.exists(info$thresholds_path)) {
    stop("Thresholds file not found: ", info$thresholds_path)
  }

  if (info$thresholds_format == "rds") {
    # RDS files are already in list format
    thresholds <- readRDS(info$thresholds_path)
  } else if (info$thresholds_format == "csv") {
    # CSV files need to be parsed and converted to list format
    thresholds_df <- readr::read_csv(info$thresholds_path, show_col_types = FALSE)

    if (model_type == "reproductive") {
      fl_row <- thresholds_df |> dplyr::filter(class == "flower")
      fr_row <- thresholds_df |> dplyr::filter(class == "fruit")
      thresholds <- list(
        fl = c(fl_row$threshold, fl_row$buffer_lower, fl_row$buffer_upper),
        fr = c(fr_row$threshold, fr_row$buffer_lower, fr_row$buffer_upper)
      )
    } else if (model_type == "leaves") {
      gr_row <- thresholds_df |> dplyr::filter(class == "green")
      cl_row <- thresholds_df |> dplyr::filter(class == "colored")
      bb_row <- thresholds_df |> dplyr::filter(class == "breaking_buds")
      thresholds <- list(
        gr = c(gr_row$threshold, gr_row$buffer_lower, gr_row$buffer_upper),
        cl = c(cl_row$threshold, cl_row$buffer_lower, cl_row$buffer_upper),
        bb = c(bb_row$threshold, bb_row$buffer_lower, bb_row$buffer_upper)
      )
    }
  } else {
    stop("Unknown thresholds format: ", info$thresholds_format)
  }

  thresholds
}


#' Get Current Model Version
#'
#' Returns the version marked as "current" in the registry for a given model type.
#'
#' @param model_type Character. Either "reproductive" or "leaves"
#' @param registry_path Character. Path to registry YAML file.
#'
#' @return Character. The current version string.
#'
#' @export
get_current_version <- function(model_type = c("reproductive", "leaves"),
                                registry_path = "model_registry.yaml") {
  model_type <- match.arg(model_type)
  registry <- yaml::read_yaml(registry_path)
  registry[[model_type]]$current
}


#' Generate HuggingFace Upload Command
#'
#' Constructs the exact Rscript command to upload a trained model to HuggingFace Hub.
#' This command is meant to be reviewed and run manually by the user.
#'
#' @param checkpoint Character. Path to the model checkpoint (.pt file)
#' @param model_type Character. Either "reproductive" or "leaves"
#' @param version Character. Model version string (e.g., "v1.1.0")
#' @param thresholds Character. Path to thresholds file (optional)
#' @param family_stats Character. Path to family stats file (optional)
#' @param report Character. Path to training report (optional)
#' @param registry_path Character. Path to registry YAML file.
#'
#' @return Character string with the full Rscript command
#'
#' @export
generate_hf_upload_command <- function(checkpoint,
                                       model_type = c("reproductive", "leaves"),
                                       version,
                                       thresholds = NULL,
                                       family_stats = NULL,
                                       report = NULL,
                                       registry_path = "model_registry.yaml") {
  model_type <- match.arg(model_type)
  info <- get_model_info(version, model_type, registry_path)

  # Normalize all paths to relative (strip project root prefix)
  make_relative <- function(path) {
    if (is.null(path)) return(NULL)
    prefix <- paste0(getwd(), "/")
    sub(prefix, "", path, fixed = TRUE)
  }

  checkpoint <- make_relative(checkpoint)
  thresholds <- make_relative(thresholds)
  family_stats <- make_relative(family_stats)
  report <- make_relative(report)

  cmd <- paste0(
    "Rscript R/push_to_hf_hub.R \\\n",
    "  --checkpoint '", checkpoint, "' \\\n",
    "  --hf-repo '", info$hf_repo, "' \\\n",
    "  --num-labels ", info$num_labels, " \\\n",
    "  --version '", version, "'"
  )

  if (!is.null(thresholds)) {
    cmd <- paste0(cmd, " \\\n  --thresholds '", thresholds, "'")
  }
  if (!is.null(family_stats)) {
    cmd <- paste0(cmd, " \\\n  --family-stats '", family_stats, "'")
  }
  if (!is.null(report)) {
    cmd <- paste0(cmd, " \\\n  --report '", report, "'")
  }

  cmd
}
