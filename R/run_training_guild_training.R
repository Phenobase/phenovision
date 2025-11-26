#' Run Training Script via Guild AI
#'
#' Wrapper function to execute training scripts through Guild AI for
#' automatic experiment tracking and parameter management.
#'
#' Based on implementation from bioclim_intrinsic_dimension project.
#' See: https://github.com/rdinnager/bioclim_intrinsic_dimension
#'
#' @param script_path Character. Path to the R script to run (e.g., "R/vit_model_train.R")
#' @param flags Named list. Parameters to override in the script. These will
#'   override any top-level scalar assignments in the script.
#'   Example: list(epochs = 100, lr = 0.0005, pretrained_model = "plantclef")
#' @param label Character. Label for this run (used for tracking). If NULL,
#'   generates timestamp-based label.
#' @param tag Character. Tag for categorizing runs (e.g., "training", "testing").
#' @param comment Character. Optional comment describing this run.
#' @param guild_home Character. Path to Guild home directory for storing runs.
#'   Default: ".guild" in project root.
#' @param capture_output Logical. If TRUE, captures stdout/stderr from the run.
#' @param test_mode Logical. If TRUE, stages the run without executing to test
#'   file resolution (what gets copied vs symlinked). Useful for verifying
#'   guild.yml configuration before running expensive training.
#' @param operation_name Character. Guild operation name from guild.yml (e.g.,
#'   "phenovision:vit_model_train"). If provided, uses guild.yml configuration
#'   instead of direct script execution. This enables better integration with
#'   guild.yml features like sourcecode control and requires management.
#'
#' @return A list containing:
#'   \item{run_info}{Full run information from runs_info()}
#'   \item{run_dir}{Path to the run directory}
#'   \item{run_id}{Guild run ID}
#'   \item{label}{Run label}
#'   \item{tag}{Run tag}
#'   \item{flags}{Flags used for this run}
#'   \item{status}{Run status (completed, error, etc.)}
#'
#' @details
#' Guild AI automatically detects "flags" (parameters) in your R script.
#' Any scalar literal assigned at the top level becomes a flag. For example:
#'
#' ```r
#' # In your training script:
#' epochs <- 100        # This is a flag
#' lr <- 0.001          # This is a flag
#' pretrained <- "mae"  # This is a flag
#' ```
#'
#' When you call run_training_guild() with flags, they override these defaults:
#'
#' ```r
#' run_training_guild(
#'   "R/vit_model_train.R",
#'   flags = list(epochs = 50, lr = 0.0005)
#' )
#' # Now the script runs with epochs=50, lr=0.0005, pretrained="mae"
#' ```
#'
#' **IMPORTANT:** This function includes a workaround for a Guild AI discovery
#' bug. The workaround may not be needed in future versions of guildai R package.
#'
#' @examples
#' \dontrun{
#' # Run training with custom parameters
#' result <- run_training_guild(
#'   script_path = "R/vit_model_train.R",
#'   flags = list(
#'     pretrained_model = "plantclef",
#'     epochs = 100,
#'     batch_size = 384,
#'     lr = 0.0005
#'   ),
#'   label = paste0("repro_training_", Sys.Date()),
#'   tag = "reproductive_structures",
#'   comment = "Training with PlantCLEF pretrained model"
#' )
#'
#' # Access run information
#' cat("Run directory:", result$run_dir, "\n")
#' cat("Run status:", result$status, "\n")
#'
#' # Get checkpoint path (example - adjust based on your script)
#' checkpoint_path <- file.path(result$run_dir, "output", "checkpoint.pt")
#' }
#'
#' @export
run_training_guild <- function(script_path,
                               flags = list(),
                               label = NULL,
                               tag = NULL,
                               comment = "",
                               guild_home = ".guild",
                               capture_output = TRUE,
                               test_mode = FALSE,
                               operation_name = NULL) {

  # Check that script exists (only if not using operation_name)
  if (is.null(operation_name) && !file.exists(script_path)) {
    stop("Script not found: ", script_path)
  }

  # Determine what to pass to guild_run
  if (!is.null(operation_name)) {
    # Use guild.yml operation (e.g., "phenovision:vit_model_train")
    # No need for subdirectory workaround - guild.yml handles everything
    script_path_for_guild <- operation_name
    temp_script_path <- NULL
    message("Using Guild operation from guild.yml: ", operation_name)
  } else {
    # WORKAROUND: Copy subdirectory scripts to root
    # The guildai R package has issues with scripts in subdirectories
    # This copies the script to root, runs it, then deletes the copy
    script_is_subdir <- grepl("/", script_path)
    if (script_is_subdir) {
      temp_script_name <- paste0(".guild_temp_", basename(script_path))
      temp_script_path <- temp_script_name
      file.copy(script_path, temp_script_path, overwrite = TRUE)
      message("Copied script to root directory for Guild AI compatibility: ", temp_script_path)
      script_path_for_guild <- temp_script_path
    } else {
      temp_script_path <- NULL
      script_path_for_guild <- script_path
    }

    # Ensure temp script is deleted on exit (even if error occurs)
    if (!is.null(temp_script_path)) {
      on.exit({
        if (file.exists(temp_script_path)) {
          file.remove(temp_script_path)
          message("Cleaned up temporary script: ", temp_script_path)
        }
      }, add = TRUE)
    }
  }

  # Load guildai package
  if (!requireNamespace("guildai", quietly = TRUE)) {
    stop("guildai package not installed. Install with: install.packages('guildai')")
  }
  library(guildai)

  # WORKAROUND: Fix Guild AI discovery bug
  # This may not be needed in future versions - test first!
  # See: https://github.com/rdinnager/bioclim_intrinsic_dimension/blob/master/R/target_functions.R
  tryCatch({
    assignInNamespace("find_guild", guildai:::find_r_guildai_guild, ns = "guildai")
    message("Applied Guild AI discovery workaround")
  }, error = function(e) {
    warning("Could not apply Guild AI workaround. May not be needed: ", e$message)
  })

  # Set Guild home directory
  guild_home_full <- normalizePath(guild_home, mustWork = FALSE)
  if (!dir.exists(guild_home_full)) {
    dir.create(guild_home_full, recursive = TRUE)
    message("Created Guild home directory: ", guild_home_full)
  }
  Sys.setenv(GUILD_HOME = guild_home_full)
  message("GUILD_HOME set to: ", guild_home_full)

  # Generate label if not provided
  if (is.null(label)) {
    label <- paste0("run_", format(Sys.time(), "%Y%m%d_%H%M%S"))
    message("Generated label: ", label)
  }

  # Log what we're doing
  message("\n=== ", if(test_mode) "TESTING" else "Running", " script via Guild AI ===")
  message("Script: ", script_path)
  message("Label: ", label)
  if (!is.null(tag)) message("Tag: ", tag)
  if (test_mode) message("TEST MODE: Will stage run without executing")
  message("Flags:")
  for (flag_name in names(flags)) {
    message("  ", flag_name, " = ", flags[[flag_name]])
  }
  message("===================================\n")

  # TEST MODE: Test sourcecode resolution without executing
  if (test_mode) {
    message("TEST MODE: Testing Guild sourcecode resolution...\n")

    # Use guild_run with test_sourcecode = TRUE
    # This should show what files/directories would be resolved without actually running
    message("Calling guild_run with test_sourcecode = TRUE...\n")

    tryCatch({
      result <- guild_run(
        script_path_for_guild,
        label = label,
        tag = tag,
        flags = flags,
        stage = TRUE,
        as_job = FALSE
      )

      message("\n=== Guild test_sourcecode Result ===")
      message("Result type: ", class(result))
      print(result)

      # Try to get the most recent run to examine its directory
      message("\n=== Checking Most Recent Run Directory ===")
      runs <- runs_info()
      if (nrow(runs) > 0) {
        latest_run <- runs[1, ]
        run_dir <- latest_run$dir
        message("Run directory: ", run_dir)

        if (dir.exists(run_dir)) {
          files <- list.files(run_dir, all.files = TRUE, no.. = TRUE)
          message("\nFiles/directories in run directory:")
          for (f in files) {
            full_path <- file.path(run_dir, f)
            info <- file.info(full_path)
            is_link <- Sys.readlink(full_path) != ""
            if (is_link) {
              link_target <- Sys.readlink(full_path)
              message(sprintf("  %s -> %s (SYMLINK)", f, link_target))
            } else if (info$isdir) {
              message(sprintf("  %s/ (DIRECTORY)", f))
            } else {
              message(sprintf("  %s (FILE, %s bytes)", f, info$size))
            }
          }
        } else {
          message("Run directory does not exist")
        }
      } else {
        message("No runs found")
      }

      return(list(
        test_mode = TRUE,
        script = script_path,
        label = label,
        flags = flags,
        test_sourcecode_result = result,
        run_info = if(nrow(runs) > 0) latest_run else NULL
      ))

    }, error = function(e) {
      message("ERROR during test_sourcecode: ", e$message)
      return(list(
        test_mode = TRUE,
        script = script_path,
        label = label,
        flags = flags,
        error = e$message
      ))
    })
  }

  # NORMAL MODE: Run the script with Guild AI (using temp copy if needed)
  run_start <- Sys.time()
  tryCatch({
    guild_run(script_path_for_guild,  # Use temp copy for subdirectory scripts
              label = label,
              tag = tag,
              as_job = FALSE,  # Run synchronously
              flags = flags,
              comment = comment)
    run_status <- "completed"
  }, error = function(e) {
    message("ERROR during guild_run: ", e$message)
    run_status <<- "error"
    stop(e)
  })
  run_end <- Sys.time()
  run_duration <- difftime(run_end, run_start, units = "mins")

  message("\nRun completed in ", round(run_duration, 2), " minutes")

  # Get run information
  message("Retrieving run information...")
  run_info <- runs_info(label = label)

  # Handle case where runs_info returns multiple runs with same label
  if (nrow(run_info) > 1) {
    warning("Multiple runs found with label '", label, "'. Using most recent.")
    run_info <- run_info[1, ]
  }

  # Extract useful information
  result <- list(
    run_info = run_info,
    run_dir = run_info$dir,
    run_id = run_info$id,
    label = label,
    tag = tag,
    flags = flags,
    status = run_status,
    duration_mins = as.numeric(run_duration),
    started = run_start,
    completed = run_end
  )

  message("Run directory: ", result$run_dir)
  message("Run ID: ", result$run_id)

  return(result)
}
