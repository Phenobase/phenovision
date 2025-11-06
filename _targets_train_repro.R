# PhenoVision Reproductive Model Training Pipeline
#
# This targets pipeline handles the full reproductive model training workflow:
# 1. Model training with Guild AI experiment tracking
# 2. Threshold tuning on validation set
# 3. Family-level accuracy statistics
# 4. HuggingFace model upload
# 5. DOI minting reminder
#
# Usage:
#   targets::tar_make(script = "_targets_train_repro.R")
#   Rscript run_pipeline.R --pipeline=train_repro

# =============================================================================
# Setup
# =============================================================================

# Load common configuration
source("_targets_common.R")

# Load functions
source_common()     # Common functions (model loading, etc.)
source_training()   # Training-specific functions

# Additional packages
library(targets)
library(tarchetypes)  # Modern targets syntax
library(conflicted)
conflicts_prefer(dplyr::filter)

# Configure targets for sequential execution (training is not parallelizable)
setup_targets_parallel(workers = 1)

# =============================================================================
# Pipeline (Modern tar_plan() Syntax)
# =============================================================================
#
# Note: train_config could be further optimized by separating parameters into
# individual targets for granular dependency tracking (see _targets_download_annots.R)
#
tar_plan(

  # ===========================================================================
  # Training Configuration
  # ===========================================================================

  # Training parameters - modify these to configure the training run
  tar_target(
    train_config,
    list(
      # Model initialization
      pretrained_model = "mae",  # Options: "plantclef", "imagenet", "imagenet21k", "dino", "mae", "clip"
      resume_from = NULL,  # Path to checkpoint to resume from (NULL for fresh start)

      # Training hyperparameters
      batch_size = 384L,
      blr = 5e-4,  # Base learning rate
      num_epochs = 100L,
      weight_decay = 0.05,
      layer_decay = 0.65,

      # Data
      train_csv = "data/inat/train.csv",
      val_csv = "data/inat/validation.csv",
      test_csv = "data/inat/test.csv",
      bad_images_rds = "data/inat/bad_images.rds",

      # Output
      output_dir = file.path(
        "output/reproductive",
        paste0(pretrained_model, "_", format(Sys.Date(), "%Y_%m_%d"))
      ),

      # Guild AI
      guild_label = paste0("repro_", pretrained_model, "_", format(Sys.Date(), "%Y%m%d")),
      guild_tag = "reproductive"
    )
  ),

  # Create output directories
  tar_target(
    output_dirs,
    {
      dir.create(train_config$output_dir, recursive = TRUE, showWarnings = FALSE)
      dir.create(file.path(train_config$output_dir, "checkpoints"),
                 recursive = TRUE, showWarnings = FALSE)
      train_config$output_dir
    },
    format = "file"
  ),

  # ===========================================================================
  # Training Run (via Guild AI)
  # ===========================================================================

  tar_target(
    training_run,
    {
      # Ensure output directory exists
      output_dirs

      # Run training via Guild AI wrapper
      run_info <- run_training_guild(
        script_path = "R/vit_model_train.R",
        flags = list(
          pretrained_model = train_config$pretrained_model,
          batch_size = train_config$batch_size,
          blr = train_config$blr,
          num_epochs = train_config$num_epochs
        ),
        label = train_config$guild_label,
        tag = train_config$guild_tag,
        comment = paste0(
          "Reproductive model training: ",
          train_config$pretrained_model, " pretrained, ",
          train_config$num_epochs, " epochs"
        ),
        guild_home = paths$guild_home
      )

      # Return path to final checkpoint
      checkpoint_path <- file.path(
        train_config$output_dir,
        "checkpoints",
        paste0(
          "vit_finetuned_",
          train_config$pretrained_model,
          "_epoch",
          train_config$num_epochs,
          ".pt"
        )
      )

      list(
        run_info = run_info,
        checkpoint_path = checkpoint_path,
        config = train_config
      )
    }
  ),

  # ===========================================================================
  # Model Testing and Threshold Tuning
  # ===========================================================================

  # TODO: Extract testing logic from R/vit_model_testing.R into a function
  # For now, this is a placeholder that assumes testing script has been run
  tar_target(
    test_results_path,
    {
      # This should call a testing function that:
      # 1. Loads the checkpoint from training_run$checkpoint_path
      # 2. Runs inference on train/val/test splits
      # 3. Computes optimal thresholds on validation set
      # 4. Applies thresholds to all splits
      # 5. Saves results

      training_run  # Dependency

      # Expected output path
      file.path(
        train_config$output_dir,
        paste0("epoch_", train_config$num_epochs, "_testing_data.rds")
      )
    }
  ),

  tar_target(
    threshold_buffers,
    {
      # TODO: Create function to extract and save threshold buffers
      # For now, placeholder

      training_run  # Dependency

      # Expected output path
      file.path(
        train_config$output_dir,
        paste0("epoch_", train_config$num_epochs, "_threshold_buffers.csv")
      )
    }
  ),

  # ===========================================================================
  # Family-Level Statistics
  # ===========================================================================

  tar_target(
    family_stats,
    {
      # TODO: Extract family stats computation from R/testing_get_family_accuracy.R
      # For now, placeholder

      test_results_path  # Dependency

      # Expected output path
      file.path(
        train_config$output_dir,
        "family_stats.csv"
      )
    }
  ),

  # ===========================================================================
  # Model Upload to HuggingFace
  # ===========================================================================

  tar_target(
    hf_model_upload,
    {
      # TODO: Create function wrapping R/phenovision_push_to_hf_hub.R
      # For now, placeholder that returns instructions

      training_run  # Dependency
      test_results_path  # Dependency
      threshold_buffers  # Dependency
      family_stats  # Dependency

      # Return instructions for user
      list(
        checkpoint = training_run$checkpoint_path,
        threshold_file = threshold_buffers,
        family_stats_file = family_stats,
        instructions = paste0(
          "Manual step required:\n",
          "1. Run: Rscript R/phenovision_push_to_hf_hub.R\n",
          "2. Provide checkpoint path: ", training_run$checkpoint_path, "\n",
          "3. Upload threshold file: ", threshold_buffers, "\n",
          "4. Upload family stats: ", family_stats
        )
      )
    }
  ),

  # ===========================================================================
  # DOI Minting Reminder
  # ===========================================================================

  tar_target(
    doi_reminder,
    {
      hf_model_upload  # Dependency

      # Return reminder message
      list(
        message = paste0(
          "\n",
          "=" = 70, "\n",
          "NEXT STEPS: Model Upload and DOI\n",
          "=" = 70, "\n",
          "1. Upload model to HuggingFace Hub:\n",
          "   - Repository: phenobase/phenovision\n",
          "   - Checkpoint: ", training_run$checkpoint_path, "\n",
          "   - Threshold file: ", threshold_buffers, "\n",
          "   - Family stats: ", family_stats, "\n\n",
          "2. Create DOI via HuggingFace:\n",
          "   - Go to model repository settings\n",
          "   - Click 'Create DOI'\n",
          "   - Record DOI for use in inference pipeline\n\n",
          "3. Update inference pipeline:\n",
          "   - Edit _targets_common.R\n",
          "   - Update config$model_doi_repro with new DOI\n\n",
          "4. Test inference with new model:\n",
          "   - Rscript run_pipeline.R --pipeline=inference\n",
          "=" = 70
        )
      )
    }
  ),

  # ===========================================================================
  # Final Summary
  # ===========================================================================

  tar_target(
    training_summary,
    {
      doi_reminder  # Ensure all steps complete first

      # Compile summary
      list(
        config = train_config,
        guild_run = training_run$run_info,
        checkpoint = training_run$checkpoint_path,
        test_results = test_results_path,
        thresholds = threshold_buffers,
        family_stats = family_stats,
        upload_instructions = hf_model_upload$instructions,
        doi_reminder = doi_reminder$message,
        completion_time = Sys.time()
      )
    }
  )
)
