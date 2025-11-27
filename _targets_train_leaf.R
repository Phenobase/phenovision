# PhenoVision Leaf Model Training Pipeline
#
# This targets pipeline handles the full leaf phenology model training workflow:
# 1. First round training (4 epochs) with balanced dataset
# 2. Second round data filtering (high-confidence predictions on additional data)
# 3. Second round training (4 epochs) with augmented dataset
# 4. Threshold tuning on validation set
# 5. Family-level accuracy statistics
# 6. HuggingFace model upload
# 7. DOI minting reminder
#
# Usage:
#   targets::tar_make(script = "_targets_train_leaf.R")
#   Rscript run_pipeline.R --pipeline=train_leaf

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

# Configure targets for sequential execution (GPU training is not parallelizable)
setup_targets_parallel(workers = 0)

# =============================================================================
# Pipeline (Modern tar_plan() Syntax)
# =============================================================================
tar_plan(

  # ===========================================================================
  # Training Configuration (Separate Targets for Granular Dependencies)
  # ===========================================================================

  # Model initialization
  checkpoint_init = "models/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth",
  resume_round1_from = NULL,  # Path to checkpoint to resume round 1 (NULL for fresh)
  resume_round2_from = NULL,  # Path to checkpoint to resume round 2 (NULL uses round 1 output)

  # Training hyperparameters (separate targets so changing one doesn't invalidate all)
  batch_size = 384L,
  blr = 5e-4,  # Base learning rate
  num_epochs_round1 = 4L,
  num_epochs_round2 = 4L,
  weight_decay = 0.05,
  layer_decay = 0.65,

  # Data filtering parameters (round 2)
  confidence_threshold = 0.95,  # Minimum prediction confidence for second round data

  # Data paths
  train_csv = "data/leaves/train.csv",
  val_csv = "data/leaves/validation.csv",
  seconds_csv = "data/leaves/seconds.csv",

  # Output directory
  output_dir = file.path(
    "output/leaves",
    paste0("model_", format(Sys.Date(), "%m_%d_%Y"))
  ),

  # Guild AI configuration
  guild_label_r1 = paste0("leaf_round1_", format(Sys.Date(), "%Y%m%d")),
  guild_label_r2 = paste0("leaf_round2_", format(Sys.Date(), "%Y%m%d")),
  guild_tag = "leaf",

  # Create output directories
  tar_target(
    output_dirs,
    {
      dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
      dir.create(file.path(output_dir, "checkpoints"),
                 recursive = TRUE, showWarnings = FALSE)
      dir.create(file.path(output_dir, "second"),
                 recursive = TRUE, showWarnings = FALSE)
      dir.create(file.path(output_dir, "second", "checkpoints"),
                 recursive = TRUE, showWarnings = FALSE)
      output_dir
    },
    format = "file"
  ),

  # ===========================================================================
  # Round 1: Initial Training
  # ===========================================================================

  # TODO: The current R/leaves_train_model.R script combines both rounds
  # For proper targets integration, we should split into separate scripts:
  # - R/leaves_train_round1.R (lines 1-278)
  # - R/leaves_filter_round2_data.R (lines 281-332)
  # - R/leaves_train_round2.R (lines 333-562)
  #
  # For now, this pipeline documents the intended workflow

  tar_target(
    training_run_round1,
    {
      # Ensure output directory exists
      output_dirs

      # TODO: Run first round training
      # This should:
      # 1. Load train/val data
      # 2. Balance dataset (resample "rest" to match "background")
      # 3. Initialize model from PlantCLEF checkpoint
      # 4. Train for num_epochs_round1 epochs
      # 5. Save checkpoints

      # Expected checkpoint path
      checkpoint_path <- file.path(
        output_dir,
        "checkpoints",
        paste0("vit_finetuned_epoch", num_epochs_round1, ".pt")
      )

      list(
        checkpoint_path = checkpoint_path,
        round = 1
      )
    }
  ),

  # ===========================================================================
  # Round 2: Data Filtering
  # ===========================================================================

  tar_target(
    round2_filtered_data,
    {
      # TODO: Create function to filter second round data
      # This should:
      # 1. Load model checkpoint from round 1
      # 2. Load "seconds" dataset
      # 3. Remove corrupt images
      # 4. Run inference on seconds dataset
      # 5. Filter to only high-confidence correct predictions (>0.95)
      # 6. Save filtered dataset

      training_run_round1  # Dependency

      # Expected output path
      file.path(
        output_dir,
        "seconds_filtered.csv"
      )
    }
  ),

  # ===========================================================================
  # Round 2: Training with Augmented Data
  # ===========================================================================

  tar_target(
    training_run_round2,
    {
      # TODO: Run second round training
      # This should:
      # 1. Load original train/val data
      # 2. Combine with filtered second round data
      # 3. Balance dataset
      # 4. Load model checkpoint from round 1
      # 5. Train for num_epochs_round2 epochs
      # 6. Save checkpoints to "second" subdirectory

      training_run_round1  # Dependency
      round2_filtered_data  # Dependency

      # Expected checkpoint path
      checkpoint_path <- file.path(
        output_dir,
        "second",
        "checkpoints",
        paste0("vit_finetuned_epoch", num_epochs_round2, ".pt")
      )

      list(
        checkpoint_path = checkpoint_path,
        round = 2
      )
    }
  ),

  # ===========================================================================
  # Model Testing and Threshold Tuning
  # ===========================================================================

  tar_target(
    test_results_path,
    {
      # TODO: Extract testing logic into function
      # Should use the final checkpoint from round 2

      training_run_round2  # Dependency

      # Expected output path
      file.path(
        output_dir,
        "second",
        paste0("epoch_", num_epochs_round2, "_testing_data.rds")
      )
    }
  ),

  tar_target(
    threshold_buffers,
    {
      # TODO: Create function to extract and save threshold buffers

      training_run_round2  # Dependency

      # Expected output path
      file.path(
        output_dir,
        "second",
        paste0("epoch_", num_epochs_round2, "_threshold_buffers.csv")
      )
    }
  ),

  # ===========================================================================
  # Family-Level Statistics
  # ===========================================================================

  tar_target(
    family_stats,
    {
      # TODO: Extract family stats computation

      test_results_path  # Dependency

      # Expected output path
      file.path(
        output_dir,
        "second",
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
      # TODO: Create function wrapping R/phenovisionL_push_to_hf_hub.R

      training_run_round2  # Dependency
      test_results_path  # Dependency
      threshold_buffers  # Dependency
      family_stats  # Dependency

      # Return instructions for user
      list(
        checkpoint = training_run_round2$checkpoint_path,
        threshold_file = threshold_buffers,
        family_stats_file = family_stats,
        instructions = paste0(
          "Manual step required:\n",
          "1. Run: Rscript R/phenovisionL_push_to_hf_hub.R\n",
          "2. Provide checkpoint path: ", training_run_round2$checkpoint_path, "\n",
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
          paste(rep("=", 70), collapse = ""), "\n",
          "NEXT STEPS: Model Upload and DOI\n",
          paste(rep("=", 70), collapse = ""), "\n",
          "1. Upload model to HuggingFace Hub:\n",
          "   - Repository: phenobase/phenovisionL\n",
          "   - Checkpoint: ", training_run_round2$checkpoint_path, "\n",
          "   - Threshold file: ", threshold_buffers, "\n",
          "   - Family stats: ", family_stats, "\n\n",
          "2. Create DOI via HuggingFace:\n",
          "   - Go to model repository settings\n",
          "   - Click 'Create DOI'\n",
          "   - Record DOI for use in inference pipeline\n\n",
          "3. Update inference pipeline:\n",
          "   - Edit _targets_common.R\n",
          "   - Update config$model_doi_leaves with new DOI\n\n",
          "4. Test inference with new model:\n",
          "   - Rscript run_pipeline.R --pipeline=inference\n",
          paste(rep("=", 70), collapse = "")
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
        config = list(
          checkpoint_init = checkpoint_init,
          resume_round1_from = resume_round1_from,
          resume_round2_from = resume_round2_from,
          batch_size = batch_size,
          blr = blr,
          num_epochs_round1 = num_epochs_round1,
          num_epochs_round2 = num_epochs_round2,
          weight_decay = weight_decay,
          layer_decay = layer_decay,
          confidence_threshold = confidence_threshold,
          train_csv = train_csv,
          val_csv = val_csv,
          seconds_csv = seconds_csv,
          output_dir = output_dir,
          guild_label_r1 = guild_label_r1,
          guild_label_r2 = guild_label_r2,
          guild_tag = guild_tag
        ),
        round1_checkpoint = training_run_round1$checkpoint_path,
        round2_checkpoint = training_run_round2$checkpoint_path,
        filtered_data = round2_filtered_data,
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
