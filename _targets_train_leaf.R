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
  # HuggingFace Upload Command
  # ===========================================================================

  tar_target(
    hf_upload_command,
    {
      # Generate the exact command to upload model to HuggingFace
      # This is NOT run automatically - requires human review first
      #
      # Note: leaf pipeline uses output_dir-based versioning rather than
      # a fixed model_version target. The version for the registry must be
      # assigned manually when adding the new entry to model_registry.yaml.
      checkpoint <- training_run_round2$checkpoint_path

      cmd <- paste0(
        "Rscript R/push_to_hf_hub.R \\\n",
        "  --checkpoint '", checkpoint, "' \\\n",
        "  --hf-repo 'phenobase/phenovisionL' \\\n",
        "  --num-labels 3 \\\n",
        "  --version '<VERSION>' \\\n",
        "  --thresholds '", threshold_buffers, "' \\\n",
        "  --family-stats '", family_stats, "'"
      )

      list(
        command = cmd,
        checkpoint = checkpoint,
        threshold_file = threshold_buffers,
        family_stats_file = family_stats
      )
    }
  ),

  # ===========================================================================
  # Upload Instructions File
  # ===========================================================================
  # Writes all upload instructions to a text file in the output directory
  # so the user can review them without loading targets in R.

  tar_target(
    upload_instructions_file,
    {
      hf_upload_command  # Dependency

      instructions_text <- paste0(
        "PhenoVision Leaf Model - Upload Instructions\n",
        "Generated: ", Sys.time(), "\n",
        strrep("=", 70), "\n\n",
        "Round 1 checkpoint: ",
        training_run_round1$checkpoint_path, "\n",
        "Round 2 checkpoint: ",
        training_run_round2$checkpoint_path, "\n",
        "Round 2 filtered data: ",
        round2_filtered_data, "\n\n",
        strrep("-", 70), "\n",
        "STEP 1: Assign a version (e.g., v1.1.0)\n",
        "  Replace <VERSION> in the command below.\n\n",
        strrep("-", 70), "\n",
        "STEP 2: Review results, then run the upload command:\n\n",
        hf_upload_command$command, "\n\n",
        strrep("-", 70), "\n",
        "STEP 3: Create DOI via HuggingFace:\n",
        "  - Go to ",
        "https://huggingface.co/phenobase/phenovisionL/settings\n",
        "  - Click 'Create DOI'\n",
        "  - Record the new DOI\n\n",
        strrep("-", 70), "\n",
        "STEP 4: Add version to model_registry.yaml:\n",
        "  - Add new version entry under leaves > versions\n",
        "  - Set doi, output_dir, thresholds_file, etc.\n",
        "  - Update current: to the new version\n\n",
        strrep("-", 70), "\n",
        "STEP 5: Update inference pipeline version:\n",
        "  - In _targets_inference.R, set: ",
        "model_version_leaves = \"<VERSION>\"\n\n",
        strrep("-", 70), "\n",
        "STEP 6: Test inference:\n",
        "  Rscript run_pipeline.R --pipeline=inference\n\n",
        strrep("=", 70), "\n\n",
        "Training Summary\n",
        strrep("-", 70), "\n",
        "Init checkpoint:      ", checkpoint_init, "\n",
        "Epochs (R1/R2):       ",
        num_epochs_round1, " / ", num_epochs_round2, "\n",
        "Batch size:           ", batch_size, "\n",
        "Base LR:              ", blr, "\n",
        "Weight decay:         ", weight_decay, "\n",
        "Layer decay:          ", layer_decay, "\n",
        "Confidence threshold: ", confidence_threshold, "\n",
        "Output dir:           ", output_dir, "\n",
        "Threshold file:       ", threshold_buffers, "\n",
        "Family stats:         ", family_stats, "\n",
        "Completion time:      ", Sys.time(), "\n"
      )

      out_path <- file.path(output_dir, "UPLOAD_INSTRUCTIONS.txt")
      writeLines(instructions_text, out_path)
      cat(instructions_text)  # Also print to console
      out_path
    },
    format = "file"
  )
)
