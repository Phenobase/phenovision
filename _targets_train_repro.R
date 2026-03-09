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
setup_targets_parallel(workers = 0)

# =============================================================================
# Pipeline (Modern tar_plan() Syntax)
# =============================================================================
tar_plan(

  # ===========================================================================
  # Versioning (Bump to Trigger Full Pipeline Rerun)
  # ===========================================================================

  # Model version following semantic versioning: vMAJOR.MINOR.PATCH
  # MAJOR: New functionality (e.g., v2.0.0 = leaf phenology added)
  # MINOR: Model logic changes (e.g., v1.1.0 = image augmentation added)
  # PATCH: Data-only updates (e.g., v1.1.1 = retrained on new data)
  model_version = "v1.1.0",

  # ===========================================================================
  # Training Configuration (Separate Targets for Granular Dependencies)
  # ===========================================================================

  # Model initialization
  pretrained_model = "doi",  # Options: "plantclef", "imagenet", "imagenet21k", "dino", "mae", "clip", "doi"
  pretrained_doi = "10.57967/hf/2763",  # DOI for HuggingFace model (only used if pretrained_model = "doi")
  reinit_head = FALSE,  # If TRUE, reinitialize classification head even when loading from DOI
  resume_from = NULL,  # Path to checkpoint to resume from (NULL for fresh start)

  # Training hyperparameters (separate targets so changing one doesn't invalidate all)
  batch_size = 384L,
  blr = 5e-4,  # Base learning rate
  num_epochs = 10L,
  weight_decay = 0.05,
  layer_decay = 0.65,

  # Data splitting configuration
  train_prop = 0.6,
  val_prop = 0.2,
  test_prop = 0.2,
  split_seed = 234987,

  # Output directory (includes version for tracking)
  output_dir = file.path("output/reproductive", model_version),

  # Guild AI configuration
  guild_label = paste0("repro_", model_version, "_", gsub("-", "", data_date)),
  guild_tag = "reproductive",

  # ===========================================================================
  # Data Preparation (Load, Filter Bad Images, Split with Versioning)
  # ===========================================================================

  data_date = "2025-10-27",

  # Load full reproductive annotations CSV from download pipeline
  tar_target(
    full_annotations_csv,
    paste0("data/inat/repro_annotations_full_", data_date, ".csv"),
    format = "file"
  ),

  # Load annotations from CSV
  full_annotations = read_csv(full_annotations_csv),

  # Detect bad images (adds "good" column)
  annotations_with_quality = detect_bad_images(
    full_annotations,
    batch_size = 1024L,
    num_workers = 16L,
    dummy_target_cols = c("flowering", "fruiting")
  ),

  # Filter to good images only
  good_annotations = dplyr::filter(annotations_with_quality, good == TRUE),

  # Split into train/val/test with versioned filenames
  data_splits = split_and_version_training_data(
    good_annotations,
    version = model_version,
    output_dir = "data/inat",
    split_type = "repro",
    train_prop = train_prop,
    val_prop = val_prop,
    test_prop = test_prop,
    seed = split_seed
  ),

  # Track versioned CSV file targets
  tar_target(train_csv, data_splits$train_path, format = "file"),
  tar_target(val_csv, data_splits$val_path, format = "file"),
  tar_target(test_csv, data_splits$test_path, format = "file"),

  # Create output directories
  tar_target(
    output_dirs,
    {
      dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
      dir.create(file.path(output_dir, "checkpoints"),
                 recursive = TRUE, showWarnings = FALSE)
      output_dir
    }
  ),

  # ===========================================================================
  # Training Run (via Guild AI)
  # ===========================================================================

  tar_target(
    training_run,
    {
      # Ensure output directory and data splits exist
      output_dirs
      train_csv
      val_csv
      test_csv

      # Run training via Guild AI wrapper with script YAML front matter
      run_info <- run_training_guild(
        script_path = "R/vit_model_train.R",
        flags = list(
          model_version = model_version,
          train_csv = train_csv,
          val_csv = val_csv,
          pretrained_model = pretrained_model,
          pretrained_doi = pretrained_doi,
          reinit_head = reinit_head,
          batch_size = batch_size,
          blr = blr,
          num_epochs = num_epochs,
          weight_decay = weight_decay,
          layer_decay = layer_decay
        ),
        label = guild_label,
        tag = guild_tag,
        comment = paste0(
          "Reproductive model training ",
          model_version, ": ",
          pretrained_model, " pretrained, ",
          num_epochs, " epochs"
        ),
        guild_home = paths$guild_home
      )

      # Return path to final checkpoint
      checkpoint_path <- file.path(
        output_dir,
        "checkpoints",
        paste0(
          "phenovision_",
          model_version,
          "_epoch",
          num_epochs,
          ".pt"
        )
      )

      list(
        run_info = run_info,
        checkpoint_path = checkpoint_path
      )
    }
  ),

  # ===========================================================================
  # Best Epoch Selection (based on average DQI, or J-Index as fallback)
  # ===========================================================================

  # Extract best epoch from GuildAI run scalars
  tar_target(
    best_epoch_info,
    {
      library(guildai)

      # Get scalars from training run
      # runs_scalars() returns columns: run, path, tag, value, step
      # - tag contains the metric name (e.g., "val_flower_dqi")
      # - step is already numeric
      scalars <- runs_scalars(runs = training_run$run_info$run_id)

      # Check which metrics are available
      available_tags <- unique(scalars$tag)
      has_dqi <- all(c("val_flower_dqi", "val_fruit_dqi") %in% available_tags)
      has_jindex <- all(c("val_flower_jindex", "val_fruit_jindex") %in% available_tags)

      # Find best epoch by average metric across both classes
      # Note: step 1 = initial model evaluation, step 2 = epoch 1, etc.
      if (has_dqi) {
        # Prefer DQI (Detection Quality Index)
        metric_tags <- c("val_flower_dqi", "val_fruit_dqi")
        metric_used <- "dqi"
        cli::cli_alert_info("Using DQI metrics for best epoch selection")
      } else if (has_jindex) {
        # Fall back to J-Index
        metric_tags <- c("val_flower_jindex", "val_fruit_jindex")
        metric_used <- "jindex"
        cli::cli_alert_info("DQI metrics not found, using J-Index for best epoch selection")
      } else {
        stop("Neither DQI nor J-Index metrics found in training run scalars")
      }

      best_step <- scalars %>%
        dplyr::filter(tag %in% metric_tags) %>%
        dplyr::group_by(step) %>%
        dplyr::summarize(avg_metric = mean(value), .groups = "drop") %>%
        dplyr::slice_max(avg_metric, n = 1, with_ties = FALSE) %>%
        dplyr::pull(step)

      # Get epoch directly from the explicit epoch scalar
      # The training script now logs both 'step' and 'epoch' scalars
      # step 1 = epoch 0 (initial model), step 2 = epoch 1, etc.
      epoch_scalar <- scalars %>%
        dplyr::filter(tag == "epoch", step == best_step) %>%
        dplyr::pull(value)

      if (length(epoch_scalar) > 0) {
        best_epoch <- as.integer(epoch_scalar)
        cli::cli_alert_info("Best epoch: {best_epoch} (from explicit epoch scalar)")
      } else {
        # Fallback for older runs without explicit epoch scalar
        cli::cli_alert_warning("No 'epoch' scalar found, calculating from step (step - 1)")
        best_epoch <- best_step - 1
      }

      # Get all metrics at best epoch
      best_metrics <- scalars %>%
        dplyr::filter(step == best_step) %>%
        tidyr::pivot_wider(names_from = tag, values_from = value)

      # Build result
      list(
        best_epoch = best_epoch,
        best_step = best_step,
        metric_used = metric_used,
        checkpoint_path = file.path(
          output_dir, "checkpoints",
          paste0("phenovision_", model_version, "_epoch", best_epoch, ".pt")
        ),
        metrics = best_metrics,
        run_id = training_run$run_info$run_id
      )
    }
  ),

  # Track the best checkpoint file for downstream dependencies
  tar_target(
    best_checkpoint_file,
    {
      checkpoint_path <- best_epoch_info$checkpoint_path
      if (!file.exists(checkpoint_path)) {
        stop("Best checkpoint not found: ", checkpoint_path)
      }
      normalizePath(checkpoint_path)
    },
    format = "file"
  ),

  # Save best epoch metadata for reproducibility
  tar_target(
    best_epoch_file,
    {
      metrics <- best_epoch_info$metrics

      # Handle both DQI and J-Index cases
      # DQI may not exist in older training runs
      has_dqi <- !is.null(metrics$val_flower_dqi) && !is.null(metrics$val_fruit_dqi)

      info <- tibble::tibble(
        run_id = best_epoch_info$run_id,
        best_epoch = best_epoch_info$best_epoch,
        best_step = best_epoch_info$best_step,
        metric_used = best_epoch_info$metric_used,
        checkpoint_path = best_epoch_info$checkpoint_path,
        avg_dqi = if (has_dqi) (metrics$val_flower_dqi + metrics$val_fruit_dqi) / 2 else NA_real_,
        val_flower_dqi = metrics$val_flower_dqi %||% NA_real_,
        val_fruit_dqi = metrics$val_fruit_dqi %||% NA_real_,
        avg_jindex = (metrics$val_flower_jindex + metrics$val_fruit_jindex) / 2,
        val_flower_jindex = metrics$val_flower_jindex,
        val_fruit_jindex = metrics$val_fruit_jindex,
        val_flower_acc = metrics$val_flower_acc,
        val_fruit_acc = metrics$val_fruit_acc,
        val_loss = metrics$val_loss
      )
      output_path <- file.path(output_dir, "best_epoch_info.csv")
      readr::write_csv(info, output_path)
      normalizePath(output_path)
    },
    format = "file"
  ),

  # ===========================================================================
  # Model Evaluation: Generate Predictions on All Splits
  # ===========================================================================

  # Generate predictions on train/val/test with logits, probabilities, and metadata
  # Uses best checkpoint (by DQI) instead of final epoch
  tar_target(
    all_predictions,
    {
      generate_all_predictions(
        checkpoint_path = best_checkpoint_file,
        train_csv = train_csv,
        val_csv = val_csv,
        test_csv = test_csv,
        batch_size = 2560L,
        num_workers = 8L,
        device = "cuda:0"
      )
    },
    format = "qs"
  ),

  # ===========================================================================
  # Taxonomic Enrichment
  # ===========================================================================

  # Open Arrow dataset for taxonomic metadata
  meta_taxa = paths$metadata_root,

  # Create taxonomy lookup: map taxon_ids to ancestry and split into components
  tar_target(
    taxonomy,
    {
      all_predictions %>%
        distinct(taxon_id) %>%
        left_join(
          open_dataset(file.path(meta_taxa, "taxa")) %>%
            select(taxon_id, ancestry),
          by = "taxon_id",
          copy = TRUE
        ) %>%
        collect() %>%
        mutate(taxa_ids = str_split(ancestry, "/")) %>%
        select(-ancestry) %>%
        unnest_longer(taxa_ids, transform = as.integer)
    }
  ),

  # Get family-level taxa
  tar_target(
    families,
    {
      open_dataset(file.path(meta_taxa, "taxa")) %>%
        filter(rank == "family") %>%
        collect()
    }
  ),

  # Get genus-level taxa
  tar_target(
    genera,
    {
      open_dataset(file.path(meta_taxa, "taxa")) %>%
        filter(rank == "genus") %>%
        collect()
    }
  ),

  # Add family and genus names to ALL predictions (train/val/test)
  tar_target(
    predictions_enriched,
    {
      # Extract genus names from taxonomy
      gen_tax <- taxonomy %>%
        filter(taxa_ids %in% genera$taxon_id) %>%
        left_join(genera, by = c("taxa_ids" = "taxon_id")) %>%
        select(taxon_id, genus = name)

      # Extract family names from taxonomy
      fam_tax <- taxonomy %>%
        filter(taxa_ids %in% families$taxon_id) %>%
        left_join(families, by = c("taxa_ids" = "taxon_id")) %>%
        select(taxon_id, family = name)

      # Add family and genus to predictions
      all_predictions %>%
        left_join(gen_tax, by = "taxon_id") %>%
        left_join(fam_tax, by = "taxon_id")
    },
    format = "qs"
  ),

  # ===========================================================================
  # Threshold Optimization (J-Index)
  # ===========================================================================

  # Find optimal thresholds on validation set
  optimal_thresholds = find_optimal_thresholds(
    predictions_enriched,  # Changed from all_predictions
    threshold_seq = seq(0, 1, by = 0.01)
  ),

  # ===========================================================================
  # Apply Thresholds to Create Prediction Classes
  # ===========================================================================

  # Apply optimal thresholds to ALL predictions (train/val/test) to create .pred_class columns
  tar_target(
    predictions_with_classes,
    {
      predictions_enriched %>%
        mutate(
          .pred_class_flower = ifelse(.pred_flower >= optimal_thresholds$flower$threshold, "detected", "not detected"),
          .pred_class_fruit = ifelse(.pred_fruit >= optimal_thresholds$fruit$threshold, "detected", "not detected")
        )
    },
    format = "qs"
  ),

  # Filter to test partition once (cached for efficient reuse)
  tar_target(
    test_predictions,
    {
      predictions_with_classes %>%
        filter(partition == "testing")
    },
    format = "qs"
  ),

  # ===========================================================================
  # Test Set Evaluation
  # ===========================================================================

  # Evaluate test predictions with optimal thresholds
  test_metrics = evaluate_test_predictions(
    test_predictions,  # Changed from all_predictions
    optimal_thresholds
  ),

  # Create simple summary table
  test_summary = create_test_summary(
    test_metrics,
    optimal_thresholds,
    training_config = list(
      model_version = model_version,
      pretrained_model = pretrained_model,
      num_epochs = num_epochs,
      batch_size = batch_size,
      blr = blr,
      weight_decay = weight_decay,
      layer_decay = layer_decay
    )
  ),

  # ===========================================================================
  # Conformal Prediction: Calibration
  # ===========================================================================

  # Calculate conformity scores on validation (calibration) set
  conformal_calibration = calculate_conformal_calibration(predictions_enriched),  # Changed from all_predictions

  # Save calibration scores to CSV for reuse in inference pipeline
  tar_target(
    conformal_calibration_file,
    {
      # Convert calibration list to long-format data frame
      cal_df <- tibble::tibble(
        class = c(
          rep("flower", conformal_calibration$flower$n_detected),
          rep("flower", conformal_calibration$flower$n_not_detected),
          rep("fruit", conformal_calibration$fruit$n_detected),
          rep("fruit", conformal_calibration$fruit$n_not_detected)
        ),
        true_label = c(
          rep("detected", conformal_calibration$flower$n_detected),
          rep("not_detected", conformal_calibration$flower$n_not_detected),
          rep("detected", conformal_calibration$fruit$n_detected),
          rep("not_detected", conformal_calibration$fruit$n_not_detected)
        ),
        calibration_score = c(
          conformal_calibration$flower$detected_scores,
          conformal_calibration$flower$not_detected_scores,
          conformal_calibration$fruit$detected_scores,
          conformal_calibration$fruit$not_detected_scores
        )
      )

      # Save to CSV
      output_path <- file.path(output_dir, "conformal_calibration_scores.csv")
      readr::write_csv(cal_df, output_path)
      output_path
    },
    format = "file"
  ),

  # ===========================================================================
  # Conformal Prediction: Apply to Test Set
  # ===========================================================================

  # Calculate conformal p-values for test predictions
  conformal_test_pvalues = apply_conformal_prediction(
    predictions_enriched,  # Changed from all_predictions
    conformal_calibration
  ),

  # ===========================================================================
  # Conformal Prediction: Coverage Check
  # ===========================================================================

  # Verify coverage at different alpha levels
  conformal_coverage = check_conformal_coverage(
    conformal_test_pvalues,
    alpha_levels = c(0.05, 0.10, 0.15, 0.20)
  ),

  # ===========================================================================
  # Conformal Prediction: Alpha Cutoff Analysis (Mondrian - Independent Optimization)
  # ===========================================================================

  # Analyze different alpha cutoffs for COMBINED approach (original)
  alpha_analysis = analyze_alpha_cutoffs(
    conformal_test_pvalues,
    alpha_values = c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
  ),

  # Analyze alpha cutoffs for POSITIVE predictions separately (PPV optimization)
  alpha_analysis_positive = analyze_alpha_positive(
    conformal_test_pvalues,
    alpha_values = c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
  ),

  # Analyze alpha cutoffs for NEGATIVE predictions separately (NPV optimization)
  alpha_analysis_negative = analyze_alpha_negative(
    conformal_test_pvalues,
    alpha_values = c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
  ),

  # ===========================================================================
  # Conformal Prediction: Factorial Alpha Optimization (NEW APPROACH)
  # ===========================================================================

  # Factorial analysis: test all combinations of (alpha_pos, alpha_neg)
  # NOTE: Alpha=0 is excluded because it's a degenerate case (removes 100% of data)
  # In conformal prediction, lower alpha = stricter coverage = MORE data removed as uncertain
  alpha_analysis_factorial = analyze_alpha_factorial(
    conformal_test_pvalues,
    alpha_pos_values = c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    alpha_neg_values = c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
  ),

  # Find optimal alpha combinations using multiple methods
  # Uses detection data loss (not overall loss) for presence-only optimization
  optimal_alpha_candidates = find_optimal_alphas(
    alpha_analysis_factorial,
    max_detection_loss = 20,  # Maximum % of true detections we're willing to lose
    efficiency_threshold = 0.5
  ),

  # Save complete factorial analysis results
  tar_target(
    factorial_analysis_file,
    {
      output_path <- file.path(output_dir, "alpha_factorial_analysis.csv")
      readr::write_csv(alpha_analysis_factorial, output_path)
      output_path
    },
    format = "file"
  ),

  # Save optimal candidates
  tar_target(
    optimal_candidates_file,
    {
      output_path <- file.path(output_dir, "alpha_optimal_candidates.csv")
      readr::write_csv(optimal_alpha_candidates, output_path)
      output_path
    },
    format = "file"
  ),

  # Get recommended alpha cutoffs (separate for positive/negative)
  recommended_alpha = get_recommended_alpha(
    alpha_analysis,
    max_removal = 15,
    min_accuracy_gain = 0.005
  ),

  # Save recommended alpha cutoffs for inference pipeline
  # Extract best alpha for positive (PPV) and negative (NPV) separately
  tar_target(
    recommended_alpha_file,
    {
      # Get alpha with best PPV for positive predictions (detected)
      flower_pos_alpha <- alpha_analysis_positive %>%
        dplyr::filter(class == "flower") %>%
        dplyr::slice_max(ppv, n = 1) %>%
        dplyr::pull(alpha)

      fruit_pos_alpha <- alpha_analysis_positive %>%
        dplyr::filter(class == "fruit") %>%
        dplyr::slice_max(ppv, n = 1) %>%
        dplyr::pull(alpha)

      # Get alpha with best NPV for negative predictions (not detected)
      flower_neg_alpha <- alpha_analysis_negative %>%
        dplyr::filter(class == "flower") %>%
        dplyr::slice_max(npv, n = 1) %>%
        dplyr::pull(alpha)

      fruit_neg_alpha <- alpha_analysis_negative %>%
        dplyr::filter(class == "fruit") %>%
        dplyr::slice_max(npv, n = 1) %>%
        dplyr::pull(alpha)

      # Create data frame with recommended alphas
      alpha_df <- tibble::tibble(
        class = c("flower", "flower", "fruit", "fruit"),
        prediction_type = c("detected", "not_detected", "detected", "not_detected"),
        recommended_alpha = c(flower_pos_alpha, flower_neg_alpha,
                             fruit_pos_alpha, fruit_neg_alpha)
      )

      # Save to CSV
      output_path <- file.path(output_dir, "recommended_alpha_cutoffs.csv")
      readr::write_csv(alpha_df, output_path)
      output_path
    },
    format = "file"
  ),

  # ===========================================================================
  # Accuracy-Based Buffer Zone Analysis (Simpler Alternative)
  # ===========================================================================
  # Based on original implementation in R/vit_model_test_threshold_new.R
  # Uses LOESS smoothing by default - important for reuse with leaf model

  # Flower buffer zone (overall accuracy metric)
  accuracy_buffer_flower_overall = calculate_accuracy_buffer_zone(
    data = predictions_enriched,
    class_name = "flower",
    optimal_threshold = optimal_thresholds$flower$threshold,
    partition_filter = "validation",
    n_intervals = 100,
    accuracy_cutoff = 0.80,
    smoother = "loess",
    smoother_span = 0.3,
    metric = "overall"
  ),

  # Flower buffer zone (PPV/NPV metric)
  accuracy_buffer_flower_ppv_npv = calculate_accuracy_buffer_zone(
    data = predictions_enriched,
    class_name = "flower",
    optimal_threshold = optimal_thresholds$flower$threshold,
    partition_filter = "validation",
    n_intervals = 100,
    accuracy_cutoff = 0.80,
    smoother = "loess",
    smoother_span = 0.3,
    metric = "ppv_npv"
  ),

  # Fruit buffer zone (overall accuracy metric)
  accuracy_buffer_fruit_overall = calculate_accuracy_buffer_zone(
    data = predictions_enriched,
    class_name = "fruit",
    optimal_threshold = optimal_thresholds$fruit$threshold,
    partition_filter = "validation",
    n_intervals = 100,
    accuracy_cutoff = 0.80,
    smoother = "loess",
    smoother_span = 0.3,
    metric = "overall"
  ),

  # Fruit buffer zone (PPV/NPV metric)
  accuracy_buffer_fruit_ppv_npv = calculate_accuracy_buffer_zone(
    data = predictions_enriched,
    class_name = "fruit",
    optimal_threshold = optimal_thresholds$fruit$threshold,
    partition_filter = "validation",
    n_intervals = 100,
    accuracy_cutoff = 0.80,
    smoother = "loess",
    smoother_span = 0.3,
    metric = "ppv_npv"
  ),

  # Combined equivocal zone plot (like original equivocal_zone_plot_new.png)
  tar_target(
    plot_equivocal_zone_combined,
    {
      p <- create_combined_accuracy_plot(
        accuracy_buffer_flower_overall,
        accuracy_buffer_fruit_overall
      )
      output_path <- file.path(output_dir, "plot_equivocal_zone_combined.png")
      ggplot2::ggsave(output_path, p, width = 10, height = 8, dpi = 300)
      normalizePath(output_path)
    },
    format = "file"
  ),

  # Flower accuracy buffer comparison plot (overall vs PPV/NPV)
  tar_target(
    plot_accuracy_buffer_flower,
    {
      p <- create_accuracy_comparison_plot(
        accuracy_buffer_flower_overall,
        accuracy_buffer_flower_ppv_npv,
        "flower"
      )
      output_path <- file.path(output_dir, "plot_accuracy_buffer_flower.png")
      ggplot2::ggsave(output_path, p, width = 14, height = 7, dpi = 300)
      normalizePath(output_path)
    },
    format = "file"
  ),

  # Fruit accuracy buffer comparison plot (overall vs PPV/NPV)
  tar_target(
    plot_accuracy_buffer_fruit,
    {
      p <- create_accuracy_comparison_plot(
        accuracy_buffer_fruit_overall,
        accuracy_buffer_fruit_ppv_npv,
        "fruit"
      )
      output_path <- file.path(output_dir, "plot_accuracy_buffer_fruit.png")
      ggplot2::ggsave(output_path, p, width = 14, height = 7, dpi = 300)
      normalizePath(output_path)
    },
    format = "file"
  ),

  # Save accuracy buffer results to CSV
  tar_target(
    accuracy_buffer_results_file,
    {
      buffer_summary <- tibble::tibble(
        class = c("flower", "flower", "fruit", "fruit"),
        metric = c("overall", "ppv_npv", "overall", "ppv_npv"),
        threshold = c(
          accuracy_buffer_flower_overall$optimal_threshold,
          accuracy_buffer_flower_ppv_npv$optimal_threshold,
          accuracy_buffer_fruit_overall$optimal_threshold,
          accuracy_buffer_fruit_ppv_npv$optimal_threshold
        ),
        buffer_lower = c(
          accuracy_buffer_flower_overall$buffer_lower,
          accuracy_buffer_flower_ppv_npv$buffer_lower,
          accuracy_buffer_fruit_overall$buffer_lower,
          accuracy_buffer_fruit_ppv_npv$buffer_lower
        ),
        buffer_upper = c(
          accuracy_buffer_flower_overall$buffer_upper,
          accuracy_buffer_flower_ppv_npv$buffer_upper,
          accuracy_buffer_fruit_overall$buffer_upper,
          accuracy_buffer_fruit_ppv_npv$buffer_upper
        ),
        equivocal_zone_lower = threshold - buffer_lower,
        equivocal_zone_upper = threshold + buffer_upper
      )

      output_path <- file.path(output_dir, "accuracy_buffer_results.csv")
      readr::write_csv(buffer_summary, output_path)
      output_path
    },
    format = "file"
  ),

  # ===========================================================================
  # Multi-Cutoff Accuracy Buffer Analysis (75% to 95% by 2%)
  # ===========================================================================

  # Multi-cutoff analysis for flower
  tar_target(
    multi_cutoff_flower,
    analyze_multi_cutoff_buffer_zones(
      data = predictions_enriched %>% dplyr::filter(partition == "validation"),
      class_name = "flower",
      optimal_threshold = optimal_thresholds$flower$threshold,
      partition_filter = "validation",
      cutoffs = seq(0.75, 0.95, by = 0.02),
      n_intervals = 100,
      smoother = "loess",
      smoother_span = 0.3
    )
  ),

  # Multi-cutoff analysis for fruit
  tar_target(
    multi_cutoff_fruit,
    analyze_multi_cutoff_buffer_zones(
      data = predictions_enriched %>% dplyr::filter(partition == "validation"),
      class_name = "fruit",
      optimal_threshold = optimal_thresholds$fruit$threshold,
      partition_filter = "validation",
      cutoffs = seq(0.75, 0.95, by = 0.02),
      n_intervals = 100,
      smoother = "loess",
      smoother_span = 0.3
    )
  ),

  # Plot: Data loss vs cutoff comparison
  tar_target(
    plot_multi_cutoff_dataloss,
    {
      p <- create_multi_cutoff_dataloss_plot(multi_cutoff_flower, multi_cutoff_fruit)
      output_path <- file.path(output_dir, "plot_multi_cutoff_dataloss.png")
      ggplot2::ggsave(output_path, p, width = 10, height = 10, dpi = 300)
      normalizePath(output_path)
    },
    format = "file"
  ),

  # Save multi-cutoff results to CSV
  tar_target(
    multi_cutoff_results_file,
    {
      combined <- dplyr::bind_rows(multi_cutoff_flower, multi_cutoff_fruit)
      output_path <- file.path(output_dir, "multi_cutoff_buffer_analysis.csv")
      readr::write_csv(combined, output_path)
      normalizePath(output_path)
    },
    format = "file"
  ),

  # ===========================================================================
  # Final Buffer Zone Parameters (for downstream pipelines)
  # ===========================================================================

  # Save final buffer zone parameters (overall accuracy at 80% cutoff)
  tar_target(
    final_buffer_params_file,
    {
      buffer_params <- tibble::tibble(
        class = c("flower", "fruit"),
        threshold = c(
          accuracy_buffer_flower_overall$optimal_threshold,
          accuracy_buffer_fruit_overall$optimal_threshold
        ),
        buffer_lower = c(
          accuracy_buffer_flower_overall$buffer_lower,
          accuracy_buffer_fruit_overall$buffer_lower
        ),
        buffer_upper = c(
          accuracy_buffer_flower_overall$buffer_upper,
          accuracy_buffer_fruit_overall$buffer_upper
        ),
        equivocal_lower = threshold - buffer_lower,
        equivocal_upper = threshold + buffer_upper,
        accuracy_cutoff = 0.80,
        method = "overall_accuracy"
      )
      output_path <- file.path(output_dir, "final_buffer_params.csv")
      readr::write_csv(buffer_params, output_path)
      normalizePath(output_path)
    },
    format = "file"
  ),

  # ===========================================================================
  # Test Set Evaluation (Before/After Buffer Zone Filtering)
  # ===========================================================================

  # Evaluate on test set with and without buffer zone filtering
  tar_target(
    test_evaluation_summary,
    {
      # Get buffer zone parameters
      flower_threshold <- accuracy_buffer_flower_overall$optimal_threshold
      flower_eq_lower <- flower_threshold - accuracy_buffer_flower_overall$buffer_lower
      flower_eq_upper <- flower_threshold + accuracy_buffer_flower_overall$buffer_upper

      fruit_threshold <- accuracy_buffer_fruit_overall$optimal_threshold
      fruit_eq_lower <- fruit_threshold - accuracy_buffer_fruit_overall$buffer_lower
      fruit_eq_upper <- fruit_threshold + accuracy_buffer_fruit_overall$buffer_upper

      # Filter to test partition
      test_data <- predictions_enriched %>%
        dplyr::filter(partition == "testing")

      # Add classification columns
      test_data <- test_data %>%
        dplyr::mutate(
          # Hard predictions at threshold
          .pred_class_flower = factor(
            ifelse(.pred_flower >= flower_threshold, "1", "0"),
            levels = c("1", "0")
          ),
          .pred_class_fruit = factor(
            ifelse(.pred_fruit >= fruit_threshold, "1", "0"),
            levels = c("1", "0")
          ),
          # Buffer zone flags
          .in_buffer_flower = .pred_flower >= flower_eq_lower & .pred_flower <= flower_eq_upper,
          .in_buffer_fruit = .pred_fruit >= fruit_eq_lower & .pred_fruit <= fruit_eq_upper
        )

      # Helper function to calculate metrics
      calc_metrics <- function(data, truth_col, pred_col, class_name, filter_desc) {
        truth <- data[[truth_col]]
        pred <- data[[pred_col]]
        tibble::tibble(
          class = class_name,
          filter = filter_desc,
          n = length(truth),
          accuracy = yardstick::accuracy_vec(truth, pred),
          sensitivity = yardstick::sens_vec(truth, pred),
          specificity = yardstick::spec_vec(truth, pred),
          ppv = yardstick::ppv_vec(truth, pred),
          npv = yardstick::npv_vec(truth, pred),
          j_index = yardstick::j_index_vec(truth, pred),
          f1 = yardstick::f_meas_vec(truth, pred),
          dqi = dqi_vec(truth, pred)
        )
      }

      # Calculate metrics for all conditions
      results <- dplyr::bind_rows(
        # Flower - all data
        calc_metrics(test_data, "flower", ".pred_class_flower", "flower", "All test data"),
        # Flower - after removing buffer zone
        calc_metrics(
          test_data %>% dplyr::filter(!.in_buffer_flower),
          "flower", ".pred_class_flower", "flower", "Buffer zone removed"
        ),
        # Fruit - all data
        calc_metrics(test_data, "fruit", ".pred_class_fruit", "fruit", "All test data"),
        # Fruit - after removing buffer zone
        calc_metrics(
          test_data %>% dplyr::filter(!.in_buffer_fruit),
          "fruit", ".pred_class_fruit", "fruit", "Buffer zone removed"
        )
      )

      # Add data loss info
      flower_loss <- sum(test_data$.in_buffer_flower) / nrow(test_data) * 100
      fruit_loss <- sum(test_data$.in_buffer_fruit) / nrow(test_data) * 100

      results <- results %>%
        dplyr::mutate(
          pct_data_loss = dplyr::case_when(
            filter == "All test data" ~ 0,
            class == "flower" ~ flower_loss,
            class == "fruit" ~ fruit_loss
          )
        )

      results
    }
  ),

  # Save test evaluation summary to file
  tar_target(
    test_evaluation_file,
    {
      output_path <- file.path(output_dir, "test_evaluation_summary.csv")
      readr::write_csv(test_evaluation_summary, output_path)
      normalizePath(output_path)
    },
    format = "file"
  ),

  # ===========================================================================
  # Visualization: Plots for Analysis
  # ===========================================================================

  # Plot threshold performance (metrics vs threshold)
  tar_target(
    plot_threshold_performance,
    {
      library(ggplot2)

      # Combine flower and fruit threshold data (in long format from probably::threshold_perf)
      threshold_data <- bind_rows(
        optimal_thresholds$flower$threshold_data %>% mutate(class = "flower"),
        optimal_thresholds$fruit$threshold_data %>% mutate(class = "fruit")
      )

      # Filter to metrics of interest and rename for plotting
      # Note: dqi and f_meas are now included in threshold_data
      threshold_data <- threshold_data %>%
        filter(.metric %in% c("dqi", "j_index", "f_meas", "sens", "spec", "ppv", "npv")) %>%
        mutate(
          metric_label = case_when(
            .metric == "dqi" ~ "DQI",
            .metric == "j_index" ~ "J-Index",
            .metric == "f_meas" ~ "F1",
            .metric == "sens" ~ "Sensitivity",
            .metric == "spec" ~ "Specificity",
            .metric == "ppv" ~ "PPV",
            .metric == "npv" ~ "NPV"
          )
        )

      # Plot metrics vs threshold
      p <- ggplot(threshold_data, aes(x = .threshold, y = .estimate, color = metric_label)) +
        geom_line(aes(linewidth = metric_label)) +
        geom_vline(data = data.frame(class = "flower", threshold = optimal_thresholds$flower$threshold),
                   aes(xintercept = threshold), linetype = "dashed", alpha = 0.5) +
        geom_vline(data = data.frame(class = "fruit", threshold = optimal_thresholds$fruit$threshold),
                   aes(xintercept = threshold), linetype = "dashed", alpha = 0.5) +
        facet_wrap(~ class, scales = "free_x") +
        scale_color_manual(
          name = "Metric",
          values = c("DQI" = "darkorange", "J-Index" = "black", "F1" = "brown",
                     "Sensitivity" = "darkgreen", "Specificity" = "blue",
                     "PPV" = "red", "NPV" = "purple")
        ) +
        scale_linewidth_manual(
          name = "Metric",
          values = c("DQI" = 1.2, "J-Index" = 1, "F1" = 1,
                     "Sensitivity" = 0.8, "Specificity" = 0.8,
                     "PPV" = 0.8, "NPV" = 0.8)
        ) +
        labs(
          title = "Threshold Performance",
          subtitle = "Dashed lines show optimal thresholds (DQI-optimized)",
          x = "Threshold",
          y = "Metric Value"
        ) +
        theme_minimal() +
        theme(legend.position = "bottom")

      output_path <- file.path(output_dir, "plot_threshold_performance.png")
      ggsave(output_path, p, width = 10, height = 5, dpi = 300)
      normalizePath(output_path)
    },
    format = "file"
  ),

  # Plot conformal positive analysis (PPV vs data retained)
  tar_target(
    plot_conformal_positive,
    {
      library(ggplot2)

      p <- ggplot(alpha_analysis_positive, aes(x = pct_removed)) +
        geom_line(aes(y = ppv, color = "PPV"), linewidth = 1) +
        geom_line(aes(y = sensitivity, color = "Sensitivity"), linewidth = 0.8) +
        geom_line(aes(y = specificity, color = "Specificity"), linewidth = 0.8) +
        geom_point(aes(y = ppv, color = "PPV"), size = 2) +
        facet_wrap(~ class) +
        scale_color_manual(
          name = "Metric",
          values = c("PPV" = "darkred", "Sensitivity" = "darkgreen", "Specificity" = "blue")
        ) +
        labs(
          title = "Conformal Prediction: Positive Detection Analysis",
          subtitle = "Metrics vs Data Removed (for samples with p_detected > alpha)",
          x = "Data Removed (%)",
          y = "Metric Value"
        ) +
        theme_minimal() +
        theme(legend.position = "bottom")

      output_path <- file.path(output_dir, "plot_conformal_positive.png")
      ggsave(output_path, p, width = 10, height = 5, dpi = 300)
      normalizePath(output_path)
    },
    format = "file"
  ),

  # Plot conformal negative analysis (NPV vs data retained)
  tar_target(
    plot_conformal_negative,
    {
      library(ggplot2)

      p <- ggplot(alpha_analysis_negative, aes(x = pct_removed)) +
        geom_line(aes(y = npv, color = "NPV"), linewidth = 1) +
        geom_line(aes(y = sensitivity, color = "Sensitivity"), linewidth = 0.8) +
        geom_line(aes(y = specificity, color = "Specificity"), linewidth = 0.8) +
        geom_point(aes(y = npv, color = "NPV"), size = 2) +
        facet_wrap(~ class) +
        scale_color_manual(
          name = "Metric",
          values = c("NPV" = "darkblue", "Sensitivity" = "darkgreen", "Specificity" = "blue")
        ) +
        labs(
          title = "Conformal Prediction: Negative Detection Analysis",
          subtitle = "Metrics vs Data Removed (for samples with p_not_detected > alpha)",
          x = "Data Removed (%)",
          y = "Metric Value"
        ) +
        theme_minimal() +
        theme(legend.position = "bottom")

      output_path <- file.path(output_dir, "plot_conformal_negative.png")
      ggsave(output_path, p, width = 10, height = 5, dpi = 300)
      normalizePath(output_path)
    },
    format = "file"
  ),

  # Plot conformal coverage by true label
  tar_target(
    plot_conformal_coverage,
    {
      library(ggplot2)

      p <- ggplot(conformal_coverage, aes(x = alpha, y = coverage, color = true_label)) +
        geom_line(linewidth = 1) +
        geom_point(size = 2) +
        geom_line(aes(y = expected_coverage), linetype = "dashed", color = "black", linewidth = 0.8) +
        facet_wrap(~ class) +
        scale_color_manual(
          name = "True Label",
          values = c("1" = "darkred", "0" = "darkblue"),
          labels = c("1" = "Detected", "0" = "Not Detected")
        ) +
        labs(
          title = "Mondrian Conformal Coverage by True Label",
          subtitle = "Dashed line shows expected coverage (1-alpha)",
          x = "Alpha",
          y = "Coverage"
        ) +
        theme_minimal() +
        theme(legend.position = "bottom")

      output_path <- file.path(output_dir, "plot_conformal_coverage.png")
      ggsave(output_path, p, width = 10, height = 5, dpi = 300)
      normalizePath(output_path)
    },
    format = "file"
  ),

  # ===========================================================================
  # Factorial Scatter Plots with Marginals (6 metrics × 2 classes = 12 plots)
  # ===========================================================================

  # PPV Scatter Plots with Marginals
  tar_target(
    plot_factorial_ppv_flower,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "flower",
      metric_col = "ppv",
      metric_label = "PPV",
      viridis_option = "plasma",
      title = "Flower PPV (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_ppv_flower.png")
    ),
    format = "file"
  ),

  tar_target(
    plot_factorial_ppv_fruit,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "fruit",
      metric_col = "ppv",
      metric_label = "PPV",
      viridis_option = "plasma",
      title = "Fruit PPV (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_ppv_fruit.png")
    ),
    format = "file"
  ),

  # NPV Scatter Plots with Marginals
  tar_target(
    plot_factorial_npv_flower,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "flower",
      metric_col = "npv",
      metric_label = "NPV",
      viridis_option = "viridis",
      title = "Flower NPV (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_npv_flower.png")
    ),
    format = "file"
  ),

  tar_target(
    plot_factorial_npv_fruit,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "fruit",
      metric_col = "npv",
      metric_label = "NPV",
      viridis_option = "viridis",
      title = "Fruit NPV (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_npv_fruit.png")
    ),
    format = "file"
  ),

  # Accuracy Scatter Plots with Marginals
  tar_target(
    plot_factorial_accuracy_flower,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "flower",
      metric_col = "accuracy",
      metric_label = "Accuracy",
      viridis_option = "cividis",
      title = "Flower Accuracy (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_accuracy_flower.png")
    ),
    format = "file"
  ),

  tar_target(
    plot_factorial_accuracy_fruit,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "fruit",
      metric_col = "accuracy",
      metric_label = "Accuracy",
      viridis_option = "cividis",
      title = "Fruit Accuracy (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_accuracy_fruit.png")
    ),
    format = "file"
  ),

  # J-Index Scatter Plots with Marginals
  tar_target(
    plot_factorial_jindex_flower,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "flower",
      metric_col = "j_index",
      metric_label = "J-Index",
      viridis_option = "mako",
      title = "Flower J-Index (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_jindex_flower.png")
    ),
    format = "file"
  ),

  tar_target(
    plot_factorial_jindex_fruit,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "fruit",
      metric_col = "j_index",
      metric_label = "J-Index",
      viridis_option = "mako",
      title = "Fruit J-Index (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_jindex_fruit.png")
    ),
    format = "file"
  ),

  # DQI Scatter Plots with Marginals (Detection Quality Index = PPV + Sensitivity - 1)
  tar_target(
    plot_factorial_dqi_flower,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "flower",
      metric_col = "dqi",
      metric_label = "DQI",
      viridis_option = "turbo",
      title = "Flower DQI (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_dqi_flower.png")
    ),
    format = "file"
  ),

  tar_target(
    plot_factorial_dqi_fruit,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "fruit",
      metric_col = "dqi",
      metric_label = "DQI",
      viridis_option = "turbo",
      title = "Fruit DQI (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_dqi_fruit.png")
    ),
    format = "file"
  ),

  # Data Loss Scatter + Marginal Plots
  tar_target(
    plot_factorial_dataloss_flower,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "flower",
      metric_col = "pct_removed",
      metric_label = "Data Loss (%)",
      viridis_option = "rocket",
      title = "Flower Data Loss (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_dataloss_flower.png"),
      is_percentage = TRUE
    ),
    format = "file"
  ),

  tar_target(
    plot_factorial_dataloss_fruit,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "fruit",
      metric_col = "pct_removed",
      metric_label = "Data Loss (%)",
      viridis_option = "rocket",
      title = "Fruit Data Loss (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_dataloss_fruit.png"),
      is_percentage = TRUE
    ),
    format = "file"
  ),

  # Detection Data Loss Scatter + Marginal Plots (key for presence-only approach)
  tar_target(
    plot_factorial_det_dataloss_flower,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "flower",
      metric_col = "pct_detected_removed",
      metric_label = "Det. Loss (%)",
      viridis_option = "magma",
      title = "Flower DETECTION Data Loss (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_det_dataloss_flower.png"),
      is_percentage = TRUE
    ),
    format = "file"
  ),

  tar_target(
    plot_factorial_det_dataloss_fruit,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "fruit",
      metric_col = "pct_detected_removed",
      metric_label = "Det. Loss (%)",
      viridis_option = "magma",
      title = "Fruit DETECTION Data Loss (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_det_dataloss_fruit.png"),
      is_percentage = TRUE
    ),
    format = "file"
  ),

  # Non-Detection Data Loss Scatter + Marginal Plots
  tar_target(
    plot_factorial_nondet_dataloss_flower,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "flower",
      metric_col = "pct_not_detected_removed",
      metric_label = "Non-Det Loss (%)",
      viridis_option = "cividis",
      title = "Flower NON-DETECTION Data Loss (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_nondet_dataloss_flower.png"),
      is_percentage = TRUE
    ),
    format = "file"
  ),

  tar_target(
    plot_factorial_nondet_dataloss_fruit,
    create_factorial_scatter_plot(
      factorial_data = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      class_name = "fruit",
      metric_col = "pct_not_detected_removed",
      metric_label = "Non-Det Loss (%)",
      viridis_option = "cividis",
      title = "Fruit NON-DETECTION Data Loss (Factorial Alpha Analysis)",
      output_path = file.path(output_dir, "plot_factorial_nondet_dataloss_fruit.png"),
      is_percentage = TRUE
    ),
    format = "file"
  ),

  # ===========================================================================
  # Conformal P-Value Density and Pareto/Elbow Plots
  # ===========================================================================

  # 2D density plot of conformal p-values (detected vs not-detected)
  tar_target(
    plot_conformal_pvalue_density,
    create_conformal_pvalue_density_plot(
      conformal_pvalues = conformal_test_pvalues,
      output_path = file.path(output_dir, "plot_conformal_pvalue_density.png")
    ),
    format = "file"
  ),

  # Pareto frontier and elbow visualization for DQI vs detection data loss
  tar_target(
    plot_pareto_elbow,
    create_pareto_elbow_plot(
      factorial_results = alpha_analysis_factorial,
      optimal_candidates = optimal_alpha_candidates,
      output_path = file.path(output_dir, "plot_pareto_elbow.png")
    ),
    format = "file"
  ),

  # Conformal p-value calibration curve (p-values vs model output)
  tar_target(
    plot_conformal_calibration_curve,
    create_conformal_calibration_curve(
      conformal_calibration = conformal_calibration,
      output_path = file.path(output_dir, "plot_conformal_calibration_curve.png"),
      n_points = 100
    ),
    format = "file"
  ),

  # ===========================================================================
  # Training Report (R Markdown)
  # ===========================================================================

  tar_target(
    training_report,
    {
      rmarkdown::render(
        input = "reports/training_report_repro.Rmd",
        output_file = file.path(
          "..",
          output_dir,
          paste0("training_report_", model_version, ".html")
        ),
        params = list(
          test_summary = test_summary,
          test_metrics = test_metrics,
          optimal_thresholds = optimal_thresholds,
          conformal_coverage = conformal_coverage,
          alpha_analysis = alpha_analysis,
          alpha_analysis_positive = alpha_analysis_positive,
          alpha_analysis_negative = alpha_analysis_negative,

          # NEW: Factorial alpha optimization
          alpha_analysis_factorial = alpha_analysis_factorial,
          optimal_alpha_candidates = optimal_alpha_candidates,
          factorial_analysis_file = factorial_analysis_file,
          optimal_candidates_file = optimal_candidates_file,

          training_config = list(
            model_version = model_version,
            pretrained_model = pretrained_model,
            num_epochs = num_epochs,
            batch_size = batch_size,
            blr = blr,
            weight_decay = weight_decay,
            layer_decay = layer_decay
          ),
          all_predictions = all_predictions,
          plot_threshold_performance = plot_threshold_performance,
          plot_conformal_positive = plot_conformal_positive,
          plot_conformal_negative = plot_conformal_negative,
          plot_conformal_coverage = plot_conformal_coverage,

          # NEW: Factorial contour plots
          plot_factorial_ppv_flower = plot_factorial_ppv_flower,
          plot_factorial_ppv_fruit = plot_factorial_ppv_fruit,
          plot_factorial_npv_flower = plot_factorial_npv_flower,
          plot_factorial_npv_fruit = plot_factorial_npv_fruit,
          plot_factorial_accuracy_flower = plot_factorial_accuracy_flower,
          plot_factorial_accuracy_fruit = plot_factorial_accuracy_fruit,
          plot_factorial_jindex_flower = plot_factorial_jindex_flower,
          plot_factorial_jindex_fruit = plot_factorial_jindex_fruit,
          plot_factorial_dqi_flower = plot_factorial_dqi_flower,
          plot_factorial_dqi_fruit = plot_factorial_dqi_fruit,
          plot_factorial_dataloss_flower = plot_factorial_dataloss_flower,
          plot_factorial_dataloss_fruit = plot_factorial_dataloss_fruit,
          plot_factorial_det_dataloss_flower = plot_factorial_det_dataloss_flower,
          plot_factorial_det_dataloss_fruit = plot_factorial_det_dataloss_fruit,
          plot_factorial_nondet_dataloss_flower = plot_factorial_nondet_dataloss_flower,
          plot_factorial_nondet_dataloss_fruit = plot_factorial_nondet_dataloss_fruit,

          # NEW: Conformal p-value and Pareto/elbow plots
          plot_conformal_pvalue_density = plot_conformal_pvalue_density,
          plot_pareto_elbow = plot_pareto_elbow,
          plot_conformal_calibration_curve = plot_conformal_calibration_curve,

          conformal_calibration_file = conformal_calibration_file,
          recommended_alpha_file = recommended_alpha_file,

          # NEW: Accuracy-based buffer zone analysis
          accuracy_buffer_flower_overall = accuracy_buffer_flower_overall,
          accuracy_buffer_flower_ppv_npv = accuracy_buffer_flower_ppv_npv,
          accuracy_buffer_fruit_overall = accuracy_buffer_fruit_overall,
          accuracy_buffer_fruit_ppv_npv = accuracy_buffer_fruit_ppv_npv,
          plot_equivocal_zone_combined = plot_equivocal_zone_combined,
          plot_accuracy_buffer_flower = plot_accuracy_buffer_flower,
          plot_accuracy_buffer_fruit = plot_accuracy_buffer_fruit,
          accuracy_buffer_results_file = accuracy_buffer_results_file,

          # Multi-cutoff sensitivity analysis
          multi_cutoff_flower = multi_cutoff_flower,
          multi_cutoff_fruit = multi_cutoff_fruit,
          plot_multi_cutoff_dataloss = plot_multi_cutoff_dataloss,
          multi_cutoff_results_file = multi_cutoff_results_file,

          # Final evaluation summary
          test_evaluation_summary = test_evaluation_summary,
          test_evaluation_file = test_evaluation_file,
          final_buffer_params_file = final_buffer_params_file
        ),
        quiet = TRUE
      )

      # Return path to output file
      file.path(
        output_dir,
        paste0("training_report_", model_version, ".html")
      )
    },
    format = "file"
  ),

  # ===========================================================================
  # Family-Level Statistics
  # ===========================================================================

  tar_target(
    family_stats,
    {
      # Compute family-level accuracy statistics with buffer zone analysis
      # Uses buffer parameters to identify equivocal predictions and compute
      # proportion of certain predictions per family (for Phenobase schema)

      # Get buffer zone parameters from accuracy-based buffer analysis
      fl_thresh <- accuracy_buffer_flower_overall$optimal_threshold
      fl_buf_lower <- accuracy_buffer_flower_overall$buffer_lower
      fl_buf_upper <- accuracy_buffer_flower_overall$buffer_upper
      fr_thresh <- accuracy_buffer_fruit_overall$optimal_threshold
      fr_buf_lower <- accuracy_buffer_fruit_overall$buffer_lower
      fr_buf_upper <- accuracy_buffer_fruit_overall$buffer_upper

      dat <- test_predictions %>%
        dplyr::mutate(
          # With buffer zones: predictions in buffer zone are equivocal (NA)
          pred_fl_eq = dplyr::case_when(
            .pred_flower >= fl_thresh + fl_buf_upper ~ "1",
            .pred_flower <= fl_thresh - fl_buf_lower ~ "0",
            TRUE ~ NA_character_
          ),
          pred_fr_eq = dplyr::case_when(
            .pred_fruit >= fr_thresh + fr_buf_upper ~ "1",
            .pred_fruit <= fr_thresh - fr_buf_lower ~ "0",
            TRUE ~ NA_character_
          ),
          # Without buffer: simple threshold on all data
          pred_fl_no_eq = ifelse(.pred_flower >= fl_thresh, "1", "0"),
          pred_fr_no_eq = ifelse(.pred_fruit >= fr_thresh, "1", "0")
        )

      # Wide format: one row per family with all metrics
      # Column names match what convert_fam_to_long() expects
      family_stats_df <- dat %>%
        dplyr::group_by(family) %>%
        dplyr::summarise(
          count = dplyr::n(),
          # Proportion of equivocal predictions per family
          equiv_prop_fl = sum(is.na(pred_fl_eq)) / dplyr::n(),
          equiv_prop_fr = sum(is.na(pred_fr_eq)) / dplyr::n(),
          # Accuracy including equivocal as errors: certain correct / total
          .accuracy_family_flower_incl_equiv = sum(pred_fl_eq == flower, na.rm = TRUE) / dplyr::n(),
          .accuracy_family_fruit_incl_equiv = sum(pred_fr_eq == fruit, na.rm = TRUE) / dplyr::n(),
          # Accuracy excluding equivocal consideration: simple threshold on all data
          .accuracy_family_flower = mean(pred_fl_no_eq == flower),
          .accuracy_family_fruit = mean(pred_fr_no_eq == fruit),
          .groups = "drop"
        )

      # Save to file
      output_path <- file.path(output_dir, "family_stats.csv")
      readr::write_csv(family_stats_df, output_path)

      output_path
    },
    format = "file"
  ),

  # ===========================================================================
  # Save Threshold Information
  # ===========================================================================

  tar_target(
    threshold_file,
    {
      # Save optimal thresholds to CSV for use in inference
      threshold_df <- tibble::tibble(
        class = c("flower", "fruit"),
        threshold = c(
          optimal_thresholds$flower$threshold,
          optimal_thresholds$fruit$threshold
        ),
        j_index = c(
          optimal_thresholds$flower$j_index,
          optimal_thresholds$fruit$j_index
        ),
        sensitivity = c(
          optimal_thresholds$flower$sensitivity,
          optimal_thresholds$fruit$sensitivity
        ),
        specificity = c(
          optimal_thresholds$flower$specificity,
          optimal_thresholds$fruit$specificity
        )
      )

      output_path <- file.path(
        output_dir,
        "optimal_thresholds.csv"
      )
      readr::write_csv(threshold_df, output_path)

      output_path
    },
    format = "file"
  ),

  # ===========================================================================
  # HuggingFace Upload Command
  # ===========================================================================

  tar_target(
    hf_upload_command,
    {
      # Generate the exact command to upload model to HuggingFace
      # This is NOT run automatically - requires human review first
      cmd <- generate_hf_upload_command(
        checkpoint = best_checkpoint_file,
        model_type = "reproductive",
        version = model_version,
        thresholds = final_buffer_params_file,
        family_stats = family_stats,
        report = training_report
      )

      list(
        command = cmd,
        checkpoint = best_checkpoint_file,
        best_epoch = best_epoch_info$best_epoch,
        threshold_file = final_buffer_params_file,
        family_stats_file = family_stats,
        report = training_report
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
        "PhenoVision Reproductive Model ", model_version, " - Upload Instructions\n",
        "Generated: ", Sys.time(), "\n",
        strrep("=", 70), "\n\n",
        "Best epoch: ", best_epoch_info$best_epoch,
        " (avg DQI: ", round((best_epoch_info$metrics$val_flower_dqi +
                              best_epoch_info$metrics$val_fruit_dqi) / 2, 4), ")\n",
        "Metric used: ", best_epoch_info$metric_used, "\n\n",
        strrep("-", 70), "\n",
        "STEP 1: Review results, then run the upload command:\n\n",
        hf_upload_command$command, "\n\n",
        strrep("-", 70), "\n",
        "STEP 2: Create DOI via HuggingFace:\n",
        "  - Go to https://huggingface.co/phenobase/phenovision/settings\n",
        "  - Click 'Create DOI'\n",
        "  - Record the new DOI (format: 10.57967/hf/XXXX)\n\n",
        strrep("-", 70), "\n",
        "STEP 3: Update model_registry.yaml:\n",
        "  - Set doi: \"<new-DOI>\" for version ", model_version, "\n\n",
        strrep("-", 70), "\n",
        "STEP 4: Update inference pipeline version:\n",
        "  - In _targets_inference.R, set: model_version_repro = \"", model_version, "\"\n\n",
        strrep("-", 70), "\n",
        "STEP 5: Test inference:\n",
        "  Rscript run_pipeline.R --pipeline=inference\n\n",
        strrep("=", 70), "\n\n",
        "Training Summary\n",
        strrep("-", 70), "\n",
        "Model version:    ", model_version, "\n",
        "Pretrained model: ", pretrained_model, "\n",
        "Pretrained DOI:   ", pretrained_doi, "\n",
        "Epochs:           ", num_epochs, "\n",
        "Batch size:       ", batch_size, "\n",
        "Base LR:          ", blr, "\n",
        "Weight decay:     ", weight_decay, "\n",
        "Layer decay:      ", layer_decay, "\n",
        "Output dir:       ", output_dir, "\n",
        "Checkpoint:       ", sub(paste0(getwd(), "/"), "", best_checkpoint_file, fixed = TRUE), "\n",
        "Threshold file:   ", final_buffer_params_file, "\n",
        "Family stats:     ", family_stats, "\n",
        "Training report:  ", training_report, "\n",
        "Completion time:  ", Sys.time(), "\n"
      )

      out_path <- file.path(output_dir, "UPLOAD_INSTRUCTIONS.txt")
      writeLines(instructions_text, out_path)
      cat(instructions_text)  # Also print to console
      out_path
    },
    format = "file"
  )
)
