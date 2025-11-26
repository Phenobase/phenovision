# Model Evaluation Functions for Training Pipeline
#
# Functions for generating predictions, finding optimal thresholds,
# and evaluating model performance on train/val/test sets.
#
# Suffix: _training.R

# =============================================================================
# Dependencies
# =============================================================================

# Required packages (loaded in targets pipeline):
# - reticulate
# - tidyverse
# - tidymodels
# - probably
# - yardstick

# =============================================================================
# Main Functions
# =============================================================================

#' Generate Predictions on All Data Splits
#'
#' Run model inference on training, validation, and test datasets.
#' Returns a data.frame with raw logits, probabilities, and original metadata.
#'
#' @param checkpoint_path Path to model checkpoint (.pt file)
#' @param train_csv Path to training CSV
#' @param val_csv Path to validation CSV
#' @param test_csv Path to test CSV
#' @param batch_size Batch size for inference (default 2560)
#' @param num_workers Number of data loader workers (default 8)
#' @param device CUDA device (default "cuda:0")
#'
#' @return data.frame with columns:
#'   - Original CSV columns (file_name, flowering, fruiting, family, etc.)
#'   - .pred_flower, .pred_fruit: Sigmoid probabilities
#'   - .logit_flower, .logit_fruit: Raw logits
#'   - .loss_flower, .loss_fruit: Per-sample BCE loss
#'   - flower, fruit: Factor versions of labels
#'   - partition: "training", "validation", or "testing"
#'
#' @export
generate_all_predictions <- function(checkpoint_path,
                                     train_csv,
                                     val_csv,
                                     test_csv,
                                     batch_size = 2560L,
                                     num_workers = 8L,
                                     device = "cuda:0") {

  # Import Python modules
  torch <- reticulate::import("torch")
  timm <- reticulate::import("timm")
  ds <- reticulate::import_from_path("custom_dataset", "py")
  eval <- reticulate::import_from_path("evaluate", "py")
  models_vit <- reticulate::import_from_path("models_vit", "PlantCLEF2022")

  # Load model checkpoint
  message("Loading model from: ", checkpoint_path)
  model <- torch$load(checkpoint_path)
  model <- model$to(device)
  model$eval()

  # Create transform using model config
  vit_base <- models_vit$vit_large_patch16(num_classes = 2L)
  config <- timm$data$resolve_data_config(model = vit_base)
  transform <- timm$data$create_transform(!!!config)

  # Define loss function
  loss_fn <- torch$nn$BCEWithLogitsLoss(reduction = 'none')
  criterion <- function(outputs, targets) {
    loss_fn(outputs, targets$type(torch$float32))
  }

  # Load data splits
  message("Loading data splits...")
  train_data <- readr::read_csv(train_csv, show_col_types = FALSE)
  val_data <- readr::read_csv(val_csv, show_col_types = FALSE)
  test_data <- readr::read_csv(test_csv, show_col_types = FALSE)

  # Helper function to run inference on one split
  run_inference_on_split <- function(data, split_name) {
    message("Running inference on ", split_name, " (", nrow(data), " images)...")

    # Prepare data for PyTorch
    img_paths <- reticulate::r_to_py(data$file_name)
    targets_matrix <- data %>%
      dplyr::select(fruiting, flowering) %>%
      as.matrix()

    # Create dataset and dataloader
    dataset <- ds$PhenoDataset(img_paths, targets_matrix, transform = transform)
    dataloader <- timm$data$create_loader(
      dataset,
      c(3L, 224L, 224L),
      as.integer(batch_size),
      num_workers = as.integer(num_workers)
    )

    # Run inference
    results <- eval$evaluate(dataloader, model, device)
    logits <- torch$cat(results[[1]])
    preds <- torch$nn$functional$sigmoid(logits)
    truth <- torch$cat(results[[2]])
    loss <- criterion(logits, truth)

    # Convert to R data.frame
    pred_df <- tibble::tibble(
      .pred_fruit = as.numeric(preds$cpu()$numpy()[, 1]),
      .pred_flower = as.numeric(preds$cpu()$numpy()[, 2]),
      .logit_fruit = as.numeric(logits$cpu()$numpy()[, 1]),
      .logit_flower = as.numeric(logits$cpu()$numpy()[, 2]),
      .loss_fruit = as.numeric(loss$cpu()$numpy()[, 1]),
      .loss_flower = as.numeric(loss$cpu()$numpy()[, 2]),
      fruit = factor(as.integer(truth$cpu()$numpy()[, 1]), levels = c("1", "0")),
      flower = factor(as.integer(truth$cpu()$numpy()[, 2]), levels = c("1", "0")),
      partition = split_name
    )

    # Combine with original metadata
    dplyr::bind_cols(data, pred_df)
  }

  # Run inference on all splits
  train_preds <- run_inference_on_split(train_data, "training")
  val_preds <- run_inference_on_split(val_data, "validation")
  test_preds <- run_inference_on_split(test_data, "testing")

  # Combine all predictions
  all_predictions <- dplyr::bind_rows(
    train_preds,
    val_preds,
    test_preds
  )

  message("Generated predictions for ", nrow(all_predictions), " images total")

  return(all_predictions)
}


#' Find Optimal Thresholds Using J-Index
#'
#' Use validation predictions to find thresholds that maximize Youden's J-index.
#'
#' @param predictions data.frame with validation predictions
#' @param threshold_seq Sequence of thresholds to test (default seq(0, 1, 0.01))
#'
#' @return list with:
#'   - flower: list(threshold, j_index, sensitivity, specificity, threshold_data)
#'   - fruit: list(threshold, j_index, sensitivity, specificity, threshold_data)
#'
#' @export
find_optimal_thresholds <- function(predictions,
                                    threshold_seq = seq(0, 1, by = 0.01)) {

  # Filter to validation set only
  val_preds <- predictions %>%
    dplyr::filter(partition == "validation")

  if (nrow(val_preds) == 0) {
    stop("No validation predictions found in data")
  }

  message("Finding optimal thresholds on ", nrow(val_preds), " validation samples")

  # Helper to find optimal threshold for one class
  find_threshold_for_class <- function(truth_col, pred_col, class_name) {
    # Compute metrics at all thresholds (including PPV, NPV, F1)
    threshold_data <- val_preds %>%
      probably::threshold_perf(
        truth = !!rlang::sym(truth_col),
        estimate = !!rlang::sym(pred_col),
        thresholds = threshold_seq,
        event_level = "first",
        metrics = yardstick::metric_set(
          yardstick::j_index,
          yardstick::sens,
          yardstick::spec,
          yardstick::ppv,
          yardstick::npv,
          yardstick::f_meas  # F1 score
        )
      )

    # Add DQI (PPV + Sensitivity - 1) to threshold_data
    # Calculate from existing PPV and Sensitivity values
    ppv_data <- threshold_data %>%
      dplyr::filter(.metric == "ppv") %>%
      dplyr::select(.threshold, ppv_estimate = .estimate)

    sens_data <- threshold_data %>%
      dplyr::filter(.metric == "sens") %>%
      dplyr::select(.threshold, sens_estimate = .estimate)

    dqi_data <- ppv_data %>%
      dplyr::inner_join(sens_data, by = ".threshold") %>%
      dplyr::mutate(
        .metric = "dqi",
        .estimator = "binary",
        .estimate = ppv_estimate + sens_estimate - 1
      ) %>%
      dplyr::select(.metric, .estimator, .threshold, .estimate)

    # Combine with original data
    threshold_data <- dplyr::bind_rows(threshold_data, dqi_data)

    # Find threshold with max DQI (instead of j_index)
    best <- threshold_data %>%
      dplyr::filter(.metric == "dqi") %>%
      dplyr::slice_max(.estimate, n = 1)

    # Get metrics at that threshold
    # Use near() for floating-point comparison instead of ==
    sens <- threshold_data %>%
      dplyr::filter(.metric == "sens", dplyr::near(.threshold, best$.threshold[1])) %>%
      dplyr::pull(.estimate)
    if (length(sens) == 0) sens <- NA_real_

    spec <- threshold_data %>%
      dplyr::filter(.metric == "spec", dplyr::near(.threshold, best$.threshold[1])) %>%
      dplyr::pull(.estimate)
    if (length(spec) == 0) spec <- NA_real_

    ppv <- threshold_data %>%
      dplyr::filter(.metric == "ppv", dplyr::near(.threshold, best$.threshold[1])) %>%
      dplyr::pull(.estimate)
    if (length(ppv) == 0) ppv <- NA_real_

    j_index <- threshold_data %>%
      dplyr::filter(.metric == "j_index", dplyr::near(.threshold, best$.threshold[1])) %>%
      dplyr::pull(.estimate)
    if (length(j_index) == 0) j_index <- NA_real_

    f1 <- threshold_data %>%
      dplyr::filter(.metric == "f_meas", dplyr::near(.threshold, best$.threshold[1])) %>%
      dplyr::pull(.estimate)
    if (length(f1) == 0) f1 <- NA_real_

    message(sprintf("  %s: threshold=%.3f, dqi=%.4f, ppv=%.4f, sens=%.4f, j_index=%.4f, f1=%.4f",
                    class_name, best$.threshold[1], best$.estimate[1], ppv, sens, j_index, f1))

    list(
      threshold = best$.threshold[1],
      dqi = best$.estimate[1],
      j_index = j_index,
      f1 = f1,
      ppv = ppv,
      sensitivity = sens,
      specificity = spec,
      threshold_data = threshold_data
    )
  }

  # Find thresholds for both classes
  flower_threshold <- find_threshold_for_class("flower", ".pred_flower", "flower")
  fruit_threshold <- find_threshold_for_class("fruit", ".pred_fruit", "fruit")

  list(
    flower = flower_threshold,
    fruit = fruit_threshold
  )
}


#' Evaluate Test Set Predictions
#'
#' Apply optimal thresholds to test predictions and compute comprehensive metrics.
#'
#' @param predictions data.frame with all predictions
#' @param thresholds list from find_optimal_thresholds()
#'
#' @return list with:
#'   - metrics: data.frame with metrics for each class
#'   - confusion_matrices: list of confusion matrices
#'   - predictions: data.frame with .pred_class columns added
#'
#' @export
evaluate_test_predictions <- function(predictions, thresholds) {

  # Filter to test set

test_preds <- predictions %>%
    dplyr::filter(partition == "testing")

  if (nrow(test_preds) == 0) {
    stop("No test predictions found in data")
  }

  message("Evaluating ", nrow(test_preds), " test samples")

  # Apply thresholds to create class predictions
  test_preds <- test_preds %>%
    dplyr::mutate(
      .pred_class_flower = factor(
        ifelse(.pred_flower >= thresholds$flower$threshold, "1", "0"),
        levels = c("1", "0")
      ),
      .pred_class_fruit = factor(
        ifelse(.pred_fruit >= thresholds$fruit$threshold, "1", "0"),
        levels = c("1", "0")
      )
    )

  # Helper to compute metrics for one class
  compute_class_metrics <- function(truth_col, pred_col, prob_col, class_name) {
    # Compute PPV and sensitivity first for DQI calculation
    ppv_val <- yardstick::ppv_vec(
      test_preds[[truth_col]],
      test_preds[[pred_col]],
      event_level = "first"
    )
    sens_val <- yardstick::sens_vec(
      test_preds[[truth_col]],
      test_preds[[pred_col]],
      event_level = "first"
    )

    metrics <- tibble::tibble(
      class = class_name,
      n = nrow(test_preds),
      n_positive = sum(test_preds[[truth_col]] == "1"),
      n_negative = sum(test_preds[[truth_col]] == "0"),
      threshold = thresholds[[class_name]]$threshold,
      accuracy = yardstick::accuracy_vec(
        test_preds[[truth_col]],
        test_preds[[pred_col]]
      ),
      j_index = yardstick::j_index_vec(
        test_preds[[truth_col]],
        test_preds[[pred_col]],
        event_level = "first"
      ),
      dqi = ppv_val + sens_val - 1,  # DQI = PPV + Sensitivity - 1
      sensitivity = sens_val,
      specificity = yardstick::spec_vec(
        test_preds[[truth_col]],
        test_preds[[pred_col]],
        event_level = "first"
      ),
      ppv = ppv_val,
      npv = yardstick::npv_vec(
        test_preds[[truth_col]],
        test_preds[[pred_col]],
        event_level = "first"
      ),
      f1 = yardstick::f_meas_vec(
        test_preds[[truth_col]],
        test_preds[[pred_col]],
        event_level = "first"
      ),
      auc_roc = yardstick::roc_auc_vec(
        test_preds[[truth_col]],
        test_preds[[prob_col]],
        event_level = "first"
      )
    )

    metrics
  }

  # Compute metrics for both classes
  flower_metrics <- compute_class_metrics(
    "flower", ".pred_class_flower", ".pred_flower", "flower"
  )
  fruit_metrics <- compute_class_metrics(
    "fruit", ".pred_class_fruit", ".pred_fruit", "fruit"
  )

  all_metrics <- dplyr::bind_rows(flower_metrics, fruit_metrics)

  # Compute confusion matrices
  cm_flower <- table(
    Truth = test_preds$flower,
    Predicted = test_preds$.pred_class_flower
  )
  cm_fruit <- table(
    Truth = test_preds$fruit,
    Predicted = test_preds$.pred_class_fruit
  )

  # Print summary
  message("\nTest Set Metrics:")
  message(sprintf("  Flower: accuracy=%.4f, j_index=%.4f, AUC=%.4f",
                  flower_metrics$accuracy, flower_metrics$j_index, flower_metrics$auc_roc))
  message(sprintf("  Fruit:  accuracy=%.4f, j_index=%.4f, AUC=%.4f",
                  fruit_metrics$accuracy, fruit_metrics$j_index, fruit_metrics$auc_roc))

  list(
    metrics = all_metrics,
    confusion_matrices = list(
      flower = cm_flower,
      fruit = cm_fruit
    ),
    predictions = test_preds
  )
}


#' Create Simple Test Summary
#'
#' Generate a concise summary of test results.
#'
#' @param test_metrics Output from evaluate_test_predictions()
#' @param thresholds Output from find_optimal_thresholds()
#' @param training_config List of training configuration parameters
#'
#' @return data.frame with summary information
#'
#' @export
create_test_summary <- function(test_metrics, thresholds, training_config = list()) {

  metrics <- test_metrics$metrics

  summary_df <- tibble::tibble(
    # Sample sizes
    n_test = metrics$n[1],
    n_flower_positive = metrics$n_positive[metrics$class == "flower"],
    n_flower_negative = metrics$n_negative[metrics$class == "flower"],
    n_fruit_positive = metrics$n_positive[metrics$class == "fruit"],
    n_fruit_negative = metrics$n_negative[metrics$class == "fruit"],

    # Flower metrics
    flower_threshold = thresholds$flower$threshold,
    flower_accuracy = metrics$accuracy[metrics$class == "flower"],
    flower_dqi = metrics$dqi[metrics$class == "flower"],
    flower_j_index = metrics$j_index[metrics$class == "flower"],
    flower_sensitivity = metrics$sensitivity[metrics$class == "flower"],
    flower_specificity = metrics$specificity[metrics$class == "flower"],
    flower_ppv = metrics$ppv[metrics$class == "flower"],
    flower_auc = metrics$auc_roc[metrics$class == "flower"],

    # Fruit metrics
    fruit_threshold = thresholds$fruit$threshold,
    fruit_accuracy = metrics$accuracy[metrics$class == "fruit"],
    fruit_dqi = metrics$dqi[metrics$class == "fruit"],
    fruit_j_index = metrics$j_index[metrics$class == "fruit"],
    fruit_sensitivity = metrics$sensitivity[metrics$class == "fruit"],
    fruit_specificity = metrics$specificity[metrics$class == "fruit"],
    fruit_ppv = metrics$ppv[metrics$class == "fruit"],
    fruit_auc = metrics$auc_roc[metrics$class == "fruit"],

    # Training info
    model_version = training_config$model_version %||% NA_character_,
    pretrained_model = training_config$pretrained_model %||% NA_character_,
    num_epochs = training_config$num_epochs %||% NA_integer_,
    batch_size = training_config$batch_size %||% NA_integer_,

    # Metadata
    evaluation_time = Sys.time()
  )

  return(summary_df)
}
