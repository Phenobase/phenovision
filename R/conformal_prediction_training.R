# Conformal Prediction Functions for Training Pipeline
#
# Functions for calibrating conformal prediction, applying to test set,
# checking coverage, and analyzing alpha cutoff optimization.
#
# Suffix: _training.R

# =============================================================================
# Dependencies
# =============================================================================

# Required packages (loaded in targets pipeline):
# - tidyverse
# - yardstick
# - probably

# =============================================================================
# Conformal Calibration
# =============================================================================

#' Calculate Conformal Calibration Scores
#'
#' Calculate conformity scores on validation (calibration) set.
#' Uses Mondrian approach: separate calibration for detected vs not detected.
#'
#' Conformity scores (lower = better conformity):
#' - For detected hypothesis: 1 - model_output (so high predictions = low score)
#' - For not_detected hypothesis: model_output (so low predictions = low score)
#'
#' @param predictions data.frame with columns: partition, flower, fruit,
#'   .pred_flower, .pred_fruit
#'
#' @return list with structure:
#'   list(
#'     flower = list(n_detected, n_not_detected, detected_scores, not_detected_scores,
#'                   detected_scores_sorted, not_detected_scores_sorted),
#'     fruit = list(n_detected, n_not_detected, detected_scores, not_detected_scores,
#'                  detected_scores_sorted, not_detected_scores_sorted)
#'   )
#'
#' @export
calculate_conformal_calibration <- function(predictions) {

  # Filter to validation set only
  val_preds <- predictions %>%
    dplyr::filter(partition == "validation")

  if (nrow(val_preds) == 0) {
    stop("No validation predictions found in data")
  }

  message("Calculating conformal calibration on ", nrow(val_preds), " validation samples")

  # Helper function for one class
  calibrate_class <- function(truth_col, pred_col, class_name) {
    # Get truth labels and predictions
    truth <- val_preds[[truth_col]]
    pred <- val_preds[[pred_col]]

    # Split by true label (Mondrian)
    detected_idx <- which(truth == "1")
    not_detected_idx <- which(truth == "0")

    # Calculate conformity scores
    # For detected samples: score = 1 - prediction (lower score = more conforming)
    detected_scores <- 1 - pred[detected_idx]

    # For not detected samples: score = prediction (lower score = more conforming)
    not_detected_scores <- pred[not_detected_idx]

    message(sprintf("  %s: %d detected, %d not_detected",
                    class_name, length(detected_scores), length(not_detected_scores)))

    # CRITICAL: Sort scores once for fast binary search later
    detected_scores_sorted <- sort(detected_scores)
    not_detected_scores_sorted <- sort(not_detected_scores)

    list(
      n_detected = length(detected_scores),
      n_not_detected = length(not_detected_scores),
      detected_scores = detected_scores,
      not_detected_scores = not_detected_scores,
      detected_scores_sorted = detected_scores_sorted,
      not_detected_scores_sorted = not_detected_scores_sorted
    )
  }

  # Calibrate both classes
  flower_cal <- calibrate_class("flower", ".pred_flower", "flower")
  fruit_cal <- calibrate_class("fruit", ".pred_fruit", "fruit")

  list(
    flower = flower_cal,
    fruit = fruit_cal
  )
}


# =============================================================================
# Apply Conformal Prediction
# =============================================================================

#' Apply Conformal Prediction to Generate P-Values
#'
#' Calculate conformal p-values for all predictions using calibration scores.
#' P-value = proportion of calibration scores >= test score.
#' Uses vectorized binary search for efficiency.
#'
#' @param predictions data.frame with all predictions
#' @param calibration list from calculate_conformal_calibration()
#'
#' @return data.frame with original columns plus:
#'   - .conf_pval_flower_detected
#'   - .conf_pval_flower_not_detected
#'   - .conf_pval_fruit_detected
#'   - .conf_pval_fruit_not_detected
#'
#' @export
apply_conformal_prediction <- function(predictions, calibration) {

  message("Applying conformal prediction to ", nrow(predictions), " samples")

  # Vectorized p-value computation using binary search
  compute_pvalues <- function(test_scores, sorted_cal) {
    n <- length(sorted_cal)
    # Count calibration scores < test_score (binary search)
    n_less <- findInterval(test_scores, sorted_cal, left.open = TRUE)
    # Count calibration scores >= test_score
    n_geq <- n - n_less
    # Conformal p-value (add 1 for conservativeness)
    (n_geq + 1) / (n + 1)
  }

  # Calculate p-values for each class (fully vectorized)
  result <- predictions %>%
    dplyr::mutate(
      # FLOWER p-values (vectorized)
      .conf_pval_flower_detected = compute_pvalues(
        1 - .pred_flower,  # conformity score for detected hypothesis
        calibration$flower$detected_scores_sorted
      ),
      .conf_pval_flower_not_detected = compute_pvalues(
        .pred_flower,  # conformity score for not_detected hypothesis
        calibration$flower$not_detected_scores_sorted
      ),

      # FRUIT p-values (vectorized)
      .conf_pval_fruit_detected = compute_pvalues(
        1 - .pred_fruit,  # conformity score for detected hypothesis
        calibration$fruit$detected_scores_sorted
      ),
      .conf_pval_fruit_not_detected = compute_pvalues(
        .pred_fruit,  # conformity score for not_detected hypothesis
        calibration$fruit$not_detected_scores_sorted
      )
    )

  message("  Added 4 p-value columns")

  return(result)
}


# =============================================================================
# Coverage Check
# =============================================================================

#' Check Conformal Coverage
#'
#' Verify that conformal prediction achieves expected coverage at different
#' alpha levels. Mondrian approach checks coverage separately for each true label.
#'
#' @param test_preds_with_pvals data.frame with test predictions and p-values
#' @param alpha_levels numeric vector of alpha values to check
#'
#' @return data.frame with coverage results for each (class, true_label, alpha)
#'
#' @export
check_conformal_coverage <- function(test_preds_with_pvals,
                                      alpha_levels = c(0.05, 0.10, 0.15, 0.20)) {

  # Filter to test set only
  test_data <- test_preds_with_pvals %>%
    dplyr::filter(partition == "testing")

  if (nrow(test_data) == 0) {
    stop("No test predictions found in data")
  }

  message("Checking conformal coverage on ", nrow(test_data), " test samples")

  results <- list()

  for (alpha in alpha_levels) {
    # ===========================================================================
    # FLOWER coverage (Mondrian - separate by true label)
    # ===========================================================================

    # Detected (y=1)
    flower_detected <- test_data %>%
      dplyr::filter(flower == "1")

    coverage_flower_detected <- mean(flower_detected$.conf_pval_flower_detected > alpha)

    results[[length(results) + 1]] <- tibble::tibble(
      class = "flower",
      true_label = "1",
      alpha = alpha,
      expected_coverage = 1 - alpha,
      coverage = coverage_flower_detected,
      n = nrow(flower_detected)
    )

    # Not detected (y=0)
    flower_not_detected <- test_data %>%
      dplyr::filter(flower == "0")

    coverage_flower_not_detected <- mean(flower_not_detected$.conf_pval_flower_not_detected > alpha)

    results[[length(results) + 1]] <- tibble::tibble(
      class = "flower",
      true_label = "0",
      alpha = alpha,
      expected_coverage = 1 - alpha,
      coverage = coverage_flower_not_detected,
      n = nrow(flower_not_detected)
    )

    # ===========================================================================
    # FRUIT coverage (Mondrian - separate by true label)
    # ===========================================================================

    # Detected (y=1)
    fruit_detected <- test_data %>%
      dplyr::filter(fruit == "1")

    coverage_fruit_detected <- mean(fruit_detected$.conf_pval_fruit_detected > alpha)

    results[[length(results) + 1]] <- tibble::tibble(
      class = "fruit",
      true_label = "1",
      alpha = alpha,
      expected_coverage = 1 - alpha,
      coverage = coverage_fruit_detected,
      n = nrow(fruit_detected)
    )

    # Not detected (y=0)
    fruit_not_detected <- test_data %>%
      dplyr::filter(fruit == "0")

    coverage_fruit_not_detected <- mean(fruit_not_detected$.conf_pval_fruit_not_detected > alpha)

    results[[length(results) + 1]] <- tibble::tibble(
      class = "fruit",
      true_label = "0",
      alpha = alpha,
      expected_coverage = 1 - alpha,
      coverage = coverage_fruit_not_detected,
      n = nrow(fruit_not_detected)
    )
  }

  coverage_df <- dplyr::bind_rows(results)

  message("Coverage check complete for ", length(alpha_levels), " alpha levels")

  return(coverage_df)
}


# =============================================================================
# Alpha Cutoff Analysis (Original Combined Approach)
# =============================================================================

#' Analyze Alpha Cutoffs (Combined Approach)
#'
#' Original approach: uses same alpha for both hypotheses (OR logic).
#' Keep sample if EITHER p_detected > alpha OR p_not_detected > alpha.
#'
#' @param test_preds_with_pvals data.frame with test predictions and p-values
#' @param alpha_values numeric vector of alpha values to test
#'
#' @return data.frame with one row per (alpha, class) combination
#'
#' @export
analyze_alpha_cutoffs <- function(test_preds_with_pvals,
                                   alpha_values = c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)) {

  message("Analyzing alpha cutoffs (combined approach): ", length(alpha_values), " values")

  results <- list()

  for (alpha in alpha_values) {
    flower_result <- analyze_single_alpha(test_preds_with_pvals, alpha, "flower")
    fruit_result <- analyze_single_alpha(test_preds_with_pvals, alpha, "fruit")

    results[[length(results) + 1]] <- flower_result
    results[[length(results) + 1]] <- fruit_result
  }

  dplyr::bind_rows(results)
}


#' Analyze Single Alpha Value (Helper Function)
#'
#' @param test_preds_with_pvals data.frame with test predictions and p-values
#' @param alpha numeric alpha value
#' @param class_name character "flower" or "fruit"
#'
#' @return tibble with metrics for this (alpha, class) combination
#'
#' @keywords internal
analyze_single_alpha <- function(test_preds_with_pvals, alpha, class_name) {

  # Filter to test set
  test_data <- test_preds_with_pvals %>%
    dplyr::filter(partition == "testing")

  # Get column names for this class
  truth_col <- class_name
  pred_col <- paste0(".pred_", class_name)
  pval_detected_col <- paste0(".conf_pval_", class_name, "_detected")
  pval_not_detected_col <- paste0(".conf_pval_", class_name, "_not_detected")

  # Apply filtering (OR logic)
  class_data <- test_data %>%
    dplyr::mutate(
      # Keep if EITHER hypothesis passes threshold (permissive)
      keep = (!!rlang::sym(pval_detected_col) > alpha) | (!!rlang::sym(pval_not_detected_col) > alpha),

      # Predict based on which has higher p-value
      .conf_pred = dplyr::case_when(
        !keep ~ NA_character_,
        !!rlang::sym(pval_detected_col) >= !!rlang::sym(pval_not_detected_col) ~ "1",
        TRUE ~ "0"
      )
    )

  # Filter to kept samples
  kept_data <- class_data %>% dplyr::filter(keep)

  n_total <- nrow(class_data)
  n_kept <- nrow(kept_data)
  n_removed <- n_total - n_kept
  pct_removed <- 100 * n_removed / n_total

  # Calculate metrics
  if (n_kept > 10) {
    truth <- factor(kept_data[[truth_col]], levels = c("1", "0"))
    pred <- factor(kept_data$.conf_pred, levels = c("1", "0"))
    prob <- kept_data[[pred_col]]

    tibble::tibble(
      alpha = alpha,
      class = class_name,
      n_total = n_total,
      n_kept = n_kept,
      n_removed = n_removed,
      pct_removed = pct_removed,
      accuracy = yardstick::accuracy_vec(truth, pred),
      j_index = yardstick::j_index_vec(truth, pred, event_level = "first"),
      sensitivity = yardstick::sens_vec(truth, pred, event_level = "first"),
      specificity = yardstick::spec_vec(truth, pred, event_level = "first"),
      ppv = yardstick::ppv_vec(truth, pred, event_level = "first"),
      npv = yardstick::npv_vec(truth, pred, event_level = "first"),
      auc_roc = yardstick::roc_auc_vec(truth, prob, event_level = "first")
    )
  } else {
    tibble::tibble(
      alpha = alpha, class = class_name,
      n_total = n_total, n_kept = n_kept, n_removed = n_removed,
      pct_removed = pct_removed,
      accuracy = NA_real_, j_index = NA_real_,
      sensitivity = NA_real_, specificity = NA_real_,
      ppv = NA_real_, npv = NA_real_, auc_roc = NA_real_
    )
  }
}


# =============================================================================
# Independent Alpha Analysis (Positive Predictions)
# =============================================================================

#' Analyze Alpha Cutoffs for Positive Predictions
#'
#' Optimize alpha cutoff specifically for positive (detected) predictions.
#' Keep ONLY samples where p_detected > alpha (ignoring p_not_detected).
#'
#' NOTE: This independent approach is FLAWED (see factorial approach below).
#'
#' @param test_preds_with_pvals data.frame with test predictions and p-values
#' @param alpha_values numeric vector of alpha values to test
#'
#' @return data.frame with one row per (alpha, class) combination
#'
#' @export
analyze_alpha_positive <- function(test_preds_with_pvals,
                                    alpha_values = c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)) {

  message("Analyzing alpha cutoffs for POSITIVE predictions: ", length(alpha_values), " values")

  # Filter to test set
  test_data <- test_preds_with_pvals %>%
    dplyr::filter(partition == "testing")

  results <- list()

  for (alpha in alpha_values) {
    # ===========================================================================
    # FLOWER (positive only)
    # ===========================================================================

    flower_data <- test_data %>%
      dplyr::mutate(
        keep = .conf_pval_flower_detected > alpha,
        .conf_pred = ifelse(keep, "1", NA_character_)
      )

    kept_flower <- flower_data %>% dplyr::filter(keep)

    if (nrow(kept_flower) > 10) {
      truth_flower <- factor(kept_flower$flower, levels = c("1", "0"))
      pred_flower <- factor(kept_flower$.conf_pred, levels = c("1", "0"))
      prob_flower <- kept_flower$.pred_flower

      flower_metrics <- tibble::tibble(
        alpha = alpha,
        class = "flower",
        n_total = nrow(flower_data),
        n_kept = nrow(kept_flower),
        n_removed = nrow(flower_data) - nrow(kept_flower),
        pct_removed = 100 * (nrow(flower_data) - nrow(kept_flower)) / nrow(flower_data),
        accuracy = yardstick::accuracy_vec(truth_flower, pred_flower),
        j_index = yardstick::j_index_vec(truth_flower, pred_flower, event_level = "first"),
        sensitivity = yardstick::sens_vec(truth_flower, pred_flower, event_level = "first"),
        specificity = yardstick::spec_vec(truth_flower, pred_flower, event_level = "first"),
        ppv = yardstick::ppv_vec(truth_flower, pred_flower, event_level = "first"),
        npv = yardstick::npv_vec(truth_flower, pred_flower, event_level = "first"),
        auc_roc = yardstick::roc_auc_vec(truth_flower, prob_flower, event_level = "first")
      )
    } else {
      flower_metrics <- tibble::tibble(
        alpha = alpha, class = "flower",
        n_total = nrow(flower_data), n_kept = nrow(kept_flower),
        n_removed = nrow(flower_data) - nrow(kept_flower),
        pct_removed = 100 * (nrow(flower_data) - nrow(kept_flower)) / nrow(flower_data),
        accuracy = NA_real_, j_index = NA_real_,
        sensitivity = NA_real_, specificity = NA_real_,
        ppv = NA_real_, npv = NA_real_, auc_roc = NA_real_
      )
    }

    results[[length(results) + 1]] <- flower_metrics

    # ===========================================================================
    # FRUIT (positive only)
    # ===========================================================================

    fruit_data <- test_data %>%
      dplyr::mutate(
        keep = .conf_pval_fruit_detected > alpha,
        .conf_pred = ifelse(keep, "1", NA_character_)
      )

    kept_fruit <- fruit_data %>% dplyr::filter(keep)

    if (nrow(kept_fruit) > 10) {
      truth_fruit <- factor(kept_fruit$fruit, levels = c("1", "0"))
      pred_fruit <- factor(kept_fruit$.conf_pred, levels = c("1", "0"))
      prob_fruit <- kept_fruit$.pred_fruit

      fruit_metrics <- tibble::tibble(
        alpha = alpha,
        class = "fruit",
        n_total = nrow(fruit_data),
        n_kept = nrow(kept_fruit),
        n_removed = nrow(fruit_data) - nrow(kept_fruit),
        pct_removed = 100 * (nrow(fruit_data) - nrow(kept_fruit)) / nrow(fruit_data),
        accuracy = yardstick::accuracy_vec(truth_fruit, pred_fruit),
        j_index = yardstick::j_index_vec(truth_fruit, pred_fruit, event_level = "first"),
        sensitivity = yardstick::sens_vec(truth_fruit, pred_fruit, event_level = "first"),
        specificity = yardstick::spec_vec(truth_fruit, pred_fruit, event_level = "first"),
        ppv = yardstick::ppv_vec(truth_fruit, pred_fruit, event_level = "first"),
        npv = yardstick::npv_vec(truth_fruit, pred_fruit, event_level = "first"),
        auc_roc = yardstick::roc_auc_vec(truth_fruit, prob_fruit, event_level = "first")
      )
    } else {
      fruit_metrics <- tibble::tibble(
        alpha = alpha, class = "fruit",
        n_total = nrow(fruit_data), n_kept = nrow(kept_fruit),
        n_removed = nrow(fruit_data) - nrow(kept_fruit),
        pct_removed = 100 * (nrow(fruit_data) - nrow(kept_fruit)) / nrow(fruit_data),
        accuracy = NA_real_, j_index = NA_real_,
        sensitivity = NA_real_, specificity = NA_real_,
        ppv = NA_real_, npv = NA_real_, auc_roc = NA_real_
      )
    }

    results[[length(results) + 1]] <- fruit_metrics
  }

  dplyr::bind_rows(results)
}


# =============================================================================
# Independent Alpha Analysis (Negative Predictions)
# =============================================================================

#' Analyze Alpha Cutoffs for Negative Predictions
#'
#' Optimize alpha cutoff specifically for negative (not detected) predictions.
#' Keep ONLY samples where p_not_detected > alpha (ignoring p_detected).
#'
#' NOTE: This independent approach is FLAWED (see factorial approach below).
#'
#' @param test_preds_with_pvals data.frame with test predictions and p-values
#' @param alpha_values numeric vector of alpha values to test
#'
#' @return data.frame with one row per (alpha, class) combination
#'
#' @export
analyze_alpha_negative <- function(test_preds_with_pvals,
                                    alpha_values = c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)) {

  message("Analyzing alpha cutoffs for NEGATIVE predictions: ", length(alpha_values), " values")

  # Filter to test set
  test_data <- test_preds_with_pvals %>%
    dplyr::filter(partition == "testing")

  results <- list()

  for (alpha in alpha_values) {
    # ===========================================================================
    # FLOWER (negative only)
    # ===========================================================================

    flower_data <- test_data %>%
      dplyr::mutate(
        keep = .conf_pval_flower_not_detected > alpha,
        .conf_pred = ifelse(keep, "0", NA_character_)
      )

    kept_flower <- flower_data %>% dplyr::filter(keep)

    if (nrow(kept_flower) > 10) {
      truth_flower <- factor(kept_flower$flower, levels = c("1", "0"))
      pred_flower <- factor(kept_flower$.conf_pred, levels = c("1", "0"))
      prob_flower <- kept_flower$.pred_flower

      flower_metrics <- tibble::tibble(
        alpha = alpha,
        class = "flower",
        n_total = nrow(flower_data),
        n_kept = nrow(kept_flower),
        n_removed = nrow(flower_data) - nrow(kept_flower),
        pct_removed = 100 * (nrow(flower_data) - nrow(kept_flower)) / nrow(flower_data),
        accuracy = yardstick::accuracy_vec(truth_flower, pred_flower),
        j_index = yardstick::j_index_vec(truth_flower, pred_flower, event_level = "first"),
        sensitivity = yardstick::sens_vec(truth_flower, pred_flower, event_level = "first"),
        specificity = yardstick::spec_vec(truth_flower, pred_flower, event_level = "first"),
        ppv = yardstick::ppv_vec(truth_flower, pred_flower, event_level = "first"),
        npv = yardstick::npv_vec(truth_flower, pred_flower, event_level = "first"),
        auc_roc = yardstick::roc_auc_vec(truth_flower, prob_flower, event_level = "first")
      )
    } else {
      flower_metrics <- tibble::tibble(
        alpha = alpha, class = "flower",
        n_total = nrow(flower_data), n_kept = nrow(kept_flower),
        n_removed = nrow(flower_data) - nrow(kept_flower),
        pct_removed = 100 * (nrow(flower_data) - nrow(kept_flower)) / nrow(flower_data),
        accuracy = NA_real_, j_index = NA_real_,
        sensitivity = NA_real_, specificity = NA_real_,
        ppv = NA_real_, npv = NA_real_, auc_roc = NA_real_
      )
    }

    results[[length(results) + 1]] <- flower_metrics

    # ===========================================================================
    # FRUIT (negative only)
    # ===========================================================================

    fruit_data <- test_data %>%
      dplyr::mutate(
        keep = .conf_pval_fruit_not_detected > alpha,
        .conf_pred = ifelse(keep, "0", NA_character_)
      )

    kept_fruit <- fruit_data %>% dplyr::filter(keep)

    if (nrow(kept_fruit) > 10) {
      truth_fruit <- factor(kept_fruit$fruit, levels = c("1", "0"))
      pred_fruit <- factor(kept_fruit$.conf_pred, levels = c("1", "0"))
      prob_fruit <- kept_fruit$.pred_fruit

      fruit_metrics <- tibble::tibble(
        alpha = alpha,
        class = "fruit",
        n_total = nrow(fruit_data),
        n_kept = nrow(kept_fruit),
        n_removed = nrow(fruit_data) - nrow(kept_fruit),
        pct_removed = 100 * (nrow(fruit_data) - nrow(kept_fruit)) / nrow(fruit_data),
        accuracy = yardstick::accuracy_vec(truth_fruit, pred_fruit),
        j_index = yardstick::j_index_vec(truth_fruit, pred_fruit, event_level = "first"),
        sensitivity = yardstick::sens_vec(truth_fruit, pred_fruit, event_level = "first"),
        specificity = yardstick::spec_vec(truth_fruit, pred_fruit, event_level = "first"),
        ppv = yardstick::ppv_vec(truth_fruit, pred_fruit, event_level = "first"),
        npv = yardstick::npv_vec(truth_fruit, pred_fruit, event_level = "first"),
        auc_roc = yardstick::roc_auc_vec(truth_fruit, prob_fruit, event_level = "first")
      )
    } else {
      fruit_metrics <- tibble::tibble(
        alpha = alpha, class = "fruit",
        n_total = nrow(fruit_data), n_kept = nrow(kept_fruit),
        n_removed = nrow(fruit_data) - nrow(kept_fruit),
        pct_removed = 100 * (nrow(fruit_data) - nrow(kept_fruit)) / nrow(fruit_data),
        accuracy = NA_real_, j_index = NA_real_,
        sensitivity = NA_real_, specificity = NA_real_,
        ppv = NA_real_, npv = NA_real_, auc_roc = NA_real_
      )
    }

    results[[length(results) + 1]] <- fruit_metrics
  }

  dplyr::bind_rows(results)
}


# =============================================================================
# Get Recommended Alpha
# =============================================================================

#' Get Recommended Alpha Cutoffs
#'
#' Select recommended alpha cutoff based on accuracy-removal trade-off.
#'
#' @param alpha_analysis data.frame from analyze_alpha_cutoffs()
#' @param max_removal maximum percent removal allowed (default 15)
#' @param min_accuracy_gain minimum accuracy gain to justify removal (default 0.005)
#'
#' @return data.frame with recommended alpha for each class
#'
#' @export
get_recommended_alpha <- function(alpha_analysis,
                                   max_removal = 15,
                                   min_accuracy_gain = 0.005) {

  message("Selecting recommended alpha cutoffs")

  # Helper for one class
  get_class_alpha <- function(class_data, class_name) {
    # Sort by alpha
    class_data <- class_data %>% dplyr::arrange(alpha)

    # Baseline accuracy (alpha = 0, no removal)
    baseline_accuracy <- class_data$accuracy[1]

    # Find alpha with best trade-off
    best_alpha <- class_data %>%
      dplyr::filter(
        pct_removed <= max_removal,
        accuracy >= baseline_accuracy + min_accuracy_gain
      ) %>%
      dplyr::arrange(dplyr::desc(accuracy), pct_removed) %>%
      dplyr::slice(1)

    if (nrow(best_alpha) == 0) {
      # No improvement found, use alpha = 0
      best_alpha <- class_data %>% dplyr::slice(1)
    }

    message(sprintf("  %s: alpha=%.2f, accuracy=%.4f, removal=%.1f%%",
                    class_name, best_alpha$alpha, best_alpha$accuracy, best_alpha$pct_removed))

    best_alpha
  }

  # Get recommended for both classes
  flower_rec <- get_class_alpha(
    alpha_analysis %>% dplyr::filter(class == "flower"),
    "flower"
  )
  fruit_rec <- get_class_alpha(
    alpha_analysis %>% dplyr::filter(class == "fruit"),
    "fruit"
  )

  dplyr::bind_rows(flower_rec, fruit_rec)
}


# =============================================================================
# FACTORIAL ALPHA OPTIMIZATION (NEW APPROACH)
# =============================================================================

#' Factorial Alpha Optimization for Conformal Prediction
#'
#' Test all combinations of positive and negative alpha cutoffs to find optimal
#' trade-off between prediction quality (PPV/NPV) and data retention.
#'
#' **Filtering logic (conservative):**
#' - Keep sample if EXACTLY ONE p-value exceeds threshold (confident about one class)
#' - Remove if BOTH p-values below thresholds (uncertain about both)
#' - Remove if BOTH p-values above thresholds (contradictory - confident about both)
#'
#' @param test_preds_with_pvals data.frame with conformal p-values (.conf_pval_*)
#' @param alpha_pos_values numeric vector of alpha values for positive (detected) hypothesis
#' @param alpha_neg_values numeric vector of alpha values for negative (not detected) hypothesis
#'
#' @return data.frame with one row per (alpha_pos, alpha_neg, class) combination containing:
#'   - alpha_pos, alpha_neg, class
#'   - n_total, n_kept, n_removed, pct_removed
#'   - n_detected_total, n_detected_removed, pct_detected_removed (detection data loss)
#'   - n_not_detected_total, n_not_detected_removed, pct_not_detected_removed (non-detection data loss)
#'   - n_both_low (uncertain), n_both_high (contradictory)
#'   - accuracy, j_index, dqi, sensitivity, specificity, ppv, npv, auc_roc
#'
#' @export
analyze_alpha_factorial <- function(test_preds_with_pvals,
                                    alpha_pos_values = c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
                                    alpha_neg_values = c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)) {

  message("Factorial alpha analysis: ", length(alpha_pos_values), " × ", length(alpha_neg_values),
          " = ", length(alpha_pos_values) * length(alpha_neg_values), " combinations per class")

  results <- list()

  for (alpha_pos in alpha_pos_values) {
    for (alpha_neg in alpha_neg_values) {

      # ===========================================================================
      # FLOWER ANALYSIS
      # ===========================================================================

      flower_data <- test_preds_with_pvals %>%
        dplyr::mutate(
          pos_above = .conf_pval_flower_detected > alpha_pos,
          neg_above = .conf_pval_flower_not_detected > alpha_neg,

          # Determine if kept or removed (CONSERVATIVE logic)
          ambiguous = dplyr::case_when(
            # Case 1: Both BELOW threshold → uncertain about both → REMOVE
            !pos_above & !neg_above ~ TRUE,
            # Case 2: Both ABOVE threshold → confident about both (contradictory) → REMOVE
            pos_above & neg_above ~ TRUE,
            # Case 3: Exactly ONE above → confident → KEEP
            TRUE ~ FALSE
          ),

          # For kept samples, predict the class with p-value ABOVE threshold
          .conf_pred = dplyr::case_when(
            ambiguous ~ NA_character_,
            pos_above & !neg_above ~ "1",   # Only detected is confident
            !pos_above & neg_above ~ "0",   # Only not-detected is confident
            TRUE ~ NA_character_  # Should never reach here
          ),

          # Track WHY sample was removed
          both_low = !pos_above & !neg_above,
          both_high = pos_above & neg_above
        )

      # Count ambiguity types
      n_both_low_flower <- sum(flower_data$both_low)
      n_both_high_flower <- sum(flower_data$both_high)

      # Filter to kept samples
      kept_flower <- flower_data %>% dplyr::filter(!ambiguous)

      n_total_flower <- nrow(flower_data)
      n_kept_flower <- nrow(kept_flower)
      n_removed_flower <- n_total_flower - n_kept_flower
      pct_removed_flower <- 100 * n_removed_flower / n_total_flower

      # Calculate detection-specific data loss (important for presence-only approach)
      n_detected_total_flower <- sum(flower_data$flower == "1")
      n_detected_removed_flower <- sum(flower_data$flower == "1" & flower_data$ambiguous)
      pct_detected_removed_flower <- if (n_detected_total_flower > 0) {
        100 * n_detected_removed_flower / n_detected_total_flower
      } else NA_real_

      n_not_detected_total_flower <- sum(flower_data$flower == "0")
      n_not_detected_removed_flower <- sum(flower_data$flower == "0" & flower_data$ambiguous)
      pct_not_detected_removed_flower <- if (n_not_detected_total_flower > 0) {
        100 * n_not_detected_removed_flower / n_not_detected_total_flower
      } else NA_real_

      # Calculate metrics on kept samples
      if (n_kept_flower > 10) {  # Need minimum samples for valid metrics
        truth_flower <- factor(kept_flower$flower, levels = c("1", "0"))
        pred_flower <- factor(kept_flower$.conf_pred, levels = c("1", "0"))
        prob_flower <- kept_flower$.pred_flower

        # Calculate PPV and Sensitivity first for DQI
        ppv_flower <- yardstick::ppv_vec(truth_flower, pred_flower, event_level = "first")
        sens_flower <- yardstick::sens_vec(truth_flower, pred_flower, event_level = "first")

        flower_metrics <- tibble::tibble(
          alpha_pos = alpha_pos,
          alpha_neg = alpha_neg,
          class = "flower",
          n_total = n_total_flower,
          n_kept = n_kept_flower,
          n_removed = n_removed_flower,
          pct_removed = pct_removed_flower,

          # Detection-specific data loss (presence-only approach)
          n_detected_total = n_detected_total_flower,
          n_detected_removed = n_detected_removed_flower,
          pct_detected_removed = pct_detected_removed_flower,
          n_not_detected_total = n_not_detected_total_flower,
          n_not_detected_removed = n_not_detected_removed_flower,
          pct_not_detected_removed = pct_not_detected_removed_flower,

          n_both_low = n_both_low_flower,
          n_both_high = n_both_high_flower,

          # Overall metrics
          accuracy = yardstick::accuracy_vec(truth_flower, pred_flower),
          j_index = yardstick::j_index_vec(truth_flower, pred_flower, event_level = "first"),
          dqi = ppv_flower + sens_flower - 1,  # Detection Quality Index

          # Detection metrics
          sensitivity = sens_flower,
          specificity = yardstick::spec_vec(truth_flower, pred_flower, event_level = "first"),
          ppv = ppv_flower,
          npv = yardstick::npv_vec(truth_flower, pred_flower, event_level = "first"),

          # AUC (uses probabilities, not conformal predictions)
          auc_roc = yardstick::roc_auc_vec(truth_flower, prob_flower, event_level = "first")
        )
      } else {
        # Not enough data for valid metrics
        flower_metrics <- tibble::tibble(
          alpha_pos = alpha_pos, alpha_neg = alpha_neg, class = "flower",
          n_total = n_total_flower, n_kept = n_kept_flower, n_removed = n_removed_flower,
          pct_removed = pct_removed_flower,
          # Detection-specific data loss
          n_detected_total = n_detected_total_flower,
          n_detected_removed = n_detected_removed_flower,
          pct_detected_removed = pct_detected_removed_flower,
          n_not_detected_total = n_not_detected_total_flower,
          n_not_detected_removed = n_not_detected_removed_flower,
          pct_not_detected_removed = pct_not_detected_removed_flower,
          n_both_low = n_both_low_flower, n_both_high = n_both_high_flower,
          accuracy = NA_real_, j_index = NA_real_, dqi = NA_real_,
          sensitivity = NA_real_, specificity = NA_real_,
          ppv = NA_real_, npv = NA_real_, auc_roc = NA_real_
        )
      }

      # ===========================================================================
      # FRUIT ANALYSIS (same logic)
      # ===========================================================================

      fruit_data <- test_preds_with_pvals %>%
        dplyr::mutate(
          pos_above = .conf_pval_fruit_detected > alpha_pos,
          neg_above = .conf_pval_fruit_not_detected > alpha_neg,

          ambiguous = dplyr::case_when(
            !pos_above & !neg_above ~ TRUE,
            pos_above & neg_above ~ TRUE,
            TRUE ~ FALSE
          ),

          .conf_pred = dplyr::case_when(
            ambiguous ~ NA_character_,
            pos_above & !neg_above ~ "1",
            !pos_above & neg_above ~ "0",
            TRUE ~ NA_character_
          ),

          both_low = !pos_above & !neg_above,
          both_high = pos_above & neg_above
        )

      n_both_low_fruit <- sum(fruit_data$both_low)
      n_both_high_fruit <- sum(fruit_data$both_high)

      kept_fruit <- fruit_data %>% dplyr::filter(!ambiguous)

      n_total_fruit <- nrow(fruit_data)
      n_kept_fruit <- nrow(kept_fruit)
      n_removed_fruit <- n_total_fruit - n_kept_fruit
      pct_removed_fruit <- 100 * n_removed_fruit / n_total_fruit

      # Calculate detection-specific data loss (important for presence-only approach)
      n_detected_total_fruit <- sum(fruit_data$fruit == "1")
      n_detected_removed_fruit <- sum(fruit_data$fruit == "1" & fruit_data$ambiguous)
      pct_detected_removed_fruit <- if (n_detected_total_fruit > 0) {
        100 * n_detected_removed_fruit / n_detected_total_fruit
      } else NA_real_

      n_not_detected_total_fruit <- sum(fruit_data$fruit == "0")
      n_not_detected_removed_fruit <- sum(fruit_data$fruit == "0" & fruit_data$ambiguous)
      pct_not_detected_removed_fruit <- if (n_not_detected_total_fruit > 0) {
        100 * n_not_detected_removed_fruit / n_not_detected_total_fruit
      } else NA_real_

      if (n_kept_fruit > 10) {
        truth_fruit <- factor(kept_fruit$fruit, levels = c("1", "0"))
        pred_fruit <- factor(kept_fruit$.conf_pred, levels = c("1", "0"))
        prob_fruit <- kept_fruit$.pred_fruit

        # Calculate PPV and Sensitivity first for DQI
        ppv_fruit <- yardstick::ppv_vec(truth_fruit, pred_fruit, event_level = "first")
        sens_fruit <- yardstick::sens_vec(truth_fruit, pred_fruit, event_level = "first")

        fruit_metrics <- tibble::tibble(
          alpha_pos = alpha_pos,
          alpha_neg = alpha_neg,
          class = "fruit",
          n_total = n_total_fruit,
          n_kept = n_kept_fruit,
          n_removed = n_removed_fruit,
          pct_removed = pct_removed_fruit,

          # Detection-specific data loss (presence-only approach)
          n_detected_total = n_detected_total_fruit,
          n_detected_removed = n_detected_removed_fruit,
          pct_detected_removed = pct_detected_removed_fruit,
          n_not_detected_total = n_not_detected_total_fruit,
          n_not_detected_removed = n_not_detected_removed_fruit,
          pct_not_detected_removed = pct_not_detected_removed_fruit,

          n_both_low = n_both_low_fruit,
          n_both_high = n_both_high_fruit,

          accuracy = yardstick::accuracy_vec(truth_fruit, pred_fruit),
          j_index = yardstick::j_index_vec(truth_fruit, pred_fruit, event_level = "first"),
          dqi = ppv_fruit + sens_fruit - 1,  # Detection Quality Index
          sensitivity = sens_fruit,
          specificity = yardstick::spec_vec(truth_fruit, pred_fruit, event_level = "first"),
          ppv = ppv_fruit,
          npv = yardstick::npv_vec(truth_fruit, pred_fruit, event_level = "first"),
          auc_roc = yardstick::roc_auc_vec(truth_fruit, prob_fruit, event_level = "first")
        )
      } else {
        fruit_metrics <- tibble::tibble(
          alpha_pos = alpha_pos, alpha_neg = alpha_neg, class = "fruit",
          n_total = n_total_fruit, n_kept = n_kept_fruit, n_removed = n_removed_fruit,
          pct_removed = pct_removed_fruit,
          # Detection-specific data loss
          n_detected_total = n_detected_total_fruit,
          n_detected_removed = n_detected_removed_fruit,
          pct_detected_removed = pct_detected_removed_fruit,
          n_not_detected_total = n_not_detected_total_fruit,
          n_not_detected_removed = n_not_detected_removed_fruit,
          pct_not_detected_removed = pct_not_detected_removed_fruit,
          n_both_low = n_both_low_fruit, n_both_high = n_both_high_fruit,
          accuracy = NA_real_, j_index = NA_real_, dqi = NA_real_,
          sensitivity = NA_real_, specificity = NA_real_,
          ppv = NA_real_, npv = NA_real_, auc_roc = NA_real_
        )
      }

      results[[length(results) + 1]] <- flower_metrics
      results[[length(results) + 1]] <- fruit_metrics
    }
  }

  message("Factorial analysis complete: ", length(results), " total combinations")

  dplyr::bind_rows(results)
}


# =============================================================================
# Find Optimal Alpha Cutoffs
# =============================================================================

#' Find Optimal Alpha Cutoffs Using Multiple Methods
#'
#' Analyze DQI-vs-detection-data-loss trade-off curve to find optimal alpha combinations
#' using multiple optimization methods. DQI (Detection Quality Index) = PPV + Sensitivity - 1.
#'
#' Uses **detection data loss** (pct_detected_removed) rather than overall data loss
#' for the presence-only approach, where losing true detections is more costly.
#'
#' **Methods:**
#' 1. Conservative: Highest DQI within max_detection_loss constraint
#' 2. Efficiency Ratio: Where ΔDQI/Δdetection_loss drops below threshold
#' 3. Inflection Point: Where second derivative = 0 (diminishing returns)
#' 4. Elbow: Maximum distance to line connecting first and last points
#'
#' @param factorial_results data.frame from analyze_alpha_factorial()
#' @param max_detection_loss maximum percent detection loss allowed (default 20)
#' @param efficiency_threshold threshold for efficiency ratio method (default 0.5)
#'
#' @return data.frame with top candidates per class, including:
#'   - class, method, rank
#'   - alpha_pos, alpha_neg
#'   - dqi, sensitivity, ppv, npv, accuracy, j_index
#'   - pct_detected_removed, pct_not_detected_removed, pct_removed
#'   - n_both_low, n_both_high
#'   - efficiency_ratio (for method 2)
#'
#' @export
find_optimal_alphas <- function(factorial_results,
                                 max_detection_loss = 20,
                                 efficiency_threshold = 0.5) {

  message("Finding optimal alpha cutoffs using 4 methods (optimizing for detection loss)")

  # Process each class separately
  classes <- unique(factorial_results$class)
  all_candidates <- list()

  for (cls in classes) {
    message("\n", toupper(cls), " optimization:")

    # Filter to this class and apply DETECTION loss constraint (not overall loss)
    class_data <- factorial_results %>%
      dplyr::filter(class == cls, pct_detected_removed <= max_detection_loss, !is.na(dqi))

    if (nrow(class_data) == 0) {
      message("  No valid combinations within constraint")
      next
    }

    # Sort by DQI (descending), then by detection loss (ascending)
    class_data <- class_data %>%
      dplyr::arrange(dplyr::desc(dqi), pct_detected_removed)

    # Find Pareto frontier (non-dominated solutions)
    pareto_points <- class_data[1, ]  # Start with highest DQI

    for (i in 2:nrow(class_data)) {
      current <- class_data[i, ]
      last_pareto <- pareto_points[nrow(pareto_points), ]

      # Add if it has lower detection loss (can't have higher DQI, already sorted)
      if (current$pct_detected_removed < last_pareto$pct_detected_removed) {
        pareto_points <- dplyr::bind_rows(pareto_points, current)
      }
    }

    # Sort Pareto frontier by detection loss for curve analysis
    pareto_points <- pareto_points %>%
      dplyr::arrange(pct_detected_removed)

    message("  Pareto frontier: ", nrow(pareto_points), " points")

    # =========================================================================
    # METHOD 1: Conservative (highest DQI within constraint)
    # =========================================================================

    conservative <- pareto_points %>%
      dplyr::slice_max(dqi, n = 1, with_ties = FALSE) %>%
      dplyr::mutate(method = "conservative", rank = 1)

    message(sprintf("  Method 1 (Conservative): DQI=%.4f, sens=%.4f, det_loss=%.1f%%",
                    conservative$dqi, conservative$sensitivity, conservative$pct_detected_removed))

    all_candidates[[length(all_candidates) + 1]] <- conservative

    # =========================================================================
    # METHOD 2: Efficiency Ratio (ΔDQI/Δdata_loss < threshold)
    # =========================================================================

    if (nrow(pareto_points) > 1) {
      # Calculate efficiency ratio between consecutive points (using detection loss)
      pareto_points <- pareto_points %>%
        dplyr::mutate(
          delta_dqi = dplyr::lead(dqi) - dqi,
          delta_loss = dplyr::lead(pct_detected_removed) - pct_detected_removed,
          efficiency_ratio = delta_dqi / delta_loss
        )

      # Find first point where efficiency drops below threshold
      efficiency_optimal <- pareto_points %>%
        dplyr::filter(efficiency_ratio < efficiency_threshold) %>%
        dplyr::slice(1) %>%
        dplyr::mutate(method = "efficiency_ratio", rank = 1)

      if (nrow(efficiency_optimal) > 0) {
        message(sprintf("  Method 2 (Efficiency): DQI=%.4f, sens=%.4f, det_loss=%.1f%%, ratio=%.3f",
                        efficiency_optimal$dqi, efficiency_optimal$sensitivity,
                        efficiency_optimal$pct_detected_removed, efficiency_optimal$efficiency_ratio))
        all_candidates[[length(all_candidates) + 1]] <- efficiency_optimal
      } else {
        message("  Method 2 (Efficiency): No point below threshold")
      }
    }

    # =========================================================================
    # METHOD 3: Inflection Point (second derivative = 0)
    # =========================================================================

    if (nrow(pareto_points) > 5) {  # Need enough points for spline
      tryCatch({
        # Fit smooth spline to DQI vs detection loss
        spline_fit <- smooth.spline(pareto_points$pct_detected_removed, pareto_points$dqi, spar = 0.5)

        # Calculate second derivative
        second_deriv <- predict(spline_fit, deriv = 2)

        # Find where second derivative crosses zero
        sign_changes <- which(diff(sign(second_deriv$y)) != 0)

        if (length(sign_changes) > 0) {
          # Use first sign change (first inflection point)
          inflection_x <- second_deriv$x[sign_changes[1]]

          # Find closest Pareto point
          inflection_optimal <- pareto_points %>%
            dplyr::mutate(dist = abs(pct_detected_removed - inflection_x)) %>%
            dplyr::slice_min(dist, n = 1, with_ties = FALSE) %>%
            dplyr::mutate(method = "inflection_point", rank = 1) %>%
            dplyr::select(-dist)

          message(sprintf("  Method 3 (Inflection): DQI=%.4f, sens=%.4f, det_loss=%.1f%%",
                          inflection_optimal$dqi, inflection_optimal$sensitivity,
                          inflection_optimal$pct_detected_removed))
          all_candidates[[length(all_candidates) + 1]] <- inflection_optimal
        } else {
          message("  Method 3 (Inflection): No inflection point found")
        }
      }, error = function(e) {
        message("  Method 3 (Inflection): Failed - ", e$message)
      })
    }

    # =========================================================================
    # METHOD 4: Elbow (maximum distance to first-last line)
    # =========================================================================

    if (nrow(pareto_points) > 2) {
      # Get first and last points
      first_pt <- pareto_points[1, ]
      last_pt <- pareto_points[nrow(pareto_points), ]

      # Calculate distance from each point to line connecting first and last (using detection loss)
      x1 <- first_pt$pct_detected_removed
      y1 <- first_pt$dqi
      x2 <- last_pt$pct_detected_removed
      y2 <- last_pt$dqi

      pareto_points <- pareto_points %>%
        dplyr::mutate(
          dist_to_line = abs((y2 - y1) * pct_detected_removed - (x2 - x1) * dqi + x2 * y1 - y2 * x1) /
            sqrt((y2 - y1)^2 + (x2 - x1)^2)
        )

      elbow_optimal <- pareto_points %>%
        dplyr::slice_max(dist_to_line, n = 1, with_ties = FALSE) %>%
        dplyr::mutate(method = "elbow", rank = 1) %>%
        dplyr::select(-dist_to_line)

      message(sprintf("  Method 4 (Elbow): DQI=%.4f, sens=%.4f, det_loss=%.1f%%",
                      elbow_optimal$dqi, elbow_optimal$sensitivity,
                      elbow_optimal$pct_detected_removed))
      all_candidates[[length(all_candidates) + 1]] <- elbow_optimal
    }
  }

  # Combine all candidates
  if (length(all_candidates) == 0) {
    message("\nNo candidates found")
    return(tibble::tibble())
  }

  candidates_df <- dplyr::bind_rows(all_candidates) %>%
    dplyr::select(class, method, rank, alpha_pos, alpha_neg,
                  dqi, sensitivity, ppv, npv, accuracy, j_index,
                  pct_detected_removed, pct_not_detected_removed, pct_removed,
                  n_both_low, n_both_high,
                  dplyr::everything())

  message("\nFound ", nrow(candidates_df), " total candidates")

  return(candidates_df)
}


# =============================================================================
# Factorial Scatter Plot Helper
# =============================================================================

#' Create Factorial Scatter Plot with Marginals and Text Labels
#'
#' Generates a scatter plot of metric values across (alpha_pos, alpha_neg)
#' combinations with marginal density plots and text labels showing values.
#'
#' @param factorial_data data.frame from analyze_alpha_factorial()
#' @param optimal_candidates data.frame from find_optimal_alphas()
#' @param class_name "flower" or "fruit"
#' @param metric_col column name for the metric to plot (unquoted or string)
#' @param metric_label display name for the metric (e.g., "PPV", "DQI")
#' @param viridis_option viridis color palette option ("plasma", "viridis", etc.)
#' @param title plot title
#' @param output_path file path for saving the plot
#' @param is_percentage if TRUE, format labels as percentages (default FALSE)
#'
#' @return file path to the saved plot (normalized)
#'
#' @export
create_factorial_scatter_plot <- function(factorial_data,
                                          optimal_candidates,
                                          class_name,
                                          metric_col,
                                          metric_label,
                                          viridis_option = "plasma",
                                          title,
                                          output_path,
                                          is_percentage = FALSE) {

  library(ggplot2)
  library(patchwork)

  # Filter data by class
  metric_col_str <- if (is.character(metric_col)) metric_col else deparse(substitute(metric_col))

  plot_data <- factorial_data %>%
    dplyr::filter(class == class_name, !is.na(.data[[metric_col_str]]))

  opt_data <- optimal_candidates %>%
    dplyr::filter(class == class_name)

  # Create text labels (rounded to 2 decimal places, or 1 for percentages)
  if (is_percentage) {
    plot_data <- plot_data %>%
      dplyr::mutate(.label = sprintf("%.1f", .data[[metric_col_str]]))
  } else {
    plot_data <- plot_data %>%
      dplyr::mutate(.label = sprintf("%.2f", .data[[metric_col_str]]))
  }

  # Calculate label offset based on data range
  x_range <- diff(range(plot_data$alpha_pos, na.rm = TRUE))
  y_range <- diff(range(plot_data$alpha_neg, na.rm = TRUE))
  x_offset <- x_range * 0.02
  y_offset <- y_range * 0.03

  # Main scatter plot with text labels
  p_main <- ggplot(plot_data, aes(x = alpha_pos, y = alpha_neg)) +
    geom_point(aes(size = .data[[metric_col_str]], color = .data[[metric_col_str]]), alpha = 0.8) +
    geom_text(aes(label = .label),
              hjust = -0.4, vjust = -0.8,
              size = 2.5, color = "gray30") +
    geom_point(data = opt_data, aes(shape = method),
               size = 5, color = "red", stroke = 1.5) +
    scale_color_viridis_c(name = metric_label, option = viridis_option) +
    scale_size_continuous(name = metric_label, range = c(2, 8), guide = "none") +
    scale_shape_manual(name = "Method",
                       values = c(conservative = 15, efficiency_ratio = 16,
                                  inflection_point = 17, elbow = 18)) +
    coord_cartesian(clip = "off") +
    labs(x = "Alpha (Detected)", y = "Alpha (Not Detected)") +
    theme_minimal() +
    theme(legend.position = "right",
          plot.margin = margin(10, 20, 10, 10))

  # Top marginal: metric vs alpha_pos
  p_top <- ggplot(plot_data, aes(x = alpha_pos, y = .data[[metric_col_str]])) +
    geom_point(aes(color = .data[[metric_col_str]]), alpha = 0.6, size = 2) +
    geom_smooth(method = "loess", se = TRUE, color = "black", linewidth = 0.8) +
    scale_color_viridis_c(option = viridis_option, guide = "none") +
    labs(x = NULL, y = metric_label) +
    theme_minimal() +
    theme(axis.text.x = ggplot2::element_blank(), axis.ticks.x = ggplot2::element_blank())

  # Right marginal: metric vs alpha_neg (flipped)
  p_right <- ggplot(plot_data, aes(x = alpha_neg, y = .data[[metric_col_str]])) +
    geom_point(aes(color = .data[[metric_col_str]]), alpha = 0.6, size = 2) +
    geom_smooth(method = "loess", se = TRUE, color = "black", linewidth = 0.8) +
    scale_color_viridis_c(option = viridis_option, guide = "none") +
    coord_flip() +
    labs(x = NULL, y = metric_label) +
    theme_minimal() +
    theme(axis.text.y = ggplot2::element_blank(), axis.ticks.y = ggplot2::element_blank())

  # Compose with patchwork
  p_combined <- (p_top + plot_spacer()) / (p_main + p_right) +
    plot_layout(widths = c(4, 1), heights = c(1, 4)) +
    plot_annotation(
      title = title,
      subtitle = "Marginal plots show smoothed relationships; values shown at each point"
    )

  ggsave(output_path, p_combined, width = 10, height = 8, dpi = 300)
  normalizePath(output_path)
}


# =============================================================================
# Conformal P-Value Calibration Curve
# =============================================================================

#' Create Conformal P-Value Calibration Curve
#'
#' Creates a plot showing how conformal p-values change across the full range
#' of possible model outputs (0 to 1). This visualizes the relationship between
#' model probability and conformal p-values.
#'
#' For each hypothetical model output value in seq(0, 1, length.out=100):
#' - Detected p-value: proportion of calibration scores <= (1 - model_output)
#' - Not detected p-value: proportion of calibration scores <= model_output
#'
#' @param conformal_calibration list from calculate_conformal_calibration()
#' @param output_path Path to save the plot
#' @param n_points Number of points to evaluate (default 100)
#'
#' @return Path to saved plot file
#'
#' @export
create_conformal_calibration_curve <- function(conformal_calibration,
                                                output_path,
                                                n_points = 100) {

  library(ggplot2)

  message("Creating conformal p-value calibration curve with ", n_points, " points")

  # Create sequence of hypothetical model outputs
  model_outputs <- seq(0, 1, length.out = n_points)

  # Helper function to compute p-value at a given output
  compute_pval <- function(model_output, scores_sorted) {
    # Conformity score for detected hypothesis: 1 - prediction
    # Conformity score for not_detected hypothesis: prediction
    # P-value = (# scores >= test_score + 1) / (n + 1)
    # Using sorted scores for efficiency

    n <- length(scores_sorted)
    if (n == 0) return(NA_real_)

    # Number of calibration scores >= test score
    # This is equivalent to n - (number of scores < test_score)
    # Use findInterval for efficiency with sorted scores
    n_larger_or_equal <- n - findInterval(model_output, scores_sorted) + 1
    (n_larger_or_equal) / (n + 1)
  }

  # Calculate p-values for flower
  flower_detected_scores <- conformal_calibration$flower$detected_scores_sorted
  flower_not_detected_scores <- conformal_calibration$flower$not_detected_scores_sorted

  flower_data <- tibble::tibble(
    model_output = model_outputs,
    class = "Flower"
  ) %>%
    dplyr::mutate(
      # For detected hypothesis: conformity score = 1 - prediction
      pval_detected = purrr::map_dbl(model_output, ~{
        test_score <- 1 - .x
        compute_pval(test_score, flower_detected_scores)
      }),
      # For not detected hypothesis: conformity score = prediction
      pval_not_detected = purrr::map_dbl(model_output, ~{
        test_score <- .x
        compute_pval(test_score, flower_not_detected_scores)
      })
    )

  # Calculate p-values for fruit
  fruit_detected_scores <- conformal_calibration$fruit$detected_scores_sorted
  fruit_not_detected_scores <- conformal_calibration$fruit$not_detected_scores_sorted

  fruit_data <- tibble::tibble(
    model_output = model_outputs,
    class = "Fruit"
  ) %>%
    dplyr::mutate(
      pval_detected = purrr::map_dbl(model_output, ~{
        test_score <- 1 - .x
        compute_pval(test_score, fruit_detected_scores)
      }),
      pval_not_detected = purrr::map_dbl(model_output, ~{
        test_score <- .x
        compute_pval(test_score, fruit_not_detected_scores)
      })
    )

  # Combine and reshape
  plot_data <- dplyr::bind_rows(flower_data, fruit_data) %>%
    tidyr::pivot_longer(
      cols = c(pval_detected, pval_not_detected),
      names_to = "hypothesis",
      values_to = "pvalue",
      names_prefix = "pval_"
    ) %>%
    dplyr::mutate(
      hypothesis = dplyr::case_match(
        hypothesis,
        "detected" ~ "P(Detected)",
        "not_detected" ~ "P(Not Detected)"
      )
    )

  # Create plot
  p <- ggplot(plot_data, aes(x = model_output, y = pvalue, color = hypothesis)) +
    geom_line(linewidth = 1.2) +
    geom_hline(yintercept = 0.1, linetype = "dashed", color = "gray40", alpha = 0.7) +
    geom_hline(yintercept = 0.2, linetype = "dotted", color = "gray40", alpha = 0.7) +
    facet_wrap(~class, ncol = 2) +
    scale_color_manual(
      values = c("P(Detected)" = "#E69F00", "P(Not Detected)" = "#56B4E9"),
      name = "Conformal Hypothesis"
    ) +
    labs(
      x = "Model Output (Probability)",
      y = "Conformal P-Value",
      title = "Conformal P-Values Across Model Output Range",
      subtitle = paste0(
        "Horizontal lines show common alpha cutoffs (dashed=0.10, dotted=0.20)\n",
        "Where curves cross alpha line determines class boundaries"
      )
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      strip.text = element_text(face = "bold", size = 12),
      plot.title = element_text(face = "bold"),
      panel.grid.minor = ggplot2::element_blank()
    ) +
    scale_x_continuous(
      breaks = seq(0, 1, 0.2),
      labels = scales::percent
    ) +
    scale_y_continuous(
      breaks = seq(0, 1, 0.2),
      labels = scales::percent
    )

  ggsave(output_path, p, width = 12, height = 5, dpi = 300)
  normalizePath(output_path)
}


# =============================================================================
# 2D Density Plot of Conformal P-Values
# =============================================================================

#' Create 2D Density Plot of Conformal P-Values
#'
#' Creates a 2D density plot showing the joint distribution of p-values for
#' the "detected" hypothesis vs the "not detected" hypothesis. This visualizes
#' how the conformal prediction separates the two classes.
#'
#' Points along the diagonal indicate uncertainty; points in corners indicate
#' confident predictions. The detected p-value being high means "this sample
#' conforms to the detected class", while not_detected p-value being high means
#' "this sample conforms to the not detected class".
#'
#' @param conformal_pvalues data.frame with columns:
#'   - .conf_pval_flower_detected, .conf_pval_flower_not_detected
#'   - .conf_pval_fruit_detected, .conf_pval_fruit_not_detected
#'   - flower, fruit (true labels)
#' @param output_path Path to save the plot
#'
#' @return Path to saved plot file
#'
#' @export
create_conformal_pvalue_density_plot <- function(conformal_pvalues,
                                                  output_path) {

  library(ggplot2)
  library(patchwork)

  # Filter to test set only (partition is "testing" not "test")
  test_data <- conformal_pvalues %>%
    dplyr::filter(partition == "testing")

  message("Creating 2D density plot for ", nrow(test_data), " test samples")

  # Create flower density plot
  p_flower <- ggplot(test_data, aes(
    x = .conf_pval_flower_detected,
    y = .conf_pval_flower_not_detected
  )) +
    geom_density_2d_filled(alpha = 0.8) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red",
                linewidth = 0.8) +
    scale_fill_viridis_d(option = "plasma", name = "Density") +
    labs(
      x = "P-value (Detected Hypothesis)",
      y = "P-value (Not Detected Hypothesis)",
      title = "Flower Conformal P-Values"
    ) +
    theme_minimal() +
    theme(
      legend.position = "right",
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +
    coord_fixed()

  # Create fruit density plot
  p_fruit <- ggplot(test_data, aes(
    x = .conf_pval_fruit_detected,
    y = .conf_pval_fruit_not_detected
  )) +
    geom_density_2d_filled(alpha = 0.8) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red",
                linewidth = 0.8) +
    scale_fill_viridis_d(option = "viridis", name = "Density") +
    labs(
      x = "P-value (Detected Hypothesis)",
      y = "P-value (Not Detected Hypothesis)",
      title = "Fruit Conformal P-Values"
    ) +
    theme_minimal() +
    theme(
      legend.position = "right",
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +
    coord_fixed()

  # Combine with patchwork
  p_combined <- p_flower + p_fruit +
    plot_layout(ncol = 2) +
    plot_annotation(
      title = "Joint Distribution of Conformal P-Values (Test Set)",
      subtitle = "Diagonal line separates detected (lower-right) from not detected (upper-left); corner points are confident predictions"
    )

  ggsave(output_path, p_combined, width = 14, height = 6, dpi = 300)
  normalizePath(output_path)
}


# =============================================================================
# Pareto Frontier / Elbow Visualization
# =============================================================================

#' Create Pareto Frontier and Elbow Visualization
#'
#' Creates a plot showing the DQI vs Detection Data Loss trade-off curve,
#' including the Pareto frontier, smoothed trend line, and marks showing how
#' the elbow method chooses its optimum.
#'
#' The elbow method finds the point on the Pareto frontier with maximum
#' perpendicular distance from the line connecting the endpoints.
#'
#' @param factorial_results data.frame from analyze_alpha_factorial()
#' @param optimal_candidates data.frame from find_optimal_alphas()
#' @param output_path Path to save the plot
#'
#' @return Path to saved plot file
#'
#' @export
create_pareto_elbow_plot <- function(factorial_results,
                                     optimal_candidates,
                                     output_path) {

  library(ggplot2)
  library(patchwork)

  # Process each class
  classes <- unique(factorial_results$class)
  plots <- list()

  for (class_name in classes) {
    # Get data for this class
    class_data <- factorial_results %>%
      dplyr::filter(class == class_name) %>%
      dplyr::arrange(dplyr::desc(dqi), pct_detected_removed)

    # Find Pareto frontier
    pareto_points <- class_data[1, ]

    for (i in 2:nrow(class_data)) {
      current <- class_data[i, ]
      last_pareto <- pareto_points[nrow(pareto_points), ]

      if (current$pct_detected_removed < last_pareto$pct_detected_removed) {
        pareto_points <- dplyr::bind_rows(pareto_points, current)
      }
    }

    pareto_points <- pareto_points %>%
      dplyr::arrange(pct_detected_removed)

    # Get elbow point for this class
    elbow_point <- optimal_candidates %>%
      dplyr::filter(class == class_name, method == "elbow")

    # Calculate line from first to last Pareto point for visualization
    first_pt <- pareto_points[1, ]
    last_pt <- pareto_points[nrow(pareto_points), ]

    # Create the plot
    p <- ggplot() +
      # All factorial combinations (faded background)
      geom_point(
        data = class_data,
        aes(x = pct_detected_removed, y = dqi),
        alpha = 0.2, color = "gray60", size = 1
      ) +
      # Pareto frontier points
      geom_point(
        data = pareto_points,
        aes(x = pct_detected_removed, y = dqi),
        color = "steelblue", size = 3, alpha = 0.8
      ) +
      # Pareto frontier line
      geom_line(
        data = pareto_points,
        aes(x = pct_detected_removed, y = dqi),
        color = "steelblue", linewidth = 1
      ) +
      # Smoothed trend line through Pareto points
      geom_smooth(
        data = pareto_points,
        aes(x = pct_detected_removed, y = dqi),
        method = "loess", se = TRUE,
        color = "darkblue", linewidth = 0.8, linetype = "dashed",
        fill = "lightblue", alpha = 0.3
      ) +
      # Line connecting endpoints (elbow reference)
      geom_segment(
        aes(x = first_pt$pct_detected_removed, y = first_pt$dqi,
            xend = last_pt$pct_detected_removed, yend = last_pt$dqi),
        linetype = "dotted", color = "red", linewidth = 0.8
      )

    # Add elbow point if found
    if (nrow(elbow_point) > 0) {
      p <- p +
        # Elbow point
        geom_point(
          data = elbow_point,
          aes(x = pct_detected_removed, y = dqi),
          color = "red", size = 5, shape = 18
        ) +
        # Perpendicular line from elbow to reference line
        geom_segment(
          aes(
            x = elbow_point$pct_detected_removed,
            y = elbow_point$dqi,
            xend = (elbow_point$pct_detected_removed + first_pt$pct_detected_removed) / 2 +
              (elbow_point$dqi - first_pt$dqi) *
              (last_pt$pct_detected_removed - first_pt$pct_detected_removed) /
              ((last_pt$dqi - first_pt$dqi)^2 +
               (last_pt$pct_detected_removed - first_pt$pct_detected_removed)^2) *
              (last_pt$dqi - first_pt$dqi) / 2,
            yend = elbow_point$dqi -
              (elbow_point$pct_detected_removed - first_pt$pct_detected_removed) *
              (last_pt$dqi - first_pt$dqi) /
              ((last_pt$dqi - first_pt$dqi)^2 +
               (last_pt$pct_detected_removed - first_pt$pct_detected_removed)^2) *
              (last_pt$dqi - first_pt$dqi) / 2
          ),
          color = "red", linewidth = 0.5, linetype = "dashed"
        ) +
        # Label for elbow point
        geom_label(
          data = elbow_point,
          aes(x = pct_detected_removed, y = dqi,
              label = sprintf("Elbow\nDQI=%.3f\nLoss=%.1f%%",
                              dqi, pct_detected_removed)),
          hjust = -0.1, vjust = 0.5,
          fill = "white", alpha = 0.8, size = 3
        )
    }

    p <- p +
      labs(
        x = "Detection Data Loss (%)",
        y = "DQI (PPV + Sensitivity - 1)",
        title = paste0(tools::toTitleCase(class_name), ": DQI vs Detection Loss Trade-off"),
        subtitle = "Blue = Pareto frontier, Red diamond = Elbow optimum, Dotted red = Reference line"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = ggplot2::element_blank()
      ) +
      coord_cartesian(clip = "off") +
      scale_x_continuous(expand = expansion(mult = c(0.02, 0.15)))

    plots[[class_name]] <- p
  }

  # Combine plots
  p_combined <- plots$flower + plots$fruit +
    plot_layout(ncol = 2) +
    plot_annotation(
      title = "Pareto Frontier and Elbow Method Optimization",
      subtitle = "The elbow point maximizes perpendicular distance from the line connecting endpoints"
    )

  ggsave(output_path, p_combined, width = 14, height = 6, dpi = 300)
  normalizePath(output_path)
}


# =============================================================================
# ACCURACY-BASED BUFFER ZONE ANALYSIS
# =============================================================================
# Simpler alternative to conformal factorial optimization
# Based on original implementation in R/vit_model_test_threshold_new.R

#' Calculate Accuracy-Based Buffer Zone
#'
#' Based on original implementation in R/vit_model_test_threshold_new.R.
#' Uses santoku::chop_evenly for interval creation and yardstick::accuracy
#' for per-interval accuracy calculation.
#'
#' This is the simpler, more intuitive approach used in the original paper.
#' It identifies buffer zones where model predictions are unreliable based
#' on accuracy within prediction probability intervals.
#'
#' @param data Data frame with truth and prediction columns
#' @param class_name "flower" or "fruit"
#' @param optimal_threshold The optimal decision threshold (e.g., 0.84 for flower)
#' @param partition_filter Which partition to use ("validation" default)
#' @param n_intervals Number of intervals (default 100, matching original)
#' @param accuracy_cutoff Minimum accuracy threshold (default 0.75)
#' @param smoother Smoothing method: "loess" (default), "gam", or "none"
#' @param smoother_span Span parameter for LOESS smoothing (default 0.3)
#' @param metric Which accuracy metric: "overall" or "ppv_npv"
#'
#' @return List with:
#'   - interval_data: data frame with interval stats (accuracy, count, midpoint)
#'   - buffer_lower: lower boundary of buffer zone (distance below threshold)
#'   - buffer_upper: upper boundary of buffer zone (distance above threshold)
#'   - optimal_threshold: the decision threshold used
#'   - accuracy_cutoff: the cutoff used
#'   - bad_intervals: intervals where accuracy <= cutoff
#'   - buffer_param: vector c(lower, upper) for make_two_class_pred
#'
#' @export
calculate_accuracy_buffer_zone <- function(
    data,
    class_name,
    optimal_threshold,
    partition_filter = "validation",
    n_intervals = 100,
    accuracy_cutoff = 0.75,
    smoother = "loess",
    smoother_span = 0.3,
    metric = "overall"
) {

  message(sprintf("Calculating accuracy-based buffer zone for %s (metric=%s, smoother=%s)",
                  class_name, metric, smoother))

  # Column names
  pred_col <- paste0(".pred_", class_name)
  truth_col <- class_name  # "flower" or "fruit"

  # Filter to partition
  filtered <- data %>%
    dplyr::filter(partition == partition_filter)

  if (nrow(filtered) == 0) {
    stop(sprintf("No data found for partition '%s'", partition_filter))
  }

  message(sprintf("  Using %d samples from '%s' partition", nrow(filtered), partition_filter))

  # Apply threshold to create class predictions
  filtered <- filtered %>%
    dplyr::mutate(
      .class_pred = probably::make_two_class_pred(
        .data[[pred_col]],
        levels(.data[[truth_col]]),
        threshold = optimal_threshold
      ),
      # Create intervals using santoku (matching original implementation)
      .cut = santoku::chop_evenly(
        .data[[pred_col]], n_intervals,
        labels = santoku::lbl_midpoints()
      )
    )

  # Calculate accuracy per interval (original approach)
  interval_acc <- filtered %>%
    dplyr::group_by(.cut) %>%
    yardstick::accuracy(
      truth = !!rlang::sym(truth_col),
      estimate = .class_pred
    ) %>%
    dplyr::mutate(
      value = as.numeric(as.character(.cut))
    ) %>%
    dplyr::ungroup()

  # Count samples per interval
  interval_counts <- filtered %>%
    dplyr::group_by(.cut) %>%
    dplyr::summarise(count = dplyr::n(), .groups = "drop") %>%
    dplyr::mutate(value = as.numeric(as.character(.cut)))

  # Combine accuracy and counts
  interval_data <- interval_acc %>%
    dplyr::left_join(interval_counts, by = c(".cut", "value")) %>%
    dplyr::mutate(prop = count / sum(count, na.rm = TRUE))

  # If PPV/NPV metric requested, calculate that instead
  if (metric == "ppv_npv") {
    ppv_npv_data <- filtered %>%
      dplyr::group_by(.cut) %>%
      dplyr::summarise(
        value = as.numeric(as.character(.cut[1])),
        n_pos_truth = sum(.data[[truth_col]] == "1"),
        n_neg_truth = sum(.data[[truth_col]] == "0"),
        n_total = dplyr::n(),
        .groups = "drop"
      ) %>%
      dplyr::mutate(
        # PPV for intervals above threshold, NPV for below
        ppv = n_pos_truth / n_total,
        npv = n_neg_truth / n_total,
        ppv_npv_accuracy = dplyr::if_else(value >= optimal_threshold, ppv, npv)
      )

    interval_data <- interval_data %>%
      dplyr::left_join(
        ppv_npv_data %>% dplyr::select(value, ppv_npv_accuracy),
        by = "value"
      )

    # Use PPV/NPV as the accuracy metric
    interval_data$.estimate <- interval_data$ppv_npv_accuracy
  }

  # Apply smoothing (default: LOESS - important for sparse data like leaf model)
  if (smoother != "none" && nrow(interval_data) > 5) {
    valid_data <- interval_data %>%
      dplyr::filter(!is.na(.estimate), count >= 5)

    if (nrow(valid_data) >= 5) {
      if (smoother == "loess") {
        fit <- stats::loess(.estimate ~ value, data = valid_data, span = smoother_span)
        interval_data$smoothed <- stats::predict(fit, newdata = interval_data)
      } else if (smoother == "gam") {
        if (!requireNamespace("mgcv", quietly = TRUE)) {
          warning("Package 'mgcv' required for GAM smoothing, falling back to LOESS")
          fit <- stats::loess(.estimate ~ value, data = valid_data, span = smoother_span)
          interval_data$smoothed <- stats::predict(fit, newdata = interval_data)
        } else {
          fit <- mgcv::gam(.estimate ~ s(value), data = valid_data)
          interval_data$smoothed <- stats::predict(fit, newdata = interval_data)
        }
      }
      message(sprintf("  Applied %s smoothing (span=%.2f)", smoother, smoother_span))
    } else {
      warning("Too few valid intervals for smoothing, using raw values")
      interval_data$smoothed <- interval_data$.estimate
    }
  } else {
    interval_data$smoothed <- interval_data$.estimate
    if (smoother != "none") {
      message("  No smoothing applied (insufficient data)")
    }
  }

  # Determine which accuracy values to use for finding buffer zone
  # Use smoothed values if available, otherwise raw
  accuracy_for_buffer <- if (smoother != "none" && "smoothed" %in% names(interval_data)) {
    interval_data$smoothed
  } else {
    interval_data$.estimate
  }

  # Find bad intervals (accuracy <= cutoff)
  bad_intervals <- interval_data %>%
    dplyr::mutate(.accuracy_for_buffer = accuracy_for_buffer) %>%
    dplyr::filter(.accuracy_for_buffer <= accuracy_cutoff) %>%
    dplyr::select(-.accuracy_for_buffer)

  message(sprintf("  Found %d intervals with accuracy <= %.0f%%",
                  nrow(bad_intervals), accuracy_cutoff * 100))

  # Calculate buffer boundaries
  # Buffer lower = how far below threshold the equivocal zone extends
  # Buffer upper = how far above threshold the equivocal zone extends
  bad_below <- bad_intervals %>% dplyr::filter(value < optimal_threshold)
  bad_above <- bad_intervals %>% dplyr::filter(value >= optimal_threshold)

  buffer_lower <- if (nrow(bad_below) > 0) {
    optimal_threshold - min(bad_below$value)
  } else {
    0
  }

  buffer_upper <- if (nrow(bad_above) > 0) {
    max(bad_above$value) - optimal_threshold
  } else {
    0
  }

  # Calculate equivocal zone boundaries
  eq_lower <- optimal_threshold - buffer_lower
  eq_upper <- optimal_threshold + buffer_upper

  message(sprintf("  Buffer zone: [%.3f, %.3f] (threshold=%.3f)",
                  eq_lower, eq_upper, optimal_threshold))
  message(sprintf("  Buffer params: lower=%.3f, upper=%.3f", buffer_lower, buffer_upper))

  list(
    interval_data = interval_data,
    bad_intervals = bad_intervals,
    buffer_lower = buffer_lower,
    buffer_upper = buffer_upper,
    optimal_threshold = optimal_threshold,
    accuracy_cutoff = accuracy_cutoff,
    metric = metric,
    smoother = smoother,
    class_name = class_name,  # Store for use in plotting functions
    # For make_two_class_pred buffer parameter: c(lower, upper)
    buffer_param = c(buffer_lower, buffer_upper),
    # Equivocal zone boundaries (for convenience)
    equivocal_zone = c(eq_lower, eq_upper)
  )
}


#' Create Accuracy Buffer Zone Visualization
#'
#' Creates a plot similar to output/equivocal_zone_plot_new.png showing:
#' - Accuracy line across model output range
#' - Sample count area (scaled)
#' - Threshold line
#' - Accuracy cutoff line
#' - Shaded buffer/equivocal zone
#'
#' @param buffer_result Output from calculate_accuracy_buffer_zone()
#' @param class_name "flower" or "fruit" for title/colors
#' @param show_counts Show sample count area (default TRUE)
#' @param show_smoothed Show smoothed line if available (default TRUE)
#'
#' @return ggplot object
#'
#' @export
create_accuracy_buffer_plot <- function(
    buffer_result,
    class_name = NULL,
    show_counts = TRUE,
    show_smoothed = TRUE
) {
  # Get class_name from buffer_result if not provided
  if (is.null(class_name)) {
    class_name <- buffer_result$class_name
    if (is.null(class_name)) {
      stop("class_name must be provided or included in buffer_result")
    }
  }

  data <- buffer_result$interval_data
  threshold <- buffer_result$optimal_threshold
  cutoff <- buffer_result$accuracy_cutoff
  buffer_lower <- buffer_result$buffer_lower
  buffer_upper <- buffer_result$buffer_upper

  # Calculate equivocal zone boundaries
  eq_lower <- threshold - buffer_lower
  eq_upper <- threshold + buffer_upper


  # Color for this class (from wesanderson FantasticFox1)
  pal <- c(flower = "#E58601", fruit = "#46ACC8")
  class_color <- pal[class_name]

  # Get max count for scaling
  max_count <- max(data$count, na.rm = TRUE)

  p <- ggplot2::ggplot(data, ggplot2::aes(x = value))

  # Shade equivocal zone
  if (buffer_lower > 0 || buffer_upper > 0) {
    p <- p + ggplot2::annotate(
      "rect",
      xmin = eq_lower, xmax = eq_upper,
      ymin = 0, ymax = 1,
      fill = "red", alpha = 0.15
    )
  }

  # Sample count area (scaled to fit)
  if (show_counts && max_count > 0) {
    p <- p + ggplot2::geom_area(
      ggplot2::aes(y = count / max_count),
      fill = class_color, alpha = 0.3
    )
  }

  # Raw accuracy line
  p <- p + ggplot2::geom_path(
    ggplot2::aes(y = .estimate),
    color = class_color, linewidth = 0.8, alpha = 0.6
  )

  # Smoothed accuracy line (if available and requested)
  if (show_smoothed && "smoothed" %in% names(data) && !all(is.na(data$smoothed))) {
    p <- p + ggplot2::geom_path(
      ggplot2::aes(y = smoothed),
      color = class_color, linewidth = 1.2
    )
  }

  # Reference lines
  p <- p +
    ggplot2::geom_vline(xintercept = threshold, linewidth = 1, linetype = 2) +
    ggplot2::geom_hline(yintercept = cutoff, linewidth = 0.5, linetype = 3, color = "red")

  # Labels and theme
  p <- p +
    ggplot2::scale_x_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.1)
    ) +
    ggplot2::scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.1),
      sec.axis = ggplot2::sec_axis(
        transform = ~. * max_count,
        name = "Image Count",
        labels = scales::label_comma()
      )
    ) +
    ggplot2::labs(
      x = "Model Output",
      y = "Accuracy",
      title = paste0(tools::toTitleCase(class_name), " Equivocal Zone Analysis"),
      subtitle = sprintf(
        "Threshold: %.2f | Buffer: [%.2f, %.2f] | Equivocal zone: [%.2f, %.2f]",
        threshold, buffer_lower, buffer_upper, eq_lower, eq_upper
      )
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank()
    )

  p
}


#' Create Combined Accuracy Plot for Both Classes
#'
#' Creates a combined visualization showing accuracy curves for both
#' flower and fruit classes on the same plot, similar to the original
#' equivocal_zone_plot_new.png from the paper.
#'
#' @param buffer_flower Result from calculate_accuracy_buffer_zone for flower
#' @param buffer_fruit Result from calculate_accuracy_buffer_zone for fruit
#' @param show_smoothed Show smoothed lines if available (default TRUE)
#'
#' @return ggplot object
#'
#' @export
create_combined_accuracy_plot <- function(buffer_flower, buffer_fruit, show_smoothed = TRUE) {

  # Combine data
  flower_data <- buffer_flower$interval_data %>%
    dplyr::mutate(type = "flower")
  fruit_data <- buffer_fruit$interval_data %>%
    dplyr::mutate(type = "fruit")

  combined <- dplyr::bind_rows(flower_data, fruit_data)

  # Get max counts for scaling
  max_count <- max(combined$count, na.rm = TRUE)

  pal <- c(flower = "#E58601", fruit = "#46ACC8")

  p <- ggplot2::ggplot(combined, ggplot2::aes(x = value, color = type, fill = type))

  # Sample count areas
  p <- p + ggplot2::geom_area(
    ggplot2::aes(y = count / max_count),
    alpha = 0.3, position = "identity"
  )

  # Raw accuracy lines (faded)
  p <- p + ggplot2::geom_path(
    ggplot2::aes(y = .estimate),
    linewidth = 0.5, alpha = 0.4
  )

  # Smoothed accuracy lines (if available and requested)
  if (show_smoothed && "smoothed" %in% names(combined) && !all(is.na(combined$smoothed))) {
    p <- p + ggplot2::geom_path(
      ggplot2::aes(y = smoothed),
      linewidth = 1
    )
  } else {
    p <- p + ggplot2::geom_path(
      ggplot2::aes(y = .estimate),
      linewidth = 0.75
    )
  }

  # Threshold lines
  p <- p +
    ggplot2::geom_vline(xintercept = buffer_fruit$optimal_threshold,
                        linewidth = 1, linetype = 2, color = pal["fruit"]) +
    ggplot2::geom_vline(xintercept = buffer_flower$optimal_threshold,
                        linewidth = 1, linetype = 2, color = pal["flower"]) +
    ggplot2::geom_hline(yintercept = 0.75, linewidth = 0.5, linetype = 3, color = "gray40")

  # Colors and scales
  p <- p +
    ggplot2::scale_color_manual(values = pal, name = "Class") +
    ggplot2::scale_fill_manual(values = pal, name = "Class") +
    ggplot2::scale_y_continuous(
      sec.axis = ggplot2::sec_axis(transform = ~. * max_count,
                                   name = "Image Count",
                                   labels = scales::label_comma())
    ) +
    ggplot2::labs(
      x = "Model Output",
      y = "Accuracy",
      title = "Equivocal Zone Analysis",
      subtitle = sprintf(
        "Flower threshold: %.2f, buffer: [%.2f, %.2f] | Fruit threshold: %.2f, buffer: [%.2f, %.2f]",
        buffer_flower$optimal_threshold,
        buffer_flower$buffer_lower, buffer_flower$buffer_upper,
        buffer_fruit$optimal_threshold,
        buffer_fruit$buffer_lower, buffer_fruit$buffer_upper
      )
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position.inside = c(0.5, 0.9))

  p
}


#' Compare Overall vs PPV/NPV Accuracy Methods
#'
#' Creates a side-by-side comparison plot showing buffer zones
#' calculated using overall accuracy vs PPV/NPV accuracy methods.
#'
#' @param buffer_overall Result from calculate_accuracy_buffer_zone with metric="overall"
#' @param buffer_ppv_npv Result from calculate_accuracy_buffer_zone with metric="ppv_npv"
#' @param class_name "flower" or "fruit"
#'
#' @return patchwork combined plot
#'
#' @export
create_accuracy_comparison_plot <- function(
    buffer_overall,
    buffer_ppv_npv,
    class_name = NULL
) {
  # Ensure patchwork is loaded for the + operator to work with ggplot objects
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    stop("Package 'patchwork' is required for this function")
  }
  if (!"patchwork" %in% .packages()) {
    attachNamespace("patchwork")
  }

  # Get class_name from buffer_result if not provided
  if (is.null(class_name)) {
    class_name <- buffer_overall$class_name
    if (is.null(class_name)) {
      class_name <- buffer_ppv_npv$class_name
    }
    if (is.null(class_name)) {
      stop("class_name must be provided or included in buffer results")
    }
  }

  p1 <- create_accuracy_buffer_plot(buffer_overall, class_name) +
    ggplot2::labs(subtitle = paste0("Overall Accuracy | Buffer: [",
                                    round(buffer_overall$buffer_lower, 2), ", ",
                                    round(buffer_overall$buffer_upper, 2), "]"))

  p2 <- create_accuracy_buffer_plot(buffer_ppv_npv, class_name) +
    ggplot2::labs(subtitle = paste0("PPV/NPV Accuracy | Buffer: [",
                                    round(buffer_ppv_npv$buffer_lower, 2), ", ",
                                    round(buffer_ppv_npv$buffer_upper, 2), "]"))

  # Use wrap_plots instead of + operator to avoid namespace issues
  patchwork::wrap_plots(p1, p2, ncol = 2) +
    patchwork::plot_annotation(
      title = paste0(tools::toTitleCase(class_name),
                     ": Accuracy-Based Buffer Zone Comparison"),
      subtitle = "Left: Overall accuracy | Right: PPV (above) / NPV (below) threshold"
    )
}


# =============================================================================
# Multi-Cutoff Buffer Zone Analysis
# =============================================================================

#' Analyze Buffer Zones Across Multiple Accuracy Cutoffs
#'
#' Tests multiple accuracy cutoffs and calculates data loss for each.
#' The expensive accuracy-per-interval computation is done ONCE, then
#' the cheap buffer zone finding is done for each cutoff.
#'
#' @param data Data frame with truth and prediction columns
#' @param class_name "flower" or "fruit"
#' @param optimal_threshold The optimal decision threshold
#' @param partition_filter Which partition to use ("validation" default)
#' @param cutoffs Vector of accuracy cutoffs to test (default: seq(0.75, 0.95, by = 0.02))
#' @param n_intervals Number of intervals (default 100)
#' @param smoother Smoothing method ("loess", "gam", or "none")
#' @param smoother_span LOESS span parameter
#'
#' @return Data frame with one row per cutoff containing:
#'   - cutoff: the accuracy cutoff value
#'   - buffer_lower, buffer_upper: buffer zone boundaries
#'   - equivocal_lower, equivocal_upper: equivocal zone boundaries
#'   - n_total: total validation samples
#'   - n_in_buffer: samples in buffer zone
#'   - pct_data_loss: % of all data in buffer zone
#'   - n_detected_in_buffer: detection-true samples in buffer
#'   - n_not_detected_in_buffer: non-detection-true samples in buffer
#'   - pct_detected_loss: % of detection-true samples in buffer
#'   - pct_not_detected_loss: % of non-detection-true samples in buffer
#'
#' @export
analyze_multi_cutoff_buffer_zones <- function(
    data,
    class_name,
    optimal_threshold,
    partition_filter = "validation",
    cutoffs = seq(0.75, 0.95, by = 0.02),
    n_intervals = 100,
    smoother = "loess",
    smoother_span = 0.3
) {
  message(sprintf("Analyzing %d cutoffs for %s (%.2f to %.2f)",
                  length(cutoffs), class_name, min(cutoffs), max(cutoffs)))

  pred_col <- paste0(".pred_", class_name)
  truth_col <- class_name

  # Filter to partition
  filtered <- data %>%
    dplyr::filter(partition == partition_filter)

  # Calculate interval data once (shared across all cutoffs)
  filtered <- filtered %>%
    dplyr::mutate(
      .class_pred = probably::make_two_class_pred(
        .data[[pred_col]],
        levels(.data[[truth_col]]),
        threshold = optimal_threshold
      ),
      .cut = santoku::chop_evenly(
        .data[[pred_col]], n_intervals,
        labels = santoku::lbl_midpoints()
      ),
      .pred_value = .data[[pred_col]],
      .truth_value = as.character(.data[[truth_col]])
    )

  # =========================================================================
  # EXPENSIVE COMPUTATION - DONE ONCE
  # Calculate accuracy per interval (this is the slow part)
  # =========================================================================
  message("  Computing accuracy per interval (done once)...")
  interval_acc <- filtered %>%
    dplyr::group_by(.cut) %>%
    yardstick::accuracy(
      truth = !!rlang::sym(truth_col),
      estimate = .class_pred
    ) %>%
    dplyr::mutate(value = as.numeric(as.character(.cut))) %>%
    dplyr::ungroup()

  # Apply smoothing if requested (also done once)
  if (smoother != "none" && nrow(interval_acc) > 5) {
    valid_data <- interval_acc %>% dplyr::filter(!is.na(.estimate))
    if (nrow(valid_data) >= 5) {
      if (smoother == "loess") {
        fit <- stats::loess(.estimate ~ value, data = valid_data, span = smoother_span)
        interval_acc$smoothed <- stats::predict(fit, newdata = interval_acc)
      } else if (smoother == "gam") {
        fit <- mgcv::gam(.estimate ~ s(value), data = valid_data)
        interval_acc$smoothed <- stats::predict(fit, newdata = interval_acc)
      }
    } else {
      warning("Too few valid intervals for smoothing, using raw values")
      interval_acc$smoothed <- interval_acc$.estimate
    }
  } else {
    interval_acc$smoothed <- interval_acc$.estimate
  }

  # Get counts for data loss calculation
  n_total <- nrow(filtered)
  n_detected <- sum(filtered$.truth_value == "1")
  n_not_detected <- sum(filtered$.truth_value == "0")

  message(sprintf("  Total samples: %d (detected: %d, not detected: %d)",
                  n_total, n_detected, n_not_detected))

  # =========================================================================
  # CHEAP COMPUTATION - DONE FOR EACH CUTOFF
  # Just filtering the pre-computed interval_acc
  # =========================================================================
  message("  Testing cutoffs (fast operation)...")
  results <- purrr::map_dfr(cutoffs, function(cutoff) {
    # Use smoothed values if available
    accuracy_col <- if ("smoothed" %in% names(interval_acc)) "smoothed" else ".estimate"

    # Find bad intervals (accuracy <= cutoff)
    bad_intervals <- interval_acc %>%
      dplyr::filter(.data[[accuracy_col]] <= cutoff)

    if (nrow(bad_intervals) == 0) {
      # No bad intervals at this cutoff
      return(tibble::tibble(
        cutoff = cutoff,
        buffer_lower = 0,
        buffer_upper = 0,
        equivocal_lower = optimal_threshold,
        equivocal_upper = optimal_threshold,
        n_total = n_total,
        n_in_buffer = 0L,
        pct_data_loss = 0,
        n_detected_in_buffer = 0L,
        n_not_detected_in_buffer = 0L,
        pct_detected_loss = 0,
        pct_not_detected_loss = 0
      ))
    }

    # Calculate buffer boundaries
    bad_below <- bad_intervals %>% dplyr::filter(value < optimal_threshold)
    bad_above <- bad_intervals %>% dplyr::filter(value >= optimal_threshold)

    buffer_lower <- if (nrow(bad_below) > 0) optimal_threshold - min(bad_below$value) else 0
    buffer_upper <- if (nrow(bad_above) > 0) max(bad_above$value) - optimal_threshold else 0

    equivocal_lower <- optimal_threshold - buffer_lower
    equivocal_upper <- optimal_threshold + buffer_upper

    # Count samples in buffer zone
    in_buffer <- filtered %>%
      dplyr::filter(.pred_value >= equivocal_lower & .pred_value <= equivocal_upper)

    n_in_buffer <- nrow(in_buffer)
    n_detected_in_buffer <- sum(in_buffer$.truth_value == "1")
    n_not_detected_in_buffer <- sum(in_buffer$.truth_value == "0")

    tibble::tibble(
      cutoff = cutoff,
      buffer_lower = buffer_lower,
      buffer_upper = buffer_upper,
      equivocal_lower = equivocal_lower,
      equivocal_upper = equivocal_upper,
      n_total = n_total,
      n_in_buffer = n_in_buffer,
      pct_data_loss = 100 * n_in_buffer / n_total,
      n_detected_in_buffer = n_detected_in_buffer,
      n_not_detected_in_buffer = n_not_detected_in_buffer,
      pct_detected_loss = 100 * n_detected_in_buffer / n_detected,
      pct_not_detected_loss = 100 * n_not_detected_in_buffer / n_not_detected
    )
  })

  message(sprintf("  Done. Data loss ranges from %.1f%% to %.1f%%",
                  min(results$pct_data_loss), max(results$pct_data_loss)))

  results %>%
    dplyr::mutate(class = class_name, threshold = optimal_threshold) %>%
    dplyr::select(class, threshold, dplyr::everything())
}


#' Create Data Loss vs Cutoff Comparison Plot
#'
#' Creates a visualization showing how data loss varies with accuracy cutoff
#' stringency. Includes both total data loss and breakdown by detected vs
#' not detected classes.
#'
#' @param multi_cutoff_flower Results from analyze_multi_cutoff_buffer_zones for flower
#' @param multi_cutoff_fruit Results from analyze_multi_cutoff_buffer_zones for fruit
#'
#' @return patchwork combined plot
#'
#' @export
create_multi_cutoff_dataloss_plot <- function(multi_cutoff_flower, multi_cutoff_fruit) {
  # Ensure patchwork is loaded for the + operator
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    stop("Package 'patchwork' is required")
  }
  if (!"patchwork" %in% .packages()) {
    attachNamespace("patchwork")
  }

  # Combine data
  combined <- dplyr::bind_rows(multi_cutoff_flower, multi_cutoff_fruit)

  # Reshape to long format for faceted plotting
  combined_long <- combined %>%
    tidyr::pivot_longer(
      cols = c(pct_data_loss, pct_detected_loss, pct_not_detected_loss),
      names_to = "loss_type",
      values_to = "pct_loss"
    ) %>%
    dplyr::mutate(
      loss_type = dplyr::case_when(
        loss_type == "pct_data_loss" ~ "Total",
        loss_type == "pct_detected_loss" ~ "Detected (truth=1)",
        loss_type == "pct_not_detected_loss" ~ "Not Detected (truth=0)"
      ),
      loss_type = factor(loss_type, levels = c("Total", "Detected (truth=1)", "Not Detected (truth=0)"))
    )

  pal <- c(flower = "#E58601", fruit = "#46ACC8")

  # Main plot: Data loss vs cutoff
  p1 <- ggplot2::ggplot(combined_long, ggplot2::aes(x = cutoff * 100, y = pct_loss,
                                                     color = class, linetype = loss_type)) +
    ggplot2::geom_line(linewidth = 1) +
    ggplot2::geom_point(size = 2) +
    ggplot2::scale_color_manual(values = pal, name = "Class") +
    ggplot2::scale_linetype_manual(
      values = c("Total" = "solid", "Detected (truth=1)" = "dashed", "Not Detected (truth=0)" = "dotted"),
      name = "Loss Type"
    ) +
    ggplot2::scale_x_continuous(breaks = seq(75, 95, by = 2)) +
    ggplot2::labs(
      x = "Accuracy Cutoff (%)",
      y = "Data Loss (%)",
      title = "Data Loss vs Accuracy Cutoff for Buffer Zone",
      subtitle = "Higher cutoffs are stricter, leading to larger buffer zones and more data loss"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      legend.position = "bottom",
      panel.grid.minor = ggplot2::element_blank()
    )

  # Buffer zone width plot
  p2 <- ggplot2::ggplot(combined, ggplot2::aes(x = cutoff * 100, y = equivocal_upper - equivocal_lower,
                                                color = class)) +
    ggplot2::geom_line(linewidth = 1) +
    ggplot2::geom_point(size = 2) +
    ggplot2::scale_color_manual(values = pal, name = "Class") +
    ggplot2::scale_x_continuous(breaks = seq(75, 95, by = 2)) +
    ggplot2::labs(
      x = "Accuracy Cutoff (%)",
      y = "Buffer Zone Width",
      title = "Buffer Zone Width vs Accuracy Cutoff"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      legend.position = "none",
      panel.grid.minor = ggplot2::element_blank()
    )

  # Combine plots
  patchwork::wrap_plots(p1, p2, ncol = 1, heights = c(2, 1)) +
    patchwork::plot_annotation(
      title = "Accuracy Cutoff Sensitivity Analysis",
      subtitle = "Effect of accuracy cutoff on buffer zone size and data loss"
    )
}
