# Custom Metrics for Training Pipeline
#
# Custom yardstick-compatible metrics for model evaluation.
#
# Suffix: _training.R

# =============================================================================
# Detection Quality Index (DQI)
# =============================================================================

#' Detection Quality Index (DQI) - Vector Implementation
#'
#' DQI = PPV + Sensitivity - 1
#'
#' A metric that balances positive predictive value (precision) with
#' detection completeness (recall/sensitivity). Unlike J-index which
#' balances sensitivity and specificity, DQI focuses on the quality
#' of positive predictions.
#'
#' @param truth Factor of true classes
#' @param estimate Factor of predicted classes
#' @param estimator Estimator type (NULL for auto-detection, "binary" or "macro")
#' @param na_rm Remove NAs before calculation (default TRUE)
#' @param event_level "first" or "second" - which factor level is the positive class.
#'   Default "first" treats the first level (e.g., "1") as the positive/detected class.
#' @param ... Additional arguments (ignored)
#'
#' @return Numeric value between -1 and 1
#'   - 1 = perfect (PPV=1, Sensitivity=1)
#'   - 0 = random performance
#'   - negative = worse than random
#'
#' @details
#' Formula: DQI = PPV + Sensitivity - 1
#'
#' Comparison with related metrics:
#' - J-index (Youden's J) = Sensitivity + Specificity - 1 (balances TPR vs TNR)
#' - F1 = 2 * (PPV * Sensitivity) / (PPV + Sensitivity) (harmonic mean)
#' - DQI uses additive combination like J-index, but focuses on positive predictions
#'
#' Use DQI when you want to:
#' - Prioritize positive prediction quality over negative prediction quality
#' - Balance detection rate (sensitivity) with detection reliability (PPV)
#' - Avoid the interpretability issues of harmonic mean (F1)
#'
#' @examples
#' \dontrun{
#' truth <- factor(c("1", "1", "0", "0", "1", "0"), levels = c("1", "0"))
#' estimate <- factor(c("1", "1", "1", "0", "0", "0"), levels = c("1", "0"))
#' dqi_vec(truth, estimate, event_level = "first")
#' }
#'
#' @export
dqi_vec <- function(truth,
                    estimate,
                    estimator = NULL,
                    na_rm = TRUE,
                    event_level = "first",
                    ...) {


  # Calculate PPV and Sensitivity using yardstick

  ppv_val <- yardstick::ppv_vec(
    truth = truth,
    estimate = estimate,
    estimator = estimator,
    na_rm = na_rm,
    event_level = event_level
  )

  sens_val <- yardstick::sens_vec(
    truth = truth,
    estimate = estimate,
    estimator = estimator,
    na_rm = na_rm,
    event_level = event_level
  )

  # DQI = PPV + Sensitivity - 1
  dqi_val <- ppv_val + sens_val - 1

  return(dqi_val)
}


#' Detection Quality Index (DQI) - Data Frame Implementation
#'
#' @inheritParams dqi_vec
#' @param data A data frame containing the columns specified by `truth` and `estimate`
#' @param truth The column name for the true class labels (factor)
#' @param estimate The column name for the predicted class labels (factor)
#' @param case_weights Optional column for case weights
#'
#' @return A tibble with columns `.metric`, `.estimator`, `.estimate`
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' data <- tibble(
#'   truth = factor(c("1", "1", "0", "0", "1", "0"), levels = c("1", "0")),
#'   pred = factor(c("1", "1", "1", "0", "0", "0"), levels = c("1", "0"))
#' )
#' dqi(data, truth, pred, event_level = "first")
#' }
#'
#' @export
dqi <- function(data, ...) {
 UseMethod("dqi")
}

#' @export
#' @rdname dqi
dqi.data.frame <- function(data,
                           truth,
                           estimate,
                           estimator = NULL,
                           na_rm = TRUE,
                           case_weights = NULL,
                           event_level = "first",
                           ...) {

  # Compute PPV and Sensitivity
  ppv_result <- yardstick::ppv(
    data = data,
    truth = {{ truth }},
    estimate = {{ estimate }},
    estimator = estimator,
    na_rm = na_rm,
    case_weights = {{ case_weights }},
    event_level = event_level
  )

  sens_result <- yardstick::sens(
    data = data,
    truth = {{ truth }},
    estimate = {{ estimate }},
    estimator = estimator,
    na_rm = na_rm,
    case_weights = {{ case_weights }},
    event_level = event_level
  )

  # Combine into DQI
  dqi_estimate <- ppv_result$.estimate + sens_result$.estimate - 1

  tibble::tibble(
    .metric = "dqi",
    .estimator = ppv_result$.estimator,
    .estimate = dqi_estimate
  )
}


#' Create DQI metric set function
#'
#' Returns a function suitable for use with `probably::threshold_perf()`
#' and `yardstick::metric_set()`.
#'
#' @param event_level Which factor level is the positive class ("first" or "second")
#'
#' @return A metric function compatible with yardstick
#'
#' @examples
#' \dontrun
#' # Use in threshold_perf
#' probably::threshold_perf(
#'   data,
#'   truth = outcome,
#'   estimate = .pred_class,
#'   metrics = yardstick::metric_set(
#'     yardstick::j_index,
#'     yardstick::sens,
#'     yardstick::ppv,
#'     dqi_metric()
#'   )
#' )
#' }
#'
#' @export
dqi_metric <- function() {
  # Return the dqi function itself for use in metric_set
  dqi
}


# =============================================================================
# Helper: Make DQI work with metric_set
# =============================================================================

# Register the metric class for yardstick compatibility
# This allows dqi to be used in metric_set() calls

#' @export
#' @keywords internal
dqi_impl <- function(data, truth, estimate, estimator, na_rm, event_level) {
  dqi.data.frame(
    data = data,
    truth = !!rlang::enquo(truth),
    estimate = !!rlang::enquo(estimate),
    estimator = estimator,
    na_rm = na_rm,
    event_level = event_level
  )
}
