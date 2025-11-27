# Training Evaluation Helper Function
#
# Runs validation evaluation and logs metrics in GuildAI format.
# Used during training to evaluate model after each epoch.
#
# Suffix: _training.R

# =============================================================================
# Dependencies (loaded in training script):
# - reticulate (torch, eval modules)
# - dplyr, tidyr
# - probably (threshold_perf, make_two_class_pred)
# - yardstick (accuracy, j_index, sensitivity, precision)
# =============================================================================

# Define custom metric set for threshold_perf that includes precision (PPV)
# Default metrics only include sensitivity, specificity, j_index - NOT precision
.dqi_metrics <- yardstick::metric_set(
 yardstick::sensitivity,
 yardstick::specificity,
 yardstick::precision,
 yardstick::j_index
)

#' Evaluate Model and Log Metrics for GuildAI
#'
#' Runs validation evaluation, calculates DQI-optimized thresholds,
#' and logs metrics in GuildAI-compatible format.
#'
#' @param model PyTorch model to evaluate
#' @param val_dl Validation dataloader
#' @param eval_module Python evaluate module (from py/evaluate.py)
#' @param torch_module Python torch module
#' @param criterion Loss function
#' @param step_num Step number for GuildAI logging (step 1 = epoch 0, etc.)
#' @param device CUDA device (default "cuda:0")
#'
#' @return list with:
#'   - val_loss: PyTorch tensor with validation loss
#'   - val_df: data.frame with predictions and truth
#'   - thresholds: list with flower and fruit threshold info
#'   - metrics: list with all computed metrics
#'
#' @export
evaluate_and_log <- function(model, val_dl, eval_module, torch_module, criterion,
                             step_num, device = "cuda:0") {

  # Run evaluation
  val_dat <- eval_module$evaluate(val_dl, model, device)
  val_preds <- torch_module$nn$functional$sigmoid(torch_module$cat(val_dat[[1]]))
  val_truth <- torch_module$cat(val_dat[[2]])
  val_loss <- criterion(val_preds, val_truth)

  # Convert to R data.frame
  val_df <- as.data.frame(val_truth$cpu()$numpy()) |>
    dplyr::mutate(fruit = as.integer(V1), flower = as.integer(V2)) |>
    dplyr::bind_cols(as.data.frame(val_preds$cpu()$numpy()) |>
                       dplyr::rename(.pred_fruit = V1, .pred_flower = V2)) |>
    dplyr::select(-V1, -V2) |>
    dplyr::mutate(fruit = factor(fruit, levels = c("1", "0")),
                  flower = factor(flower, levels = c("1", "0")))

  # =========================================================================
  # Threshold optimization using DQI (Detection Quality Index)
  # DQI = PPV + Sensitivity - 1 (better for presence-only approach)
  # Using finer threshold grid (0.01) for more precise optimization
  # =========================================================================

  # Fruit: Calculate threshold performance with finer grid
  # Use custom metrics that include precision (PPV) for DQI calculation
  threshold_data_fr <- val_df %>%
    probably::threshold_perf(fruit, .pred_fruit, thresholds = seq(0, 1, by = 0.01),
                             event_level = "first", metrics = .dqi_metrics)

  # Add DQI metric for fruit (DQI = PPV + Sensitivity - 1)
  # Note: yardstick uses "precision" not "ppv", "sensitivity" not "sens"
  ppv_fr <- threshold_data_fr %>% dplyr::filter(.metric == "precision")
  sens_fr <- threshold_data_fr %>% dplyr::filter(.metric == "sensitivity")
  dqi_fr <- ppv_fr %>%
    dplyr::inner_join(sens_fr, by = ".threshold", suffix = c("_ppv", "_sens")) %>%
    dplyr::transmute(
      .threshold = .threshold,
      .metric = "dqi",
      .estimator = "binary",
      .estimate = .estimate_ppv + .estimate_sens - 1
    )
  threshold_data_fr <- dplyr::bind_rows(threshold_data_fr, dqi_fr)

  # Find best threshold by DQI for fruit
  max_dqi_fr <- threshold_data_fr %>%
    dplyr::filter(.metric == "dqi") %>%
    dplyr::slice_max(.estimate, n = 1, with_ties = FALSE)

  # Also keep J-Index for logging
  max_ji_fr <- threshold_data_fr %>%
    dplyr::filter(.metric == "j_index") %>%
    dplyr::slice_max(.estimate, n = 1, with_ties = FALSE)

  # Flower: Calculate threshold performance with finer grid
  # Use custom metrics that include precision (PPV) for DQI calculation
  threshold_data_fl <- val_df %>%
    probably::threshold_perf(flower, .pred_flower, thresholds = seq(0, 1, by = 0.01),
                             event_level = "first", metrics = .dqi_metrics)

  # Add DQI metric for flower (DQI = PPV + Sensitivity - 1)
  # Note: yardstick uses "precision" not "ppv", "sensitivity" not "sens"
  ppv_fl <- threshold_data_fl %>% dplyr::filter(.metric == "precision")
  sens_fl <- threshold_data_fl %>% dplyr::filter(.metric == "sensitivity")
  dqi_fl <- ppv_fl %>%
    dplyr::inner_join(sens_fl, by = ".threshold", suffix = c("_ppv", "_sens")) %>%
    dplyr::transmute(
      .threshold = .threshold,
      .metric = "dqi",
      .estimator = "binary",
      .estimate = .estimate_ppv + .estimate_sens - 1
    )
  threshold_data_fl <- dplyr::bind_rows(threshold_data_fl, dqi_fl)

  # Find best threshold by DQI for flower
  max_dqi_fl <- threshold_data_fl %>%
    dplyr::filter(.metric == "dqi") %>%
    dplyr::slice_max(.estimate, n = 1, with_ties = FALSE)

  # Also keep J-Index for logging
  max_ji_fl <- threshold_data_fl %>%
    dplyr::filter(.metric == "j_index") %>%
    dplyr::slice_max(.estimate, n = 1, with_ties = FALSE)

  # Apply thresholds optimized by DQI
  val_df <- val_df |>
    dplyr::mutate(
      .pred_fl_max = probably::make_two_class_pred(
        estimate = .pred_flower,
        levels = levels(flower),
        threshold = max_dqi_fl$.threshold[1],
        buffer = 0.025
      ),
      .pred_fr_max = probably::make_two_class_pred(
        estimate = .pred_fruit,
        levels = levels(fruit),
        threshold = max_dqi_fr$.threshold[1],
        buffer = 0.025
      )
    )

  val_acc_fl <- yardstick::accuracy(val_df, flower, .pred_fl_max)
  val_acc_fr <- yardstick::accuracy(val_df, fruit, .pred_fr_max)

  val_jind_fl <- yardstick::j_index(val_df, flower, .pred_fl_max)
  val_jind_fr <- yardstick::j_index(val_df, fruit, .pred_fr_max)

  # =========================================================================
  # Log metrics in GuildAI-compatible format
  # CRITICAL: Log step FIRST, then metrics (one per line: "key: value")
  # =========================================================================
  cat("step:", step_num, "\n")
  cat("epoch:", step_num - 1, "\n")  # Explicit epoch for clarity
  cat("val_flower_acc:", val_acc_fl$.estimate[1], "\n")
  cat("val_fruit_acc:", val_acc_fr$.estimate[1], "\n")
  cat("val_flower_jindex:", val_jind_fl$.estimate[1], "\n")
  cat("val_fruit_jindex:", val_jind_fr$.estimate[1], "\n")
  cat("val_flower_dqi:", max_dqi_fl$.estimate[1], "\n")
  cat("val_fruit_dqi:", max_dqi_fr$.estimate[1], "\n")
  cat("val_loss:", val_loss$cpu()$numpy(), "\n")
  cat("\n")  # Blank line for readability

  # Return all computed values
  list(
    val_loss = val_loss,
    val_df = val_df,
    thresholds = list(
      flower = list(
        threshold = max_dqi_fl$.threshold[1],
        dqi = max_dqi_fl$.estimate[1],
        jindex = max_ji_fl$.estimate[1]
      ),
      fruit = list(
        threshold = max_dqi_fr$.threshold[1],
        dqi = max_dqi_fr$.estimate[1],
        jindex = max_ji_fr$.estimate[1]
      )
    ),
    metrics = list(
      flower_acc = val_acc_fl$.estimate[1],
      fruit_acc = val_acc_fr$.estimate[1],
      flower_jindex = val_jind_fl$.estimate[1],
      fruit_jindex = val_jind_fr$.estimate[1],
      flower_dqi = max_dqi_fl$.estimate[1],
      fruit_dqi = max_dqi_fr$.estimate[1],
      val_loss = as.numeric(val_loss$cpu()$numpy())
    )
  )
}
