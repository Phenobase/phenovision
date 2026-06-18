#!/usr/bin/env Rscript
# =============================================================================
# xAI Training Script: Pretraining as Preadaptation Experiment
# =============================================================================
#
# Two-phase training for ecological fitting decomposition:
#   Phase 1 (Equalization): Frozen backbone, train head only → reveals true starting fitness
#   Phase 2 (Evolution): Full model unfrozen → measures evolvability
#
# Collects: frequent validation (AUC-ROC, AUC-PR, loss), gradient norms,
#           latent representations, and checkpoints with hybrid frequency.
#
# Usage:
#   Rscript xAI/R/xai_train.R                           # defaults to MAE
#   Rscript xAI/R/xai_train.R --pretrained_model plantclef
#   Rscript xAI/R/xai_train.R --pretrained_model mae
# =============================================================================

library(reticulate)
library(tidyverse)
library(yardstick)

# =============================================================================
# Flags / Configuration
# =============================================================================

# Parse command-line args (simple flag parsing)
args <- commandArgs(trailingOnly = TRUE)
pretrained_model <- "mae"  # default
for (i in seq_along(args)) {
  if (args[i] == "--pretrained_model" && i < length(args)) {
    pretrained_model <- args[i + 1]
  }
}

cat("=== xAI Training: Pretraining as Preadaptation ===\n")
cat("Pretrained model:", pretrained_model, "\n")

# Training configuration
train_csv <- "data/inat/train_v1.1.0.csv"
val_csv <- "data/inat/val_v1.1.0.csv"
num_epochs <- 20L
batch_size <- 384L
val_every_n <- 100L    # Validate on 10K subset every N batches
repr_every_n <- 500L   # Extract representations every N batches
val_subset_size <- 10000L
repr_subset_size <- 1000L
val_seed <- 42L

# LR configuration (same as original training)
blr <- 5e-4
weight_decay <- 0.05
layer_decay <- 0.65
min_lr <- 1e-6
warmup_epochs <- 5L

# Plateau detection for Phase 1
plateau_window <- 5L     # rolling window size
plateau_threshold <- 0.001  # min improvement to not be "plateau"
plateau_patience <- 3L   # consecutive plateau checks before stopping

# Data loader workers
num_workers <- {
  slurm_cpus <- Sys.getenv("SLURM_CPUS_PER_TASK", unset = "")
  if (nchar(slurm_cpus) > 0) as.integer(slurm_cpus) else 8L
}

# Output directories
run_name <- paste0(pretrained_model, "_run")
output_dir <- file.path("xAI/output", run_name)
ckpt_dir <- file.path(output_dir, "checkpoints")
repr_dir <- file.path(output_dir, "representations")
metrics_dir <- file.path(output_dir, "metrics")
for (d in c(output_dir, ckpt_dir, repr_dir, metrics_dir)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

# =============================================================================
# Python imports
# =============================================================================

torch <- import("torch")
timm <- import("timm")
PIL <- import("PIL")
np <- import("numpy")

# Import from project
ds <- import_from_path("custom_dataset", "py")
eval_module <- import_from_path("evaluate", "py")

# Import from PlantCLEF2022
models_vit <- import_from_path("models_vit", "PlantCLEF2022")
pos_embed <- import_from_path("pos_embed", "PlantCLEF2022/util")
lrd <- import_from_path("lr_decay", "PlantCLEF2022/util")
misc <- import_from_path("misc", "PlantCLEF2022/util")
NativeScaler <- misc$NativeScalerWithGradNormCount

# Import xAI engine
xai_engine <- import_from_path("xai_engine", "xAI/py")

types <- import("types")

# =============================================================================
# Data Loading
# =============================================================================

cat("Loading training data...\n")
inat_train <- read_csv(train_csv, show_col_types = FALSE)
inat_val <- read_csv(val_csv, show_col_types = FALSE)

# Fixed validation subset (10K)
set.seed(val_seed)
val_subset_idx <- sample(nrow(inat_val), min(val_subset_size, nrow(inat_val)))
repr_subset_idx <- val_subset_idx[seq_len(min(repr_subset_size, length(val_subset_idx)))]

# Save indices for reproducibility
write_csv(
  tibble(idx = val_subset_idx),
  file.path(output_dir, "val_subset_indices.csv")
)
write_csv(
  tibble(idx = repr_subset_idx),
  file.path(output_dir, "repr_subset_indices.csv")
)

inat_val_subset <- inat_val[val_subset_idx, ]
inat_repr_subset <- inat_val[repr_subset_idx, ]

cat("Training images:", nrow(inat_train), "\n")
cat("Validation subset:", nrow(inat_val_subset), "\n")
cat("Representation subset:", nrow(inat_repr_subset), "\n")

# Create PyTorch datasets and loaders
train_img <- r_to_py(inat_train$file_name)
train_labels <- inat_train |> select(fruiting, flowering) |> as.matrix()

val_sub_img <- r_to_py(inat_val_subset$file_name)
val_sub_labels <- inat_val_subset |> select(fruiting, flowering) |> as.matrix()

repr_sub_img <- r_to_py(inat_repr_subset$file_name)
repr_sub_labels <- inat_repr_subset |> select(fruiting, flowering) |> as.matrix()

# Training data loader (with augmentation)
train_dataset <- ds$PhenoDataset(train_img, r_to_py(train_labels))
transform_train <- timm$data$create_transform(
  input_size = 224L, is_training = TRUE,
  auto_augment = "rand-m9-mstd0.5-inc1",
  re_prob = 0.25, re_mode = "pixel", re_count = 1L
)
train_dataset$transform <- transform_train

train_dl <- torch$utils$data$DataLoader(
  train_dataset,
  batch_size = batch_size,
  shuffle = TRUE,
  num_workers = num_workers,
  pin_memory = TRUE,
  drop_last = TRUE
)

# Validation subset loader (no augmentation)
val_sub_dataset <- ds$PhenoDataset(val_sub_img, r_to_py(val_sub_labels))
transform_val <- timm$data$create_transform(input_size = 224L, is_training = FALSE)
val_sub_dataset$transform <- transform_val

val_sub_dl <- torch$utils$data$DataLoader(
  val_sub_dataset,
  batch_size = batch_size,
  shuffle = FALSE,
  num_workers = num_workers,
  pin_memory = TRUE,
  drop_last = FALSE
)

# Representation subset loader
repr_sub_dataset <- ds$PhenoDataset(repr_sub_img, r_to_py(repr_sub_labels))
repr_sub_dataset$transform <- transform_val

repr_sub_dl <- torch$utils$data$DataLoader(
  repr_sub_dataset,
  batch_size = batch_size,
  shuffle = FALSE,
  num_workers = num_workers,
  pin_memory = TRUE,
  drop_last = FALSE
)

# =============================================================================
# Model Initialization
# =============================================================================

cat("Initializing model with pretrained:", pretrained_model, "\n")

vit <- models_vit$vit_large_patch16(num_classes = 2L)

if (pretrained_model == "plantclef") {
  model_path <- "models/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth"
  if (!file.exists(model_path)) {
    stop("PlantCLEF base model not found at: ", model_path,
         "\nPlease recover this file and place it in models/")
  }
  model <- torch$load(model_path, map_location = "cpu")
  checkpoint_model <- model$model
  checkpoint_model$head.bias <- NULL
  checkpoint_model$head.weight <- NULL
  vit$load_state_dict(checkpoint_model, strict = FALSE)
  pos_embed$interpolate_pos_embed(vit, checkpoint_model)
} else if (pretrained_model == "mae") {
  timm_model <- timm$create_model("vit_large_patch16_224.mae", pretrained = TRUE)
  timm_state_dict <- timm_model$state_dict()
  vit$load_state_dict(timm_state_dict, strict = FALSE)
} else {
  stop("Unsupported pretrained_model: ", pretrained_model)
}

# Initialize classification head (random)
torch$nn$init$trunc_normal_(vit$head$weight, std = 1e-5)

device <- torch$device("cuda:0")
vit <- vit$to(device)

# Loss function
criterion <- torch$nn$BCEWithLogitsLoss()$to(device)

# Loss scaler for mixed precision
loss_scaler <- NativeScaler()

# Gradient layers of interest
gradient_layers <- list(
  "head" = "head.",
  "last_block" = "blocks.23.",
  "first_block" = "blocks.0."
)

cat("Model initialized. Total parameters:", sum(sapply(vit$parameters(), function(p) p$numel())), "\n")

# =============================================================================
# Checkpoint helpers
# =============================================================================

save_checkpoint <- function(model, path, step, epoch, phase, extra = list()) {
  state <- list(
    model_state_dict = model$state_dict(),
    global_step = step,
    epoch = epoch,
    phase = phase
  )
  state <- c(state, extra)
  torch$save(state, path)
  cat("Checkpoint saved:", path, "\n")
}

# Compute checkpoint schedule for Phase 2
compute_phase2_ckpt_schedule <- function(phase2_start_epoch, num_epochs, batches_per_epoch) {
  schedule <- c()
  for (ep in phase2_start_epoch:(num_epochs - 1)) {
    phase2_ep <- ep - phase2_start_epoch
    if (phase2_ep <= 5) {
      # Every 240 batches (10 per epoch)
      schedule <- c(schedule, ep * batches_per_epoch + seq(240, batches_per_epoch, by = 240))
    } else if (phase2_ep <= 10) {
      # Every 500 batches (5 per epoch)
      schedule <- c(schedule, ep * batches_per_epoch + seq(500, batches_per_epoch, by = 500))
    } else {
      # End of epoch only
      schedule <- c(schedule, (ep + 1) * batches_per_epoch)
    }
  }
  sort(unique(schedule))
}

# =============================================================================
# Validation callback
# =============================================================================

# Metrics accumulator
all_metrics <- tibble()

validate_and_log <- function(model, global_step, epoch, batch_idx, train_loss, phase) {
  val_result <- xai_engine$fast_validate(model, val_sub_dl, criterion, device)

  # Convert to R
  outputs <- as.matrix(val_result$outputs$numpy())
  targets <- as.matrix(val_result$targets$numpy())
  val_loss <- val_result$val_loss

  # Compute AUC-ROC and AUC-PR for each class
  # Column 0 = fruiting, column 1 = flowering
  classes <- c("fruit", "flower")
  metrics_row <- tibble(
    global_step = global_step,
    epoch = epoch,
    batch = batch_idx,
    phase = phase,
    train_loss = train_loss,
    val_loss = val_loss
  )

  for (i in seq_along(classes)) {
    cls <- classes[i]
    pred_probs <- outputs[, i]
    truth <- factor(targets[, i], levels = c(1, 0))

    df <- tibble(.pred_1 = pred_probs, truth = truth)

    auc_roc <- tryCatch(
      roc_auc(df, truth, .pred_1)$.estimate,
      error = function(e) NA_real_
    )
    auc_pr <- tryCatch(
      pr_auc(df, truth, .pred_1)$.estimate,
      error = function(e) NA_real_
    )

    metrics_row[[paste0("auc_roc_", cls)]] <- auc_roc
    metrics_row[[paste0("auc_pr_", cls)]] <- auc_pr
  }

  # Mean AUC across classes
  metrics_row$auc_roc_mean <- mean(c(metrics_row$auc_roc_fruit, metrics_row$auc_roc_flower), na.rm = TRUE)
  metrics_row$auc_pr_mean <- mean(c(metrics_row$auc_pr_fruit, metrics_row$auc_pr_flower), na.rm = TRUE)

  # Log to stdout (GuildAI format)
  cat("step:", global_step, "\n")
  cat("val_loss:", val_loss, "\n")
  cat("auc_roc_flower:", metrics_row$auc_roc_flower, "\n")
  cat("auc_roc_fruit:", metrics_row$auc_roc_fruit, "\n")
  cat("auc_roc_mean:", metrics_row$auc_roc_mean, "\n")
  cat("auc_pr_flower:", metrics_row$auc_pr_flower, "\n")
  cat("auc_pr_fruit:", metrics_row$auc_pr_fruit, "\n")
  cat("phase:", phase, "\n")
  cat("\n")

  # Append to accumulator
  all_metrics <<- bind_rows(all_metrics, metrics_row)

  # Save incrementally
  write_csv(all_metrics, file.path(metrics_dir, "validation_metrics.csv"))

  return(val_loss)
}

# =============================================================================
# PHASE 1: Frozen Backbone (Equalization)
# =============================================================================

cat("\n=== PHASE 1: Frozen Backbone (Equalization) ===\n")

# Freeze all params except head (use Python helper — R can't iterate generators)
xai_engine$freeze_backbone(vit)

# Save epoch-0 checkpoint (random head, frozen backbone)
save_checkpoint(vit, file.path(ckpt_dir, "epoch0_init.pt"), 0L, 0L, "init")

# Phase 1 optimizer (only head parameters)
head_params <- xai_engine$get_head_parameters(vit)
phase1_optimizer <- torch$optim$Adam(head_params, lr = 1e-3, weight_decay = 0.0)

# Phase 1 args (simplified)
phase1_args <- types$SimpleNamespace(
  accum_iter = 1L,
  warmup_epochs = 0L,
  lr = 1e-3,
  min_lr = 1e-4,
  epochs = num_epochs
)

# Plateau detection state
plateau_val_losses <- c()
plateau_count <- 0L
phase1_converged <- FALSE
global_step <- 0L

for (epoch in seq_len(num_epochs) - 1L) {  # 0-indexed
  cat("\n--- Phase 1, Epoch", epoch, "---\n")

  # Phase 1 validation callback
  phase1_val_cb <- function(model, gs, ep, bi, tl) {
    vl <- validate_and_log(model, gs, ep, bi, tl, phase = "frozen")

    # Plateau detection
    plateau_val_losses <<- c(plateau_val_losses, vl)
    if (length(plateau_val_losses) >= plateau_window) {
      recent <- tail(plateau_val_losses, plateau_window)
      improvement <- recent[1] - recent[length(recent)]
      if (improvement < plateau_threshold) {
        plateau_count <<- plateau_count + 1L
        cat("Plateau detected (", plateau_count, "/", plateau_patience, ")\n")
      } else {
        plateau_count <<- 0L
      }
    }
  }

  callbacks <- list(
    val = list(phase1_val_cb, val_every_n)
  )

  result <- xai_engine$train_one_epoch_with_callbacks(
    model = vit,
    criterion = criterion,
    data_loader = train_dl,
    optimizer = phase1_optimizer,
    device = device,
    epoch = epoch,
    loss_scaler = loss_scaler,
    args = phase1_args,
    callbacks = callbacks,
    global_step_offset = global_step,
    gradient_layers = gradient_layers
  )

  global_step <- as.integer(result$global_step)

  # Save Phase 1 checkpoint (1 per epoch)
  save_checkpoint(vit, file.path(ckpt_dir, sprintf("phase1_epoch%d.pt", epoch)),
                  global_step, epoch, "frozen")

  # Check convergence
  if (plateau_count >= plateau_patience) {
    cat("\n*** Phase 1 converged at epoch", epoch, "step", global_step, "***\n")
    phase1_converged <- TRUE
    break
  }
}

if (!phase1_converged) {
  cat("\n*** Phase 1 did not converge within", num_epochs, "epochs — proceeding anyway ***\n")
}

phase1_end_epoch <- epoch
phase1_end_step <- global_step

# Run final Phase 1 validation
cat("\nFinal Phase 1 validation...\n")
vit$eval()
final_p1_loss <- validate_and_log(vit, global_step, phase1_end_epoch, -1L, NA_real_, "frozen_final")

# Save the crucial phase1_final checkpoint (enables rerunning Phase 2)
save_checkpoint(vit, file.path(ckpt_dir, "phase1_final.pt"),
                global_step, phase1_end_epoch, "frozen_final",
                extra = list(val_loss = final_p1_loss))

cat("\nPHASE_TRANSITION: step=", global_step, " epoch=", phase1_end_epoch,
    " val_loss=", final_p1_loss, "\n")

# =============================================================================
# PHASE 2: Full Model (The Evolutionary Experiment)
# =============================================================================

cat("\n=== PHASE 2: Full Model Unfrozen (Evolution) ===\n")

# Unfreeze all parameters (use Python helper)
xai_engine$unfreeze_all(vit)

# Rebuild optimizer with layer-wise LR decay for all parameters
effective_lr <- blr * batch_size / 256
phase2_param_groups <- lrd$param_groups_lrd(
  vit, effective_lr,
  weight_decay = weight_decay,
  no_weight_decay_list = list(),
  layer_decay = layer_decay
)
phase2_optimizer <- torch$optim$AdamW(phase2_param_groups, lr = effective_lr)

# Phase 2 args
phase2_args <- types$SimpleNamespace(
  accum_iter = 1L,
  warmup_epochs = warmup_epochs,
  lr = effective_lr,
  min_lr = min_lr,
  epochs = num_epochs
)

# Compute checkpoint schedule
batches_per_epoch <- length(train_dl)
phase2_start_epoch <- phase1_end_epoch + 1L
ckpt_schedule <- compute_phase2_ckpt_schedule(phase2_start_epoch, num_epochs, batches_per_epoch)
cat("Phase 2 checkpoint schedule:", length(ckpt_schedule), "checkpoints planned\n")

# Save phase2_start checkpoint
save_checkpoint(vit, file.path(ckpt_dir, "phase2_start.pt"),
                global_step, phase2_start_epoch, "phase2_start")

# Phase 2 training loop
for (epoch in phase2_start_epoch:(num_epochs - 1L)) {
  cat("\n--- Phase 2, Epoch", epoch, "(Phase 2 epoch", epoch - phase2_start_epoch, ") ---\n")

  # Validation callback
  phase2_val_cb <- function(model, gs, ep, bi, tl) {
    validate_and_log(model, gs, ep, bi, tl, phase = "unfrozen")
  }

  # Representation callback
  phase2_repr_cb <- function(model, gs, ep, bi, tl) {
    features <- xai_engine$extract_features(model, repr_sub_dl, device)
    feat_path <- file.path(repr_dir, sprintf("features_step%06d.pt", gs))
    torch$save(features, feat_path)
    cat("Representations saved:", feat_path, "(shape:", paste(features$shape, collapse = "x"), ")\n")
  }

  # Checkpoint callback
  phase2_ckpt_cb <- function(model, gs, ep, bi, tl) {
    if (gs %in% ckpt_schedule) {
      ckpt_path <- file.path(ckpt_dir, sprintf("phase2_step%06d.pt", gs))
      save_checkpoint(model, ckpt_path, gs, ep, "unfrozen")
    }
  }

  # Determine checkpoint frequency for this epoch
  phase2_ep <- epoch - phase2_start_epoch
  if (phase2_ep <= 5) {
    ckpt_freq <- 240L
  } else if (phase2_ep <= 10) {
    ckpt_freq <- 500L
  } else {
    ckpt_freq <- batches_per_epoch  # end of epoch only
  }

  callbacks <- list(
    val = list(phase2_val_cb, val_every_n),
    repr = list(phase2_repr_cb, repr_every_n),
    ckpt = list(phase2_ckpt_cb, ckpt_freq)
  )

  result <- xai_engine$train_one_epoch_with_callbacks(
    model = vit,
    criterion = criterion,
    data_loader = train_dl,
    optimizer = phase2_optimizer,
    device = device,
    epoch = epoch,
    loss_scaler = loss_scaler,
    args = phase2_args,
    callbacks = callbacks,
    global_step_offset = global_step,
    gradient_layers = gradient_layers
  )

  global_step <- as.integer(result$global_step)

  # Always save end-of-epoch checkpoint
  save_checkpoint(vit, file.path(ckpt_dir, sprintf("phase2_epoch%d.pt", epoch)),
                  global_step, epoch, "unfrozen")

  # Save gradient norms for this epoch
  grad_norms_df <- as_tibble(do.call(rbind, lapply(result$gradient_norms, as.data.frame)))
  grad_norms_df$epoch <- epoch
  grad_norms_df$batch <- seq_len(nrow(grad_norms_df)) - 1L
  grad_norms_df$global_step <- seq(global_step - nrow(grad_norms_df), global_step - 1)

  grad_path <- file.path(metrics_dir, sprintf("gradient_norms_epoch%d.csv", epoch))
  write_csv(grad_norms_df, grad_path)
}

# =============================================================================
# Finalize
# =============================================================================

cat("\n=== Training Complete ===\n")
cat("Phase 1 ended at epoch", phase1_end_epoch, "step", phase1_end_step, "\n")
cat("Phase 2 ran epochs", phase2_start_epoch, "to", num_epochs - 1, "\n")
cat("Total global steps:", global_step, "\n")
cat("Output directory:", output_dir, "\n")

# Save run summary
summary <- tibble(
  pretrained_model = pretrained_model,
  phase1_end_epoch = phase1_end_epoch,
  phase1_end_step = phase1_end_step,
  phase1_converged = phase1_converged,
  phase2_start_epoch = phase2_start_epoch,
  total_epochs = num_epochs,
  total_steps = global_step,
  val_subset_size = val_subset_size,
  repr_subset_size = repr_subset_size,
  train_csv = train_csv,
  val_csv = val_csv
)
write_csv(summary, file.path(output_dir, "run_summary.csv"))

cat("\nAll done.\n")
