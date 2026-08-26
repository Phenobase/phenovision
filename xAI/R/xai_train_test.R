#!/usr/bin/env Rscript
# =============================================================================
# Quick validation test for xAI training pipeline
# Runs on L4 GPU (24 GB) with small batch size and minimal data
# =============================================================================

library(reticulate)
library(tidyverse)
library(yardstick)

cat("=== xAI Training Pipeline Test ===\n")

# Override for testing: small batch, few batches, MAE only
pretrained_model <- "mae"
train_csv <- "data/inat/train_v1.1.0.csv"
val_csv <- "data/inat/val_v1.1.0.csv"
num_epochs <- 2L          # Just 2 epochs to test both phases
batch_size <- 16L          # Small for L4 24GB
val_every_n <- 20L         # Validate frequently to test callback
repr_every_n <- 40L        # Test representation extraction
val_subset_size <- 200L    # Tiny for speed
repr_subset_size <- 50L
val_seed <- 42L
warmup_epochs <- 1L
blr <- 5e-4
weight_decay <- 0.05
layer_decay <- 0.65
min_lr <- 1e-6

# Plateau: converge quickly for testing
plateau_window <- 2L
plateau_threshold <- 0.01
plateau_patience <- 2L

num_workers <- 4L

# Output to a test directory
output_dir <- "xAI/output/test_run"
ckpt_dir <- file.path(output_dir, "checkpoints")
repr_dir <- file.path(output_dir, "representations")
metrics_dir <- file.path(output_dir, "metrics")
for (d in c(output_dir, ckpt_dir, repr_dir, metrics_dir)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

# =============================================================================
# Python imports
# =============================================================================

cat("Importing Python modules...\n")
torch <- import("torch")
timm <- import("timm")
np <- import("numpy")

ds <- import_from_path("custom_dataset", "py")

models_vit <- import_from_path("models_vit", "PlantCLEF2022")
pos_embed_mod <- import_from_path("pos_embed", "PlantCLEF2022/util")
lrd <- import_from_path("lr_decay", "PlantCLEF2022/util")
misc <- import_from_path("misc", "PlantCLEF2022/util")
NativeScaler <- misc$NativeScalerWithGradNormCount

xai_engine <- import_from_path("xai_engine", "xAI/py")

types <- import("types")
cat("Imports OK\n")

# =============================================================================
# Data Loading (use small subsets for testing)
# =============================================================================

cat("Loading data...\n")
inat_train_full <- read_csv(train_csv, show_col_types = FALSE)
inat_val_full <- read_csv(val_csv, show_col_types = FALSE)

# Use tiny subsets for testing
set.seed(val_seed)
train_idx <- sample(nrow(inat_train_full), min(500, nrow(inat_train_full)))
val_idx <- sample(nrow(inat_val_full), min(val_subset_size, nrow(inat_val_full)))
repr_idx <- val_idx[seq_len(min(repr_subset_size, length(val_idx)))]

inat_train <- inat_train_full[train_idx, ]
inat_val_subset <- inat_val_full[val_idx, ]
inat_repr_subset <- inat_val_full[repr_idx, ]

cat("Test training images:", nrow(inat_train), "\n")
cat("Test validation subset:", nrow(inat_val_subset), "\n")
cat("Test representation subset:", nrow(inat_repr_subset), "\n")

# Create datasets
train_img <- r_to_py(inat_train$file_name)
train_labels <- inat_train |> select(fruiting, flowering) |> as.matrix()

val_sub_img <- r_to_py(inat_val_subset$file_name)
val_sub_labels <- inat_val_subset |> select(fruiting, flowering) |> as.matrix()

repr_sub_img <- r_to_py(inat_repr_subset$file_name)
repr_sub_labels <- inat_repr_subset |> select(fruiting, flowering) |> as.matrix()

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

transform_val <- timm$data$create_transform(input_size = 224L, is_training = FALSE)

val_sub_dataset <- ds$PhenoDataset(val_sub_img, r_to_py(val_sub_labels))
val_sub_dataset$transform <- transform_val
val_sub_dl <- torch$utils$data$DataLoader(
  val_sub_dataset, batch_size = batch_size, shuffle = FALSE,
  num_workers = num_workers, pin_memory = TRUE, drop_last = FALSE
)

repr_sub_dataset <- ds$PhenoDataset(repr_sub_img, r_to_py(repr_sub_labels))
repr_sub_dataset$transform <- transform_val
repr_sub_dl <- torch$utils$data$DataLoader(
  repr_sub_dataset, batch_size = batch_size, shuffle = FALSE,
  num_workers = num_workers, pin_memory = TRUE, drop_last = FALSE
)

cat("DataLoaders created. Batches per epoch:", length(train_dl), "\n")

# =============================================================================
# Model
# =============================================================================

cat("Initializing MAE pretrained model...\n")
vit <- models_vit$vit_large_patch16(num_classes = 2L)
timm_model <- timm$create_model("vit_large_patch16_224.mae", pretrained = TRUE)
vit$load_state_dict(timm_model$state_dict(), strict = FALSE)
torch$nn$init$trunc_normal_(vit$head$weight, std = 1e-5)

device <- torch$device("cuda:0")
vit <- vit$to(device)
criterion <- torch$nn$BCEWithLogitsLoss()$to(device)
loss_scaler <- NativeScaler()

gradient_layers <- list(
  "head" = "head.",
  "last_block" = "blocks.23.",
  "first_block" = "blocks.0."
)

cat("Model on GPU. VRAM after model load:\n")
system("nvidia-smi --query-gpu=memory.used --format=csv,noheader")

# =============================================================================
# Test validation function
# =============================================================================

all_metrics <- tibble()

validate_and_log <- function(model, global_step, epoch, batch_idx, train_loss, phase) {
  val_result <- xai_engine$fast_validate(model, val_sub_dl, criterion, device)

  outputs <- as.matrix(val_result$outputs$numpy())
  targets <- as.matrix(val_result$targets$numpy())
  val_loss <- val_result$val_loss

  classes <- c("fruit", "flower")
  metrics_row <- tibble(
    global_step = global_step, epoch = epoch, batch = batch_idx,
    phase = phase, train_loss = train_loss, val_loss = val_loss
  )

  for (i in seq_along(classes)) {
    cls <- classes[i]
    pred_probs <- outputs[, i]
    truth <- factor(targets[, i], levels = c(1, 0))
    df <- tibble(.pred_1 = pred_probs, truth = truth)

    auc_roc <- tryCatch(roc_auc(df, truth, .pred_1)$.estimate, error = function(e) NA_real_)
    auc_pr <- tryCatch(pr_auc(df, truth, .pred_1)$.estimate, error = function(e) NA_real_)

    metrics_row[[paste0("auc_roc_", cls)]] <- auc_roc
    metrics_row[[paste0("auc_pr_", cls)]] <- auc_pr
  }

  metrics_row$auc_roc_mean <- mean(c(metrics_row$auc_roc_fruit, metrics_row$auc_roc_flower), na.rm = TRUE)

  cat("  VAL step:", global_step, "val_loss:", round(val_loss, 4),
      "auc_roc_mean:", round(metrics_row$auc_roc_mean, 4), "phase:", phase, "\n")

  all_metrics <<- bind_rows(all_metrics, metrics_row)
  return(val_loss)
}

# =============================================================================
# Test Phase 1 (Frozen backbone)
# =============================================================================

cat("\n=== TEST: Phase 1 (Frozen Backbone) ===\n")

# Use Python-side helpers for parameter manipulation (R can't iterate generators)
xai_engine$freeze_backbone(vit)

save_state <- list(model_state_dict = vit$state_dict(), phase = "init")
torch$save(save_state, file.path(ckpt_dir, "test_init.pt"))
cat("Init checkpoint saved\n")

head_params <- xai_engine$get_head_parameters(vit)
phase1_optimizer <- torch$optim$Adam(head_params, lr = 1e-3)

phase1_args <- types$SimpleNamespace(
  accum_iter = 1L, warmup_epochs = 0L, lr = 1e-3, min_lr = 1e-4, epochs = num_epochs
)

# Phase 1 val callback
phase1_val_cb <- function(model, gs, ep, bi, tl) {
  validate_and_log(model, gs, ep, bi, tl, phase = "frozen")
}

callbacks <- list(val = list(phase1_val_cb, val_every_n))

cat("Running Phase 1 epoch 0...\n")
result <- xai_engine$train_one_epoch_with_callbacks(
  model = vit, criterion = criterion, data_loader = train_dl,
  optimizer = phase1_optimizer, device = device, epoch = 0L,
  loss_scaler = loss_scaler, args = phase1_args,
  callbacks = callbacks, global_step_offset = 0L,
  gradient_layers = gradient_layers
)

global_step <- as.integer(result$global_step)
cat("Phase 1 epoch 0 complete. Global step:", global_step, "\n")
cat("Gradient norms logged:", length(result$gradient_norms), "batches\n")

# Save phase1_final
torch$save(list(model_state_dict = vit$state_dict(), phase = "frozen_final"),
           file.path(ckpt_dir, "test_phase1_final.pt"))
cat("Phase 1 final checkpoint saved\n")

# =============================================================================
# Test Phase 2 (Full model)
# =============================================================================

cat("\n=== TEST: Phase 2 (Full Model Unfrozen) ===\n")

xai_engine$unfreeze_all(vit)

effective_lr <- blr * batch_size / 256
phase2_param_groups <- lrd$param_groups_lrd(
  vit, effective_lr, weight_decay = weight_decay,
  no_weight_decay_list = list(), layer_decay = layer_decay
)
phase2_optimizer <- torch$optim$AdamW(phase2_param_groups, lr = effective_lr)

phase2_args <- types$SimpleNamespace(
  accum_iter = 1L, warmup_epochs = warmup_epochs,
  lr = effective_lr, min_lr = min_lr, epochs = num_epochs
)

# Phase 2 callbacks: val + repr + ckpt
phase2_val_cb <- function(model, gs, ep, bi, tl) {
  validate_and_log(model, gs, ep, bi, tl, phase = "unfrozen")
}

phase2_repr_cb <- function(model, gs, ep, bi, tl) {
  features <- xai_engine$extract_features(model, repr_sub_dl, device)
  feat_path <- file.path(repr_dir, sprintf("test_features_step%06d.pt", gs))
  torch$save(features, feat_path)
  cat("  REPR saved:", feat_path, "shape:", paste(as.integer(features$shape), collapse = "x"), "\n")
}

phase2_ckpt_cb <- function(model, gs, ep, bi, tl) {
  ckpt_path <- file.path(ckpt_dir, sprintf("test_phase2_step%06d.pt", gs))
  torch$save(list(model_state_dict = model$state_dict(), step = gs, epoch = ep, phase = "unfrozen"), ckpt_path)
  cat("  CKPT saved:", ckpt_path, "\n")
}

callbacks2 <- list(
  val = list(phase2_val_cb, val_every_n),
  repr = list(phase2_repr_cb, repr_every_n),
  ckpt = list(phase2_ckpt_cb, 20L)  # frequent for testing
)

cat("Running Phase 2 epoch 1...\n")
result2 <- xai_engine$train_one_epoch_with_callbacks(
  model = vit, criterion = criterion, data_loader = train_dl,
  optimizer = phase2_optimizer, device = device, epoch = 1L,
  loss_scaler = loss_scaler, args = phase2_args,
  callbacks = callbacks2, global_step_offset = global_step,
  gradient_layers = gradient_layers
)

global_step <- as.integer(result2$global_step)
cat("Phase 2 epoch 1 complete. Global step:", global_step, "\n")

# =============================================================================
# Summary
# =============================================================================

cat("\n=== TEST COMPLETE ===\n")
cat("Metrics collected:", nrow(all_metrics), "validation points\n")
write_csv(all_metrics, file.path(metrics_dir, "test_validation_metrics.csv"))
cat("Metrics saved to:", file.path(metrics_dir, "test_validation_metrics.csv"), "\n")

cat("\nCheckpoints:\n")
system(paste("ls -lh", ckpt_dir))
cat("\nRepresentations:\n")
system(paste("ls -lh", repr_dir))
cat("\nGPU memory at end:\n")
system("nvidia-smi --query-gpu=memory.used --format=csv,noheader")

cat("\nAll tests passed! Pipeline is working.\n")
