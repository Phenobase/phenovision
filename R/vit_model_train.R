#!/usr/bin/env Rscript
#| requires:
#|     - file: data
#|       target-type: link
#|     - file: models
#|       target-type: link
#|     - file: output
#|       target-type: link

library(reticulate)
library(tidyverse)
library(tidymodels)
library(probably)

# Source evaluation helper function
source("R/training_eval_helper_training.R")

# =============================================================================
# GuildAI Flags (adjustable parameters)
# =============================================================================

# Model versioning and data paths
model_version <- "v1.1.0"  # Version string for this training run
train_csv <- "data/inat/train_v1.1.0.csv"  # Path to versioned training data
val_csv <- "data/inat/val_v1.1.0.csv"  # Path to versioned validation data

# Model initialization
pretrained_model <- "mae"  # Options: "plantclef", "imagenet", "imagenet21k", "dino", "mae", "clip", "doi"
pretrained_doi <- ""  # DOI for HuggingFace model (only used if pretrained_model = "doi")
reinit_head <- FALSE  # If TRUE, reinitialize classification head even when loading from DOI

# Data loader workers (reads from SLURM_CPUS_PER_TASK env var if available)
num_workers <- {
  slurm_cpus <- Sys.getenv("SLURM_CPUS_PER_TASK", unset = "")
  if (nchar(slurm_cpus) > 0) as.integer(slurm_cpus) else 8L
}

torch <- import("torch")
timm <- import("timm")
PIL <- import("PIL")
ds <- import_from_path("custom_dataset", "py")
del <- import_from_path("remove_bad_images", "py")
engine <- import_from_path("engine_finetune", "PlantCLEF2022")
lrd <- import_from_path("lr_decay", "PlantCLEF2022/util")
misc <- import_from_path("misc", "PlantCLEF2022/util")
eval <- import_from_path("evaluate", "py")
NativeScaler <- misc$NativeScalerWithGradNormCount
types <- import("types")

# Create model output folder using version
model_folder <- file.path("output/reproductive", model_version)
cat("Model will be saved to:", model_folder, "\n")
cat("Model version:", model_version, "\n")
if(!dir.exists(model_folder)) dir.create(model_folder, recursive = TRUE)
if(!dir.exists(file.path(model_folder, "checkpoints"))) dir.create(file.path(model_folder, "checkpoints"))

# Load training and validation data from versioned CSV files
# Note: Bad images have been filtered out during data splitting
cat("Loading training data from:", train_csv, "\n")
cat("Loading validation data from:", val_csv, "\n")
inat_train <- read_csv(train_csv)
inat_val <- read_csv(val_csv)

train_img <- r_to_py(inat_train$file_name)
train_fruit_flower <- inat_train |>
  select(fruiting, flowering) |>
  as.matrix()

val_img <- r_to_py(inat_val$file_name)
val_fruit_flower <- inat_val |>
  select(fruiting, flowering) |>
  as.matrix()

# Modified section for pretrained model selection
models_vit <- import_from_path("models_vit", "PlantCLEF2022")
pos_embed <- import_from_path("pos_embed", "PlantCLEF2022/util")

if (pretrained_model == "plantclef") {
  # Original approach - load PlantCLEF model
  model <- torch$load("models/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth")
  vit <- models_vit$vit_large_patch16(num_classes = 2L)
  checkpoint_model <- model$model

  # Load weights from PlantCLEF checkpoint
  checkpoint_model$head.bias <- NULL
  checkpoint_model$head.weight <- NULL
  vit$load_state_dict(checkpoint_model, strict = FALSE)

  # Use custom position embedding interpolation
  pos_embed$interpolate_pos_embed(vit, checkpoint_model)
} else if (pretrained_model == "doi") {
  # Load from HuggingFace DOI
  if (pretrained_doi == "" || is.null(pretrained_doi)) {
    stop("pretrained_doi must be specified when pretrained_model = 'doi'")
  }

  cat("Loading model from HuggingFace DOI:", pretrained_doi, "\n")

  # Source the model loading function
  source("R/model_loading_training.R")

  # Load model from DOI (already on CUDA)
  vit <- load_phenovision_for_training(
    doi = pretrained_doi,
    num_classes = 2L,  # Reproductive model has 2 classes
    reinit_head = reinit_head,
    device = "cuda"
  )

  cat("Model loaded successfully from DOI\n")
} else {
  # Use timm for other pretrained models
  model_name <- switch(
    pretrained_model,
    "imagenet" = "vit_large_patch16_224",
    "imagenet21k" = "vit_large_patch16_224_in21k",
    "dino" = "vit_large_patch16_224.dino",
    "mae" = "vit_large_patch16_224.mae",
    "clip" = "vit_large_patch16_clip_224.openai",
    # Default to ImageNet if unknown option
    "vit_large_patch16_224"
  )

  # For timm models, still use the PlantCLEF architecture
  # but initialize with timm's pretrained weights
  vit <- models_vit$vit_large_patch16(num_classes = 2L)

  # Load pretrained weights from timm
  timm_model <- timm$create_model(model_name, pretrained = TRUE)

  # Extract state_dict from timm model
  timm_state_dict <- timm_model$state_dict()

  # Load the timm state_dict into our model (with strict=FALSE to handle key differences)
  vit$load_state_dict(timm_state_dict, strict = FALSE)
}

# Initialize the classification head (unless we're using DOI and kept the head)
if (pretrained_model != "doi" || reinit_head) {
  torch$nn$init$trunc_normal_(vit$head$weight, std = 1e-5)
}

# Ensure model is on CUDA
if (!vit$training) {
  vit$train()  # Set to training mode
}
vit <- vit$cuda()

# Continue with the original script
config <- timm$data$resolve_data_config(model = vit)

# Training transform with explicit augmentation
transform <- timm$data$create_transform(
  input_size = list(3L, 224L, 224L),
  is_training = TRUE,
  color_jitter = FALSE,
  auto_augment = 'rand-m9-mstd0.5-inc1',  # RandAugment
  interpolation = 'bicubic',
  re_prob = 0.25,  # Random erasing probability
  re_mode = 'pixel',
  re_count = 1,
  mean = list(0.485, 0.456, 0.406),
  std = list(0.229, 0.224, 0.255)
)

# Validation transform without augmentation
val_transform <- timm$data$create_transform(!!!config)

batch_size <- 384L

train_ds <- ds$PhenoDataset(train_img, train_fruit_flower, transform = transform)
train_dl <- timm$data$create_loader(train_ds, c(3L, 224L, 224L), batch_size, num_workers = num_workers,
                                    is_training = TRUE)

val_ds <- ds$PhenoDataset(val_img, val_fruit_flower, transform = val_transform)
val_dl <- timm$data$create_loader(val_ds, c(3L, 224L, 224L), batch_size, num_workers = num_workers)

#test_it <- as_iterator(train_dl)
#test_dat <- iter_next(test_it)

weight_decay <- 0.05
layer_decay <- 0.65

param_groups <- lrd$param_groups_lrd(vit, weight_decay,
                                     no_weight_decay_list=vit$no_weight_decay(),
                                     layer_decay = layer_decay
)

blr <- 5e-4

lr <- blr * batch_size / 256

min_lr <- 1e-6

optimizer <- torch$optim$AdamW(param_groups, lr = lr)
loss_scaler <- NativeScaler()

loss <- torch$nn$BCEWithLogitsLoss()
criterion <- function(outputs, targets) {
  loss(outputs, targets$type(torch$float32))
}

#log_writer <- torch$utils$tensorboard$SummaryWriter(log_dir = model_folder)
log_writer <- NULL

clip_grad <- py_none()

num_epochs <- 100

# =============================================================================
# Training Loop with num_epochs=0 Support
# =============================================================================

# Save initial model as epoch 0 (enables num_epochs=0 testing)
checkpoint_name <- paste0("phenovision_", model_version, "_epoch0.pt")
torch$save(vit, file.path(model_folder, "checkpoints", checkpoint_name))
cat("Saved initial model as:", checkpoint_name, "\n")

# Initial evaluation (step 1 = epoch 0)
cat("\n=== Initial Evaluation (epoch 0) ===\n")
evaluate_and_log(vit, val_dl, eval, torch, criterion, step_num = 1)

# Training loop (only if num_epochs > 0)
if (num_epochs > 0) {
  for (i in 1:num_epochs) {
    cat("\n=== Training epoch", i, "of", num_epochs, "===\n")

    # Train one epoch
    train_stats <- engine$train_one_epoch(
      vit, criterion, train_dl,
      optimizer, "cuda", i, loss_scaler,
      clip_grad, mixup_fn = py_none(),
      log_writer = log_writer,
      args = types$SimpleNamespace(accum_iter = 1L, warmup_epochs = 5L, lr = lr, min_lr = min_lr,
                                   epochs = num_epochs)
    )

    # Save checkpoint
    checkpoint_name <- paste0("phenovision_", model_version, "_epoch", i, ".pt")
    torch$save(vit, file.path(model_folder, "checkpoints", checkpoint_name))
    cat("Saved checkpoint:", checkpoint_name, "\n")

    # Evaluate (step i+1 = epoch i)
    cat("\n=== Evaluation (epoch", i, ") ===\n")
    evaluate_and_log(vit, val_dl, eval, torch, criterion, step_num = i + 1)
  }
}

cat("\n=== Training Complete ===\n")
cat("Final checkpoint: epoch", num_epochs, "\n")
cat("Total evaluations logged:", num_epochs + 1, "(step 1 to", num_epochs + 1, ")\n")
