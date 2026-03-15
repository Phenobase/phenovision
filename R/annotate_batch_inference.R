#' Annotate a Batch of Images with PhenoVision Model
#'
#' Run inference on a batch of plant images using a PhenoVision model loaded
#' from HuggingFace Hub. Returns predictions with probabilities and logits for
#' each phenological trait.
#'
#' @param images_batch_split Data frame with image paths and metadata
#' @param model_doi Character. DOI of model to use (default: reproductive model)
#' @param trait Character. Which traits to predict: "flower/fruit" or "leaves"
#' @param num_workers Integer. Number of data loader workers (default: 6)
#' @param done Character vector. Photo IDs already processed (for filtering)
#'
#' @return Data frame with predictions:
#'   - Original image metadata
#'   - `.pred_*` columns: Probabilities [0, 1] for each trait
#'   - `.logit_*` columns: Raw logits for each trait
#'   - `model_version`: DOI of model used
#'   - `corrupt`: Boolean indicating if image failed to load
#'
#' @details
#' **For flower/fruit trait:**
#' - `.pred_fruit`: Probability of fruiting structures
#' - `.pred_flower`: Probability of flowering structures
#'
#' **For leaves trait:**
#' - `.pred_leaves_green`: Probability of green leaves
#' - `.pred_leaves_colored`: Probability of colored/senescing leaves
#' - `.pred_leaves_breaking_buds`: Probability of breaking leaf buds
#'
#' **Note:** Typo fixed from original - was `breaking_nuds`, now correctly `breaking_buds`
#'
#' The function:
#' 1. Loads the model from DOI
#' 2. Validates model output dimensions match trait type
#' 3. Creates PyTorch dataset and dataloader
#' 4. Runs inference on GPU
#' 5. Converts predictions to probabilities via sigmoid
#' 6. Marks corrupt images that failed to load
#'
#' @examples
#' \dontrun{
#' # Annotate batch with reproductive model
#' results <- annotate_batch(
#'   images_batch,
#'   model_doi = "10.57967/hf/2763",
#'   trait = "flower/fruit"
#' )
#'
#' # Annotate batch with leaf model
#' results_leaves <- annotate_batch(
#'   images_batch,
#'   model_doi = "10.57967/hf/5785",
#'   trait = "leaves"
#' )
#' }
#'
#' @export
annotate_batch <- function(images_batch_split,
                           model_doi = "10.57967/hf/2763",
                           trait = c("flower/fruit", "leaves"),
                           num_workers = 6L,
                           batch_size = 1280L,
                           done = NULL) {

  trait <- match.arg(trait)

  # Assign this worker to a specific GPU using CUDA_VISIBLE_DEVICES.
  # MUST happen before ANY Python imports (including load_phenovision) so
  # CUDA initializes with only the assigned GPU visible.
  # Once CUDA is initialized in a persistent worker, the GPU assignment sticks
  # for all subsequent calls, so we only need to acquire on first call.
  #
  # Note: SLURM may pre-set CUDA_VISIBLE_DEVICES (e.g., "0,1,2" for 3 GPUs),
  # so we use a separate flag to track whether we've already assigned a single GPU.
  if (Sys.getenv("PHENOVISION_GPU_ASSIGNED") == "") {
    gpu_lock_dir <- file.path(here::here(), ".gpu_locks")
    dir.create(gpu_lock_dir, showWarnings = FALSE, recursive = TRUE)

    ngpus <- as.integer(system("nvidia-smi -L 2>/dev/null | wc -l",
                               intern = TRUE))
    if (is.na(ngpus) || ngpus < 1L) ngpus <- 1L

    # Acquire a GPU by creating a lock file (gpu_0.lock, gpu_1.lock, etc.)
    # Each persistent worker holds its lock for its entire lifetime.
    gpu_id <- NA_integer_
    for (i in seq_len(ngpus) - 1L) {
      lf <- file.path(gpu_lock_dir, paste0("gpu_", i, ".lock"))
      lock_result <- filelock::lock(lf, timeout = 0)
      if (!is.null(lock_result)) {
        gpu_id <- i
        # Do NOT unlock — hold lock for worker lifetime.
        # Lock is auto-released when the worker process exits.
        break
      }
    }
    if (is.na(gpu_id)) gpu_id <- 0L  # fallback if all locked

    Sys.setenv(CUDA_VISIBLE_DEVICES = as.character(gpu_id))
    Sys.setenv(PHENOVISION_GPU_ASSIGNED = "1")
    message("Worker ", Sys.getpid(), " acquired GPU ", gpu_id,
            " (CUDA_VISIBLE_DEVICES=", gpu_id, ")")
  }

  inf_images <- images_batch_split$path

  # Load model from DOI (imports transformers/torch — CUDA_VISIBLE_DEVICES
  # must already be set before this call)
  phenovision <- load_phenovision(model_doi)

  # Validate model output dimensions
  if (trait == "leaves") {
    if (phenovision$num_labels != 3) {
      stop("Model loaded from doi does not have expected output size! ",
           "Expected 3 labels for leaves, got ", phenovision$num_labels)
    }
  }
  if (trait == "flower/fruit") {
    if (phenovision$num_labels != 2) {
      stop("Model loaded from doi does not have expected output size! ",
           "Expected 2 labels for flower/fruit, got ", phenovision$num_labels)
    }
  }

  # Filter to undone images if provided
  if (!is.null(done)) {
    inf_images <- inf_images[which(inf_images %chin% done)]
  }

  # Convert to Python
  inf_img <- reticulate::r_to_py(inf_images)

  # Create dummy trait data for dataset
  if (trait == "flower/fruit") {
    inf_trait <- tibble::tibble(
      flowers = rep(0.5, length(inf_images)),
      fruits = rep(0.5, length(inf_images))
    ) |>
      as.matrix()
    n_lab <- 2L
  } else {
    inf_trait <- tibble::tibble(
      leaves_green = rep(0.5, length(inf_images)),
      leaves_colored = rep(0.5, length(inf_images)),
      leaves_breaking_buds = rep(0.5, length(inf_images))
    ) |>
      as.matrix()
    n_lab <- 3L
  }

  # Import required Python modules
  models_vit <- reticulate::import_from_path("models_vit", "PlantCLEF2022")
  pos_embed <- reticulate::import_from_path("pos_embed", "PlantCLEF2022/util")
  torch <- reticulate::import("torch")
  timm <- reticulate::import("timm")
  ds <- reticulate::import_from_path("custom_dataset", "py")
  eval <- reticulate::import_from_path("evaluate", "py")

  # Create ViT model architecture for config
  vit2 <- models_vit$vit_large_patch16(num_classes = n_lab)

  # Get data transforms
  config <- timm$data$resolve_data_config(model = vit2)
  transform <- timm$data$create_transform(!!!config)

  # Move model to GPU (always cuda:0 since CUDA_VISIBLE_DEVICES limits visibility)
  phenovision <- phenovision$cuda()

  # Create dataset and dataloader
  inf_ds <- ds$PhenoDatasetInf(
    inf_img,
    inf_trait,
    transform = transform,
    inference_mode = TRUE
  )

  inf_dl <- timm$data$create_loader(
    inf_ds,
    c(3L, 224L, 224L),
    batch_size,
    num_workers = num_workers,
    is_training = FALSE
  )

  # Run inference
  timing <- system.time({
    inf_dat <- eval$infer_hfhub(inf_dl, phenovision, "cuda:0")
  })

  # Extract logits and predictions
  inf_logits <- torch$cat(inf_dat[[1]])
  inf_preds <- torch$nn$functional$sigmoid(inf_logits)
  inf_truth <- torch$cat(inf_dat[[2]])

  # Identify corrupt images (marked with -999999999)
  noskip <- inf_truth[, 0] != -999999999

  # Convert to R matrices/vectors
  inf_preds <- as.matrix(inf_preds$numpy())
  noskip <- as.vector(noskip$numpy())

  # Combine with input data
  if (trait == "flower/fruit") {
    inf_res <- images_batch_split |>
      dplyr::bind_cols(
        as.data.frame(inf_preds) |>
          dplyr::rename(.pred_fruit = V1, .pred_flower = V2)
      ) |>
      dplyr::bind_cols(
        as.data.frame(inf_logits$cpu()$numpy()) |>
          dplyr::rename(.logit_fruit = V1, .logit_flower = V2)
      ) |>
      dplyr::mutate(
        model_version = model_doi,
        corrupt = !noskip
      )
  } else {
    inf_res <- images_batch_split |>
      dplyr::bind_cols(
        as.data.frame(inf_preds) |>
          dplyr::rename(
            .pred_leaves_green = V1,
            .pred_leaves_colored = V2,
            .pred_leaves_breaking_buds = V3  # TYPO FIXED HERE
          )
      ) |>
      dplyr::bind_cols(
        as.data.frame(inf_logits$cpu()$numpy()) |>
          dplyr::rename(
            .logit_leaves_green = V1,
            .logit_leaves_colored = V2,
            .logit_leaves_breaking_buds = V3
          )
      ) |>
      dplyr::mutate(
        model_version = model_doi,
        corrupt = !noskip
      )
  }

  # Attach timing as attribute
  attr(inf_res, "timing") <- timing

  # Free GPU memory so persistent crew workers don't accumulate CUDA allocations.
  # Without this, PyTorch's memory cache keeps the model + tensors on GPU,
  # which would cause OOM on the next branch or confuse GPU auto-selection.
  rm(phenovision, inf_logits, inf_preds, inf_truth, inf_dl, inf_ds, inf_dat)
  gc()
  torch$cuda$empty_cache()

  inf_res
}
