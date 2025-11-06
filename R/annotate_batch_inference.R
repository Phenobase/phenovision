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
                           done = NULL) {

  trait <- match.arg(trait)

  inf_images <- images_batch_split$path

  # Load model from DOI
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
      leaves_breaking_buds = rep(0.5, length(inf_images))  # TYPO FIXED HERE
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

  # Move model to GPU
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
    batch_size_inference,  # From _targets_common.R
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

  inf_res
}
