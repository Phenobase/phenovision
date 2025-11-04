require(reticulate)
require(tidyverse)
require(data.table)

use_condaenv("reticulate-gpu2")

torch <- import("torch")
timm <- import("timm")
PIL <- import("PIL")
ds <- import_from_path("custom_dataset", "py")
#del <- import_from_path("remove_bad_images", "py")
engine <- import_from_path("engine_finetune", "PlantCLEF2022")
lrd <- import_from_path("lr_decay", "PlantCLEF2022/util")
misc <- import_from_path("misc", "PlantCLEF2022/util")
eval <- import_from_path("evaluate", "py")
NativeScaler <- misc$NativeScalerWithGradNormCount
types <- import("types")

#source("R/model_loading.R")

annotate_batch <- function(images_batch_split, model_doi = "10.57967/hf/2763",
                           trait = c("flower/fruit", "leaves"),
                           num_workers = 6L, done = NULL) {

    trait <- match.arg(trait)
    #browser()

    #images_batch_split <- images_batch_split[[1]]

    inf_images <- images_batch_split$path

    ## see if there are any completed annotations already

    #complete_folder <- file.path(inat_folder, model_doi)
    # if(dir.exists(complete_folder)) {
    #   files <- list.files(complete_folder, full.names = TRUE)
    #   done <- purrr::map(files, read_csv) |>
    #     list_rbind() |>
    #     pull(photo_id)
    # }

    phenovision <- load_phenovision(model_doi)

    if(trait == "leaves") {
      if(phenovision$num_labels != 3) {
        stop("Model loaded from doi does not have expected output size!")
      }
    }
    if(trait == "flower/fruit") {
      if(phenovision$num_labels != 2) {
        stop("Model loaded from doi does not have expected output size!")
      }
    }

    if(!is.null(done)) {
        inf_images <- inf_images[which(inf_images %chin% done)]
    }
    inf_img <- r_to_py(inf_images)
    if(trait == "flower/fruit") {
      inf_trait <- tibble(flowers = rep(0.5, length(inf_images)),
                          fruits = rep(0.5, length(inf_images))) |>
        as.matrix()
      n_lab <- 2L
    } else {
      inf_trait <- tibble(leaves_green = rep(0.5, length(inf_images)),
                          leaves_colored = rep(0.5, length(inf_images)),
                          leaves_breaking_buds = rep(0.5, length(inf_images))) |>
        as.matrix()
      n_lab <- 3L
    }

    models_vit <- import_from_path("models_vit", "PlantCLEF2022")
    pos_embed <- import_from_path("pos_embed", "PlantCLEF2022/util")

    vit2 <- models_vit$vit_large_patch16(num_classes = n_lab)

    config <- timm$data$resolve_data_config(model = vit2)
    transform <- timm$data$create_transform(!!!config)

    batch_size <- batch_size_inference
    phenovision <- phenovision$cuda()

    inf_ds <- ds$PhenoDatasetInf(inf_img, inf_trait, transform = transform, inference_mode = TRUE)
    inf_dl <- timm$data$create_loader(inf_ds, c(3L, 224L, 224L), batch_size,
                                      num_workers = num_workers,
                                      is_training = FALSE)

    #inf_ds <- ds$PhenoDataset(inf_img, inf_fruit_flower, transform = transform)
    #inf_dl <- timm$data$create_loader(inf_ds, c(3L, 224L, 224L), batch_size, num_workers = 8L)

    timing <- system.time({
      inf_dat <- eval$infer_hfhub(inf_dl, phenovision, "cuda:0")
    })

    #browser()

    inf_logits <- torch$cat(inf_dat[[1]])
    inf_preds <- torch$nn$functional$sigmoid(inf_logits)
    inf_truth <- torch$cat(inf_dat[[2]])

    noskip <- inf_truth[ , 0] != -999999999

    inf_preds <- as.matrix(inf_preds$numpy())
    noskip <- as.vector(noskip$numpy())

    if(trait == "flower/fruit") {
      inf_res <- images_batch_split |>
          bind_cols(as.data.frame(inf_preds) |>
                    rename(.pred_fruit = V1, .pred_flower = V2)) |>
          bind_cols(as.data.frame(inf_logits$cpu()$numpy()) |>
                    rename(.logit_fruit = V1, .logit_flower = V2)) |>
          mutate(model_version = model_doi,
                 corrupt = !noskip)
    } else {
      inf_res <- images_batch_split |>
          bind_cols(as.data.frame(inf_preds) |>
                    rename(.pred_leaves_green = V1,
                           .pred_leaves_colored = V2,
                           .pred_leaves_breaking_nuds = V3)) |> #todo: must fix this silly typo, 'breaking_nuds'
          bind_cols(as.data.frame(inf_logits$cpu()$numpy()) |>
                    rename(.logit_leaves_green = V1,
                           .logit_leaves_colored = V2,
                           .logit_leaves_breaking_buds = V3)) |>
          mutate(model_version = model_doi,
                 corrupt = !noskip)
    }

    attr(inf_res, "timing") <- timing

    inf_res

}


