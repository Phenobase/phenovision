require(reticulate)
require(tidyverse)
require(data.table)

use_condaenv("rstudio-gpu")

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

annotate_batch <- function(inf_images, model_doi = "10.57967/hf/2763", done = NULL) {
    
    phenovision <- load_phenovision(model_doi)
    
    if(!is.null(done)) {
        inf_images <- inf_images[which(inf_images %chin% done)]
    }
    inf_img <- r_to_py(inf_images)
    inf_fruit_flower <- matrix(0, nrow = length(inf_images), ncol = 2L)
    inf_fruit_flower[ , 1] <- seq_len(length(inf_images))

    models_vit <- import_from_path("models_vit", "PlantCLEF2022")
    pos_embed <- import_from_path("pos_embed", "PlantCLEF2022/util")

    vit2 <- models_vit$vit_large_patch16(num_classes = 2L)

    config <- timm$data$resolve_data_config(model = vit2)
    transform <- timm$data$create_transform(!!!config)

    batch_size <- 2560L
    phenovision <- phenovision$cuda()

    inf_ds <- ds$PhenoDataset(inf_img, inf_fruit_flower, transform = transform)
    inf_dl <- timm$data$create_loader(inf_ds, c(3L, 224L, 224L), batch_size, num_workers = 8L)

    timing <- system.time({
    inf_dat <- eval$infer_hfhub(inf_dl, phenovision, "cuda:0")
    })

    inf_logits <- torch$cat(inf_dat[[1]])
    inf_indexes <- torch$cat(inf_dat[[2]])$cpu()$numpy()
    inf_preds <- torch$nn$functional$sigmoid(inf_logits)

    corrupt <- inf_fruit_flower[ , 1] != inf_indexes[ , 1]
    #sum(corrupt)
    #corrupt_images <- basename(inf_images)[corrupt]

    inf_res <- tibble(file_name = basename(inf_images)) |>
        bind_cols(as.data.frame(inf_preds$cpu()$numpy()) |>
                  rename(.pred_fruit = V1, .pred_flower = V2)) |>
        bind_cols(as.data.frame(inf_logits$cpu()$numpy()) |>
                  rename(.logit_fruit = V1, .logit_flower = V2)) |>
        mutate(corrupt = corrupt,
               model_version = model_doi)

    attr(inf_res, "timing") <- timing
    
    inf_res

}