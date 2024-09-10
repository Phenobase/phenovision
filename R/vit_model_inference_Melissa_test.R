library(reticulate)
library(tidyverse)
library(tidymodels)
library(probably)
library(phyf)
library(arrow)
library(fs)

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

source("R/model_loading.R")

current_model <- "10.57967/hf/2763"

phenovision <- load_phenovision(current_model)
vers <- model_version(current_model)

mel_photos <- read_csv("/blue/guralnick/share/melissag/melissa_photoIDs_all.csv")
meta <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/angio_photos")

mel_photos <- mel_photos |>
  left_join(meta |>
    select(photo_uuid, batch_j) |>
    filter(photo_uuid %in% mel_photos$photo_uuid),
  copy = TRUE)

mel_photos <- mel_photos |>
  drop_na(batch_j) |>
  mutate(path = file.path("/blue/guralnick/share/phenobase_inat_data/images/medium",
  paste0("batch_", batch_j),
  paste0(photo_id, ".", extension)))

inf_images <- mel_photos$path

length(inf_images)

inf_img <- r_to_py(inf_images)
inf_fruit_flower <- matrix(0, nrow = length(inf_images), ncol = 2L)
inf_fruit_flower[ , 1] <- seq_len(length(inf_images))

models_vit <- import_from_path("models_vit", "PlantCLEF2022")
pos_embed <- import_from_path("pos_embed", "PlantCLEF2022/util")

vit2 <- models_vit$vit_large_patch16(num_classes = 2L)

config <- timm$data$resolve_data_config(model = vit2)
transform <- timm$data$create_transform(!!!config)

batch_size <- 2560L

#test_ds <- ds$PhenoDataset(inf_img[1:(batch_size*10)], inf_fruit_flower[1:(batch_size*10), ], transform = transform)
#test_dl <- timm$data$create_loader(test_ds, c(3L, 224L, 224L), batch_size)

phenovision <- phenovision$cuda()

# test_dat = eval$infer(test_dl, vit, "cuda:0")
#
# test_logits <- torch$cat(test_dat[[1]])
# test_preds <- torch$nn$functional$sigmoid(test_logits)
#
# test_res <- tibble(file_name = basename(inf_images[1:(batch_size*10)])) |>
#   bind_cols(as.data.frame(test_preds$cpu()$numpy()) |>
#               rename(.pred_fruit = V1, .pred_flower = V2)) |>
#   bind_cols(as.data.frame(test_logits$cpu()$numpy()) |>
#               rename(.logit_fruit = V1, .logit_flower = V2))

inf_ds <- ds$PhenoDataset(inf_img, inf_fruit_flower, transform = transform)
inf_dl <- timm$data$create_loader(inf_ds, c(3L, 224L, 224L), batch_size, num_workers = 8L)

system.time({
inf_dat <- eval$infer_hfhub(inf_dl, phenovision, "cuda:0")
})

## (2056*2009) / (6085/60) = 40,728 images per minute

inf_logits <- torch$cat(inf_dat[[1]])
inf_indexes <- torch$cat(inf_dat[[2]])$cpu()$numpy()
inf_preds <- torch$nn$functional$sigmoid(inf_logits)

corrupt <- inf_fruit_flower[ , 1] != inf_indexes[ , 1]
sum(corrupt)
#corrupt_images <- basename(inf_images)[corrupt]

inf_res <- tibble(file_name = basename(inf_images)) |>
    bind_cols(as.data.frame(inf_preds$cpu()$numpy()) |>
              rename(.pred_fruit = V1, .pred_flower = V2)) |>
    bind_cols(as.data.frame(inf_logits$cpu()$numpy()) |>
              rename(.logit_fruit = V1, .logit_flower = V2)) |>
    mutate(corrupt = corrupt,
           model_version = current_model)

vers_path <- file.path("output", vers)
if(!dir.exists(vers_path)) {
  dir.create(vers_path)
}

csv_path <- file.path(vers_path, "melissa_inference_results.csv")

write_csv(inf_res, csv_path)