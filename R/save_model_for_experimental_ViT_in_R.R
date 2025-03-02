library(reticulate)
library(tidyverse)

torch <- import("torch")
timm <- import("timm")
transformers <- import("transformers")
convert <- import_from_path("convert", "py")

ds <- import_from_path("custom_dataset", "py")
lrd <- import_from_path("lr_decay", "PlantCLEF2022/util")
misc <- import_from_path("misc", "PlantCLEF2022/util")

model_file <- "output/model_04_13_2024/checkpoints/vit_finetuned_epoch4.pt"

models_vit <- import_from_path("models_vit", "PlantCLEF2022")
pos_embed <- import_from_path("pos_embed", "PlantCLEF2022/util")

vit <- torch$load(model_file)

config <- transformers$models$vit$ViTConfig()
config$hidden_size <- 1024L
config$intermediate_size <- 4096L
config$num_hidden_layers <- 24L
config$num_attention_heads <- 16L

vit_tr <- transformers$models$vit$ViTForImageClassification(config)

state_dict <- vit$state_dict()
new_state_dict <- convert$convert_state_dict(state_dict, config)
#names(new_state_dict) <- gsub("vit.", "", names(new_state_dict), fixed = TRUE)

vit_tr_state_dict <- vit_tr$state_dict()

in_there <- names(vit_tr_state_dict) %in% names(new_state_dict)

names(new_state_dict) <- gsub("head.", "classifier.", names(new_state_dict), fixed = TRUE)

#dims <- map(new_state_dict, ~list(.x$shape))
#dims_tr <- map(vit_tr_state_dict, ~.x$size())

vit_tr$load_state_dict(new_state_dict)

vit_tr_R <- map(py_to_r(vit_tr_state_dict),
                ~ .x$cpu()$numpy())
params <- sum(map_int(vit_tr_R, length))

write_rds(list(config = config$to_dict(), weights = vit_tr_R),
          "output/model_04_13_2024/R/vit_finetuned_epoch4.rds")
