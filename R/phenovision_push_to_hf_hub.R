## A lot of this code based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/convert_vit_mae_to_pytorch.py

library(reticulate)
library(tidyverse)

torch <- import("torch")
timm <- import("timm")
transformers <- import("transformers")
convert <- import_from_path("convert", "py")

ds <- import_from_path("custom_dataset", "py")
lrd <- import_from_path("lr_decay", "PlantCLEF2022/util")
misc <- import_from_path("misc", "PlantCLEF2022/util")
eval <- import_from_path("evaluate", "py")

rocksdb <- import("rocksdb")
rdb_options = rocksdb$Options(create_if_missing = TRUE)
db <- "/blue/guralnick/share/phenobase_inat_data/images_rocksdb/test_batch.db/"
image_db <- rocksdb$DB(db, rdb_options)

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

#### test to make sure model output is the same for a sample of images
keys <- image_db$iterkeys()
keys$seek_to_first()

inf_images <- coro::collect(keys, 10)
inf_fruit_flower <- matrix(0, nrow = length(inf_images), ncol = 2L)

data_config <- timm$data$resolve_data_config(model = vit)

inf_ds <- ds$PhenoDatasetRocksDB(inf_images, inf_fruit_flower, image_db, transform = transform)
inf_dl <- timm$data$create_loader(inf_ds, c(3L, 224L, 224L), 10L, num_workers = 6L)

ims <- coro::collect(inf_dl)

vit_tr <- vit_tr$cuda()

test_orig <- vit(ims[[1]][[1]])
test_new <- vit_tr(ims[[1]][[1]])

all.equal(as.matrix(test_orig$cpu()$detach()$numpy()), as.matrix(test_new$logits$cpu()$detach()$numpy()))
## Mean relative difference: 2.388019e-05; This seems close enough to me

vit_tr$push_to_hub("phenobase/phenovision")



