library(ape)
library(tidyverse)
library(arrow)
library(reticulate)

tree <- read.nexus("data/phylogeny/oo_330891.tre")
bad_images <- read_rds("data/inat/bad_images.rds")

inat_train <- read_csv("data/inat/train.csv") |>
  filter(!file_name %in% bad_images)

inat_val <- read_csv("data/inat/validation.csv") |>
  filter(!file_name %in% bad_images)

inat_test <- read_csv("data/inat/test.csv") |>
  filter(!file_name %in% bad_images)

inat_all <- bind_rows(inat_train, inat_val, inat_test)

angio_meta <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/angio_photos")
taxa_meta <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/taxa/part-0.parquet")

inat_all <- inat_all |>
  left_join(angio_meta |>
              select(photo_id, taxon_id) |>
              filter(photo_id %in% inat_all$photo_id),
            copy = TRUE)

inat_all <- inat_all |>
  left_join(taxa_meta |>
              select(taxon_id, name) |>
              filter(taxon_id %in% inat_all$taxon_id),
            copy = TRUE)

specs <- gsub(" ", "_", unique(inat_all$name))
sum(specs %in% tree$tip.label) / length(specs)

genera_in_tree <- unique(str_split_i(tree$tip.label, "_", 1))

prop_genera <- sum(unique(inat_all$genus) %in% genera_in_tree) / length(unique(inat_all$genus))

inat_all <- inat_all |>
  filter(genus %in% genera_in_tree)

counts <- inat_all |> group_by(name) |> summarise(count = n()) |> pull(count)

hist(log(counts), breaks = 40)
sum(counts > 10)


####### get latent codes ############
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

model_file <- "output/model_04_13_2024/checkpoints/vit_finetuned_epoch4.pt"

inf_img <- r_to_py(inat_all$file_name)
inf_fruit_flower <- matrix(0, nrow = length(inf_img), ncol = 2L)

models_vit <- import_from_path("models_vit", "PlantCLEF2022")
pos_embed <- import_from_path("pos_embed", "PlantCLEF2022/util")

vit <- torch$load(model_file)

vit2 <- models_vit$vit_large_patch16(num_classes = 2L)

config <- timm$data$resolve_data_config(model = vit2)
transform <- timm$data$create_transform(!!!config)

batch_size <- 2056L

vit <- vit$cuda()

inf_ds <- ds$PhenoDataset(inf_img, inf_fruit_flower, transform = transform)
inf_dl <- timm$data$create_loader(inf_ds, c(3L, 224L, 224L), batch_size, num_workers = 5L)

codes <- eval$get_codes(inf_dl, vit, "cuda:0")

codes_tens <- torch$cat(codes)

code_df <- as.data.frame(codes_tens$cpu()$numpy())
colnames(code_df) <- paste0("MAE_", str_pad(1:1024, 4, pad = "0"))

code_df <- inat_all |>
  bind_cols(code_df)

write_csv(code_df, "output/model_04_13_2024/mae_codes_w_meta.csv")
write_rds(code_df, "output/model_04_13_2024/mae_codes_w_meta.rds", compress = "gz")
