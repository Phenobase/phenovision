library(reticulate)
library(tidyverse)
library(tidymodels)
library(probably)
library(zeallot)
library(arrow)
library(data.table)

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

model_file <- file.path("output/leaves/phenovision-init_model2_04_11_2025/", "checkpoints", paste0("vit_finetuned_round2_epoch", 1, ".pt"))

meta <- read_csv("data/leaves/inference_metadata_03-06-2025.csv")

rob_annot <- read_csv("data/leaves/phenobase_dwca_annotation/rob_leaf_breaking_buds_annotation.csv")
rob_annot <- rob_annot |>
  mutate(genus = word(taxon.name))

meta <- meta |>
  select(file_name, genus) |>
  filter(genus != "Logfia")

inf_img <- r_to_py(meta$file_name)
inf_leaves <- tibble(leaves_green = rep(0.5, nrow(meta)),
                       leaves_colored = rep(0.5, nrow(meta)),
                       leaves_breaking_buds = rep(0.5, nrow(meta))) |>
  as.matrix()

models_vit <- import_from_path("models_vit", "PlantCLEF2022")
pos_embed <- import_from_path("pos_embed", "PlantCLEF2022/util")

vit <- torch$load(model_file)

vit2 <- models_vit$vit_large_patch16(num_classes = 3L)

config <- timm$data$resolve_data_config(model = vit2)
transform <- timm$data$create_transform(!!!config)

batch_size <- 2560L

#c(train_ds, train_ind) %<-% ds$prepare_inference_dataset(train_img, train_leaves, transform = transform)
inf_ds <- ds$PhenoDatasetInf(inf_img, inf_leaves, transform = transform, inference_mode = TRUE)
inf_dl <- timm$data$create_loader(inf_ds, c(3L, 224L, 224L), batch_size, num_workers = 6L,
                                  is_training = FALSE)

inf_dat <- eval$evaluate(inf_dl, vit, "cuda:0")
inf_logits <- torch$cat(inf_dat[[1]])
inf_preds <- torch$nn$functional$sigmoid(inf_logits)
inf_truth <- torch$cat(inf_dat[[2]])

noskip <- inf_truth[ , 0] != -999999999

inf_preds <- as.matrix(inf_preds[noskip, ]$numpy())
noskip <- as.vector(noskip$numpy())

leaf_buffers <- read_rds("output/leaves/phenovision-init_model2_04_11_2025/epoch_1_threshold_buffers.csv")

inf_df <- as.data.frame(inf_preds) |>
  rename(.pred_leaves_green = V1, .pred_leaves_colored = V2,
         .pred_leaves_breaking_buds = V3) |>
  bind_cols(meta[noskip, ])

inf_df <- inf_df |>
  mutate(
    .class_gr = make_two_class_pred(
      estimate = .pred_leaves_green,
      levels = c("Detected", "Not Detected"),
      threshold = leaf_buffers$gr[1],
      buffer = leaf_buffers$gr[-1]
    ),
    .class_cl = make_two_class_pred(
      estimate = .pred_leaves_colored,
      levels = c("Detected", "Not Detected"),
      threshold = leaf_buffers$cl[1],
      buffer = leaf_buffers$cl[-1]
    ),
    .class_bb = make_two_class_pred(
      estimate = .pred_leaves_breaking_buds,
      levels = c("Detected", "Not Detected"),
      threshold = leaf_buffers$bb[1],
      buffer = leaf_buffers$bb[-1]
    ),
    .equivocal_gr = ifelse(is_equivocal(.class_gr), "Equivocal", "Unequivocal"),
    .equivocal_cl = ifelse(is_equivocal(.class_cl), "Equivocal", "Unequivocal"),
    .equivocal_bb = ifelse(is_equivocal(.class_bb), "Equivocal", "Unequivocal")
  ) |>
  mutate(
    .class_gr = make_two_class_pred(
      estimate = .pred_leaves_green,
      levels = c("Detected", "Not Detected"),
      threshold = leaf_buffers$gr[1]
    ),
    .class_cl = make_two_class_pred(
      estimate = .pred_leaves_colored,
      levels = c("Detected", "Not Detected"),
      threshold = leaf_buffers$cl[1]
    ),
    .class_bb = make_two_class_pred(
      estimate = .pred_leaves_breaking_buds,
      levels = c("Detected", "Not Detected"),
      threshold = leaf_buffers$bb[1]
    ))


inf_df <- inf_df |>
  mutate(.class_no = ifelse(.class_gr == "Not Detected" & .class_cl == "Not Detected" & .class_bb == "Not Detected",
                            "Detected", "Not Detected"),
         .equivocal_no = ifelse(.equivocal_gr == "Unequivocal" & .equivocal_cl == "Unequivocal" & .equivocal_bb == "Unequivocal",
                                "Unequivocal", "Equivocal"))

write_rds(inf_df, "output/leaves/phenovision-init_model2_04_11_2025/all_inference_results_round2_epoch_1.rds")

###### add metadata back in ###########

meta <- read_csv("data/leaves/inference_metadata_03-06-2025.csv")
inf_df <- read_rds("output/leaves/phenovision-init_model2_04_11_2025/all_inference_results_round2_epoch_1.rds")
meta_taxa <- open_dataset("data/phenobase_inat_data/metadata/taxa")

inf_df <- inf_df |>
  left_join(meta)

families <- meta_taxa |>
  filter(rank == "family") |>
  collect()

genera <- meta_taxa |>
  filter(rank == "genus") |>
  collect()

taxonomy <- inf_df |>
  select(photo_id, ancestry) |>
  mutate(taxa_ids = str_split(ancestry, "/")) |>
  select(-ancestry) |>
  unnest_longer(taxa_ids, transform = as.integer)

fams <- taxonomy |>
  left_join(families |> select(taxon_id, family = name), by = c(taxa_ids = "taxon_id")) |>
  drop_na()

# gens <- taxonomy |>
#   left_join(genera |> select(taxon_id, genus = name), by = c(taxa_ids = "taxon_id")) |>
#   drop_na()

inf_df <- inf_df |>
  left_join(fams |> select(photo_id, family)) #|>
#  left_join(gens |> select(photo_id, genus))

write_rds(inf_df, "output/leaves/phenovision-init_model2_04_11_2025/all_inference_results_round2_epoch_1.rds")
