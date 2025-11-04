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

genus_props <- rob_annot |>
  group_by(genus) |>
  summarise(count = n()) |>
  ungroup() |>
  mutate(prop = ((count / sum(count)) + 1/150) / 2) |>
  filter(genus != "Logfia")

## sample images for validation
meta <- meta |>
  select(file_name, genus) |>
  filter(genus != "Logfia") |>
  slice_sample(n = ceiling(1e6/150), by = genus) |>
  left_join(genus_props) |>
  slice_sample(n = 50000, weight_by = prop)

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

batch_size <- 1280L

#c(train_ds, train_ind) %<-% ds$prepare_inference_dataset(train_img, train_leaves, transform = transform)
inf_ds <- ds$PhenoDatasetInf(inf_img, inf_leaves, transform = transform, inference_mode = TRUE)
inf_dl <- timm$data$create_loader(inf_ds, c(3L, 224L, 224L), batch_size, num_workers = 7L,
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

inf_gr <- inf_df |>
  group_by(.equivocal_gr) |>
  summarise(count = n(), file_list = list(file_name)) |>
  ungroup() |>
  mutate(prop = count / sum(count))

inf_cl <- inf_df |>
  group_by(.equivocal_cl) |>
  summarise(count = n(), file_list = list(file_name)) |>
  ungroup() |>
  mutate(prop = count / sum(count))

inf_bb <- inf_df |>
  group_by(.equivocal_bb) |>
  summarise(count = n(), file_list = list(file_name)) |>
  ungroup() |>
  mutate(prop = count / sum(count))

inf_df <- inf_df |>
  mutate(.class_no = ifelse(.class_gr == "Not Detected" & .class_cl == "Not Detected" & .class_bb == "Not Detected",
                            "Detected", "Not Detected"),
         .equivocal_no = ifelse(.equivocal_gr == "Unequivocal" & .equivocal_cl == "Unequivocal" & .equivocal_bb == "Unequivocal",
                                "Unequivocal", "Equivocal"))

inf_no <- inf_df |>
  group_by(.equivocal_no) |>
  summarise(count = n(), file_list = list(file_name)) |>
  ungroup() |>
  mutate(prop = count / sum(count))

inf_gr_files <- inf_gr |>
  filter(.equivocal_gr == "Equivocal") |>
  rowwise() |>
  mutate(file_samp = list(sample(file_list, min(250, length(file_list)))),
         type = "gr")

inf_cl_files <- inf_cl |>
  filter(.equivocal_cl == "Equivocal") |>
  rowwise() |>
  mutate(file_samp = list(sample(file_list, min(250, length(file_list)))),
         type = "cl")

inf_bb_files <- inf_bb |>
  filter(.equivocal_bb == "Equivocal") |>
  rowwise() |>
  mutate(file_samp = list(sample(file_list, min(250, length(file_list)))),
         type = "bb")

inf_files <- bind_rows(inf_gr_files, inf_cl_files, inf_bb_files)

pwalk(list(inf_files$file_samp, inf_files$type),
      ~ file.copy(..1, file.path("output/leaves/phenovision-init_model2_04_11_2025/image_annotation_test_equiv_only2", ..2, basename(..1)),
                  copy.mode = FALSE, copy.date = TRUE))


