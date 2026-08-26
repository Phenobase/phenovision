library(reticulate)
library(tidyverse)
library(tidymodels)
library(probably)
library(phyf)

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

inf_images <- list.files("/blue/guralnick/share/pollinators/Pollinator_Photos/images",
                         full.names = TRUE)

inf_img <- r_to_py(inf_images)
inf_fruit_flower <- matrix(0, nrow = length(inf_images), ncol = 2L)

models_vit <- import_from_path("models_vit", "PlantCLEF2022")
pos_embed <- import_from_path("pos_embed", "PlantCLEF2022/util")

vit <- torch$load(model_file)

vit2 <- models_vit$vit_large_patch16(num_classes = 2L)

config <- timm$data$resolve_data_config(model = vit2)
transform <- timm$data$create_transform(!!!config)

batch_size <- 2560L

#test_ds <- ds$PhenoDataset(inf_img[1:(batch_size*10)], inf_fruit_flower[1:(batch_size*10), ], transform = transform)
#test_dl <- timm$data$create_loader(test_ds, c(3L, 224L, 224L), batch_size)

vit <- vit$cuda()

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
inf_dl <- timm$data$create_loader(inf_ds, c(3L, 224L, 224L), batch_size)

system.time({
inf_dat = eval$infer(inf_dl, vit, "cuda:0")
})

inf_logits <- torch$cat(inf_dat[[1]])
inf_preds <- torch$nn$functional$sigmoid(inf_logits)

inf_res <- tibble(file_name = basename(inf_images)) |>
  bind_cols(as.data.frame(inf_preds$cpu()$numpy()) |>
              rename(.pred_fruit = V1, .pred_flower = V2)) |>
  bind_cols(as.data.frame(inf_logits$cpu()$numpy()) |>
              rename(.logit_fruit = V1, .logit_flower = V2))

write_csv(inf_res, "side/pollinators/inference_results.csv")

inf_res <- inf_res |>
  mutate(.class_flower = make_two_class_pred(.pred_flower, c("Detected", "Not Detected"),
                                             threshold = 0.84,
                                             buffer = c(0.56, 0.01)),
         .class_fruit = make_two_class_pred(.pred_fruit, c("Detected", "Not Detected"),
                                            threshold = 0.53,
                                            buffer = c(0.3, 0.22)),
         .equivocal_flower = ifelse(is_equivocal(.class_flower), "Equivocal", "Unequivocal"),
         .equivocal_fruit = ifelse(is_equivocal(.class_fruit), "Equivocal", "Unequivocal")) |>
  mutate(.class_flower = make_two_class_pred(.pred_flower, c("Detected", "Not Detected"),
                                             threshold = 0.84),
         .class_fruit = make_two_class_pred(.pred_fruit, c("Detected", "Not Detected"),
                                            threshold = 0.53))

write_csv(inf_res, "side/pollinators/inference_results.csv")

metadat <- read_csv("/blue/guralnick/share/pollinators/pollinator_image_data.csv")

inf_res <- inf_res |>
  mutate(id = fs::path_ext_remove(file_name)) |>
  left_join(metadat |>
              mutate(id = as.character(id)))

inf_summ <- inf_res |>
  group_by(.class_flower, `Flower Visible?`, .equivocal_flower) |>
  summarise(count = n()) |>
  group_by(`Flower Visible?`) |>
  mutate(prop = count / sum(count))

inf_res_all <- inf_res |>
  filter(`Flower Visible?` != "I don't know") |>
  mutate(flower_vis = ifelse(`Flower Visible?` == "Yes", "Detected", "Not Detected") |>
           factor(levels = c("Detected", "Not Detected")))

inf_res_uneq <- inf_res |>
  filter(`Flower Visible?` != "I don't know", .equivocal_flower == "Unequivocal") |>
  mutate(flower_vis = ifelse(`Flower Visible?` == "Yes", "Detected", "Not Detected") |>
           factor(levels = c("Detected", "Not Detected")))

accuracy(inf_res_all,
         flower_vis, .class_flower)

accuracy(inf_res_uneq,
         flower_vis, .class_flower)

sens(inf_res_all,
     flower_vis, .class_flower)

spec(inf_res_all,
     flower_vis, .class_flower)

ppv(inf_res_all,
     flower_vis, .class_flower)

npv(inf_res_all,
    flower_vis, .class_flower)

write_csv(inf_res, "side/pollinators/flower_predictions_results_pollinators.csv")
write_csv(inf_summ, "side/pollinators/flower_predictions_summary_pollinators.csv")

########### pollinator threshold? ###########

poll_thresh <- inf_res_all |>
  threshold_perf(flower_vis, .pred_flower, thresholds = seq(0, 1, by = 0.02),
                 event_level = "first")
