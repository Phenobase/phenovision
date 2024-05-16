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
bad_images <- read_rds("data/inat/inference/bad_images_all.rds")

inf_images <- read_rds("output/all_inference_images.rds")

inf_images <- setdiff(inf_images, bad_images)
length(inf_images)

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
inf_dl <- timm$data$create_loader(inf_ds, c(3L, 224L, 224L), batch_size, num_workers = 8L)

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

write_csv(inf_res, "output/model_04_13_2024/inference_results.csv")

inf_res <- inf_res |>
  mutate(.class_flower = make_two_class_pred(.pred_flower, c("Detected", "Not Detected"),
                                             threshold = 0.84,
                                             buffer = 0.025),
         .class_fruit = make_two_class_pred(.pred_fruit, c("Detected", "Not Detected"),
                                            threshold = 0.175,
                                            buffer = 0.025),
         .equivocal_flower = ifelse(is_equivocal(.class_flower), "Equivocal", "Unequivocal"),
         .equivocal_fruit = ifelse(is_equivocal(.class_fruit), "Equivocal", "Unequivocal")) |>
  mutate(.class_flower = make_two_class_pred(.pred_flower, c("Detected", "Not Detected"),
                                             threshold = 0.84),
         .class_fruit = make_two_class_pred(.pred_fruit, c("Detected", "Not Detected"),
                                            threshold = 0.175))

write_csv(inf_res, "output/model_04_13_2024/inference_results.csv")

## flower threshold = 0.84
## fruit threshold = 0.175


val_logits <- torch$cat(val_dat[[1]])
val_preds <- torch$nn$functional$sigmoid(val_logits)
val_truth <- torch$cat(val_dat[[2]])
val_loss <- criterion(val_preds, val_truth)
val_df <- as.data.frame(val_truth$cpu()$numpy()) |>
  mutate(fruit = as.integer(V1), flower = as.integer(V2)) |>
  bind_cols(as.data.frame(val_preds$cpu()$numpy()) |>
              rename(.pred_fruit = V1, .pred_flower = V2)) |>
  bind_cols(as.data.frame(val_loss$cpu()$numpy()) |>
              rename(.loss_fruit = V1, .loss_flower = V2)) |>
  select(-V1, -V2) |>
  mutate(fruit = factor(fruit, levels = c("1", "0")),
         flower = factor(flower, levels = c("1", "0")))

threshold_data_fr <- val_df %>%
  threshold_perf(fruit, .pred_fruit, thresholds = seq(0, 1, by = 0.02),
                 event_level = "first")

max_ji_fr <- threshold_data_fr |>
  filter(.metric == "j_index") |>
  slice_max(.estimate, n = 5)

threshold_data_fl <- val_df %>%
  threshold_perf(flower, .pred_flower, thresholds = seq(0, 1, by = 0.02),
                 event_level = "first")
  max_ji_fr <- threshold_data_fr |>
  filter(.metric == "j_index") |>
  slice_max(.estimate, n = 5)
  max_ji_fl <- threshold_data_fl |>
  filter(.metric == "j_index") |>
  slice_max(.estimate, n = 5)

val_df <- val_df |>
  mutate(
    .pred_fl_max = make_two_class_pred(
      estimate = .pred_flower,
      levels = levels(flower),
      threshold = max_ji_fl$.threshold[1],
      buffer = 0.025
    ),
    .pred_fr_max = make_two_class_pred(
      estimate = .pred_fruit,
      levels = levels(fruit),
      threshold = max_ji_fr$.threshold[1],
      buffer = 0.025
    )
  )

val_acc_fl <- accuracy(val_df, flower, .pred_fl_max)
val_acc_fr <- accuracy(val_df, fruit, .pred_fr_max)

val_jind_fl <- j_index(val_df, flower, .pred_fl_max)
val_jind_fr <- j_index(val_df, fruit, .pred_fr_max)


test_dat = eval$evaluate(test_dl, vit, "cuda:0")
test_logits <- torch$cat(test_dat[[1]])
test_preds <- torch$nn$functional$sigmoid(test_logits)
test_truth <- torch$cat(test_dat[[2]])
test_loss <- criterion(test_preds, test_truth)
test_df <- as.data.frame(test_truth$cpu()$numpy()) |>
  mutate(fruit = as.integer(V1), flower = as.integer(V2)) |>
  bind_cols(as.data.frame(test_preds$cpu()$numpy()) |>
              rename(.pred_fruit = V1, .pred_flower = V2)) |>
  bind_cols(as.data.frame(test_loss$cpu()$numpy()) |>
              rename(.loss_fruit = V1, .loss_flower = V2)) |>
  select(-V1, -V2) |>
  mutate(fruit = factor(fruit, levels = c("1", "0")),
         flower = factor(flower, levels = c("1", "0")))

test_df <- test_df |>
  mutate(
    .pred_fl_max = make_two_class_pred(
      estimate = .pred_flower,
      levels = levels(flower),
      threshold = max_ji_fl$.threshold[1],
      buffer = 0.025
    ),
    .pred_fr_max = make_two_class_pred(
      estimate = .pred_fruit,
      levels = levels(fruit),
      threshold = max_ji_fr$.threshold[1],
      buffer = 0.025
    )
  )

test_acc_fl <- accuracy(test_df, flower, .pred_fl_max)
test_acc_fr <- accuracy(test_df, fruit, .pred_fr_max)

test_jind_fl <- j_index(test_df, flower, .pred_fl_max)
test_jind_fr <- j_index(test_df, fruit, .pred_fr_max)


############# training data ##################

train_dat = eval$evaluate(train_dl, vit, "cuda:0")
train_logits <- torch$cat(train_dat[[1]])
train_preds <- torch$nn$functional$sigmoid(train_logits)
train_truth <- torch$cat(train_dat[[2]])

train_logits$requires_grad <- FALSE
train_preds$requires_grad <- FALSE
train_truth$requires_grad <- FALSE

train_loss <- criterion(train_preds, train_truth)
train_df <- as.data.frame(train_truth$cpu()$numpy()) |>
  mutate(fruit = as.integer(V1), flower = as.integer(V2)) |>
  bind_cols(as.data.frame(train_preds$cpu()$numpy()) |>
              rename(.pred_fruit = V1, .pred_flower = V2)) |>
  bind_cols(as.data.frame(train_loss$cpu()$numpy()) |>
              rename(.loss_fruit = V1, .loss_flower = V2)) |>
  select(-V1, -V2) |>
  mutate(fruit = factor(fruit, levels = c("1", "0")),
         flower = factor(flower, levels = c("1", "0")))

train_df <- train_df |>
  mutate(
    .pred_fl_max = make_two_class_pred(
      estimate = .pred_flower,
      levels = levels(flower),
      threshold = max_ji_fl$.threshold[1],
      buffer = 0.025
    ),
    .pred_fr_max = make_two_class_pred(
      estimate = .pred_fruit,
      levels = levels(fruit),
      threshold = max_ji_fr$.threshold[1],
      buffer = 0.025
    )
  )

train_acc_fl <- accuracy(train_df, flower, .pred_fl_max)
train_acc_fr <- accuracy(train_df, fruit, .pred_fr_max)

train_jind_fl <- j_index(train_df, flower, .pred_fl_max)
train_jind_fr <- j_index(train_df, fruit, .pred_fr_max)


train_df <- train_df |>
  bind_cols(as.data.frame(train_logits$cpu()$numpy()) |>
              rename(.logit_fruit = V1, .logit_flower = V2))

val_df <- val_df |>
  bind_cols(as.data.frame(val_logits$cpu()$numpy()) |>
              rename(.logit_fruit = V1, .logit_flower = V2))

test_df <- test_df |>
  bind_cols(as.data.frame(test_logits$cpu()$numpy()) |>
              rename(.logit_fruit = V1, .logit_flower = V2))

train_df <- inat_train |>
  bind_cols(train_df)

val_df <- inat_val |>
  bind_cols(val_df)

test_df <- inat_test |>
  bind_cols(test_df)


testing_df <- val_df |>
  mutate(partition = "validation") |>
  bind_rows(test_df |>
              mutate(partition = "testing")) |>
  bind_rows(train_df |>
              mutate(partition = "training"))

write_rds(testing_df, "output/model_04_13_2024/epoch_4_testing_data.rds")
