library(reticulate)
library(tidyverse)
library(tidymodels)
library(probably)
library(zeallot)
#library(phyf)

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

model_file <- file.path("output/leaves/phenovision-init_model_02_12_2025", "checkpoints", paste0("vit_finetuned_epoch", 4, ".pt"))

leaves_train <- read_csv("data/leaves/train.csv")
leaves_val <- read_csv("data/leaves/validation.csv")
leaves_test <- read_csv("data/leaves/test.csv")

train_img <- r_to_py(leaves_train$file_name)
train_leaves <- leaves_train |>
  select(leaves_green, leaves_colored, leaves_breaking_buds) |>
  as.matrix()

val_img <- r_to_py(leaves_val$file_name)
val_leaves <- leaves_val |>
  select(leaves_green, leaves_colored, leaves_breaking_buds) |>
  as.matrix()

test_img <- r_to_py(leaves_test$file_name)
test_leaves <- leaves_test |>
  select(leaves_green, leaves_colored, leaves_breaking_buds) |>
  as.matrix()

models_vit <- import_from_path("models_vit", "PlantCLEF2022")
pos_embed <- import_from_path("pos_embed", "PlantCLEF2022/util")

vit <- torch$load(model_file)

vit2 <- models_vit$vit_large_patch16(num_classes = 3L)

config <- timm$data$resolve_data_config(model = vit2)
transform <- timm$data$create_transform(!!!config)

batch_size <- 2560L

#c(train_ds, train_ind) %<-% ds$prepare_inference_dataset(train_img, train_leaves, transform = transform)
train_ds <- ds$PhenoDatasetInf(train_img, train_leaves, transform = transform, inference_mode = TRUE)
train_dl <- timm$data$create_loader(train_ds, c(3L, 224L, 224L), batch_size, num_workers = 7L,
                                    is_training = FALSE)

test_ds <- ds$PhenoDatasetInf(test_img, test_leaves, transform = transform, inference_mode = TRUE)
test_dl <- timm$data$create_loader(test_ds, c(3L, 224L, 224L), batch_size, num_workers = 7L,
                                   is_training = FALSE)

val_ds <- ds$PhenoDatasetInf(val_img, val_leaves, transform = transform, inference_mode = TRUE)
val_dl <- timm$data$create_loader(val_ds, c(3L, 224L, 224L), batch_size, num_workers = 7L,
                                  is_training = FALSE)

vit <- vit$cuda()

loss <- torch$nn$BCEWithLogitsLoss(reduction = 'none')
criterion <- function(outputs, targets) {
  loss(outputs, targets$type(torch$float32))
}

val_dat = eval$evaluate(val_dl, vit, "cuda:0")
val_logits <- torch$cat(val_dat[[1]])
val_preds <- torch$nn$functional$sigmoid(val_logits)
val_truth <- torch$cat(val_dat[[2]])

noskip <- val_truth[ , 0] != -999999999

val_loss <- criterion(val_preds[noskip, ], val_truth[noskip, ])
val_df <- as.data.frame(val_truth[noskip, ]$cpu()$numpy()) |>
  mutate(leaves_green = as.integer(V1), leaves_colored = as.integer(V2),
         leaves_breaking_buds = as.integer(V3)) |>
  bind_cols(as.data.frame(val_preds[noskip, ]$cpu()$numpy()) |>
              rename(.pred_leaves_green = V1, .pred_leaves_colored = V2,
                     .pred_leaves_breaking_buds = V3)) |>
  bind_cols(as.data.frame(val_loss$cpu()$numpy()) |>
              rename(.loss_leaves_green = V1, .loss_leaves_colored = V2,
                     .loss_leaves_breaking_buds = V3)) |>
  select(-V1, -V2, -V3) |>
  mutate(leaves_green = factor(leaves_green, levels = c("1", "0")),
         leaves_colored = factor(leaves_colored, levels = c("1", "0")),
         leaves_breaking_buds = factor(leaves_breaking_buds, levels = c("1", "0")))

threshold_data_gr <- val_df %>%
  threshold_perf(leaves_green, .pred_leaves_green, thresholds = seq(0, 1, by = 0.005),
                 event_level = "first")

max_ji_gr <- threshold_data_gr |>
  filter(.metric == "j_index") |>
  slice_max(.estimate, n = 5)

threshold_data_cl <- val_df %>%
  threshold_perf(leaves_colored, .pred_leaves_colored, thresholds = seq(0, 1, by = 0.02),
                 event_level = "first")

max_ji_cl <- threshold_data_cl |>
  filter(.metric == "j_index") |>
  slice_max(.estimate, n = 5)

threshold_data_bb <- val_df %>%
  threshold_perf(leaves_breaking_buds, .pred_leaves_breaking_buds, thresholds = seq(0, 1, by = 0.02),
                 event_level = "first")

max_ji_bb <- threshold_data_bb |>
  filter(.metric == "j_index") |>
  slice_max(.estimate, n = 5)

val_df <- val_df |>
  mutate(
    .pred_gr_max = make_two_class_pred(
      estimate = .pred_leaves_green,
      levels = levels(leaves_green),
      threshold = max_ji_gr$.threshold[1],
      buffer = NULL
    ),
    .pred_cl_max = make_two_class_pred(
      estimate = .pred_leaves_colored,
      levels = levels(leaves_colored),
      threshold = max_ji_cl$.threshold[1],
      buffer = NULL
    ),
    .pred_bb_max = make_two_class_pred(
      estimate = .pred_leaves_breaking_buds,
      levels = levels(leaves_breaking_buds),
      threshold = max_ji_bb$.threshold[1],
      buffer = NULL
    )
  )

val_noskip <- which(py_to_r(noskip$cpu()$numpy()))

val_acc_gr <- accuracy(val_df, leaves_green, .pred_gr_max)
val_acc_cl <- accuracy(val_df, leaves_colored, .pred_cl_max)
val_acc_bb <- accuracy(val_df, leaves_breaking_buds, .pred_bb_max)

test_dat = eval$evaluate(test_dl, vit, "cuda:0")
test_logits <- torch$cat(test_dat[[1]])
test_preds <- torch$nn$functional$sigmoid(test_logits)
test_truth <- torch$cat(test_dat[[2]])

noskip <- test_truth[ , 0] != -999999999

test_loss <- criterion(test_preds[noskip, ], test_truth[noskip, ])
test_df <- as.data.frame(test_truth[noskip, ]$cpu()$numpy()) |>
  mutate(leaves_green = as.integer(V1), leaves_colored = as.integer(V2),
         leaves_breaking_buds = as.integer(V3)) |>
  bind_cols(as.data.frame(test_preds[noskip, ]$cpu()$numpy()) |>
              rename(.pred_leaves_green = V1, .pred_leaves_colored = V2,
                     .pred_leaves_breaking_buds = V3)) |>
  bind_cols(as.data.frame(test_loss$cpu()$numpy()) |>
              rename(.loss_leaves_green = V1, .loss_leaves_colored = V2,
                     .loss_leaves_breaking_buds = V3)) |>
  select(-V1, -V2, -V3) |>
  mutate(leaves_green = factor(leaves_green, levels = c("1", "0")),
         leaves_colored = factor(leaves_colored, levels = c("1", "0")),
         leaves_breaking_buds = factor(leaves_breaking_buds, levels = c("1", "0")))

# threshold_data_gr <- test_df %>%
#   threshold_perf(leaves_green, .pred_leaves_green, thresholds = seq(0, 1, by = 0.005),
#                  event_level = "first")
#
# max_ji_gr <- threshold_data_gr |>
#   filter(.metric == "j_index") |>
#   slice_max(.estimate, n = 5)
#
# threshold_data_cl <- test_df %>%
#   threshold_perf(leaves_colored, .pred_leaves_colored, thresholds = seq(0, 1, by = 0.02),
#                  event_level = "first")

# max_ji_cl <- threshold_data_cl |>
#   filter(.metric == "j_index") |>
#   slice_max(.estimate, n = 5)

# threshold_data_bb <- test_df %>%
#   threshold_perf(leaves_breaking_buds, .pred_leaves_breaking_buds, thresholds = seq(0, 1, by = 0.02),
#                  event_level = "first")

# max_ji_bb <- threshold_data_bb |>
#   filter(.metric == "j_index") |>
#   slice_max(.estimate, n = 5)

test_df <- test_df |>
  mutate(
    .pred_gr_max = make_two_class_pred(
      estimate = .pred_leaves_green,
      levels = levels(leaves_green),
      threshold = max_ji_gr$.threshold[1],
      buffer = NULL
    ),
    .pred_cl_max = make_two_class_pred(
      estimate = .pred_leaves_colored,
      levels = levels(leaves_colored),
      threshold = max_ji_cl$.threshold[1],
      buffer = NULL
    ),
    .pred_bb_max = make_two_class_pred(
      estimate = .pred_leaves_breaking_buds,
      levels = levels(leaves_breaking_buds),
      threshold = max_ji_bb$.threshold[1],
      buffer = NULL
    ),
    .pred_no_max = factor(as.numeric(.pred_gr_max == "0" & .pred_cl_max == "0" & .pred_bb_max == "0"),
                          levels = c("1", "0")),
    leaves_none = factor(as.numeric(!(leaves_green == "1" | leaves_colored == "1" | leaves_breaking_buds == "1")),
                         levels = c("1", "0"))
  )

test_noskip <- which(py_to_r(noskip$cpu()$numpy()))

test_acc_gr <- accuracy(test_df, leaves_green, .pred_gr_max)
test_acc_cl <- accuracy(test_df, leaves_colored, .pred_cl_max)
test_acc_bb <- accuracy(test_df, leaves_breaking_buds, .pred_bb_max)
test_acc_no <- accuracy(test_df, leaves_none, .pred_no_max)

cat("Testing Accuracy:\n", "Green Leaves: ", test_acc_gr$.estimate * 100, "%\n",
    "Colored Leaves: ", test_acc_cl$.estimate * 100, "%\n",
    "Breaking Buds: ", test_acc_bb$.estimate * 100, "%\n",
    "No Leaves: ", test_acc_no$.estimate * 100, "%", sep = "")

test_jind_gr <- j_index(test_df, leaves_green, .pred_gr_max)
test_jind_cl <- j_index(test_df, leaves_colored, .pred_cl_max)
test_jind_bb <- j_index(test_df, leaves_breaking_buds, .pred_bb_max)
test_jind_no <- j_index(test_df, leaves_none, .pred_no_max)

############## save it #######################

val_df <- val_df |>
  bind_cols(as.data.frame(val_logits[val_noskip - 1, ]$cpu()$numpy()) |>
              rename(.logit_leaves_green = V1, .logit_leaves_colored = V2,
                     .logit_leaves_breaking_buds = V3))

test_df <- test_df |>
  bind_cols(as.data.frame(test_logits[test_noskip - 1, ]$cpu()$numpy()) |>
              rename(.logit_leaves_green = V1, .logit_leaves_colored = V2,
                     .logit_leaves_breaking_buds = V3))

val_df <- leaves_val[val_noskip, ] |>
  select(-starts_with("leaves_")) |>
  bind_cols(val_df)

test_df <- leaves_test[test_noskip, ] |>
  select(-starts_with("leaves_")) |>
  bind_cols(test_df)

testing_df <- val_df |>
  mutate(partition = "validation") |>
  bind_rows(test_df |>
              mutate(partition = "testing"))

write_rds(testing_df, "output/leaves/phenovision-init_model_02_12_2025/epoch_4_test_results.rds")





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

