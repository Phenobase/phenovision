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

model_folder <- "output/model_05_13_2024"
if(!dir.exists(model_folder)) dir.create(model_folder)
if(!dir.exists(file.path(model_folder, "checkpoints"))) dir.create(file.path(model_folder, "checkpoints"))

bad_images <- read_rds("data/inat/bad_images.rds")

inat_train <- read_csv("data/inat/train.csv") |>
  filter(!file_name %in% bad_images)
inat_val <- read_csv("data/inat/validation.csv") |>
  filter(!file_name %in% bad_images)

train_img <- r_to_py(inat_train$file_name)
train_fruit_flower <- inat_train |>
  select(fruiting, flowering) |>
  as.matrix()

val_img <- r_to_py(inat_val$file_name)
val_fruit_flower <- inat_val |>
  select(fruiting, flowering) |>
  as.matrix()

model <- torch$load("models/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth")

models_vit <- import_from_path("models_vit", "PlantCLEF2022")
pos_embed <- import_from_path("pos_embed", "PlantCLEF2022/util")

vit <- models_vit$vit_large_patch16(num_classes = 2L)
checkpoint_model <- model$model

config <- timm$data$resolve_data_config(model = vit)
transform <- timm$data$create_transform(!!!config)

batch_size <- 384L

train_ds <- ds$PhenoDataset(train_img, train_fruit_flower, transform = transform)
train_dl <- timm$data$create_loader(train_ds, c(3L, 224L, 224L), batch_size, num_workers = 10L,
                                    is_training = TRUE)

val_ds <- ds$PhenoDataset(val_img, val_fruit_flower, transform = transform)
val_dl <- timm$data$create_loader(val_ds, c(3L, 224L, 224L), batch_size, num_workers = 10L)

#test_it <- as_iterator(train_dl)
#test_dat <- iter_next(test_it)

checkpoint_model$head.bias <- NULL
checkpoint_model$head.weight <- NULL
vit$load_state_dict(checkpoint_model, strict = FALSE)
torch$nn$init$trunc_normal_(vit$head$weight, std = 1e-5)

vit <- vit$cuda()

pos_embed$interpolate_pos_embed(vit, checkpoint_model)

weight_decay <- 0.05
layer_decay <- 0.65

param_groups <- lrd$param_groups_lrd(vit, weight_decay,
                                     no_weight_decay_list=vit$no_weight_decay(),
                                     layer_decay = layer_decay
)

blr <- 5e-4

lr <- blr * batch_size / 256

min_lr <- 1e-6

optimizer <- torch$optim$AdamW(param_groups, lr = lr)
loss_scaler <- NativeScaler()

loss <- torch$nn$BCEWithLogitsLoss()
criterion <- function(outputs, targets) {
  loss(outputs, targets$type(torch$float32))
}

log_writer <- torch$utils$tensorboard$SummaryWriter(log_dir = model_folder)

clip_grad <- py_none()

num_epochs <- 100

for(i in 1:num_epochs) {

  val_dat = eval$evaluate(val_dl, vit, "cuda:0")
  val_preds <- torch$nn$functional$sigmoid(torch$cat(val_dat[[1]]))
  val_truth <- torch$cat(val_dat[[2]])
  val_loss <- criterion(val_preds, val_truth)
  val_df <- as.data.frame(val_truth$cpu()$numpy()) |>
    mutate(fruit = as.integer(V1), flower = as.integer(V2)) |>
    bind_cols(as.data.frame(val_preds$cpu()$numpy()) |>
                rename(.pred_fruit = V1, .pred_flower = V2)) |>
    select(-V1, -V2) |>
    mutate(fruit = factor(fruit, levels = c("1", "0")),
           flower = factor(flower, levels = c("1", "0")))

  threshold_data_fr <- val_df %>%
    threshold_perf(fruit, .pred_fruit, thresholds = seq(0, 1, by = 0.05),
                   event_level = "first")

  max_ji_fr <- threshold_data_fr |>
    filter(.metric == "j_index") |>
    slice_max(.estimate, n = 5)

  threshold_data_fl <- val_df %>%
    threshold_perf(flower, .pred_flower, thresholds = seq(0, 1, by = 0.05),
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

  # sensitivity(val_df, flower, .pred_fl_max)
  # specificity(val_df, flower, .pred_fl_max)
  #
  # sensitivity(val_df, fruit, .pred_fr_max)
  # specificity(val_df, fruit, .pred_fr_max)

  cat("Epoch ", i, " Test:",
      "\nflower acc: ", val_acc_fl$.estimate[1],
      "\nfruit acc: ", val_acc_fr$.estimate[1],
      "\nflower j-index: ", val_jind_fl$.estimate[1],
      "\nfruit j-index: ", val_jind_fr$.estimate[1],
      "\nloss: ", val_loss$cpu()$numpy(),
      "\n")

  log_writer$add_scalar('perf/val_acc_fr', val_acc_fr$.estimate[1], i)
  log_writer$add_scalar('perf/val_acc_fl', val_acc_fl$.estimate[1], i)
  log_writer$add_scalar('perf/val_loss', val_loss, i)

  train_stats <- engine$train_one_epoch(
    vit, criterion, train_dl,
    optimizer, "cuda", i, loss_scaler,
    clip_grad, mixup_fn = py_none(),
    log_writer = log_writer,
    args = types$SimpleNamespace(accum_iter = 1L, warmup_epochs = 5L, lr = lr, min_lr = min_lr,
                                 epochs = num_epochs)
  )

  torch$save(vit, file.path(model_folder, "checkpoints", paste0("vit_finetuned_epoch", i, ".pt")))

}
