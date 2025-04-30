library(reticulate)
library(tidyverse)
library(tidymodels)
library(probably)
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

model_folder <- "output/leaves/phenovision-init_model2_04_11_2025"
if(!dir.exists(model_folder)) dir.create(model_folder)
if(!dir.exists(file.path(model_folder, "checkpoints"))) dir.create(file.path(model_folder, "checkpoints"))

#bad_images <- read_rds("data/inat/bad_images.rds")

leaves_train <- read_csv("data/leaves/train.csv")
leaves_val <- read_csv("data/leaves/validation.csv")

## setup two datasets for background state and 'the rest'
leaves_train_background <- leaves_train |>
  filter(leaves_green == 1 & leaves_colored == 0 & leaves_breaking_buds == 0)
leaves_train_rest <- leaves_train |>
  filter(!(leaves_green == 1 & leaves_colored == 0 & leaves_breaking_buds == 0))

# table(leaves_train_rest$leaves_breaking_buds, leaves_train_rest$leaves_green,
#       leaves_train_rest$leaves_no_live, leaves_train_rest$leaves_colored)

leaves_train_resampled <- leaves_train_rest |>
  slice_sample(n = nrow(leaves_train_background), replace = TRUE)

leaves_train_new <- bind_rows(leaves_train_resampled,
                              leaves_train_background) |>
  slice_sample(prop = 1)

leaves_val_background <- leaves_val |>
  filter(leaves_green == 1 & leaves_colored == 0 & leaves_breaking_buds == 0)
leaves_val_rest <- leaves_val |>
  filter(!(leaves_green == 1 & leaves_colored == 0 & leaves_breaking_buds == 0))

leaves_val_resampled <- leaves_val_rest |>
  slice_sample(n = nrow(leaves_val_background), replace = TRUE)

leaves_val_new <- bind_rows(leaves_val_resampled,
                            leaves_val_background) |>
  slice_sample(prop = 1)

train_img <- r_to_py(leaves_train_new$file_name)
train_leaves <- leaves_train_new |>
  select(starts_with("leaves_")) |>
  as.matrix()

# train_img_bg <- r_to_py(leaves_train_background$file_name)
# train_leaves_bg <- leaves_train_background |>
#   select(starts_with("leaves_")) |>
#   as.matrix()

val_img <- r_to_py(leaves_val_new$file_name)
val_leaves <- leaves_val_new |>
  select(starts_with("leaves_")) |>
  as.matrix()

# val_img_bg <- r_to_py(leaves_val_background$file_name)
# val_leaves_bg <- leaves_val_background |>
#   select(starts_with("leaves_")) |>
#   as.matrix()

models_vit <- import_from_path("models_vit", "PlantCLEF2022")
pos_embed <- import_from_path("pos_embed", "PlantCLEF2022/util")

checkpoint_init <- "output/model_04_13_2024/checkpoints/vit_finetuned_epoch4.pt"
model <- torch$load(checkpoint_init)

vit <- models_vit$vit_large_patch16(num_classes = 3L)
checkpoint_model <- model$state_dict()

config <- timm$data$resolve_data_config(model = vit)
transform = timm$data$create_transform(
  input_size = list(3L, 224L, 224L),
  is_training = TRUE,
  color_jitter = FALSE,
  auto_augment = 'rand-m9-mstd0.5-inc1',
  interpolation = 'bicubic',
  re_prob = 0.25,
  re_mode = 'pixel',
  re_count = 1,
  mean = list(0.485, 0.456, 0.406),
  std = list(0.229, 0.224, 0.255),
)
val_transform <- timm$data$create_transform(!!!config)

batch_size <- 384L

train_ds <- ds$PhenoDataset(train_img, train_leaves, transform = transform)
train_dl <- timm$data$create_loader(train_ds, c(3L, 224L, 224L), batch_size, num_workers = 7L,
                                    is_training = TRUE)

# train_ds_bg <- ds$PhenoDataset(train_img_bg, train_leaves_bg, transform = transform)
# train_dl_bg <- timm$data$create_loader(train_ds_bg, c(3L, 224L, 224L), batch_size, num_workers = 7L,
#                                        is_training = TRUE)

val_ds <- ds$PhenoDataset(val_img, val_leaves, transform = val_transform)
val_dl <- timm$data$create_loader(val_ds, c(3L, 224L, 224L), batch_size, num_workers = 7L,
                                  is_training = FALSE)

# val_ds_bg <- ds$PhenoDataset(val_img_bg, val_leaves_bg, transform = transform)
# val_dl_bg <- timm$data$create_loader(val_ds_bg, c(3L, 224L, 224L), batch_size, num_workers = 7L)

#test_it <- as_iterator(train_dl)
#test_dat <- iter_next(test_it)

checkpoint_model$head.bias <- NULL
checkpoint_model$head.weight <- NULL
#torch$nn$init$trunc_normal_(vit$head$weight, std = 1e-5)
vit$load_state_dict(checkpoint_model, strict = FALSE)
torch$nn$init$trunc_normal_(vit$head$weight, std = 1e-5)
torch$nn$init$trunc_normal_(vit$head$bias, std = 1e-5)

vit <- vit$cuda()

#pos_embed$interpolate_pos_embed(vit, checkpoint_model)

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

#log_writer <- torch$utils$tensorboard$SummaryWriter(log_dir = model_folder)
log_writer <- NULL

clip_grad <- py_none()

num_epochs <- 4

for(i in 1:num_epochs) {

  val_dat = eval$evaluate(val_dl, vit, "cuda:0")
  val_preds <- torch$nn$functional$sigmoid(torch$cat(val_dat[[1]]))
  val_truth <- torch$cat(val_dat[[2]])
  val_loss <- criterion(val_preds, val_truth)
  val_df <- as.data.frame(val_truth$cpu()$numpy()) |>
    mutate(leaves_green = as.integer(V1), leaves_colored = as.integer(V2),
           leaves_breaking_buds = as.integer(V3), leaves_no_live = as.integer((V1 + V2 + V3) == 0)) |>
    bind_cols(as.data.frame(val_preds$cpu()$numpy()) |>
                rename(.pred_leaves_green = V1, .pred_leaves_colored = V2,
                       .pred_leaves_breaking_buds = V3)) |>
    select(-V1, -V2, -V3) |>
    mutate(leaves_green = factor(leaves_green, levels = c("1", "0")),
           leaves_colored = factor(leaves_colored, levels = c("1", "0")),
           leaves_breaking_buds = factor(leaves_breaking_buds, levels = c("1", "0")),
           leaves_no_live = factor(leaves_no_live, levels = c("1", "0")))

  threshold_data_gr <- val_df %>%
    threshold_perf(leaves_green, .pred_leaves_green, thresholds = seq(0, 1, by = 0.05),
                   event_level = "first")

  max_ji_gr <- threshold_data_gr |>
    filter(.metric == "j_index") |>
    slice_max(.estimate, n = 5)

  threshold_data_cl <- val_df %>%
    threshold_perf(leaves_colored, .pred_leaves_colored, thresholds = seq(0, 1, by = 0.05),
                   event_level = "first")

  max_ji_cl <- threshold_data_cl |>
    filter(.metric == "j_index") |>
    slice_max(.estimate, n = 5)

  # threshold_data_no <- val_df %>%
  #   threshold_perf(leaves_no_live, .pred_leaves_no_live, thresholds = seq(0, 1, by = 0.05),
  #                  event_level = "first")
  #
  # max_ji_no <- threshold_data_no |>
  #   filter(.metric == "j_index") |>
  #   slice_max(.estimate, n = 5)

  threshold_data_bb <- val_df %>%
    threshold_perf(leaves_breaking_buds, .pred_leaves_breaking_buds, thresholds = seq(0, 1, by = 0.05),
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
      ),
      .pred_no_max = factor(as.integer((as.numeric(as.character(.pred_gr_max)) +
                                          as.numeric(as.character(.pred_cl_max)) +
                                          as.numeric(as.character(.pred_bb_max)) == 0)),
                            levels = levels(leaves_no_live))
    )

  val_acc_gr <- accuracy(val_df, leaves_green, .pred_gr_max)
  val_acc_cl <- accuracy(val_df, leaves_colored, .pred_cl_max)
  val_acc_no <- accuracy(val_df, leaves_no_live, .pred_no_max)
  val_acc_bb <- accuracy(val_df, leaves_breaking_buds, .pred_bb_max)

  val_jind_gr <- j_index(val_df, leaves_green, .pred_gr_max)
  val_jind_cl <- j_index(val_df, leaves_colored, .pred_cl_max)
  val_jind_no <- j_index(val_df, leaves_no_live, .pred_no_max)
  val_jind_bb <- j_index(val_df, leaves_breaking_buds, .pred_bb_max)

  # sensitivity(val_df, flower, .pred_fl_max)
  # specificity(val_df, flower, .pred_fl_max)
  #
  # sensitivity(val_df, fruit, .pred_fr_max)
  # specificity(val_df, fruit, .pred_fr_max)

  cat("Epoch ", i, " Test:",
      "\ngreen leaves acc: ", val_acc_gr$.estimate[1],
      "\ncolored leaves acc: ", val_acc_cl$.estimate[1],
      "\nno live leaves acc: ", val_acc_no$.estimate[1],
      "\nbreaking leaf bud acc: ", val_acc_bb$.estimate[1],
      "\ngreen leaves j-index: ", val_jind_gr$.estimate[1],
      "\ncolored leaves j-index: ", val_jind_cl$.estimate[1],
      "\nno live leaves j-index: ", val_jind_no$.estimate[1],
      "\nbreaking leaf buds j-index: ", val_jind_bb$.estimate[1],
      "\nloss: ", val_loss$cpu()$numpy(),
      "\n")

  # log_writer$add_scalar('perf/val_acc_gr', val_acc_gr$.estimate[1], i)
  # log_writer$add_scalar('perf/val_acc_cl', val_acc_cl$.estimate[1], i)
  # log_writer$add_scalar('perf/val_acc_no', val_acc_no$.estimate[1], i)
  # log_writer$add_scalar('perf/val_acc_bb', val_acc_bb$.estimate[1], i)
  # log_writer$add_scalar('perf/val_loss', val_loss, i)

  train_stats <- engine$train_one_epoch(
    vit, criterion, train_dl,
    optimizer, "cuda", i, loss_scaler,
    clip_grad, mixup_fn = py_none(),
    log_writer = log_writer,
    args = types$SimpleNamespace(accum_iter = 1L, warmup_epochs = 2L, lr = lr, min_lr = min_lr,
                                 epochs = num_epochs)
  )

  torch$save(vit, file.path(model_folder, "checkpoints", paste0("vit_finetuned_epoch", i, ".pt")))

}

###### add a second round of annotated data
vit <- torch$load("output/leaves/phenovision-init_model_02_12_2025/checkpoints/vit_finetuned_epoch4.pt")
vit <- vit$cuda()

leaves_seconds <- read_csv("data/leaves/seconds.csv")

sec_img <- r_to_py(leaves_seconds$file_name)
sec_leaves <- leaves_seconds |>
  select(starts_with("leaves_")) |>
  as.matrix()

bad_ds <- del$PhenoDatasetDeleter(sec_img, sec_leaves)
bad_dl <- torch$utils$data$DataLoader(bad_ds, 1024L, num_workers = 7L)

bad_img <- iterate(bad_dl)
bad_imgs <- map(bad_img, 1) |>
  list_flatten() |>
  list_c()
sec_bad <- map(bad_img, ~ as.vector(.x[[2]]$numpy())) |>
  list_flatten() |>
  list_c()
bad_imgs <- bad_imgs[which(sec_bad == 1)]

leaves_seconds <- leaves_seconds |>
  filter(!file_name %in% bad_imgs)

sec_img <- r_to_py(leaves_seconds$file_name)
sec_leaves <- leaves_seconds |>
  select(starts_with("leaves_")) |>
  as.matrix()

sec_ds <- ds$PhenoDataset(sec_img, sec_leaves, transform = val_transform)
sec_dl <- timm$data$create_loader(sec_ds, c(3L, 224L, 224L), 2560L, num_workers = 7L,
                                  is_training = FALSE)

sec_infer = eval$infer(sec_dl, vit, "cuda:0")
sec_logits <- torch$cat(sec_infer[[1]])
sec_indexes <- torch$cat(sec_infer[[2]])$cpu()$numpy()
sec_preds <- torch$nn$functional$sigmoid(sec_logits)$cpu()$numpy() |>
  as.matrix() |>
  as.data.frame() |>
  rename(infer_leaves_green = V1, infer_leaves_colored = V2, infer_leaves_breaking_buds = V3)

leaves_seconds_preds <- leaves_seconds |>
  bind_cols(sec_preds)

leaves_seconds_good_annots <- leaves_seconds_preds |>
  rowwise() |>
  mutate(good = all(as.numeric(c(infer_leaves_green > 0.95, infer_leaves_colored > 0.95, infer_leaves_breaking_buds > 0.95)) ==
           c(leaves_green, leaves_colored, leaves_breaking_buds))) |>
  filter(good)

### finetune model further with new data

leaves_train <- read_csv("data/leaves/train.csv")
leaves_val <- read_csv("data/leaves/validation.csv")

leaves_train <- leaves_train |>
  bind_rows(leaves_seconds_good_annots |>
              select(-starts_with("infer"), -good)) |>
  slice_sample(prop = 1)

## setup two datasets for background state and 'the rest'
leaves_train_background <- leaves_train |>
  filter(leaves_green == 1 & leaves_colored == 0 & leaves_breaking_buds == 0)
leaves_train_rest <- leaves_train |>
  filter(!(leaves_green == 1 & leaves_colored == 0 & leaves_breaking_buds == 0))

leaves_train_resampled <- leaves_train_rest |>
  slice_sample(n = nrow(leaves_train_background), replace = TRUE)

leaves_train_new <- bind_rows(leaves_train_resampled,
                              leaves_train_background) |>
  slice_sample(prop = 1)

leaves_val_background <- leaves_val |>
  filter(leaves_green == 1 & leaves_colored == 0 & leaves_breaking_buds == 0)
leaves_val_rest <- leaves_val |>
  filter(!(leaves_green == 1 & leaves_colored == 0 & leaves_breaking_buds == 0))

leaves_val_resampled <- leaves_val_rest |>
  slice_sample(n = nrow(leaves_val_background), replace = TRUE)

leaves_val_new <- bind_rows(leaves_val_resampled,
                            leaves_val_background) |>
  slice_sample(prop = 1)

train_img <- r_to_py(leaves_train_new$file_name)
train_leaves <- leaves_train_new |>
  select(starts_with("leaves_")) |>
  as.matrix()

val_img <- r_to_py(leaves_val_new$file_name)
val_leaves <- leaves_val_new |>
  select(starts_with("leaves_")) |>
  as.matrix()


config <- timm$data$resolve_data_config(model = vit)
transform = timm$data$create_transform(
  input_size = list(3L, 224L, 224L),
  is_training = TRUE,
  color_jitter = FALSE,
  auto_augment = 'rand-m9-mstd0.5-inc1',
  interpolation = 'bicubic',
  re_prob = 0.25,
  re_mode = 'pixel',
  re_count = 1,
  mean = list(0.485, 0.456, 0.406),
  std = list(0.229, 0.224, 0.255),
)
val_transform <- timm$data$create_transform(!!!config)

batch_size <- 384L

train_ds <- ds$PhenoDataset(train_img, train_leaves, transform = transform)
train_dl <- timm$data$create_loader(train_ds, c(3L, 224L, 224L), batch_size, num_workers = 7L,
                                    is_training = TRUE)

# train_ds_bg <- ds$PhenoDataset(train_img_bg, train_leaves_bg, transform = transform)
# train_dl_bg <- timm$data$create_loader(train_ds_bg, c(3L, 224L, 224L), batch_size, num_workers = 7L,
#                                        is_training = TRUE)

val_ds <- ds$PhenoDataset(val_img, val_leaves, transform = val_transform)
val_dl <- timm$data$create_loader(val_ds, c(3L, 224L, 224L), batch_size, num_workers = 7L,
                                  is_training = FALSE)

#model_folder <- "output/leaves/phenovision-init_model_02_12_2025/second"
#if(!dir.exists(model_folder)) dir.create(model_folder)
#if(!dir.exists(file.path(model_folder, "checkpoints"))) dir.create(file.path(model_folder, "checkpoints"))

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

#log_writer <- torch$utils$tensorboard$SummaryWriter(log_dir = model_folder)
log_writer <- NULL

clip_grad <- py_none()

num_epochs <- 4

for(i in 1:num_epochs) {

  val_dat = eval$evaluate(val_dl, vit, "cuda:0")
  val_preds <- torch$nn$functional$sigmoid(torch$cat(val_dat[[1]]))
  val_truth <- torch$cat(val_dat[[2]])
  val_loss <- criterion(val_preds, val_truth)
  val_df <- as.data.frame(val_truth$cpu()$numpy()) |>
    mutate(leaves_green = as.integer(V1), leaves_colored = as.integer(V2),
           leaves_breaking_buds = as.integer(V3), leaves_no_live = as.integer((V1 + V2 + V3) == 0)) |>
    bind_cols(as.data.frame(val_preds$cpu()$numpy()) |>
                rename(.pred_leaves_green = V1, .pred_leaves_colored = V2,
                       .pred_leaves_breaking_buds = V3)) |>
    select(-V1, -V2, -V3) |>
    mutate(leaves_green = factor(leaves_green, levels = c("1", "0")),
           leaves_colored = factor(leaves_colored, levels = c("1", "0")),
           leaves_breaking_buds = factor(leaves_breaking_buds, levels = c("1", "0")),
           leaves_no_live = factor(leaves_no_live, levels = c("1", "0")))

  threshold_data_gr <- val_df %>%
    threshold_perf(leaves_green, .pred_leaves_green, thresholds = seq(0, 1, by = 0.05),
                   event_level = "first")

  max_ji_gr <- threshold_data_gr |>
    filter(.metric == "j_index") |>
    slice_max(.estimate, n = 5)

  threshold_data_cl <- val_df %>%
    threshold_perf(leaves_colored, .pred_leaves_colored, thresholds = seq(0, 1, by = 0.05),
                   event_level = "first")

  max_ji_cl <- threshold_data_cl |>
    filter(.metric == "j_index") |>
    slice_max(.estimate, n = 5)

  # threshold_data_no <- val_df %>%
  #   threshold_perf(leaves_no_live, .pred_leaves_no_live, thresholds = seq(0, 1, by = 0.05),
  #                  event_level = "first")
  #
  # max_ji_no <- threshold_data_no |>
  #   filter(.metric == "j_index") |>
  #   slice_max(.estimate, n = 5)

  threshold_data_bb <- val_df %>%
    threshold_perf(leaves_breaking_buds, .pred_leaves_breaking_buds, thresholds = seq(0, 1, by = 0.05),
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
      ),
      .pred_no_max = factor(as.integer((as.numeric(as.character(.pred_gr_max)) +
                                          as.numeric(as.character(.pred_cl_max)) +
                                          as.numeric(as.character(.pred_bb_max)) == 0)),
                            levels = levels(leaves_no_live))
    )

  val_acc_gr <- accuracy(val_df, leaves_green, .pred_gr_max)
  val_acc_cl <- accuracy(val_df, leaves_colored, .pred_cl_max)
  val_acc_no <- accuracy(val_df, leaves_no_live, .pred_no_max)
  val_acc_bb <- accuracy(val_df, leaves_breaking_buds, .pred_bb_max)

  val_jind_gr <- j_index(val_df, leaves_green, .pred_gr_max)
  val_jind_cl <- j_index(val_df, leaves_colored, .pred_cl_max)
  val_jind_no <- j_index(val_df, leaves_no_live, .pred_no_max)
  val_jind_bb <- j_index(val_df, leaves_breaking_buds, .pred_bb_max)

  # sensitivity(val_df, flower, .pred_fl_max)
  # specificity(val_df, flower, .pred_fl_max)
  #
  # sensitivity(val_df, fruit, .pred_fr_max)
  # specificity(val_df, fruit, .pred_fr_max)

  cat("Epoch ", i, " Test:",
      "\ngreen leaves acc: ", val_acc_gr$.estimate[1],
      "\ncolored leaves acc: ", val_acc_cl$.estimate[1],
      "\nno live leaves acc: ", val_acc_no$.estimate[1],
      "\nbreaking leaf bud acc: ", val_acc_bb$.estimate[1],
      "\ngreen leaves j-index: ", val_jind_gr$.estimate[1],
      "\ncolored leaves j-index: ", val_jind_cl$.estimate[1],
      "\nno live leaves j-index: ", val_jind_no$.estimate[1],
      "\nbreaking leaf buds j-index: ", val_jind_bb$.estimate[1],
      "\nloss: ", val_loss$cpu()$numpy(),
      "\n")

  # log_writer$add_scalar('perf/val_acc_gr', val_acc_gr$.estimate[1], i)
  # log_writer$add_scalar('perf/val_acc_cl', val_acc_cl$.estimate[1], i)
  # log_writer$add_scalar('perf/val_acc_no', val_acc_no$.estimate[1], i)
  # log_writer$add_scalar('perf/val_acc_bb', val_acc_bb$.estimate[1], i)
  # log_writer$add_scalar('perf/val_loss', val_loss, i)

  train_stats <- engine$train_one_epoch(
    vit, criterion, train_dl,
    optimizer, "cuda", i, loss_scaler,
    clip_grad, mixup_fn = py_none(),
    log_writer = log_writer,
    args = types$SimpleNamespace(accum_iter = 1L, warmup_epochs = 2L, lr = lr, min_lr = min_lr,
                                 epochs = num_epochs)
  )

  torch$save(vit, file.path(model_folder, "checkpoints", paste0("vit_finetuned_round2_epoch", i, ".pt")))

}


## This is the one!
# Epoch 1 (round 2)
# green leaves j-index:  0.9203717
# colored leaves j-index:  0.8911916
# no live leaves j-index:  0.9012072
# breaking leaf buds j-index:  0.9815574
mean(c(0.9203717, 0.8911916, 0.9012072, 0.9815574))
# green leaves acc:  0.962261
# colored leaves acc:  0.9473975
# no live leaves acc:  0.987844
# breaking leaf bud acc:  0.9920433
# Epoch 2 (round 2)
# green leaves j-index:  0.9124636
# colored leaves j-index:  0.8481586
# no live leaves j-index:  0.8892018
# breaking leaf buds j-index:  0.9646507
# Epoch 3 (round 2)
# green leaves j-index:  0.9221385
# colored leaves j-index:  0.8534621
# no live leaves j-index:  0.9242712
# breaking leaf buds j-index:  0.9645432
