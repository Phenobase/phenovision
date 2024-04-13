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

inat_train <- read_csv("data/inat/train.csv")
inat_val <- read_csv("data/inat/validation.csv")
inat_test <- read_csv("data/inat/test.csv")

train_img <- r_to_py(inat_train$file_name)
train_fruit_flower <- inat_train |>
  select(fruiting, flowering) |>
  as.matrix()

val_img <- r_to_py(inat_val$file_name)
val_fruit_flower <- inat_val |>
  select(fruiting, flowering) |>
  as.matrix()

test_img <- r_to_py(inat_test$file_name)
test_fruit_flower <- inat_test |>
  select(fruiting, flowering) |>
  as.matrix()


train_ds <- del$PhenoDatasetDeleter(train_img, train_fruit_flower)
train_dl <- torch$utils$data$DataLoader(train_ds, 1024L, num_workers = 16L)

bad_img <- iterate(train_dl)
train_imgs <- map(bad_img, 1) |>
  list_c()
train_bad <- map(bad_img, ~ as.vector(.x[[2]]$numpy())) |>
  list_c()
bad_imgs <- unlist(train_imgs[train_bad == 1])

val_ds <- del$PhenoDatasetDeleter(val_img, val_fruit_flower)
val_dl <- torch$utils$data$DataLoader(val_ds, 1024L, num_workers = 16L)

bad_img2 <- iterate(val_dl)
val_imgs <- map(bad_img2, 1) |>
  list_c()
val_bad <- map(bad_img2, ~ as.vector(.x[[2]]$numpy())) |>
  list_c()
bad_imgs <- c(bad_imgs, unlist(val_imgs[val_bad == 1]))

test_ds <- del$PhenoDatasetDeleter(test_img, test_fruit_flower)
test_dl <- torch$utils$data$DataLoader(test_ds, 1024L, num_workers = 16L)

bad_img3 <- iterate(test_dl)
test_imgs <- map(bad_img3, 1) |>
  list_c()
test_bad <- map(bad_img3, ~ as.vector(.x[[2]]$numpy())) |>
  list_c()
bad_imgs <- c(bad_imgs, unlist(test_imgs[test_bad == 1]))

write_rds(bad_imgs, "data/inat/bad_images.rds")
