library(reticulate)
library(tidyverse)

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

# inf_images <- list.files("/blue/guralnick/share/phenobase_inat_data/images/medium",
#                          full.names = TRUE)
#
# write_rds(inf_images, "output/all_inference_images.rds")

inf_images <- read_rds("output/all_inference_images.rds")

img <- r_to_py(inf_images)
fruit_flower <- r_to_py(matrix(1, ncol = 2, nrow = length(img)))

del_ds <- del$PhenoDatasetDeleter(img, fruit_flower)
del_dl <- torch$utils$data$DataLoader(del_ds, 1024L, num_workers = 10L)

bad_img <- iterate(del_dl)
imgs <- map(bad_img, 1) |>
  list_c()
bad <- map(bad_img, ~ as.vector(.x[[2]]$numpy())) |>
  list_c()
bad_imgs <- unlist(imgs[bad == 1])

write_rds(bad_imgs, "data/inat/inference/bad_images_all.rds")

file_info <- file.info(inf_images, extra_cols = FALSE)
write_rds(file_info, "data/inat/inference/file_info_all.rds")

#tar -xvf medium.tar medium/9999990.jpeg --strip-components 1
#tar -xvf medium.tar -C /blue/guralnick/share/r.dinnage/testing --occurrence medium/100.jpg
