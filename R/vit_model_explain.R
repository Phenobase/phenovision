Sys.setenv(TIMM_FUSED_ATTN = 0)

library(reticulate)
library(tidyverse)
library(imager)

torch <- import("torch")
timm <- import("timm")
PIL <- import("PIL")
cv2 <- import("cv2")
np <- import("numpy")
ds <- import_from_path("custom_dataset", "py")
del <- import_from_path("remove_bad_images", "py")
engine <- import_from_path("engine_finetune", "PlantCLEF2022")
lrd <- import_from_path("lr_decay", "PlantCLEF2022/util")
misc <- import_from_path("misc", "PlantCLEF2022/util")
eval <- import_from_path("evaluate", "py")
NativeScaler <- misc$NativeScalerWithGradNormCount
types <- import("types")
vit_rollout <- import_from_path("vit_rollout", "vit-explain")
vit_explain <- import_from_path("vit_explain", "vit-explain")

models_vit <- import_from_path("models_vit", "PlantCLEF2022")
pos_embed <- import_from_path("pos_embed", "PlantCLEF2022/util")

vit <- torch$load("output/plantclef_2025_03_09/checkpoints/vit_finetuned_plantclef_epoch23.pt")

bad_images <- read_rds("data/inat/bad_images.rds")

inat_test <- read_csv("data/inat/test.csv") |>
  filter(!file_name %in% bad_images) |>
  slice_sample(n = 500)

test_img <- r_to_py(inat_test$file_name)
test_fruit_flower <- inat_test |>
  select(fruiting, flowering) |>
  as.matrix()

vit <- vit$cuda()

vit2 <- models_vit$vit_large_patch16(num_classes = 2L)
config <- timm$data$resolve_data_config(model = vit2)
transform <- timm$data$create_transform(!!!config)

vit2$load_state_dict(vit$state_dict())
vit2 <- vit2$cuda()

att_rollout <- vit_rollout$VITAttentionRollout(vit2, #attention_layer_name = 'k_norm',
                                               head_fusion = 'mean', discard_ratio = 0.9)

batch_size <- 1L

#test_ds <- ds$PhenoDataset(test_img, test_fruit_flower, transform = transform)
#test_dl <- timm$data$create_loader(test_ds, c(3L, 224L, 224L), batch_size, num_workers = 8L)

#ims <- iter_next(as_iterator(test_dl))
#ims[[1]]$shape

img <- load.image(inat_test$file_name[378])
plot(img)

timg <- transform(torch$tensor(aperm(img, c(3, 4, 2, 1)), dtype = torch$float32))

mask <- att_rollout(timg$cuda())

numpy_image = np_array(timg$squeeze(0L)$cpu()$numpy())

# 4. Change from channels-first (CHW) to channels-last (HWC)
numpy_image = numpy_image$transpose(2L, 1L, 0L)

# 5. Denormalize if needed (example using ImageNet stats)
# Adjust these values based on the normalization used in your timm dataloader
mean = np_array(c(0.485, 0.456, 0.406))
std = np_array(c(0.229, 0.224, 0.225))
numpy_image = std * numpy_image + mean

# 6. Scale to [0, 255] and convert to uint8
numpy_image = (numpy_image * 255.0)$clip(0, 255)$astype(np$uint8)

# 7. Convert RGB to BGR for OpenCV
opencv_image = cv2$cvtColor(numpy_image, cv2$COLOR_RGB2BGR)

mask2 = cv2$resize(mask, c(numpy_image$shape[1], numpy_image$shape[0]))
#opencv_image = cv2$cvtColor(nimg*255, cv2$COLOR_RGB2BGR)
new_img = vit_explain$show_mask_on_image(opencv_image, mask2)

plot(as.cimg(new_img/255))

##### tests ############
test <- vit2$forward_intermediates(timg$cuda(), return_prefix_tokens = TRUE, output_fmt = 'NLC')
