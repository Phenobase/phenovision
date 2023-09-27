
library(torch)
library(torchvision)
library(GPUmatrix)

library(tidyverse)
library(pins)
library(arrow)
library(phyf)

meta <- read_parquet("data/inat/inaturalist-phenobase-data.parquet")
meta <- meta |>
  mutate(file_name = file.path("data", "inat", "images", paste0(photo_id, "_medium.", extension))) |>
  select(photo_id, class, order, family, genus, split, flowering, fruiting,
         file_name) |>
  drop_na()

image_files <- list.files("data/inat/images", full.names = TRUE)
missing <- !meta$file_name %in% image_files
meta <- meta[!missing, ]

## remove some wonky images
meta <- meta |>
  filter(!file_name %in% c("data/inat/images/50430_medium.png"))

#### create sparse phylogenetic matrix from taxonomy
class_pfc <- pf_as_pfc(as.factor(meta$class))
order_pfc <- pf_as_pfc(as.factor(meta$order))
family_pfc <- pf_as_pfc(as.factor(meta$family))
genus_pfc <- pf_as_pfc(as.factor(meta$genus))

phylo_sparse <- cbind(pf_as_sparse(class_pfc), pf_as_sparse(order_pfc),
                      pf_as_sparse(family_pfc), pf_as_sparse(genus_pfc))
rownames(phylo_sparse) <- meta$photo_id

device <- if (cuda_is_available()) "cuda" else "cpu"

train_transforms <- function(img) {
  img %>%
    # first convert image to tensor
    transform_to_tensor() %>%
    # data augmentation
    transform_random_resized_crop(size = c(224, 224), scale = c(0.7, 1)) %>%
    # data augmentation
    transform_color_jitter() %>%
    # data augmentation
    transform_random_horizontal_flip() %>%
    # normalize according to what is expected by resnet
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}



val_transforms <- function(img) {
  img %>%
    transform_to_tensor() %>%
    transform_resize(256) %>%
    transform_center_crop(224) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}

test_transforms <- val_transforms

pheno_image_dataset <- dataset(name = "pheno_image_ds",
                               initialize = function(file_names, flowering, fruiting, phylo, transform = NULL,
                                                     loader = base_loader, device = "cpu") {
                                 self$file_names <- file_names
                                 self$flowering <- torch_tensor(flowering, dtype = torch_float32())
                                 self$fruiting <- torch_tensor(fruiting, dtype = torch_float32())
                                 self$pheno <- torch_cat(list(self$flowering$unsqueeze(-1), self$fruiting$unsqueeze(-1)), dim = 2L)
                                 self$phylo <- phylo
                                 self$device <- device

                                 self$transform <- transform
                                 self$loader <- loader
                                 },
  .getbatch = function(i) {
    path <- self$file_names[i]
    samples <- map(path, ~ (self$loader(.x)[ , , 1:3] |> self$transform())$unsqueeze(1))
    sample <- torch_cat(samples, dim = 1)$to(device = device)

    phylo <- gpu.matrix(self$phylo[i, ], dtype = "float32", device = device)@gm
    pheno <- self$pheno[i, ]$to(device = device)

    list(pheno = pheno, sample = sample, phylo = phylo)
  },
  .length = function() {
    length(self$file_names)
  }
)

train_meta <- meta |>
  filter(split == "train")

val_meta <- meta |>
  filter(split == "val")

test_meta <- meta |>
  filter(split == "test")

batch_size <- 64

train_ds <- pheno_image_dataset(train_meta$file_name,
                                train_meta$flowering,
                                train_meta$fruiting,
                                phylo = phylo_sparse[as.character(train_meta$photo_id), ],
                                transform = train_transforms,
                                device = device)

train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)

### test dataloader

# batch <- train_dl$.iter()$.next()
#
# images <- as_array(batch[[2]]$cpu()) %>% aperm(perm = c(1, 3, 4, 2))
# mean <- c(0.485, 0.456, 0.406)
# std <- c(0.229, 0.224, 0.225)
# images <- std * images + mean
# images <- images * 255
# images[images > 255] <- 255
# images[images < 0] <- 0
#
# plot(as.raster(images[10, , ,], max = 255))

val_ds <- pheno_image_dataset(val_meta$file_name,
                              val_meta$flowering,
                              val_meta$fruiting,
                              phylo = phylo_sparse[as.character(val_meta$photo_id), ],
                              transform = val_transforms,
                              device = device)

val_dl <- dataloader(val_ds, batch_size = batch_size)

test_ds <- pheno_image_dataset(test_meta$file_name,
                              test_meta$flowering,
                              test_meta$fruiting,
                              phylo = phylo_sparse[as.character(test_meta$photo_id), ],
                              transform = test_transforms,
                              device = device)

test_dl <- dataloader(test_ds, batch_size = batch_size)

# batch <- test_dl$.iter()$.next()
# images <- as_array(batch[[2]]$cpu()) %>% aperm(perm = c(1, 3, 4, 2))
# mean <- c(0.485, 0.456, 0.406)
# std <- c(0.229, 0.224, 0.225)
# images <- std * images + mean
# images <- images * 255
# images[images > 255] <- 255
# images[images < 0] <- 0
#
# plot(as.raster(images[10, , ,], max = 255))

#### setup model:: we will start by trying resnet18

im_model <- model_resnext50_32x4d(pretrained = TRUE)

## freeze model weights
im_model$parameters %>% purrr::walk(function(param) param$requires_grad_(FALSE))

## replace final fully connected layer with an identity function so
## we can just use the next to last output as input to a new model

im_model$fc <- nn_identity()
#im_model <- im_model$to(device = device)

batch <- train_dl$.iter()$.next()
#test <- im_model$cpu()(batch[[2]]$cpu())
## gpu is not working, will have to prototype model on the cpu :(
#test <- im_model(batch[[2]])
test <- im_model(batch[[2]]$cpu())

phylo_finetuner <- nn_module("phylo_finetuner",
                             initialize = function(embed_edges, embed_dim, im_model, im_model_output_dim, att_breadth = 32) {
                               self$phylo_embedding <- nn_parameter(torch_randn(embed_edges, embed_dim) * 0.01)
                               self$im_model <- im_model
                               self$attention_nn <- nn_sequential(nn_linear(embed_dim, att_breadth),
                                                                  nn_tanh(),
                                                                  nn_linear(att_breadth, att_breadth),
                                                                  nn_tanh(),
                                                                  nn_linear(att_breadth, im_model_output_dim))
                               self$final_fc <- nn_linear(im_model_output_dim, 2)
                               #self$final_activation <- nn_sigmoid()
                             },
                             forward = function(x, phylo) {

                               phylo_layer <- call_torch_function("torch__sparse_mm", phylo, self$phylo_embedding, quiet = TRUE)
                               x <- self$im_model(x)
                               x <- x * self$attention_nn(phylo_layer)
                               x <- self$final_fc(x)
                               #self$final_activation(x)
                               x

                             })

mod <- phylo_finetuner(ncol(phylo_sparse), 64, im_model, 512)

# test <- mod(batch[[2]]$cpu(), batch[[3]]$cpu())
# loss_fun <- nn_bce_with_logits_loss()
# loss <- loss_fun(test, batch[[1]]$cpu())
# l2_loss <- torch_mean(torch_norm(mod$phylo_embedding, dim = 2))

####### training loop ############

loss_fun <- nn_bce_with_logits_loss()

n_epoch <- 10000
lr <- 0.05
#save_every <- 50

optim1 <- optim_adam(mod$parameters, lr = lr)
scheduler <- lr_one_cycle(optim1, max_lr = lr,
                          epochs = n_epoch, steps_per_epoch = length(train_dl),
                          cycle_momentum = FALSE)

phylo_w <- 0.001

for(epoch in 1:n_epoch) {


  batch_num <- 0
  coro::loop(for (b in train_dl) {
    batch_num <- batch_num + 1

    optim1$zero_grad()
    res <- mod(b[[2]]$cpu(), b[[3]]$cpu())

    binary_loss <- loss_fun(res, b[[1]]$cpu())
    phylo_loss <- torch_mean(torch_norm(mod$phylo_embedding, dim = 2))

    loss <- binary_loss + phylo_w * phylo_loss

    loss$backward()
    optim1$step()
    scheduler$step()

    if(batch_num %% 10 == 0) {
      cat("Epoch: ", epoch,
          "Batch: ", batch_num,
          "    binary loss: ", as.numeric(binary_loss$cpu()),
          "    phylo loss: ", as.numeric(phylo_loss$cpu()),
          "\n")
    }

  })


}



