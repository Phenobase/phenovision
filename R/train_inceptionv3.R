
library(torch)
library(torchvision)
library(GPUmatrix)

library(tidyverse)
library(pins)
library(arrow)
library(phyf)

options(torch.serialization_version = 2)

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
  filter(!file_name %in% c("data/inat/images/1358947_medium.png",
                           "data/inat/images/1880488_medium.PNG",
                           "data/inat/images/20629_medium.png",
                           "data/inat/images/20725_medium.png",
                           "data/inat/images/3536455_medium.jpg",
                           "data/inat/images/56513457_medium.jpeg"
                           ))

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
    #(function(x) x$to(device = "cuda")) %>%
    # data augmentation
    transform_random_resized_crop(size = c(299, 299), scale = c(0.7, 1)) %>%
    # data augmentation
    transform_color_jitter() %>%
    # data augmentation
    transform_random_horizontal_flip() %>%
    # normalize according to what is expected by resnet
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)) #%>%
    #(function(x) x$to(device = "cpu"))
}



val_transforms <- function(img) {
  img %>%
    transform_to_tensor() %>%
    #(function(x) x$to(device = "cuda")) %>%
    transform_resize(340) %>%
    transform_center_crop(299) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))# %>%
    #(function(x) x$to(device = "cpu"))
}

val_transforms <- train_transforms

test_transforms <- val_transforms

pheno_image_dataset <- dataset(name = "pheno_image_ds",
                               initialize = function(file_names, flowering, fruiting, phylo, transform = NULL,
                                                     loader = magick_loader,
                                                     infinite = FALSE) {
                                 self$file_names <- file_names
                                 self$pheno <- cbind(flowering, fruiting)
                                 self$phylo <- phylo
                                 #self$device <- device

                                 self$transform <- transform
                                 self$loader <- loader
                                 self$infinite <- infinite
                                 },
  .getbatch = function(i) {
    path <- self$file_names[i]
    samples <- map(path, ~ (self$loader(.x) |> self$transform())$unsqueeze(1))
    sample <- torch_cat(samples, dim = 1)#$to(device = self$device)

    phylo <- gpu.matrix(self$phylo[i, ], dtype = "float32", device = "cpu")@gm
    pheno <- torch_tensor(self$pheno[i, ], dtype = torch_float32())#$to(device = self$device)

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

batch_size <- 128

train_ds <- pheno_image_dataset(train_meta$file_name,
                                train_meta$flowering,
                                train_meta$fruiting,
                                phylo = phylo_sparse[as.character(train_meta$photo_id), ],
                                transform = train_transforms)

train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE,
                       num_workers = 6, worker_packages = c("purrr", "torchvision", "GPUmatrix",
                                                            "Matrix"),
                       worker_init_fn = function(w) options(torch.serialization_version = 2))

### test dataloader

#batch <- train_dl$.iter()$.next()
#itr <- train_dl$.iter()
#batch <- itr$.next()
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
                              infinite = TRUE)

val_dl <- dataloader(val_ds, batch_size = batch_size, shuffle = TRUE,
                       num_workers = 6, worker_packages = c("purrr", "torchvision", "GPUmatrix",
                                                            "Matrix"),
                       worker_init_fn = function(w) options(torch.serialization_version = 2))

test_ds <- pheno_image_dataset(test_meta$file_name,
                              test_meta$flowering,
                              test_meta$fruiting,
                              phylo = phylo_sparse[as.character(test_meta$photo_id), ],
                              transform = test_transforms)

test_dl <- dataloader(test_ds, batch_size = batch_size,
                       num_workers = 6, worker_packages = c("purrr", "torchvision", "GPUmatrix",
                                                            "Matrix"),
                       worker_init_fn = function(w) options(torch.serialization_version = 2))

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

im_model <- model_inception_v3(pretrained = TRUE)

## freeze model weights
im_model$parameters %>% purrr::walk(function(param) param$requires_grad_(FALSE))

## replace final fully connected layer with an identity function so
## we can just use the next to last output as input to a new model

#im_model$aux_logits <- FALSE

im_model$fc <- nn_identity()
im_model$AuxLogits$fc <- nn_identity()
im_model <- im_model$to(device = device)

#batch <- train_dl$.iter()$.next()
#test <- im_model$cpu()(batch[[2]]$cpu())
## gpu is not working, will have to prototype model on the cpu :(
#test <- im_model(batch[[2]]$cuda())
#test <- im_model(batch[[2]]$cpu())

get_conv_dims <- function(mod) {

}

ic_block <- function(in_features, out_features = ceiling(in_features / 2), p = 0.5) {
  nn_sequential(#nn_batch_norm1d(in_features),
                #nn_dropout(p),
                #nn_linear(in_features, in_features),
                #nn_gelu(),
                nn_batch_norm1d(in_features),
                nn_dropout(p),
                nn_linear(in_features, out_features),
                nn_gelu())
}

phylo_finetuner <- nn_module("phylo_finetuner",
                             initialize = function(embed_edges, embed_dim, im_model, im_model_output_dim, att_breadth = 64, class_breadth = 1024) {
                               self$phylo_embedding <- nn_parameter(torch_randn(embed_edges, embed_dim) * 0.01)
                               self$im_model <- im_model

                               input_sizes <- ceiling(im_model_output_dim / (2^c(0:2)))
                               output_sizes <- ceiling(im_model_output_dim / (2^c(1:3)))

                               names(input_sizes) <- paste0("enc_", 1:3)
                               names(output_sizes) <- paste0("dec_", 3:1)

                               p_input_sizes <- input_sizes
                               p_output_sizes <- output_sizes

                               p_input_sizes[1] <- embed_dim

                               self$p_input_sizes <- p_input_sizes
                               self$p_output_sizes <- p_output_sizes

                               self$input_sizes <- input_sizes
                               self$output_sizes <- output_sizes

                               self$class_encode_layers <- nn_module_dict(
                                 map2(input_sizes, output_sizes,
                                      ~ ic_block(.x, .y, p = 0.05))
                               )

                               self$class_decode_layers <- nn_module_dict(
                                 map2(rev(output_sizes), rev(input_sizes),
                                      ~ ic_block(.x, .y, p = 0.05))
                               )

                               self$phylo_encode_layers <- nn_module_dict(
                                 map2(p_input_sizes, p_output_sizes,
                                      ~ ic_block(.x, .y, p = 0.05))
                               )

                               self$phylo_decode_layers <- nn_module_dict(
                                 map2(rev(p_output_sizes), rev(input_sizes),
                                      ~ ic_block(.x, .y, p = 0.05))
                               )

                               self$final_fc <- nn_linear(im_model_output_dim, 2)
                               #self$final_activation <- nn_sigmoid()
                             },
                             forward = function(x, phylo) {

                               phylo_layer <- call_torch_function("torch__sparse_mm", phylo, self$phylo_embedding, quiet = TRUE)
                               x <- self$im_model(x)

                               e1 <- self$class_encode_layers$enc_1(torch_cat(x, dim = 2L))
                               pe1 <- self$phylo_encode_layers$enc_1(phylo_layer)

                               e2 <- self$class_encode_layers$enc_2(e1 * pe1)
                               pe2 <- self$phylo_encode_layers$enc_2(pe1)

                               e3 <- self$class_encode_layers$enc_3(e2 * pe2)
                               pe3 <- self$phylo_encode_layers$enc_3(pe2)

                               d1 <- self$class_decode_layers$dec_1((e3 * pe3))
                               pd1 <- self$phylo_decode_layers$dec_1(pe3)

                               d2 <- self$class_decode_layers$dec_2((d1 * pd1) + e2)
                               pd2 <- self$phylo_decode_layers$dec_2(pd1 + pe2)

                               d3 <- self$class_decode_layers$dec_3((d2 * pd2) + e1)
                               pd3 <- self$phylo_decode_layers$dec_3(pd2 + pe1)

                               x <- self$final_fc(d3 * pd3)
                               #self$final_activation(x)
                               x

                             })

mod <- phylo_finetuner(ncol(phylo_sparse), 128, im_model, 2048 + 768)
mod <- mod$to(device = device)

# test <- mod(batch[[2]]$cpu(), batch[[3]]$cpu())
# loss_fun <- nn_bce_with_logits_loss()
# loss <- loss_fun(test, batch[[1]]$cpu())
# l2_loss <- torch_mean(torch_norm(mod$phylo_embedding, dim = 2))

####### training loop ############

loss_fun <- nn_bce_with_logits_loss()

n_epoch <- 10
lr <- 0.005
#save_every <- 50

optim1 <- optim_adam(mod$parameters, lr = lr)
scheduler <- lr_one_cycle(optim1, max_lr = lr,
                          epochs = n_epoch, steps_per_epoch = length(train_dl),
                          cycle_momentum = FALSE)

phylo_w <- 0.001
unfreeze_epoch <- 3

val_iter <- val_dl$.iter()

for(epoch in 1:n_epoch) {

  if(epoch == unfreeze_epoch) {
    mod$im_model$parameters[grep("Mixed_7", names(mod$im_model$parameters), fixed = TRUE)] %>%
      purrr::walk(function(param) param$requires_grad_(TRUE))
  }
  batch_num <- 0
  binary_losses <- c()
  phylo_losses <- c()
  valid_losses <- c()
  valid_losses <- c()
  valid_accs_fl <- c()
  valid_accs_fr <- c()
  coro::loop(for (b in train_dl) {
    batch_num <- batch_num + 1

    optim1$zero_grad()
    b_phylo <- b[[3]]$to(device = device)
    res <- mod(b[[2]]$to(device = device), b_phylo)

    binary_loss <- loss_fun(res, b[[1]]$to(device = device))

    phylo_mask <- torch_max(b_phylo$to_dense(), dim = 1)[[1]]
    new_embed <- mod$phylo_embedding * phylo_mask$unsqueeze(-1)
    phylo_loss <- torch_sum(new_embed^2) / batch_size

    loss <- binary_loss + phylo_w * phylo_loss


    loss$backward()
    optim1$step()
    scheduler$step()

    binary_losses <- c(binary_losses, as.numeric(binary_loss$cpu()))
    phylo_losses <- c(phylo_losses, as.numeric(phylo_loss$cpu()))

    if(batch_num %% 10 == 0) {
      mod$eval()
      mod$im_model$training <- TRUE
      v <- val_iter$.next()
      if(rlang::is_symbol(v)) {
        val_iter <- val_dl$.iter()
        v <- val_iter$.next()
      }
      with_no_grad({
        v1 <- v[[1]]$to(device = device)
        res2 <- mod(v[[2]]$to(device = device), v[[3]]$to(device = device))
        valid_loss <- loss_fun(res2, v1)
        valid_acc <- (1 - torch_mean(torch_abs(torch_round(nnf_sigmoid(res2)) - v1), dim = 1))
        valid_losses <- c(valid_losses, as.numeric(valid_loss$cpu()))
        valid_accs_fl <- c(valid_accs_fl, as.numeric(valid_acc[1]$cpu()))
        valid_accs_fr <- c(valid_accs_fr, as.numeric(valid_acc[2]$cpu()))

      })
      cat("Epoch: ", epoch,
          "Batch: ", batch_num,
          "    mean binary loss: ", mean(binary_losses),
          "    mean phylo loss: ", mean(phylo_losses),
          "    validation loss: ", mean(valid_losses),
          "    flowering validation accuracy: ", mean(valid_accs_fl),
          "    fruiting validation accuracy: ", mean(valid_accs_fr),
          "\n")
      mod$train()
    }

  })


}

torch_save(mod, "data/models/first_model_inceptionv3_10_epochs_with-finetune.to")



