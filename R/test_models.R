
library(torch)
library(torchvision)
library(GPUmatrix)

library(tidyverse)
library(pins)
library(arrow)
library(phyf)

library(tidymodels)
library(probably)

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
    transform_random_resized_crop(size = c(320, 320), scale = c(0.3, 0.7)) %>%
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
    transform_resize(364) %>%
    transform_center_crop(320) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))# %>%
    #(function(x) x$to(device = "cpu"))
}

test_transforms <- train_transforms

testing_image_dataset <- dataset(name = "pheno_image_ds",
                                 initialize = function(file_names, flowering, fruiting, phylo, transform = NULL,
                                                       loader = magick_loader,
                                                       rep = 10,
                                                       infinite = FALSE) {
                                 self$file_names <- file_names
                                 self$pheno <- cbind(flowering, fruiting)
                                 self$phylo <- phylo
                                 #self$device <- device

                                 self$transform <- transform
                                 self$loader <- loader
                                 self$rep <- rep
                                 self$infinite <- infinite
                                 },
  .getbatch = function(i) {
    path <- self$file_names[i]
    paths <- torch_tensor(rep(seq_along(path), each = self$rep))
    samples <- rep(map(path, ~ self$loader(.x)), each = self$rep)
    samples <- map(samples, ~ self$transform(.x)$unsqueeze(1))
    sample <- torch_cat(samples, dim = 1)#$to(device = self$device)

    pheno <- torch_tensor(self$pheno[i, ], dtype = torch_float32())#$to(device = self$device)
    i <- rep(i, each = self$rep)
    phylo <- gpu.matrix(self$phylo[i, ], dtype = "float32", device = "cpu")@gm

    list(pheno = pheno, sample = sample, phylo = phylo, paths = paths)
  },
  .length = function() {
    length(self$file_names)
  }
)

train_meta <- meta |>
  filter(split == "train")

# val_meta <- meta |>
#   filter(split == "val")

test_meta <- meta |>
  filter(split %in% c("test", "val"))

batch_size <- 24

predictions <- read_rds("data/preds_progress.rds")
pred_len <- nrow(do.call(rbind, predictions))

test_meta1 <- test_meta[1:pred_len, ]
test_meta2 <- test_meta[pred_len:nrow(test_meta), ]

test_meta2 <- test_meta2 |>
  filter(!file_name %in% c("data/inat/images/85640454_medium.png"
                           ))
test_meta <- test_meta2

# train_ds <- testing_image_dataset(train_meta$file_name,
#                                 train_meta$flowering,
#                                 train_meta$fruiting,
#                                 phylo = phylo_sparse[as.character(train_meta$photo_id), ],
#                                 transform = train_transforms)
#
# train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE,
#                        num_workers = 6, worker_packages = c("purrr", "torchvision", "GPUmatrix",
#                                                             "Matrix"),
#                        worker_init_fn = function(w) options(torch.serialization_version = 2))


# val_ds <- pheno_image_dataset(val_meta$file_name,
#                               val_meta$flowering,
#                               val_meta$fruiting,
#                               phylo = phylo_sparse[as.character(val_meta$photo_id), ],
#                               transform = val_transforms,
#                               infinite = TRUE)
#
# val_dl <- dataloader(val_ds, batch_size = batch_size, shuffle = TRUE,
#                        num_workers = 6, worker_packages = c("purrr", "torchvision", "GPUmatrix",
#                                                             "Matrix"),
#                        worker_init_fn = function(w) options(torch.serialization_version = 2))
#
reps <- 8
test_ds <- testing_image_dataset(test_meta$file_name,
                              test_meta$flowering,
                              test_meta$fruiting,
                              phylo = phylo_sparse[as.character(test_meta$photo_id), ],
                              transform = test_transforms,
                              rep = reps)

test_dl <- dataloader(test_ds, batch_size = batch_size,
                       num_workers = 4, worker_packages = c("purrr", "torchvision", "GPUmatrix",
                                                            "Matrix"),
                       worker_init_fn = function(w) options(torch.serialization_version = 2),
                      pin_memory = TRUE)

#batch <- test_dl$.iter()$.next()

mod <- torch_load("data/models/first_model_resnext50_10_epochs_with_last_layer_finetune_fixed.pt")
mod <- mod$cuda()
mod$eval()

library(cli)
library(plotly)

predictions <- list()
i <- length(predictions)
cli_progress_bar("Predicting", total = length(test_dl), clear = TRUE)
coro::loop(for (batch in test_dl) {

  i <- i + 1
  with_no_grad({
    preds <- mod(batch[[2]]$cuda(), batch[[3]]$cuda())
    ps <- torch_split(preds, reps)
    # dat <- torch_split(batch[[1]]$cuda(), 10) %>%
    #   map(~ .x$mean(dim = 1)$unsqueeze(1)) %>%
    #   torch_cat()

    psa <- map(ps, ~ nnf_sigmoid(.x$mean(dim = 1))$unsqueeze(1))
    res <- torch_cat(psa)
    predictions[[i]] <- as.matrix(res$cpu())

  })
  if(i %% 1000 == 0) {
    write_rds(predictions, "data/preds_progress2.rds")
  }
  cli_progress_update()

})

preds <- rbind(do.call(rbind, read_rds("data/preds_progress.rds")),
               do.call(rbind, read_rds("data/preds_progress2.rds"))) |>
  as.data.frame() |>
  rename(flowering_pred = V1, fruiting_pred = V2)

test_dat <- bind_rows(test_meta1,
                      test_meta2) |>
  bind_cols(preds) |>
  mutate(flowering = factor(flowering, levels = c("1", "0")),
         fruiting = factor(fruiting, levels = c("1", "0")))

######### basic stats ##############
test_dat <- test_dat |>
  mutate(
    .pred_fl = make_two_class_pred(
      estimate = flowering_pred,
      levels = levels(flowering),
      threshold = .5
    ),
    .pred_fr = make_two_class_pred(
      estimate = fruiting_pred,
      levels = levels(fruiting),
      threshold = .5
    )
  )

acc_fl <- accuracy(test_dat, flowering, .pred_fl)
acc_fr <- accuracy(test_dat, fruiting, .pred_fr)

test_dat %>%
  count(.truth = flowering, .pred_fl)

test_dat %>%
  count(.truth = fruiting, .pred_fr)

sensitivity(test_dat, flowering, .pred_fl)
sensitivity(test_dat, fruiting, .pred_fr)

specificity(test_dat, flowering, .pred_fl)
specificity(test_dat, fruiting, .pred_fr)

threshold_data_fl <- test_dat %>%
  threshold_perf(flowering, flowering_pred, thresholds = seq(0, 1, by = 0.005))

max_ji_fl <- threshold_data_fl |>
  filter(.metric == "j_index") |>
  slice_max(.estimate, n = 5)

plot(threshold_data_fl |> filter(.metric == "j_index") |> pull(.estimate))

threshold_data_fr <- test_dat %>%
  threshold_perf(fruiting, fruiting_pred, thresholds = seq(0, 1, by = 0.005))

max_ji_fr <- threshold_data_fr |>
  filter(.metric == "j_index") |>
  slice_max(.estimate, n = 5)

plot(threshold_data_fr |> filter(.metric == "j_index") |> pull(.estimate))

test_dat <- test_dat |>
  mutate(
    .pred_fl_max = make_two_class_pred(
      estimate = flowering_pred,
      levels = levels(flowering),
      threshold = max_ji_fl$.threshold[1],
      buffer = 0.025
    ),
    .pred_fr_max = make_two_class_pred(
      estimate = fruiting_pred,
      levels = levels(fruiting),
      threshold = max_ji_fr$.threshold[1],
      buffer = 0.025
    )
  )

accuracy(test_dat, flowering, .pred_fl_max)
accuracy(test_dat, fruiting, .pred_fr_max)

j_index(test_dat, flowering, .pred_fl_max)
j_index(test_dat, fruiting, .pred_fr_max)

test_dat %>%
  count(.truth = flowering, .pred_fl_max)

test_dat %>%
  count(.truth = fruiting, .pred_fr_max)

acc_fl <- test_dat |>
  group_by(genus) |>
  accuracy(flowering, .pred_fl_max)

acc_fr <- test_dat |>
  group_by(genus) |>
  accuracy(fruiting, .pred_fr_max)

tss_fl <- test_dat |>
  group_by(family) |>
  j_index(flowering, .pred_fl_max)

tss_fr <- test_dat |>
  group_by(family) |>
  j_index(fruiting, .pred_fr_max)

mean(tss_fl$.estimate, na.rm = TRUE)
median(tss_fl$.estimate, na.rm = TRUE)

mean(tss_fr$.estimate, na.rm = TRUE)
median(tss_fr$.estimate, na.rm = TRUE)

test_dat <- test_dat |>
  mutate(fl_fr = as.factor(paste(flowering, fruiting)),
         .pred_fl_fr = as.factor(paste(.pred_fl_max, .pred_fr_max)))

cm <- conf_mat(test_dat |> filter(!is.na(as.factor(.pred_fl_max)),
                                  !is.na(as.factor(.pred_fr_max))) |>
                 mutate(.pred_fl_fr = droplevels(.pred_fl_fr)),
               fl_fr,
               .pred_fl_fr)

autoplot(cm)
autoplot(cm, type = "heatmap")

cells_per_family <- test_dat |> filter(!is.na(as.factor(.pred_fl_max)),
                                       !is.na(as.factor(.pred_fr_max))) |>
  mutate(.pred_fl_fr = droplevels(.pred_fl_fr)) |>
  group_by(order, family) %>%
  conf_mat(fl_fr,
           .pred_fl_fr) |>
  mutate(tidied = lapply(conf_mat, tidy)) %>%
  unnest(tidied)

hist(acc_fl$.estimate, breaks = 50)
hist(acc_fr$.estimate, breaks = 50)

genus_summ <- train_meta |>
  group_by(order, family, genus) |>
  summarise(count = n(),
            imbalance_fl = (mean(as.numeric(as.character(flowering)), na.rm = TRUE) - 0.5) * 2,
            imbalance_fr = (mean(as.numeric(as.character(fruiting)), na.rm = TRUE) - 0.5) * 2,
            imb_fl_abs = abs(imbalance_fl),
            imb_fr_abs = abs(imbalance_fr)) |>
  left_join(acc_fl |> select(genus, acc_fl = .estimate)) |>
  left_join(acc_fr |> select(genus, acc_fr = .estimate))

family_summ <- genus_summ |>
  group_by(order, family) |>
  mutate(w = count / sum(count)) |>
  summarise(acc_fl = sum(acc_fl * w, na.rm = TRUE),
            acc_fr = sum(acc_fr * w, na.rm = TRUE),
            imbalance_fl = sum(imbalance_fl * w, na.rm = TRUE),
            imbalance_fr = sum(imbalance_fr * w, na.rm = TRUE),
            imb_fl = sum(imb_fl_abs * w, na.rm = TRUE),
            imb_fr = sum(imb_fr_abs * w, na.rm = TRUE),
            count = sum(count)) |>
  left_join(tss_fl |> select(family, tss_fl = .estimate)) |>
  left_join(tss_fr |> select(family, tss_fr = .estimate))

mod_fl <- lm(asin(acc_fl) ~ poly(count, imbalance_fl, imb_fl, degree = 3), data = family_summ)
summary(mod_fl)

mod_fr <- lm(asin(acc_fr) ~ poly(count, imbalance_fr, imb_fr, degree = 3), data = family_summ)
summary(mod_fr)

mod_fl2 <- glm(good ~ poly(count, imbalance_fl, degree = 3), data = family_summ |>
                mutate(good = ifelse(tss_fl > 0.6, 1, 0)),
              family = "binomial")
summary(mod_fl2)
cor(predict(mod_fl2, type = "response"), mod_fl2$y)^2

mod_fr <- lm(asin(acc_fr) ~ poly(count, imbalance_fr, imb_fr, degree = 3), data = family_summ)
summary(mod_fr)

family_preds_fl <- predict(mod_fl)
plot(sin(family_preds_fl), family_summ$acc_fl)
abline(0, 1)

family_preds_fr <- predict(mod_fr)
plot(sin(family_preds_fr), family_summ$acc_fr)
abline(0, 1)

family_summ <- family_summ |>
  ungroup() |>
  mutate(acc_fl_e = sin(family_preds_fl),
         acc_fr_e = sin(family_preds_fr))

test_dat <- test_dat |>
  mutate(acc_fl = 1 - abs(as.numeric(as.character(flowering)) - flowering_pred),
         acc_fr = 1 - abs(as.numeric(as.character(fruiting)) - fruiting_pred)) |>
  left_join(family_summ |> select(family, acc_fl_e, acc_fr_e)) |>
  mutate(acc_fl_diff = acc_fl - acc_fl_e,
         acc_fr_diff = acc_fr - acc_fr_e)

worst <- test_dat |>
  arrange(acc_fl_diff)

get_images <- function(file_names) {
  magick::image_read(file_names)
}

im <- get_images(worst$file_name[1:6])

try <- 12

plot(get_images(worst$file_name[try]))
worst$flowering[try]
worst$.pred_fl[try]
worst$fl_fr[try]
worst$.pred_fl_fr[try]

p <- ggplot(family_summ, aes(count, acc_fl)) +
  geom_point(aes(colour = order, text = family)) +
  geom_smooth(colour = "darkgreen", fill = "grey70") +
  theme_minimal()

p

plotly::ggplotly(p, tooltip = c("text", "order", "count"))

p_tss <- ggplot(family_summ, aes(count, tss_fl)) +
  geom_point(aes(colour = order, text = family)) +
  geom_smooth(colour = "darkgreen", fill = "grey70") +
  geom_hline(yintercept = c(0.6, 0.8)) +
  theme_minimal()

p_tss

plotly::ggplotly(p_tss, tooltip = c("text", "order", "count"))

p_tss <- ggplot(family_summ, aes(imbalance_fl, tss_fl)) +
  geom_point(aes(colour = order, text = family)) +
  geom_smooth(colour = "darkgreen", fill = "grey70") +
  geom_hline(yintercept = c(0.6, 0.8)) +
  theme_minimal()

p_tss

plotly::ggplotly(p_tss, tooltip = c("text", "order", "imb_fl"))

p1 <- ggplot(family_summ, aes(count, acc_fr)) +
  geom_point(aes(colour = order, text = family)) +
  geom_smooth(colour = "darkgreen", fill = "grey70") +
  theme_minimal()

p1

plotly::ggplotly(p1, tooltip = c("text", "order"))

p1_tss <- ggplot(family_summ, aes(count, tss_fr)) +
  geom_point(aes(colour = order, text = family)) +
  geom_smooth(colour = "darkgreen", fill = "grey70") +
  geom_hline(yintercept = c(0.6, 0.8)) +
  theme_minimal()

p1_tss

plotly::ggplotly(p1_tss, tooltip = c("text", "order", "count"))

p1_tss <- ggplot(family_summ, aes(imbalance_fr, tss_fr)) +
  geom_point(aes(colour = order, text = family)) +
  geom_smooth(colour = "darkgreen", fill = "grey70") +
  geom_hline(yintercept = c(0.6, 0.8)) +
  theme_minimal()

p1_tss

plotly::ggplotly(p1_tss, tooltip = c("text", "order", "count"))


p2 <- ggplot(family_summ, aes(acc_fr, acc_fl)) +
  geom_point(aes(colour = order, text = family, size = count)) +
  geom_smooth(colour = "darkgreen", fill = "grey70") +
  theme_minimal()

p2

plotly::ggplotly(p2, tooltip = c("text", "order", "count"))

p2_tss <- ggplot(family_summ, aes(tss_fr, tss_fl)) +
  geom_point(aes(colour = order, text = family, size = count)) +
  geom_smooth(colour = "darkgreen", fill = "grey70") +
  geom_hline(yintercept = c(0.6, 0.8)) +
  geom_vline(xintercept = c(0.6, 0.8)) +
  theme_minimal()

p2_tss

plotly::ggplotly(p2_tss, tooltip = c("text", "order", "count"))

p3 <- ggplot(family_summ |> filter(count > 100), aes(acc_fr, acc_fl)) +
  geom_point(aes(colour = order, text = family, size = count)) +
  geom_smooth(colour = "darkgreen", fill = "grey70", method = "lm") +
  theme_minimal()

p3

plotly::ggplotly(p3, tooltip = c("text", "order", "count"))




state <- mod$state_dict()
torch_save(state, "data/models/first_model_resnext50_10_epochs_with_last_layer_finetune_params.pt")
torch_save(mod, "data/models/first_model_resnext50_10_epochs_with_last_layer_finetune_fixed.pt")


im_model <- model_resnext50_32x4d(pretrained = TRUE)

## freeze model weights
im_model$parameters %>% purrr::walk(function(param) param$requires_grad_(FALSE))

## replace final fully connected layer with an identity function so
## we can just use the next to last output as input to a new model

#im_model$aux_logits <- FALSE

im_model$fc <- nn_identity()
im_model <- im_model$to(device = device)

#batch <- train_dl$.iter()$.next()
#test <- im_model$cpu()(batch[[2]]$cpu())
## gpu is not working, will have to prototype model on the cpu :(
#test <- im_model(batch[[2]]$cuda())
#test <- im_model(batch[[2]]$cpu())

get_conv_dims <- function(mod) {

}

ic_block <- function(in_features, out_features = ceiling(in_features / 2), p = 0.5) {
  nn_sequential(nn_batch_norm1d(in_features),
                nn_dropout(p),
                nn_linear(in_features, in_features),
                nn_gelu(),
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

                               e1 <- self$class_encode_layers$enc_1(x)
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

mod <- phylo_finetuner(ncol(phylo_sparse), 128, im_model, 2048)

mod <- mod$load_state_dict(state)
