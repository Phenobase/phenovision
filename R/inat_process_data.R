library(tidyverse)
library(tidymodels)
library(arrow)
library(phyf)

meta <- read_parquet("data/inat/inaturalist-phenobase-data.parquet")
meta <- meta |>
  mutate(file_name = file.path("data", "inat", "images", paste0(photo_id, "_medium.", extension))) |>
  select(photo_id, class, order, family, genus, split, flowering, fruiting,
         file_name) |>
  drop_na() |>
  mutate(fruit_flower = paste(fruiting, flowering, sep = "_"),
         fruit_flower_taxa = paste(fruit_flower, genus, sep = "_"))

image_files <- list.files("data/inat/images", full.names = TRUE)
missing <- !meta$file_name %in% image_files
meta <- meta[!missing, ]

## remove some wonky images
# meta <- meta |>
#   filter(!file_name %in% c("data/inat/images/1358947_medium.png",
#                            "data/inat/images/1880488_medium.PNG",
#                            "data/inat/images/20629_medium.png",
#                            "data/inat/images/20725_medium.png",
#                            "data/inat/images/3536455_medium.jpg",
#                            "data/inat/images/56513457_medium.jpeg"
#   ))

inat_split <- initial_validation_split(meta, strata = fruit_flower_taxa)

inat_train <- training(inat_split)
inat_val <- validation(inat_split)
inat_test <- testing(inat_split)

write_csv(inat_train, "data/inat/train.csv")
write_csv(inat_val, "data/inat/validation.csv")
write_csv(inat_test, "data/inat/test.csv")

#### create sparse phylogenetic matrix from taxonomy
class_pfc <- pf_as_pfc(as.factor(meta$class))
order_pfc <- pf_as_pfc(as.factor(meta$order))
family_pfc <- pf_as_pfc(as.factor(meta$family))
genus_pfc <- pf_as_pfc(as.factor(meta$genus))
id_pfc <- pf_as_pfc(as.factor(meta$photo_id))

phylo_sparse <- cbind(pf_as_sparse(class_pfc), pf_as_sparse(order_pfc),
                      pf_as_sparse(family_pfc), pf_as_sparse(genus_pfc))
rownames(phylo_sparse) <- meta$photo_id

# ps2 <- cbind(phylo_sparse, pf_as_sparse(id_pfc))
# pf <- pf_as_pfc(ps2, is_tip = rep(TRUE, nrow(ps2)),
#                  internal = !colnames(ps2) %in% unique(meta$photo_id))
# phy <- pf_as_phylo(pf)
# ape::write.tree(phy, "data/inat/taxo_tree.tre")
# phy <- ape::read.tree("data/inat/taxo_tree.tre")
# plot(phy)

##### collect imbalance data ##########
count_df <- meta |>
  group_by(class, order, family, genus) |>
  summarise(fruit = sum(fruiting), flowers = sum(flowering),
            both = sum(fruiting * flowering),
            neither = sum((1 - fruiting) * (1 - flowering)),
            total = n())

g_class_pfc <- pf_as_pfc(as.factor(count_df$class))
g_order_pfc <- pf_as_pfc(as.factor(count_df$order))
g_family_pfc <- pf_as_pfc(as.factor(count_df$family))
g_genus_pfc <- pf_as_pfc(as.factor(count_df$genus))

g_phylo_sparse <- cbind(pf_as_sparse(g_class_pfc), pf_as_sparse(g_order_pfc),
                        pf_as_sparse(g_family_pfc), pf_as_sparse(g_genus_pfc))


train_df <- inat_train |>
  group_by(class, order, family, genus) |>
  summarise(fruit = sum(fruiting), flowers = sum(flowering),
            both = sum(fruiting * flowering),
            neither = sum((1 - fruiting) * (1 - flowering)),
            total = n())

val_df <- inat_val |>
  group_by(class, order, family, genus) |>
  summarise(fruit = sum(fruiting), flowers = sum(flowering),
            both = sum(fruiting * flowering),
            neither = sum((1 - fruiting) * (1 - flowering)),
            total = n())


train_ord_df <- inat_train |>
  group_by(class, order) |>
  summarise(fruit = sum(fruiting), flowers = sum(flowering),
            both = sum(fruiting * flowering),
            neither = sum((1 - fruiting) * (1 - flowering)),
            total = n()) |>
  ungroup() |>
  mutate(prop = total / sum(total))

val_ord_df <- inat_val |>
  group_by(class, order) |>
  summarise(fruit = sum(fruiting), flowers = sum(flowering),
            both = sum(fruiting * flowering),
            neither = sum((1 - fruiting) * (1 - flowering)),
            total = n()) |>
  ungroup() |>
  mutate(prop = total / sum(total))

overall_balance <- meta |>
  summarise(flower = mean(flowering),
            fruit = mean(fruiting))
