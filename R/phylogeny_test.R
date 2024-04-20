library(ape)
library(tidyverse)

tree <- read.nexus("data/phylogeny/oo_330891.tre")
bad_images <- read_rds("data/inat/bad_images.rds")

inat_train <- read_csv("data/inat/train.csv") |>
  filter(!file_name %in% bad_images)

genera_in_tree <- unique(str_split_i(tree$tip.label, "_", 1))

prop_genera <- sum(unique(inat_train$genus) %in% genera_in_tree) / length(unique(inat_train$genus))
