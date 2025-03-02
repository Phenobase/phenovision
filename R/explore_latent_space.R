library(tidyverse)
library(phateR)
library(torch)
library(ape)
library(ggtree)
library(scico)
library(phytools)

set.seed(47856)

code_df <- read_rds("output/model_04_13_2024/mae_codes_w_meta.rds")
tree <- read.tree("data/phylogeny/genus_tree_inat2.tre")

n_distinct(code_df$family)
n_distinct(code_df$genus)

code_samp <- code_df |>
  group_by(family, genus, flowering, fruiting) |>
  slice_sample(n = 50)

code_samp <- code_samp |> ungroup()

code_samp <- code_samp |>
  filter(genus %in% tree$tip.label)

tree2 <- drop.tip(tree, which(!tree$tip.label %in% unique(code_samp$genus)))

code_phate <- phate(code_samp |> select(starts_with("MAE")) |> as.matrix(),
                    gamma = 0,
                    n.jobs = 6)

code_phate_3d <- phate(code_samp |> select(starts_with("MAE")) |> as.matrix(),
                       ndim = 3,
                       gamma = 0,
                       n.jobs = 6)

write_rds(code_phate, "output/model_04_13_2024/mae_samp_phate_gamma_0.rds")
write_rds(code_phate_3d, "output/model_04_13_2024/mae_samp_phate_gamma_0_3d.rds")

phate_df <- code_phate$embedding |>
  as.data.frame() |>
  bind_cols(code_samp) |>
  bind_cols(code_phate_3d$embedding |>
              as.data.frame() |>
              rename(PHATE_1 = PHATE1,
                     PHATE_2 = PHATE2,
                     PHATE_3 = PHATE3))

write_rds(phate_df, "output/model_04_13_2024/mae_samp_phate_df_gamma_0_w_3d.rds")

genera <- phate_df |>
  select(order, family, genus) |>
  distinct(genus, .keep_all = TRUE)

genera <- genera |>
  filter(genus %in% tree$tip.label)

sum(genera$genus %in% tree$tip.label)

families <- genera |>
  group_by(order, family) |>
  summarise(count = n()) |>
  group_by(order) |>
  mutate(order_sum = sum(count))

families_label <- families |>
  filter(count >= 100)
orders_label <- families |>
  filter(count < 100 & order_sum > 100) |>
  distinct(order, .keep_all = TRUE)

# phate_df <- phate_df |>
#   filter(genus %in% tree$tip.label)

tree_df <- fortify(tree)

tree_p <- ggtree(tree_df, aes(colour = y), layout = "circular") +
  scale_color_scico(palette = "hawaii") +
  theme(legend.position = 'none')

tree_p

phate_df <- phate_df |>
  left_join(tree_df, by = c("genus" = "label"))

pal <- c("0" = "#B7E1A1", "1" = "#8A6B9E")
leg <- ggplot(tibble(x = c(1, 1, 1, 1),
                     y = c(1, 2, 3, 4),
                     flowering = c(0, 1, 0, 1),
                     fruiting = c(0, 0, 1, 1),
                     label = c("No Flower, No Fruit",
                               "Flower, No Fruit",
                               "Fruit, No Flower",
                               "Flower and Fruit")),
              aes(x, y)) +
  annotate("rect", xmin = 0.9, xmax = 1.1, ymin = 0, ymax = 5, fill = "grey90") +
  geom_point(aes(fill = factor(flowering), colour = factor(fruiting)), alpha = 0.7,
             shape = 21, size = 7) +
  geom_text(aes(x = x + 0.5, label = label), size = 8) +
  scale_colour_manual(values = c("0" = "white", "1" = "black")) +
  scale_fill_manual(values = pal) +
  scale_x_continuous(expand = c(0.75, 0.75)) +
  scale_y_continuous(expand = c(0.05, 0.05)) +
  theme_void() +
  theme(legend.position = 'none')

p <- ggplot(phate_df, aes(PHATE1, PHATE2)) +
  geom_point(aes(fill = factor(flowering), colour = factor(fruiting)), alpha = 0.7,
             shape = 21) +
  scale_colour_manual(values = c("0" = "white", "1" = "black")) +
  scale_fill_manual(values = pal) +
  guides(custom = guide_custom(ggplotGrob(leg), title = "Phenology Annotations",
                               width = grid::unit(0.8, "npc"),
                               height = grid::unit(0.2, "npc")),
         colour = 'none', fill = 'none') +
  theme_minimal() +
  theme(legend.position = "inside",
        legend.position.inside = c(0.1, 0.15),
        legend.title = element_text(hjust = 0.5, size = 28),
        axis.title = element_text(size = 28),
        axis.text = element_text(size = 24),
        legend.text = element_text(size = 24))

ragg::agg_supertransparent("figures/MAE_PHATE.png", scaling = 1.5,
                           width = 1800, height = 1600)
p
dev.off()


ggplot(phate_df, aes(PHATE_1, PHATE_2)) +
  geom_point(aes(colour = y), alpha = 0.2) +
  scale_color_scico(palette = "hawaii") +
  theme_minimal() +
  theme(legend.position = 'none')

ggplot(phate_df, aes(PHATE_3, PHATE_2)) +
  geom_point(aes(colour = y), alpha = 0.2) +
  scale_color_scico(palette = "hawaii") +
  theme_minimal() +
  theme(legend.position = 'none')





get_tips <- function(x) {
  tips <- genera |>
    filter(family == x) |>
    distinct(genus) |>
    pull(genus) |>
    str_replace_all(" ", "_")
  #tips
  findMRCA(tree, tips)
}
fam_tips <- purrr::map_int(families_label$family,
                ~ try(get_tips(.x)),
                .progress = TRUE)
get_tips_order <- function(x) {
  tips <- genera |>
    filter(order == x) |>
    distinct(genus) |>
    pull(genus) |>
    str_replace_all(" ", "_")
  #tips
  findMRCA(tree, tips)
}
order_tips <- purrr::map_int(families_label$order,
                           ~ try(get_tips_order(.x)),
                           .progress = TRUE)

####### plot #######
tree_p <- ggtree(tree, aes(colour = y), layout = "circular") +
  scale_color_scico(palette = "hawaii") +
  theme(legend.position = 'none')

tree_p

for(i in seq_along(fam_tips)) {

  tree_p <- tree_p +
    geom_cladelab(fam_tips[i], label = families_label$family[i], angle = "auto")

}
