library(tidyverse)

fam_stats <- read_csv("output/family_stats.csv") |>
  filter(test == "test")
fam_stats_fl <- fam_stats |>
  select(family, equiv_prop_fl, .accuracy_family_flower,
         .accuracy_family_flower_incl_equiv) |>
  mutate(trait = "flower")
fam_stats_fr <- fam_stats |>
  select(family, equiv_prop_fr, .accuracy_family_fruit,
         .accuracy_family_fruit_incl_equiv) |>
  mutate(trait = "fruit")
fam_stats2 <- bind_rows(fam_stats_fl |>
                          rename(proportion_low_certainty_family = equiv_prop_fl,
                                 accuracy_excluding_low_certainty_family = .accuracy_family_flower,
                                 accuracy_family = .accuracy_family_flower_incl_equiv),
                        fam_stats_fr |>
                          rename(proportion_low_certainty_family = equiv_prop_fr,
                                 accuracy_excluding_low_certainty_family = .accuracy_family_fruit,
                                 accuracy_family = .accuracy_family_fruit_incl_equiv))
## regenerate full data with header this time and update family stats
file_name <- "output/annotations_all_w_headers_9cf8ad8.csv"
annotation_outputs <- list.files("output", pattern = "inference_results_", full.names = TRUE)

x <- annotation_outputs[[1]]
fix_and_save <- function(x, file_name, append = TRUE) {
  d <- read_csv(x)
  d <- d |>
    select(-proportion_low_certainty_family,
           -accuracy_excluding_low_certainty_family,
           -accuracy_family) |>
    left_join(fam_stats2)

  write_csv(d, file_name, append = append)
  rm(d)
  gc()
}

fix_and_save(annotation_outputs[[1]], file_name, append = FALSE)
gc()
walk(annotation_outputs[-1], ~ fix_and_save(.x, file_name), .progress = TRUE)

dat <- read_csv(file_name, n_max = 10000)
dat

######## get test data full metadata ##########

# library(tidyverse)
# test_dat <- read_rds("output/epoch_4_testing_data.rds") |>
#   filter(partition == "testing")
# annotation_outputs <- list.files("output/inference_full", pattern = "inference_results_", full.names = TRUE)
#
# x <- annotation_outputs[[1]]
# get_data <- function(x) {
#   d <- read_csv(x)
#   photo_ids <- str_remove(d$observed_image_url, "https://www.inaturalist.org/photos/")
#   d_new <- d |>
#     mutate(photo_id = photo_ids) |>
#     filter(photo_id %in% as.character(test_dat$photo_id)) |>
#     select(photo_id, longitude, latitude, genus)
#   rm(d)
#   gc()
#   d_new
# }
#
# test_spat <- map(annotation_outputs, get_data, .progress = TRUE)
# test_spat <- list_rbind(test_spat)
# write_csv(test_spat, "output/test_spat_data.csv")
#
# fam_stats <- read_csv("output/family_stats.csv") |>
#   filter(test == "test") |>
#   select(-test)

## Figures based on Daijiang's
library(ape)
library(treeio)
library(ggtree)
library(ggnewscale)
library(ggtreeExtra)
genus_tree_inat2 = read.tree("data/genus_tree_inat2.tre")

coverage_inat = read_csv("output/data_for_coverage_analysis_9cf8ad8.csv")

coverage_inat = mutate(coverage_inat, trait2 = ifelse(trait %in% c("flower", "flowering"), "Flower", "Fruit"))
count(coverage_inat, trait2)

fam <- read_csv("output/family_stats.csv") |>
  filter(test == "test")

count(fam, test)

fam_genus = read_rds("output/epoch_4_testing_data.rds") |>
  select(family, genus) |>
  distinct()

n_distinct(fam_genus$family) # 408
n_distinct(filter(fam_genus, genus %in% genus_tree_inat2$tip.label)$family) # 402

fam_genus2 = filter(fam_genus, genus %in% genus_tree_inat2$tip.label) |>
  group_by(family) |>
  sample_n(1) |> ungroup()

setdiff(fam$family, fam_genus2$family)

fam_genus2 = filter(fam_genus2, family %in% filter(fam, test == "test")$family)

fam_tree_inat = keep.tip(genus_tree_inat2, fam_genus2$genus)
plot(fam_tree_inat)

fam_tree_inat$tip.label = tibble(genus = fam_tree_inat$tip.label) |>
  left_join(fam_genus2) |> pull(family)

# number of total records?
coverage_inat
n_total_family = filter(coverage_inat, family %in% fam_tree_inat$tip.label) |>
  mutate(trait = recode(trait, "flowering" = "flower", "fruiting" = "fruit")) |>
  group_by(trait, family) |> tally() |>
  ungroup() |> pivot_wider(names_from = "trait", values_from = "n", values_fill = 0)
n_total_family = rename(n_total_family, n_total_flower = flower, n_total_fruit = fruit)

d_family_flower = filter(fam, test == "test", count >= 10) |>
  select(-test) |>
  left_join(n_total_family) |>
  arrange(desc(n_total_flower))

sort(d_family_flower$.accuracy_family_flower)

d_family_flower = drop_na(d_family_flower, .accuracy_family_flower)

fam_tree_inat_flower = keep.tip(fam_tree_inat, d_family_flower$family)

fam_flower_tbl_tree = as.treedata(as_tibble(fam_tree_inat_flower) |> full_join(
  d_family_flower |>
    rename(label = family) |>
    mutate(n_total_flower_log = log10(n_total_flower))
) |> distinct())

#' Highlight all branches linking a subset of species in a phylogeny
#'
#' @param phy The large phylogeny to plot.
#' @param subset_sp A vector of species names to be highlighted on the plot of phylogeny.
#' @param highlight_color The color to highlight the branches, the default is red.
#' @return A `ggplot2` object.
#'
highlight_subset_sp_in_phylogeny = function(phy, subset_sp, highlight_color = "red", layout = "fan"){
  phy_df = tidytree::as_tibble(phy)
  sp_node = filter(phy_df, label %in% subset_sp)$node
  # all possible combinations
  sp_comb = as.data.frame(t(combn(as.character(sp_node), 2))) %>%
    set_names(c("s1", "s2")) %>%
    mutate(s1 = as.integer(s1), s2 = as.integer(s2))
  nodes_to_connect = vector("list", length = nrow(sp_comb))
  for(i in 1:nrow(sp_comb)){
    nodes_to_connect[[i]] = ggtree::get.path(phy, sp_comb[i, 1], sp_comb[i, 2])
  }
  nodes_to_highlight = unique(unlist(nodes_to_connect))

  phy_df2 = mutate(phy_df, member = ifelse(node %in% nodes_to_highlight, "Yes", "No")) %>%
    tidytree::as.treedata()

  p = ggtree(phy_df2, layout = layout, aes(color = member)) +
    scale_color_manual("Top 20 family", values = c("No" = "black","Yes" = highlight_color))

  return(p)
}

p1 = highlight_subset_sp_in_phylogeny(fam_flower_tbl_tree, d_family_flower$family[1:20], layout = "fan")

p2 = p1 + new_scale_color() + geom_tippoint(aes(color = n_total_flower_log), size = 1) +
  scale_color_viridis_c() + labs(color = "Number of total \nflower records (log10)")

p3 = p2 +
  geom_fruit(data = rename(d_family_flower, Accuracy = .accuracy_family_flower, id = family) |>
               select(id, Accuracy),
             geom = geom_col, size = 0.5,
             aes(x = Accuracy, y = id), color = NA, fill = "lightblue", alpha = 0.6,
             axis.params=list(
               axis = "x", text.size = 2.3, vjust = 0.3, nbreak = 4
             ),
             grid.params=list())

p4 = p3 +
  theme(legend.position = c(0.83, 0.9), legend.box = "horizontal",
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 13))

ggsave("figures/inat_flower_accuracy.pdf", plot = p4, height = 11, width = 11)

# =====

d_family_fruit = filter(fam, test == "test", count >= 10) |>
  select(-test) |>
  left_join(n_total_family) |>
  arrange(desc(n_total_fruit))

sort(d_family_fruit$.accuracy_family_flower)

d_family_fruit = drop_na(d_family_flower, .accuracy_family_fruit)

fam_tree_inat_fruit = keep.tip(fam_tree_inat, d_family_fruit$family)

fam_fruit_tbl_tree = as.treedata(as_tibble(fam_tree_inat_fruit) |> full_join(
  d_family_fruit |>
    rename(label = family) |>
    mutate(n_total_fruit_log = log10(n_total_fruit))
) |> distinct())

p1 = highlight_subset_sp_in_phylogeny(fam_fruit_tbl_tree, d_family_fruit$family[1:20], layout = "fan")

p2 = p1 + new_scale_color() + geom_tippoint(aes(color = n_total_fruit_log), size = 1) +
  scale_color_viridis_c() + labs(color = "Number of total \nfruit records (log10)")

p3 = p2 +
  geom_fruit(data = rename(d_family_fruit, Accuracy = .accuracy_family_fruit, id = family) |>
               select(id, Accuracy),
             geom = geom_col, size = 0.5,
             aes(x = Accuracy, y = id), color = NA, fill = "lightblue", alpha = 0.6,
             axis.params=list(
               axis = "x", text.size = 2.3, vjust = 0.3, title = "Accuracy",
               nbreak = 4
             ),
             grid.params=list())

p4 = p3 +
  theme(legend.position = c(0.83, 0.9), legend.box = "horizontal",
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 13))

ggsave("figures/inat_fruit_accuracy.pdf", plot = p4, height = 11, width = 11)

## spatial mapping =====
library(tidyverse)
library(tidymodels)
library(sf)
library(rnaturalearth)
library(probably)

# Turn off S2
sf_use_s2(FALSE)
wld = ne_countries(scale = 'medium')
wld = filter(wld, continent != "Antarctica") |> st_make_valid()
wld = st_union(wld)
wld = st_transform(wld, crs = "ESRI:54009") # MollWeide

plot(wld)

p1 = ggplot() + geom_sf(data = wld, fill = "gray", color = "gray") + theme_void()


grids = st_make_grid(wld, cellsize = c(100000, 100000), crs = st_crs(wld))
grids = mutate(st_sf(geometry = grids), id_cells = 1:n())

ggplot() +
  geom_sf(data = grids, inherit.aes = F) +
  geom_sf(data = wld, fill = "gray", color = "gray")

g2 = st_intersects(wld, grids, sparse = FALSE)
grids2 <- grids |>
  filter(g2[1,])

test_sf <- read_rds("output/epoch_4_testing_data_only_w_coords.rds") |>
  drop_na(longitude, latitude) |>
  st_as_sf(coords = c("longitude", "latitude"), crs = st_crs(4326))

test_sf <- st_transform(test_sf, crs = "ESRI:54009")

test_sf <- test_sf |>
  st_join(grids2)

test_sf <- test_sf |>
  mutate(.class_flower = make_two_class_pred(.pred_flower, c("Detected", "Not Detected"),
                                             threshold = 0.84,
                                             buffer = c(0.56, 0.01)),
         .class_fruit = make_two_class_pred(.pred_fruit, c("Detected", "Not Detected"),
                                            threshold = 0.53,
                                            buffer = c(0.3, 0.22)),
         .equivocal_flower = ifelse(is_equivocal(.class_flower), "Equivocal", "Unequivocal"),
         .equivocal_fruit = ifelse(is_equivocal(.class_fruit), "Equivocal", "Unequivocal")) |>
  mutate(.class_flower_noeq = make_two_class_pred(.pred_flower, c("Detected", "Not Detected"),
                                             threshold = 0.84),
         .class_fruit_noeq = make_two_class_pred(.pred_fruit, c("Detected", "Not Detected"),
                                            threshold = 0.53)) |>
  mutate(.truth_flower = factor(ifelse(flower == "1", "Detected", "Not Detected"), levels = c("Detected", "Not Detected")),
         .truth_fruit = factor(ifelse(fruit == "1", "Detected", "Not Detected"), levels = c("Detected", "Not Detected")))

grids_5 <- test_sf |>
  as_tibble() |>
  group_by(id_cells) |>
  summarise(count = n()) |>
  filter(count >= 5) |>
  pull(id_cells)

acc_by_space_fl <- test_sf |>
  as_tibble() |>
  group_by(id_cells) |>
  accuracy(.truth_flower, .class_flower) |>
  ungroup() |>
  filter(id_cells %in% grids_5)

acc_by_space_fr <- test_sf |>
  as_tibble() |>
  group_by(id_cells) |>
  accuracy(.truth_fruit, .class_fruit) |>
  ungroup() |>
  filter(id_cells %in% grids_5)

equiv_by_space <- test_sf |>
  as_tibble() |>
  group_by(id_cells) |>
  summarise(equiv_prop_fl = sum(is.na(as.character(.class_flower))) / n(),
            equiv_prop_fr = sum(is.na(as.character(.class_fruit))) / n(),
            count = n()) |>
  ungroup() |>
  filter(id_cells %in% grids_5)

grid_acc_fl <- grids2 |>
  left_join(acc_by_space_fl) |>
  drop_na(.estimate)

grid_acc_fr <- grids2 |>
  left_join(acc_by_space_fr) |>
  drop_na(.estimate)

grid_equiv <- grids2 |>
  left_join(equiv_by_space) |>
  drop_na(equiv_prop_fl, equiv_prop_fr)

p_fl <- ggplot() +
  geom_sf(data = wld, fill = "gray", color = "gray") +
  geom_sf(aes(fill = .estimate), colour = NA,
          data = grid_acc_fl, inherit.aes = F) +
  scale_fill_viridis_c(name = "Accuracy\non test data") +
  ggtitle("Flowers") +
  theme_void() +
  theme(plot.title = element_text(size = 12, hjust = 0.5))

p_fr <- ggplot() +
  geom_sf(data = wld, fill = "gray", color = "gray") +
  geom_sf(aes(fill = .estimate), colour = NA,
          data = grid_acc_fr, inherit.aes = F) +
  scale_fill_viridis_c(name = "Accuracy\non test data") +
  ggtitle("Fruit") +
  theme_void() +
  theme(plot.title = element_text(size = 12, hjust = 0.5))

library(patchwork)
p <- p_fl + p_fr + plot_layout(nrow = 2, guides = 'collect')

ggsave("figures/accuracy_by_space_plot.png", p, height = 9, width = 9)

p_eq_fl <- ggplot() +
  geom_sf(data = wld, fill = "gray", color = "gray") +
  geom_sf(aes(fill = equiv_prop_fl), colour = NA,
          data = grid_equiv, inherit.aes = F) +
  scale_fill_viridis_c(name = "Proportion\nuncertain") +
  ggtitle("Flowers") +
  theme_void() +
  theme(plot.title = element_text(size = 12, hjust = 0.5))

p_eq_fr <- ggplot() +
  geom_sf(data = wld, fill = "gray", color = "gray") +
  geom_sf(aes(fill = equiv_prop_fr), colour = NA,
          data = grid_equiv, inherit.aes = F) +
  scale_fill_viridis_c(name = "Proportion\nuncertain") +
  ggtitle("Fruit") +
  theme_void() +
  theme(plot.title = element_text(size = 12, hjust = 0.5))

pe <- p_eq_fl + p_eq_fr + plot_layout(nrow = 2, guides = 'collect')

ggsave("figures/uncertainty_by_space_plot.png", pe, height = 9, width = 9)

##### family-level sample size and imbalance figures ##############

library(tidyverse)
library(patchwork)

imb_dat <- read_csv("output/family_stats_w_imbalance.csv")
samp_size <- imb_dat |>
  filter(test == "train") |>
  select(family, samp_size = count)
imb_dat <- imb_dat |>
  filter(test == "test")
imb_dat <- imb_dat |>
  mutate(abs_imb_fl = abs(imbalance_fl),
         abs_imb_fr = abs(imbalance_fr)) |>
  left_join(samp_size)

p_fl <- ggplot(imb_dat, aes(samp_size, .accuracy_family_flower)) +
  geom_smooth(data = imb_dat |>
                filter(.accuracy_family_flower != 1.0), colour = "grey30") +
  geom_point(aes(colour = .accuracy_family_flower), alpha = 0.8, size = 2) +
  scale_x_log10(labels = scales::label_comma()) +
  scale_colour_viridis_c(option = "D") +
  ylab("Accuracy on Test Data") +
  xlab("Training Data Sample Size") +
  ggtitle("Flowers") +
  ylim(c(0, 1)) +
  theme_minimal() +
  theme(legend.position = 'none',
        plot.title = element_text(size = 12, hjust = 0.5))
p_fl

p_fr <- ggplot(imb_dat, aes(samp_size, .accuracy_family_fruit)) +
  geom_smooth(data = imb_dat |>
                filter(.accuracy_family_fruit != 1.0), colour = "grey30") +
  geom_point(aes(colour = .accuracy_family_fruit), alpha = 0.8, size = 2) +
  scale_x_log10(labels = scales::label_comma()) +
  scale_colour_viridis_c(option = "D") +
  ylab("") +
  xlab("Training Data Sample Size") +
  ggtitle("Fruit") +
  ylim(c(0, 1)) +
  theme_minimal() +
  theme(legend.position = 'none',
        plot.title = element_text(size = 12, hjust = 0.5))
p_fr

p <- p_fl + p_fr
ggsave("figures/fam_samp_size_accuracy.png", p, height = 6, width = 9)

ggplot(imb_dat, aes(abs_imb_fl, .accuracy_family_flower)) +
  geom_point(aes(colour = .accuracy_family_flower)) +
  geom_smooth() +
  scale_colour_viridis_c(option = "D") +
  theme_minimal() +
  theme(legend.position = 'none')

ggplot(imb_dat, aes(abs_imb_fl, .accuracy_family_flower)) +
  geom_point(aes(size = abs_imb_fl, colour = abs_imb_fl)) +
  theme_minimal()

summary(lm(.accuracy_family_flower ~ samp_size * abs_imb_fl,
           data = imb_dat))

summary(lm(.accuracy_family_flower ~ samp_size * imbalance_fl,
           data = imb_dat))

### make spreadsheet for family-level stats ####
write_csv(imb_dat, "output/family_level_stats.csv")
