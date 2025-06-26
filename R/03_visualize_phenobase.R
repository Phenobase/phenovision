if(!require("xfun")) install.packages("xfun")
xfun::pkg_attach2(c("tidyverse", "googlesheets4", "googledrive", "ape", "rtrees",
                    "ggtree", "ggtreeExtra", "treeio", "ggnewscale")) 

drive_download("https://drive.google.com/file/d/11H_56UO08-SBr0n1LmWD4PMtNVKJheiI/view?usp=sharing", 
               path = "data/annotations_all_9cf8ad8.csv")

drive_download("https://drive.google.com/file/d/1YcHaEFhdc0q3N_mUn2YzC5smiuDI8TDs/view?usp=drive_link", 
               path = "data/data_for_coverage_analysis_9cf8ad8.csv")

coverage_inat = read_csv("data/data_for_coverage_analysis_9cf8ad8.csv")

coverage_inat = mutate(coverage_inat, trait2 = ifelse(trait %in% c("flower", "flowering"), "Flower", "Fruit"))
count(coverage_inat, trait2)

dd = filter(coverage_inat, trait2 == "Flower")
length(intersect(filter(dd, post_ml_only)$photo_id, filter(dd, !post_ml_only)$photo_id))

if(!file.exists("data/genus_tree_inat2.tre")){
  annotation_inat = read_csv("data/annotations_all_9cf8ad8.csv", col_names = FALSE)
  
  filter(coverage_inat, genus == "Dalbergia") |> count(post_ml_only)
  head(annotation_inat)
  names(annotation_inat)[11] = "genus"
  names(annotation_inat)[12] = "sp"
  annotation_inat_sp = select(annotation_inat, genus, sp) |> distinct()
  nrow(annotation_inat_sp) # 119,340. species
  annotation_inat_sp = mutate(annotation_inat_sp, sp = str_replace_all(sp, " ", "_"))
  
  annotation_inat_sp$sp_in_tree = annotation_inat_sp$sp %in% megatrees::tree_plant_otl$tip.label
  n_distinct(annotation_inat_sp$genus) # 10,135
  n_distinct(filter(annotation_inat_sp, sp_in_tree)$genus) # 7,327
  
  sp1 = filter(annotation_inat_sp, sp_in_tree) |> 
    group_by(genus) |> 
    sample_n(size = 1) |> ungroup()
  
  sp2 = filter(annotation_inat_sp, !genus %in% sp1$genus) |> 
    group_by(genus) |> 
    sample_n(size = 1) |> ungroup()
  
  sp_inat = bind_rows(sp1, sp2)
  any(duplicated(sp_inat$genus))
  
  sp_df = rtrees::sp_list_df(sp_inat$sp, "plant")
  n_distinct(sp_df$genus) # 10,133
  n_distinct(sp_df$family) # 402
  filter(sp_df, genus %in% c("Inula", "Ocotea"))$species
  filter(sp_inat, sp %in% filter(sp_df, genus %in% c("Inula", "Ocotea"))$species)
  
  
  genus_tree_inat = rtrees::get_tree(sp_list = sp_inat$sp, taxon = "plant")
  plot(genus_tree_inat, show.tip.label = F, type = "fan")
  genus_tree_inat = ape::drop.tip(genus_tree_inat, tip = c("Inula_japonica", "Ocotea_cymosa")) # duplicated genus
  
  genus_tree_inat2 = genus_tree_inat
  genus_tree_inat2$tip.label = str_extract(genus_tree_inat$tip.label, "^[^_]+")
  genus_tree_inat$tip.label[which(duplicated(genus_tree_inat2$tip.label))]
  grep("Inula", genus_tree_inat$tip.label, value = T)
  genus_tree_inat2$tip.label
  
  plot(genus_tree_inat2, show.tip.label = F, type = "fan")
  write.tree(genus_tree_inat2, "data/genus_tree_inat2.tre")
} else {
  genus_tree_inat2 = read.tree("data/genus_tree_inat2.tre")
}


# summarize genus level coverage
n_distinct(coverage_inat$genus) # 10,206
n_distinct(coverage_inat$family) # 408

n_distinct(annotation_inat$`Leucadendron grandiflorum`)

n_distinct(filter(coverage_inat, post_ml_only)$genus) # 10117
n_distinct(filter(coverage_inat, post_ml_only)$family) # 408

n_distinct(filter(coverage_inat, !post_ml_only)$genus) # 6583
n_distinct(filter(coverage_inat, !post_ml_only)$family) # 363

coverage_inat2 = filter(coverage_inat, genus %in% genus_tree_inat2$tip.label) |> 
  mutate(trait = recode(trait, "flowering" = "flower", "fruiting" = "fruit")) |> 
  group_by(trait, genus) |> 
  count(post_ml_only) |> ungroup()

summary(filter(coverage_inat2, post_ml_only, trait == "flower", n >= 10)$n) 
count(coverage_inat, trait)
count(coverage_inat, trait, post_ml_only)

coverage_inat3 = coverage_inat2 |> 
  mutate(post_ml_only = as.character(post_ml_only),
         post_ml_only = paste0("ml_", post_ml_only)) |> 
  pivot_wider(names_from = "post_ml_only", values_from = "n", values_fill = 0) |> 
  mutate(n_total = ml_TRUE + ml_FALSE,
         ml_prop = ml_TRUE / n_total,
         human_prop = ml_FALSE / n_total)

saveRDS(coverage_inat3, "data/coverage_inat3.rds")


# read saved data ====
# genus_tree_inat2 = read.tree("data/genus_tree_inat2.tre")
coverage_inat3 = readRDS("data/coverage_inat3.rds")

summary(filter(coverage_inat3, trait == "flower", n_total >= 10)$ml_prop)
filter(coverage_inat3, trait == "flower", n_total >= 10) |> arrange(ml_prop)
filter(coverage_inat3, trait == "flower", n_total >= 10) |> arrange(desc(ml_TRUE))

summary(filter(coverage_inat3, trait == "fruit", n_total >= 10)$ml_prop)
filter(coverage_inat3, trait == "fruit", n_total >= 10) |> arrange(ml_prop)
filter(coverage_inat3, trait == "fruit", n_total >= 10) |> arrange(desc(ml_TRUE))

coverage_inat4 = pivot_longer(coverage_inat3, 
                              cols = c("ml_prop", "human_prop"),
                              names_to = "source", values_to = "Proportion")


d_flower = filter(coverage_inat4, trait == "flower", n_total >= 10) |> 
  rename(id = genus) |> 
  select(id, trait, Proportion, source) |> 
  mutate(source = recode(source, "ml_prop" = "Machine annotation",
                         "human_prop" = "Human annotation"),
         source = factor(source, levels = c("Machine annotation", "Human annotation"))) |> 
  rename(`Data source (%)` = source)
# mutate(id = factor(id, levels = genus_tree_inat2$tip.label))

t1 = keep.tip(genus_tree_inat2, as.character(d_flower$id)) |> ladderize()
setdiff(d_flower$id, t1$tip.label)
setdiff(t1$tip.label, d_flower$id)
t1.tbl = as_tibble(t1) |> full_join(
  filter(coverage_inat4, trait == "flower", n_total >= 10) |> 
    rename(label = genus) |> 
    select(label, n_total) |> 
    mutate(n_total_log = log10(n_total))
) |> distinct()
t1_tbl_tree = as.treedata(t1.tbl)

## rectanglar layout
# p1 = ggtree(t1) + scale_x_continuous(expand = c(0, 0))
# 
# p2 = p1 + 
#   geom_facet(panel = "Flower", data = d_flower, geom = geom_col,
#              aes(x = Proportion, color = source, fill = source), 
#              orientation = 'y', width = .1) + 
#   theme(strip.background = element_blank(),
#         legend.position.inside = c(0.1, 0.9))
# facet_labeller(p2, c(Tree = "Genus Phylogeny")) |> 
#   facet_widths(widths = c(2, 1)) 

p1 = ggtree(t1_tbl_tree, layout = "fan", size = 0.2, aes(color = n_total_log)) +
  scale_color_viridis_c() + labs(color = "Number of total \nflower records (log10)")

p1

p2 = p1 + new_scale_color() +
  geom_fruit(data = d_flower, geom = geom_col,
             aes(x = Proportion, y = id, color = `Data source (%)`, 
                 fill = `Data source (%)`), 
             width = 0.2, inherit.aes = F)  +
  theme(legend.position = c(0.23, 0.9), legend.box = "horizontal",
        legend.title = element_text(size = 14), 
        legend.text = element_text(size = 13))
p2 = p2 + scale_fill_manual(values = c("lightblue", "darkred")) +
  scale_color_manual(values = c("lightblue", "darkred"))
p2


ggsave("figures/inat_flower_coverage.pdf", plot = p2, height = 9, width = 9)
# ggsave("figures/inat_flower_coverage.png", plot = p2, height = 10, width = 10)

# fruit ----
d_fruit = filter(coverage_inat4, trait == "fruit", n_total >= 10) |> 
  rename(id = genus) |> 
  select(id, Proportion, source) |> 
  mutate(source = recode(source, "ml_prop" = "Machine annotation",
                         "human_prop" = "Human annotation"),
         source = factor(source, levels = c("Machine annotation", "Human annotation"))) |> 
  rename(`Data source (%)` = source)
# mutate(id = factor(id, levels = genus_tree_inat2$tip.label))

t2 = keep.tip(genus_tree_inat2, as.character(d_fruit$id)) |> ladderize()
t2.tbl = as_tibble(t2) |> full_join(
  filter(coverage_inat4, trait == "fruit", n_total >= 10) |> 
    rename(label = genus) |> 
    select(label, n_total) |> 
    mutate(n_total_log = log10(n_total))
) |> distinct()
t2_tbl_tree = as.treedata(t2.tbl)

p1_fruit = ggtree(t2_tbl_tree, layout = "fan", size = 0.2, aes(color = n_total_log)) +
  scale_color_viridis_c() + labs(color = "Number of total \nfruit records (log10)")

p2_fruit = p1_fruit + new_scale_color() +
  geom_fruit(data = d_fruit, geom = geom_col,
             aes(x = Proportion, y = id, color = `Data source (%)`, 
                 fill = `Data source (%)`), 
             width = 0.2, inherit.aes = F)  +
  theme(legend.position = c(0.23, 0.9), legend.box = "horizontal",
        legend.title = element_text(size = 14), 
        legend.text = element_text(size = 13))
p2_fruit = p2_fruit + scale_fill_manual(values = c("lightblue", "darkred")) +
  scale_color_manual(values = c("lightblue", "darkred"))
p2_fruit


ggsave("figures/inat_fruit_coverage.pdf", plot = p2_fruit, height = 9, width = 9)

# family level stats ====
fam = read_csv("data/family_stats.csv")
count(fam, test)

fam_genus = coverage_inat |> 
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
library(sf)
library(rnaturalearth)
coverage_inat
# Turn off S2
sf_use_s2(FALSE)
wld = ne_countries(scale = 'medium')
wld = filter(wld, continent != "Antarctica") |> st_make_valid()
wld = st_union(wld)
wld = st_transform(wld, crs = "ESRI:54009") # MollWeide

plot(wld)

grids = st_make_grid(wld, cellsize = c(100000, 100000), crs = st_crs(wld))
grids = mutate(st_sf(geometry = grids), id_cells = 1:n())


g2 = st_intersects(wld, grids)

coverage_inat = drop_na(coverage_inat, latitude, longitude)
dat = st_transform(st_as_sf(coverage_inat, coords = c("longitude", "latitude"), crs = 4326, remove = F), 
                   crs = "ESRI:54009")

dat2 = st_join(dat, grids)

grids2 = grids[sort(unique(c(dat2$id_cells, g2[[1]]))),]

dat2 = st_drop_geometry(dat2) 

dat2_summary = group_by(dat2, id_cells, trait2, post_ml_only) |> 
  tally() 

dat2_summary = pivot_wider(dat2_summary, names_from = "post_ml_only", values_from = "n", values_fill = 0)
dat2_summary = rename(dat2_summary, ml_annotated = `TRUE`, human_annotated= `FALSE`)
dat2_summary = mutate(dat2_summary, total_records = ml_annotated + human_annotated,
                      ml_prop = ml_annotated / total_records) |> ungroup()

base_map = ggplot() + 
  geom_sf(data = wld, fill = "gray", color = "gray") +
  # geom_sf(data = grids2, fill = "gray80") +
  theme_void()

grids2_dat = left_join(grids2, dat2_summary)

grids2_dat = mutate(grids2_dat, trait2 = recode(trait2, "flower" = "Flower", "fruit" = "Fruit"))

st_drop_geometry(grids2_dat) |> 
  drop_na(trait2) |> 
  mutate(log10_ml = log10(ml_annotated + 1)) |> 
  group_by(trait2) |> 
  summarise(across(log10_ml, .fns = 
                     list(min = min,
                          median = median,
                          mean = mean,
                          stdev = sd,
                          q25 = ~quantile(., 0.25),
                          q75 = ~quantile(., 0.75),
                          max = max)))

st_drop_geometry(grids2_dat) |> 
  drop_na(trait2) |> 
  filter(human_annotated > 0) |> 
  group_by(trait2) |> 
  summarise(across(ml_prop, .fns = 
                     list(min = min,
                          median = median,
                          mean = mean,
                          stdev = sd,
                          q25 = ~quantile(., 0.25),
                          q75 = ~quantile(., 0.75),
                          max = max)))


# d3 = grids2_dat |> 
#   drop_na(trait2) |> 
#   mutate(`Cell types` = ifelse(human_annotated > 10, "Human annotated data available (n > 10)",
#                             ifelse(ml_annotated > 10, "New grid cells with phenoVision data (n > 10)", 
#                                    "Grid cells with < 10 human or machine annotated data"))) 

d3 = grids2_dat |> 
  drop_na(trait2) |> 
  mutate(`(A): Cell types` = ifelse(human_annotated > 0, "Human annotated data available",
                                    ifelse(ml_annotated > 0, "New grid cells with phenoVision data", 
                                           NA))) 

p1_flower_cells = base_map +
  geom_sf(data = d3, aes(fill = `(A): Cell types`), color = NA) +
  scale_fill_manual(values= c("lightblue", "darkred")) +
  facet_wrap(~trait2) +
  # scale_fill_viridis_d(option = "D", direction = -1) +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +
  # scale_color_viridis_d(option = "D", direction = -1) +
  theme(legend.position = "top", legend.text = element_text(size = 12), 
        legend.title = element_text(size = 12))

d5 = drop_na(grids2_dat, trait2) |> 
  mutate(log10_total_records = log10(total_records),
         log10_ml_records = log10(ml_annotated))

p1_flower_counts = base_map +
  geom_sf(data = d5, aes(fill = log10_ml_records), color = NA) +
  scale_fill_viridis_c(name = "(B): Total number of machine annotated records (log10)") +
  facet_wrap(~trait2) +
  # scale_fill_viridis_d(option = "D", direction = -1) +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +
  # scale_color_viridis_d(option = "D", direction = -1) +
  theme(legend.position = "top", legend.text = element_text(size = 12), 
        legend.title = element_text(size = 12))

logit_trans = function(x){
  stopifnot(all(x >= 0) & all(x <= 1))
  x = ifelse(x == 0, x + 0.01, x)
  x = ifelse(x == 1, x - 0.01, x)
  log(x / (1 - x))
}
d4 = drop_na(grids2_dat, trait2) |> 
  mutate(ml_prop_logit = logit_trans(ml_prop),
         ml_prop_catg = case_when(#ml_prop > 0.99 ~ "> 99%",
           #ml_prop > 0.97 ~ "> 97%",
           ml_prop > 0.95 ~ "> 95%",
           ml_prop > 0.90 ~ "> 90%",
           ml_prop > 0.70 ~ "> 70%",
           ml_prop >= 0.5 ~ "> 50%",
           .default = "< 50%"))

p2_flower_prop = base_map +
  geom_sf(data = d4, aes(fill = ml_prop_catg), color = NA, linesize = 0.1) +
  colorspace::scale_fill_discrete_sequential(name = "(C): Proportion of machine annotated data", palette = "Viridis") +
  facet_wrap(~trait2) +
  # scale_color_viridis_d(option = "D", direction = -1) +
  theme(legend.position = "top", legend.text = element_text(size = 12), 
        legend.title = element_text(size = 12))

library(patchwork)

p12_flower_cells = p1_flower_cells + p1_flower_counts +
  p2_flower_prop + plot_layout(nrow = 3)

ggsave("figures/flower_spatial_cells.pdf", plot = p12_flower_cells, width = 10, height = 7.5)
# ggsave("figures/flower_spatial_cells.png", plot = p12_flower_cells, width = 9, height = 9)


