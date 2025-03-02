library(tidyverse)
library(tidymodels)
library(arrow)
library(phyf)
library(jsonlite)
library(data.table)

set.seed(13459)

leaf_annot <- read_parquet("data/leaves/phenobase_dwca_annotation/inat_annotation/part-0.parquet") |>
  filter(dynamicProperties != "")
## fix weird double double quotes issue
leaf_annot <- leaf_annot |>
  mutate(dynamicProperties = str_replace_all(dynamicProperties, stringr::fixed("\"\""), "\""))
leaf_json <- map(leaf_annot$dynamicProperties, possibly(fromJSON), .progress = TRUE)

rob_annot <- read_csv("data/leaves/phenobase_dwca_annotation/rob_leaf_breaking_buds_annotation.csv")
rob_annot <- rob_annot |>
  mutate(genus = word(taxon.name))

keep_genera <- unique(rob_annot$genus)

meta <- leaf_annot |>
  mutate(leaves = map(leaf_json, "leaves"),
         leaves_green = as.numeric(map_lgl(leaves, ~"green leaves" %chin% .x, .progress = TRUE)),
         leaves_colored = as.numeric(map_lgl(leaves, ~"colored leaves" %chin% .x, .progress = TRUE)),
         leaves_no_live = as.numeric(map_lgl(leaves, ~"no live leaves" %chin% .x, .progress = TRUE)),
         leaves_breaking_buds = as.numeric(map_lgl(leaves, ~"breaking leaf buds" %chin% .x, .progress = TRUE)))

meta <- meta |>
  filter(!leaves_breaking_buds == 1)

meta <- meta |>
  filter(genus %chin% keep_genera)

photo_meta <- open_dataset("data/phenobase_inat_data/metadata/angio_photos")
photo_meta <- photo_meta |>
  select(observation_uuid, photo_uuid, photo_id, batch_j, extension) |>
  filter(observation_uuid %in% unique(c(meta$observation_uuid, rob_annot$observation_uuid))) |>
  collect()

meta <- meta |>
  left_join(photo_meta)

meta <- meta |>
  drop_na(batch_j)

meta <- meta |>
  mutate(file_name = file.path("data", "phenobase_inat_data", "images", "medium", paste0("batch_", batch_j), paste0(photo_id, ".", extension)))

fams <- meta |>
  select(genus, family) |>
  distinct()

rob_photos <- list.files("data/leaves/phenobase_dwca_annotation/images_breaking_leaf_buds")
rob_photo_ids <- rob_photos |>
  fs::path_ext_remove()

rob_annot <- rob_annot |>
  left_join(tibble(photo_id = rob_photos,
                   file_name = file.path("data/leaves/phenobase_dwca_annotation/images_breaking_leaf_buds", rob_photos)))

rob_annot <- rob_annot |>
  drop_na(file_name)

rob_annot <- rob_annot |>
  left_join(fams)

rob_annot <- rob_annot |>
  mutate(leaves_green = as.numeric(map_lgl(leaves, ~ grepl("Green leaves", .x, fixed = TRUE), .progress = TRUE)),
         leaves_colored = as.numeric(map_lgl(leaves, ~ grepl("Colored leaves", .x, fixed = TRUE), .progress = TRUE)),
         leaves_no_live = as.numeric(map_lgl(leaves, ~ grepl("No live leaves", .x, fixed = TRUE), .progress = TRUE)),
         leaves_breaking_buds = as.numeric(map_lgl(leaves, ~ grepl("Breaking leaf buds", .x, fixed = TRUE), .progress = TRUE)))

all_meta <- meta |>
  select(observation_uuid, scientificName, genus, family, file_name,
         leaves_green, leaves_colored, leaves_no_live, leaves_breaking_buds) |>
  mutate(strata = paste(family, leaves_green, leaves_colored,
                        leaves_no_live, leaves_breaking_buds, sep = "|"))

all_meta <- all_meta |>
  bind_rows(rob_annot |>
              select(observation_uuid, scientificName = taxon.name, genus, family, file_name,
                     leaves_green, leaves_colored, leaves_no_live,
                     leaves_breaking_buds) |>
              mutate(strata = paste(family, leaves_green, leaves_colored,
                                    leaves_no_live, leaves_breaking_buds, sep = "|")))

all_meta <- all_meta |>
  group_by(observation_uuid) |>
  mutate(count = n()) |>
  ungroup()

summ <- all_meta |>
  group_by(leaves_green, leaves_colored, leaves_breaking_buds, leaves_no_live) |>
  summarise(count = n())

all_meta <- all_meta |>
  filter(!(leaves_green == 0 & leaves_colored == 0 & leaves_breaking_buds == 0 & leaves_no_live == 0)) |>
  select(-leaves_no_live)

summ <- all_meta |>
  group_by(leaves_green, leaves_colored, leaves_breaking_buds) |>
  summarise(count = n())


all_meta_singles <- all_meta |>
  filter(count == 1)

all_meta_multiple <- all_meta |>
  filter(count > 1)

#
# image_files <- list.files("data/inat/images", full.names = TRUE)
# missing <- !meta$file_name %in% image_files
# meta <- meta[!missing, ]


leaves_split <- initial_validation_split(all_meta_singles, strata = strata,
                                         prop = c(0.8, 0.1), pool = 0.025)

leaves_train <- training(leaves_split)
leaves_val <- validation(leaves_split)
leaves_test <- testing(leaves_split)

write_csv(leaves_train, "data/leaves/train.csv")
write_csv(leaves_val, "data/leaves/validation.csv")
write_csv(leaves_test, "data/leaves/test.csv")

write_csv(all_meta_multiple, "data/leaves/seconds.csv")

