library(reticulate)
library(tidyverse)
library(tidymodels)
library(probably)
library(zeallot)
library(arrow)
library(data.table)

meta <- open_dataset("data/phenobase_inat_data/metadata/angio_photos")

taxa_meta <- open_dataset("data/phenobase_inat_data/metadata/taxa/part-0.parquet")

rob_annot <- read_csv("data/leaves/phenobase_dwca_annotation/rob_leaf_breaking_buds_annotation.csv")
rob_annot <- rob_annot |>
  mutate(genus = word(taxon.name))

keep_genera <- unique(rob_annot$genus)
keep_genera <- setdiff(keep_genera, "Logfia")

genus_taxon_ids <- taxa_meta |>
  filter(rank == "genus") |>
  filter(name %in% keep_genera) |>
  collect() |>
  filter(active)

plantae_id <- taxa_meta |>
  filter(rank == "kingdom") |>
  collect() |>
  filter(active, name == "Plantae")

taxa_dataset <- meta |>
  distinct(taxon_id, .keep_all = TRUE) |>
  select(taxon_id) |>
  left_join(taxa_meta |>
              select(taxon_id, ancestry),
            by = "taxon_id",
            copy = TRUE) |>
  collect() |>
  select(taxon_id, ancestry) |>
  mutate(taxa_ids = str_split(ancestry, "/")) |>
  select(-ancestry) |>
  unnest_longer(taxa_ids, transform = as.integer) |>
  filter(taxa_ids %in% genus_taxon_ids$taxon_id)

taxonomy <- taxa_dataset

meta_filtered <- meta |>
  filter(taxon_id %in% taxonomy$taxon_id) |>
  collect()

taxonomy <- taxonomy |>
  left_join(taxa_meta |>
              select(taxa_ids = taxon_id, genus = name) |>
              filter(taxa_ids %in% taxonomy$taxa_ids),
            copy = TRUE) |>
  left_join(taxa_meta |>
              select(name, taxon_id, ancestry) |>
              filter(taxon_id %in% taxonomy$taxon_id),
            copy = TRUE)

meta_filtered <- meta_filtered |>
  left_join(taxonomy)
meta_filtered <- meta_filtered |>
  mutate(file_name = file.path("data", "phenobase_inat_data", "images", "medium", paste0("batch_", batch_j), paste0(photo_id, ".", extension)))

write_csv(meta_filtered, "data/leaves/inference_metadata_04-11-2025.csv")

