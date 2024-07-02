library(arrow)
library(tidyverse)
library(reticulate)
library(santoku)

angio_meta <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/angio_photos")
taxa_meta <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/taxa/part-0.parquet")

test_meta <- angio_meta |>
  slice_head(n = 100000) |>
  slice_sample(n = 1000) |>
  as_tibble()

test_meta <- test_meta |>
  mutate(year = year(observed_on))

test_dataset <- test_meta |>
  select(photo_id, observation_uuid, latitude, longitude, taxon_id, observed_on, year) |>
  left_join(taxa_meta |>
              select(taxon_id, name, rank_level, rank) |>
              filter(taxon_id %in% test_meta$taxon_id),
            by = "taxon_id",
            copy = TRUE)

taxon_ids <- angio_meta |>
  select(taxon_id) |>
  pull(taxon_id)
#batches <- gl(ceiling(length(taxon_ids) / 2560L), k = 2560L, length = length(taxon_ids))
angio_meta <- angio_meta |>
  mutate(year = year(observed_on),
         month = month(observed_on)) |>
  left_join(taxa_meta |>
              select(taxon_id, name, rank_level, rank) |>
              filter(taxon_id %in% taxon_ids),
            by = "taxon_id",
            copy = TRUE) |>
  group_by(year, month)

write_dataset(angio_meta, "/blue/guralnick/share/phenobase_inat_data/images2")

############ get image data #####################
photo_ids <- angio_meta$photo_id[1:5000]
get_images_as_bytes <- function(photo_ids) {

}


test <- tibble(x = as.raw(c(1:1000)))
write_dataset(test, "test")
