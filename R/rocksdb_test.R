library(reticulate)
library(tidyverse)
library(arrow)

PIL <- import("PIL")
rocksdb <- import("rocksdb")
rdb <- import_from_path("rocksdb_funs", "py")
ds <- import_from_path("custom_dataset", "py")

image_path <- "/blue/guralnick/share/phenobase_inat_data/images/medium"

rdb_options = rocksdb$Options(create_if_missing = TRUE)

angio_meta <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/angio_photos")

## test 1 million images
photos <- angio_meta |>
  select(photo_id, extension) |>
  collect() |>
  sample_n(3e+6)

file_names <- file.path(image_path, paste0(photos$photo_id, ".", photos$extension))

db <- "/blue/guralnick/share/phenobase_inat_data/images_rocksdb/test_batch.db"

system.time({
  test <- rdb$rocks_store_img_batch(db, file_names, rdb_options)
})

## ~ 46 minutes for 1 million images: but actually only ~ 333,000: so that is 7240 images per minute

library(reticulate)
library(tidyverse)
library(arrow)

PIL <- import("PIL")
rocksdb <- import("rocksdb")
rdb <- import_from_path("rocksdb_funs", "py")
ds <- import_from_path("custom_dataset", "py")

rdb_options = rocksdb$Options(create_if_missing = TRUE)
db <- "/blue/guralnick/share/phenobase_inat_data/images_rocksdb/test_batch.db"

image_db <- rocksdb$DB(db, rdb_options)
keys <- image_db$iterkeys()
keys$seek_to_first()
test_key <- iter_next(keys)
key_str = test_key$decode('utf-8')

keys2 <- py_eval('list(r.keys)')


