library(tidyverse)
library(arrow)

## double check numbers match between internal and ingest formats

ing <- read_csv("output/leaves/10.57967_hf_5785/annotations_ingest_all.csv")
inter <- read_csv("output/leaves/10.57967_hf_5785/annotations_internal_all.csv")

summ <- inter |>
  group_by(trait, predictionClass, certainty) |>
  summarise(count = n())

summ2 <- ing |>
  group_by(trait) |>
  summarise(count = n())

## get most recent observation date for training data

train <- read_csv("data/leaves/train.csv")
seconds <- read_csv("data/leaves/seconds.csv")
meta_obs <- open_dataset("data/phenobase_inat_data/metadata/angio_photos/")

train <- train |>
  left_join(meta_obs |>
              filter(observation_uuid %in% train$observation_uuid) |>
              collect() |>
              group_by(observation_uuid) |>
              summarise(observed_on = observed_on[1]))

seconds <- seconds |>
  left_join(meta_obs |>
              filter(observation_uuid %in% seconds$observation_uuid) |>
              collect() |>
              group_by(observation_uuid) |>
              summarise(observed_on = observed_on[1]))

max(c(train$observed_on, seconds$observed_on), na.rm = TRUE)
