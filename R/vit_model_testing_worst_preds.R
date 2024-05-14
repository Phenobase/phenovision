library(tidyverse)

test_df <- read_rds("output/model_04_13_2024/epoch_4_testing_data.rds")

worst_preds_fl <- test_df |>
  group_by(flower) |>
  slice_max(.loss_flower, n = 500) |>
  arrange(flower, desc(.loss_flower)) |>
  mutate(num = str_pad(1:n(), 3, pad = "0"),
         new_filename = paste0(ifelse(flower == "1", "output/worst_predictions/flower/false_negatives",
                                      "output/worst_predictions/flower/false_positives"),
                               "/", num, "_", basename(file_name)))

worst_preds_fr <- test_df |>
  group_by(fruit) |>
  slice_max(.loss_fruit, n = 500) |>
  arrange(fruit, desc(.loss_fruit)) |>
  mutate(num = str_pad(1:n(), 3, pad = "0"),
         new_filename = paste0(ifelse(fruit == "1", "output/worst_predictions/fruit/false_negatives",
                                      "output/worst_predictions/fruit/false_positives"), "/", num, "_", basename(file_name)))

file.copy(worst_preds_fl$file_name, worst_preds_fl$new_filename, overwrite = FALSE)
file.copy(worst_preds_fr$file_name, worst_preds_fr$new_filename, overwrite = FALSE)

