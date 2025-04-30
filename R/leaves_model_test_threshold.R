library(tidyverse)
library(tidymodels)
library(probably)
library(ggforce)
library(santoku)
library(wesanderson)

leaves_test_df <- read_rds("output/leaves/phenovision-init_model2_04_11_2025/epoch_1_test_results.rds")

val_dat <- leaves_test_df |> filter(partition == "validation")
test_dat <- leaves_test_df |> filter(partition == "testing")

thresholds <- read_csv("output/leaves/phenovision-init_model2_04_11_2025/epoch_1_test_thresholds.csv")

test_dat <- bind_rows(val_dat, test_dat) |>
  mutate(.class_leaves_green = make_two_class_pred(.pred_leaves_green,
                                                   levels(leaves_green),
                                                   threshold = thresholds |> filter(var == "gr") |> pull(.threshold)),
         .class_leaves_colored = make_two_class_pred(.pred_leaves_colored,
                                                     levels(leaves_colored),
                                                     threshold = thresholds |> filter(var == "cl") |> pull(.threshold)),
         .class_leaves_breaking_buds = make_two_class_pred(.pred_leaves_breaking_buds,
                                                           levels(leaves_breaking_buds),
                                                           threshold = thresholds |> filter(var == "bb") |> pull(.threshold))) |>
  mutate(.cut_leaves_green = chop_evenly(.pred_leaves_green, 100,
                                         labels = lbl_midpoints()),
         .cut_leaves_colored = chop_evenly(.pred_leaves_colored, 100,
                                           labels = lbl_midpoints()),
         .cut_leaves_breaking_buds = chop_evenly(.pred_leaves_breaking_buds, 100,
                                                 labels = lbl_midpoints()))

leaves_green_acc <- test_dat |>
  group_by(.cut_leaves_green) |>
  accuracy(leaves_green, .class_leaves_green) |>
  mutate(type = "leaves_green", value = as.numeric(as.character(.cut_leaves_green))) |>
  ungroup() |>
  left_join(test_dat |>
              group_by(.cut_leaves_green) |>
              summarise(count = n()))

leaves_green_acc <- leaves_green_acc |>
  mutate(cummean = cummean(.estimate))

ggplot(leaves_green_acc, aes(value, .estimate)) +
  geom_point() +
  geom_smooth() +
  scale_y_continuous(breaks = seq(0, 1, by = 0.05)) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.05)) +
  theme_minimal()

### lower threshold = 0.115
leaves_green_buffer <- c(thresholds |> filter(var == "gr") |> pull(.threshold), thresholds |> filter(var == "gr") |> pull(.threshold) - 0.115, 0.001)

leaves_colored_acc <- test_dat |>
  group_by(.cut_leaves_colored) |>
  accuracy(leaves_colored, .class_leaves_colored) |>
  mutate(type = "leaves_colored", value = as.numeric(as.character(.cut_leaves_colored))) |>
  ungroup() |>
  left_join(test_dat |>
              group_by(.cut_leaves_colored) |>
              summarise(count = n())) |>
  mutate(cummean = cummean(.estimate),
         cummean_rev = rev(cummean(rev(.estimate))))

### upper threshold = 0.995
leaves_colored_buffer <- c(thresholds |> filter(var == "cl") |> pull(.threshold), 0.001, 0.995 - thresholds |> filter(var == "cl") |> pull(.threshold))

leaves_breaking_buds_acc <- test_dat |>
  group_by(.cut_leaves_breaking_buds) |>
  accuracy(leaves_breaking_buds, .class_leaves_breaking_buds) |>
  mutate(type = "leaves_breaking_buds", value = as.numeric(as.character(.cut_leaves_breaking_buds))) |>
  ungroup() |>
  left_join(test_dat |>
              group_by(.cut_leaves_breaking_buds) |>
              summarise(count = n())) |>
  mutate(cummean = cummean(.estimate),
         cummean_rev = rev(cummean(rev(.estimate))))

### upper threshold = 0.835
leaves_breaking_buds_buffer <- c(thresholds |> filter(var == "bb") |> pull(.threshold), 0.001, 0.835 - thresholds |> filter(var == "bb") |> pull(.threshold))

buffers <- list(gr = leaves_green_buffer, cl = leaves_colored_buffer, bb = leaves_breaking_buds_buffer)
write_rds(buffers, "output/leaves/phenovision-init_model2_04_11_2025/epoch_1_threshold_buffers.csv")
