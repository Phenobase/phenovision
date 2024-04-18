library(tidyverse)
library(tidymodels)
library(probably)

test_dat <- read_rds("output/epoch_4_testing_data.rds")

val_dat <- test_dat |> filter(partition == "validation")
test_dat <- test_dat |> filter(partition == "testing")

acc_fl <- accuracy(test_dat |>
                     group_by(partition),
                   flower, .pred_fl_max)

acc_fr <- accuracy(test_dat |>
                     group_by(partition),
                   fruit, .pred_fr_max)

jind_fl <- j_index(test_dat |>
                     group_by(partition),
                   flower, .pred_fl_max)

jind_fr <- j_index(test_dat |>
                     group_by(partition),
                   fruit, .pred_fr_max)


genus_summ <- test_dat |>
  group_by(order, family, genus) |>
  summarise(n = n(), flower_prop = mean(flowering),
            fruit_prop = mean(fruiting))

family_summ <- test_dat |>
  group_by(order, family) |>
  summarise(n = n(), flower_prop = mean(flowering),
            fruit_prop = mean(fruiting))

genus_fl_acc <- test_dat |>
  group_by(genus) |>
  accuracy(flower, .pred_fl_max) |>
  left_join(genus_summ)

genus_fr_acc <- test_dat |>
  group_by(genus) |>
  accuracy(fruit, .pred_fr_max) |>
  left_join(genus_summ)

family_fl_acc <- test_dat |>
  group_by(family) |>
  accuracy(flower, .pred_fl_max) |>
  left_join(family_summ)

family_fl_acc_val <- val_dat |>
  group_by(family) |>
  accuracy(flower, .pred_fl_max) |>
  left_join(family_summ)

family_fr_acc <- test_dat |>
  group_by(family) |>
  accuracy(fruit, .pred_fr_max) |>
  left_join(family_summ)

ggplot(genus_fl_acc, aes(.estimate)) +
  geom_histogram(bins = 100) +
  geom_vline(xintercept = 0.9) +
  theme_minimal()

ggplot(genus_fr_acc, aes(.estimate)) +
  geom_histogram(bins = 50) +
  geom_vline(xintercept = 0.9) +
  theme_minimal()

ggplot(family_fl_acc, aes(.estimate)) +
  geom_histogram(bins = 50) +
  geom_vline(xintercept = 0.9) +
  theme_minimal()

ggplot(family_fr_acc, aes(.estimate)) +
  geom_histogram(bins = 50) +
  geom_vline(xintercept = 0.9) +
  theme_minimal()

ggplot(family_fl_acc, aes(flower_prop)) +
  geom_histogram(bins = 40) +
  geom_vline(xintercept = 0.5) +
  theme_minimal()

ggplot(genus_fl_acc, aes(n, .estimate)) +
  geom_point(alpha = 0.5) +
  geom_smooth(se = FALSE, colour = "red") +
  geom_hline(yintercept = 0.9) +
  #scale_x_sqrt() +
  theme_minimal()

ggplot(genus_fl_acc, aes(n, .estimate)) +
  geom_vline(xintercept = 600, colour = "green") +
  geom_point(alpha = 0.5) +
  geom_smooth(se = FALSE, colour = "red") +
  geom_hline(yintercept = 0.9) +
  xlim(c(0, 3500)) +
  theme_minimal()

sum(genus_fl_acc$.estimate[genus_fl_acc$n < 10], na.rm = TRUE) / sum(genus_fl_acc$n < 10)

ggplot(family_fl_acc, aes(n, .estimate)) +
  geom_point() +
  geom_smooth(se = FALSE, colour = "red") +
  geom_hline(yintercept = 0.9) +
  #scale_x_sqrt() +
  theme_minimal()

ggplot(family_fl_acc, aes(n, .estimate)) +
  geom_point() +
  geom_smooth(se = FALSE, colour = "red") +
  geom_hline(yintercept = 0.9) +
  geom_vline(xintercept = 1000, colour = "green") +
  scale_x_log10() +
  theme_minimal()

ggplot(genus_fl_acc, aes(flower_prop, .estimate)) +
  geom_abline(intercept = 1, slope = -1, colour = "blue",
              linewidth = 2) +
  geom_abline(intercept = 0, slope = 1, colour = "blue",
              linewidth = 2) +
  geom_point(size = 1.4) +
  geom_smooth(se = FALSE, colour = "red") +
  geom_hline(yintercept = 0.9) +
  theme_minimal()

ggplot(family_fl_acc, aes(flower_prop, .estimate)) +
  geom_abline(intercept = 1, slope = -1, colour = "blue",
              linewidth = 2) +
  geom_abline(intercept = 0, slope = 1, colour = "blue",
              linewidth = 2) +
  geom_point(size = 1.4) +
  geom_smooth(se = FALSE, colour = "red") +
  geom_hline(yintercept = 0.9) +
  theme_minimal()

quantile(family_fl_acc$.estimate, seq(0, 1, by = 0.05))

## 85% of families > 85% accuracy
## 75% of families > 90% accuracy
## 55% of families > 95% accuracy

ggplot(genus_fr_acc, aes(n, .estimate)) +
  geom_point() +
  geom_smooth(se = FALSE, colour = "red") +
  geom_hline(yintercept = 0.9) +
  #scale_x_sqrt() +
  theme_minimal()

sum(genus_fr_acc$.estimate[genus_fr_acc$n < 10], na.rm = TRUE) / sum(genus_fr_acc$n < 10)

ggplot(family_fr_acc, aes(n, .estimate)) +
  geom_point() +
  geom_smooth(se = FALSE, colour = "red") +
  geom_hline(yintercept = 0.9) +
  #scale_x_sqrt() +
  theme_minimal()

p <- ggplot(family_fl_acc, aes(n, .estimate)) +
  geom_point(aes(colour = order, text = family)) +
  geom_smooth(se = FALSE, colour = "red") +
  geom_hline(yintercept = 0.9) +
  geom_vline(xintercept = 600, colour = "green") +
  xlim(c(0, 3500)) +
  theme_minimal()

plotly::ggplotly(p, tooltip = c("text", "order", "n", ".estimate"))


quantile(family_fr_acc$.estimate, seq(0, 1, by = 0.05), na.rm = TRUE)

## 75% of families > 85% accuracy
## 60% of families > 90% accuracy
## 45% of families > 95% accuracy

######### problematic genera #######
prob_gen <- genus_fl_acc |>
  filter(flower_prop == .estimate |
         flower_prop == 1 - .estimate)

######### problematic families #######
prob_fam <- family_fl_acc |>
  filter(flower_prop == .estimate |
         flower_prop == 1 - .estimate)


genus_fl_jind <- test_dat |>
  filter(partition == "testing") |>
  group_by(genus) |>
  j_index(flower, .pred_fl_max) |>
  left_join(genus_summ)

family_fl_acc <- test_dat |>
  filter(partition == "testing") |>
  group_by(family) |>
  accuracy(flower, .pred_fl_max) |>
  left_join(test_dat |>
              filter(partition == "testing") |>
              group_by(family) |>
              summarise(n = n()))

family_fl_jind <- test_dat |>
  filter(partition == "testing") |>
  group_by(family) |>
  j_index(flower, .pred_fl_max) |>
  left_join(test_dat |>
              filter(partition == "testing") |>
              group_by(family) |>
              summarise(n = n()))

mean(genus_fl_acc$.estimate, na.rm = TRUE)
mean(family_fl_acc$.estimate, na.rm = TRUE)

ggplot(family_fl_acc, aes(.estimate)) +
  geom_histogram() +
  theme_minimal()

ggplot(genus_fl_acc, aes(.estimate)) +
  geom_histogram() +
  theme_minimal()

ggplot(family_fl_jind, aes(.estimate)) +
  geom_histogram() +
  theme_minimal()

sum(family_fl_jind$.estimate > 0.6, na.rm = TRUE)/nrow(family_fl_jind)

