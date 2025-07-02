library(tidyverse)
library(tidymodels)
library(probably)

test_dat <- read_rds("output/model_04_13_2024/epoch_4_testing_data.rds")
#test_dat <- read_rds("output/epoch_4_testing_data.rds")

test_dat <- test_dat |>
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

test_dat <- test_dat |>
  mutate(test = ifelse(partition == "training", "train", "test"))

imb_fl <- test_dat |>
  filter(test == "train") |>
  group_by(family) |>
  summarise(imbalance = (mean(as.numeric(as.character(flowering)), na.rm = TRUE) - 0.5) * 2)

imb_fr <- test_dat |>
  filter(test == "train") |>
  group_by(family) |>
  summarise(imbalance = (mean(as.numeric(as.character(fruiting)), na.rm = TRUE) - 0.5) * 2)

acc_by_fam_fl <- test_dat |>
  group_by(test, family) |>
  accuracy(.truth_flower, .class_flower) |>
  ungroup()

acc_by_fam_fr <- test_dat |>
  group_by(test, family) |>
  accuracy(.truth_fruit, .class_fruit) |>
  ungroup()

equiv_by_fam <- test_dat |>
  group_by(test, family) |>
  summarise(equiv_prop_fl = sum(is.na(as.character(.class_flower))) / n(),
            equiv_prop_fr = sum(is.na(as.character(.class_fruit))) / n(),
            count = n()) |>
  ungroup()

acc_by_fam_fr_total <- test_dat |>
  group_by(test, family) |>
  accuracy(.truth_fruit, .class_fruit_noeq) |>
  ungroup()

acc_by_fam_fl_total <- test_dat |>
  group_by(test, family) |>
  accuracy(.truth_flower, .class_flower_noeq) |>
  ungroup()

fam_dat <- equiv_by_fam |>
  left_join(acc_by_fam_fl |>
              select(test, family, .accuracy_family_flower = .estimate)) |>
  left_join(acc_by_fam_fr |>
              select(test, family, .accuracy_family_fruit = .estimate)) |>
  left_join(acc_by_fam_fl_total |>
              select(test, family, .accuracy_family_flower_incl_equiv = .estimate)) |>
  left_join(acc_by_fam_fr_total |>
              select(test, family, .accuracy_family_fruit_incl_equiv = .estimate)) |>
  left_join(imb_fl |>
              select(family, imbalance_fl = imbalance)) |>
  left_join(imb_fr |>
              select(family, imbalance_fr = imbalance))

write_csv(fam_dat, "output/model_04_13_2024/family_stats.csv")
#write_csv(fam_dat, "output/family_stats_w_imbalance.csv")

test_dat <- test_dat |>
  left_join(equiv_by_fam |>
              filter(test == "test") |>
              select(-test)) |>
  left_join(acc_by_fam_fl |>
              filter(test == "test") |>
              select(family, .accuracy_flower = .estimate)) |>
  left_join(acc_by_fam_fr |>
              filter(test == "test") |>
              select(family, .accuracy_fruit = .estimate))
