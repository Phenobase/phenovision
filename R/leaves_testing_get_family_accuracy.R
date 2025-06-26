library(tidyverse)
library(tidymodels)
library(probably)

#test_dat <- read_rds("output/model_04_13_2024/epoch_4_testing_data.rds")
test_dat <- read_rds("output/leaves/phenovision-init_model2_04_11_2025/epoch_1_test_results.rds")
leaf_buffers <- read_rds("output/leaves/phenovision-init_model2_04_11_2025/epoch_1_threshold_buffers.csv")

test_dat <- test_dat |>
  mutate(.class_leaves_green = make_two_class_pred(.pred_leaves_green, c("Detected", "Not Detected"),
                                             threshold = leaf_buffers$gr[1],
                                             buffer = leaf_buffers$gr[-1]),
         .class_leaves_colored = make_two_class_pred(.pred_leaves_colored, c("Detected", "Not Detected"),
                                            threshold = leaf_buffers$cl[1],
                                            buffer = leaf_buffers$cl[-1]),
         .class_leaves_breaking_buds = make_two_class_pred(.pred_leaves_breaking_buds, c("Detected", "Not Detected"),
                                                     threshold = leaf_buffers$bb[1],
                                                     buffer = leaf_buffers$bb[-1]),
         .equivocal_leaves_green = ifelse(is_equivocal(.class_leaves_green), "Equivocal", "Unequivocal"),
         .equivocal_leaves_colored = ifelse(is_equivocal(.class_leaves_colored), "Equivocal", "Unequivocal"),
         .equivocal_leaves_breaking_buds = ifelse(is_equivocal(.class_leaves_breaking_buds), "Equivocal", "Unequivocal")) |>
  mutate(.class_leaves_green_noeq = make_two_class_pred(.pred_leaves_green, c("Detected", "Not Detected"),
                                             threshold = leaf_buffers$gr[1]),
         .class_leaves_colored_noeq = make_two_class_pred(.pred_leaves_colored, c("Detected", "Not Detected"),
                                            threshold = leaf_buffers$cl[1]),
         .class_leaves_breaking_buds_noeq = make_two_class_pred(.pred_leaves_breaking_buds, c("Detected", "Not Detected"),
                                                          threshold = leaf_buffers$bb[1])) |>
  mutate(.truth_leaves_green = factor(ifelse(leaves_green == "1", "Detected", "Not Detected"), levels = c("Detected", "Not Detected")),
         .truth_leaves_colored = factor(ifelse(leaves_colored == "1", "Detected", "Not Detected"), levels = c("Detected", "Not Detected")),
         .truth_leaves_breaking_buds = factor(ifelse(leaves_breaking_buds == "1", "Detected", "Not Detected"), levels = c("Detected", "Not Detected")))

test_dat <- test_dat |>
  mutate(test = ifelse(partition == "training", "train", "test"))

acc_by_fam_gr <- test_dat |>
  group_by(test, family) |>
  accuracy(.truth_leaves_green, .class_leaves_green) |>
  ungroup()

acc_by_fam_cl <- test_dat |>
  group_by(test, family) |>
  accuracy(.truth_leaves_colored, .class_leaves_colored) |>
  ungroup()

acc_by_fam_bb <- test_dat |>
  group_by(test, family) |>
  accuracy(.truth_leaves_breaking_buds, .class_leaves_breaking_buds) |>
  ungroup()

equiv_by_fam <- test_dat |>
  group_by(test, family) |>
  summarise(equiv_prop_gr = sum(is.na(as.character(.class_leaves_green))) / n(),
            equiv_prop_cl = sum(is.na(as.character(.class_leaves_colored))) / n(),
            equiv_prop_bb = sum(is.na(as.character(.class_leaves_breaking_buds))) / n(),
            count = n()) |>
  ungroup()

acc_by_fam_gr_total <- test_dat |>
  group_by(test, family) |>
  accuracy(.truth_leaves_green, .class_leaves_green_noeq) |>
  ungroup()

acc_by_fam_cl_total <- test_dat |>
  group_by(test, family) |>
  accuracy(.truth_leaves_colored, .class_leaves_colored_noeq) |>
  ungroup()

acc_by_fam_bb_total <- test_dat |>
  group_by(test, family) |>
  accuracy(.truth_leaves_breaking_buds, .class_leaves_breaking_buds_noeq) |>
  ungroup()

fam_dat <- equiv_by_fam |>
  left_join(acc_by_fam_gr |>
              select(test, family, .accuracy_family_leaves_green = .estimate)) |>
  left_join(acc_by_fam_cl |>
              select(test, family, .accuracy_family_leaves_colored = .estimate)) |>
  left_join(acc_by_fam_bb |>
              select(test, family, .accuracy_family_leaves_breaking_buds = .estimate)) |>
  left_join(acc_by_fam_gr_total |>
              select(test, family, .accuracy_family_leaves_green_incl_equiv = .estimate)) |>
  left_join(acc_by_fam_cl_total |>
              select(test, family, .accuracy_family_leaves_colored_incl_equiv = .estimate)) |>
  left_join(acc_by_fam_bb_total |>
              select(test, family, .accuracy_family_leaves_breaking_buds_incl_equiv = .estimate))

write_csv(fam_dat, "output/leaves/phenovision-init_model2_04_11_2025/family_stats.csv")

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
