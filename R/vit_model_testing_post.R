library(tidyverse)
library(tidymodels)
library(probably)
library(ggforce)
library(patchwork)

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

############## Find new threshold ##############

thresholds_flower <- threshold_perf(val_dat,
                                    flower,
                                    .pred_flower,
                                    thresholds = seq(0, 1, by = 0.005),
                                    metrics = metric_set(sens, spec, accuracy,
                                                         precision, j_index, npv))

thresholds_fruit <- threshold_perf(val_dat,
                                    fruit,
                                    .pred_fruit,
                                    thresholds = seq(0, 1, by = 0.005),
                                    metrics = metric_set(sens, spec, accuracy,
                                                         precision, j_index, npv))

ppv_npv_fl <- thresholds_flower |>
  filter(.metric %in% c("precision", "npv")) |>
  group_by(.threshold) |>
  summarise(.estimate = sum(.estimate) / 2) |>
  ungroup() |>
  mutate(.metric = "ppv+npv", .estimator = "binary")

thresholds_flower <- thresholds_flower |>
  bind_rows(ppv_npv_fl)

ppv_npv_fr <- thresholds_fruit |>
  filter(.metric %in% c("precision", "npv")) |>
  group_by(.threshold) |>
  summarise(.estimate = sum(.estimate) / 2) |>
  ungroup() |>
  mutate(.metric = "ppv+npv", .estimator = "binary")

thresholds_fruit <- thresholds_fruit |>
  bind_rows(ppv_npv_fr)



## we want to maximize specificity, which minimizes commission errors
## that is error where the model predicts a flower when there was not one

threshold_flower_spec <- thresholds_flower |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  filter(sens > 0.90) |>
  arrange(desc(spec))

max_j_fl <- threshold_flower_spec |>
  slice_max(j_index, n = 1)

max_spec_fl <- threshold_flower_spec |>
  slice_head(n = 1)

threshold_flower_spec <- threshold_flower_spec |>
  mutate(sens_dist = sens - max_j_fl$sens[1],
         spec_dist = spec - max_j_fl$spec[1],
         dist_dist = (sens_dist + spec_dist))

max_dist <- threshold_flower_spec |>
  slice_max(dist_dist)

min_dist <- threshold_flower_spec |>
  slice_min(distance, n = 1)

threshold_fruit_spec <- thresholds_fruit |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  filter(sens > 0.90) |>
  arrange(desc(spec))

threshold_fruit_spec_85 <- thresholds_fruit |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  filter(sens > 0.85) |>
  arrange(desc(spec))

threshold_fruit_spec_none <- thresholds_fruit |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  arrange(desc(spec))

max_j_fr <- threshold_fruit_spec_none |>
  slice_max(j_index, n = 1)

max_spec_fr <- threshold_fruit_spec |>
  slice_head(n = 1)

max_j_90_fl <- threshold_flower_spec |>
  filter(j_index >= 0.9) |>
  slice_max(spec, n = 1)

max_j_80_fr <- threshold_fruit_spec_85 |>
  filter(j_index >= 0.8) |>
  slice_max(spec, n = 1)

thresholds_flower <- thresholds_flower |>
  mutate(label = case_when(.metric == "sens" ~ "True Positive Rate (sensitivity)",
                           .metric == "spec" ~ "True Negative Rate (specificity)",
                           .metric == "precision" ~ "Positive Predictive Value (precision)",
                           .metric == "npv" ~ "Negative Predictive Value",
                           .metric == "accuracy" ~ "Accuracy",
                           .metric == "j_index" ~ "True Skill Statistic (TSS)",
                           .metric == "ppv+npv" ~ "PPV + NPV"))

thresholds_fruit <- thresholds_fruit |>
  mutate(label = case_when(.metric == "sens" ~ "True Positive Rate (sensitivity)",
                           .metric == "spec" ~ "True Negative Rate (specificity)",
                           .metric == "precision" ~ "Positive Predictive Value (precision)",
                           .metric == "npv" ~ "Negative Predictive Value",
                           .metric == "accuracy" ~ "Accuracy",
                           .metric == "j_index" ~ "True Skill Statistic (TSS)",
                           .metric == "ppv+npv" ~ "PPV + NPV"))

ggplot(thresholds_flower |>
         filter(.metric %in% c("sens", "spec", "precision", "j_index", "npv", "ppv+npv")),
       aes(.threshold, .estimate)) +
  geom_path(aes(colour = label), size = 1.25, alpha = 0.75) +
  geom_vline(xintercept = max_j_fl$.threshold[1]) +
  geom_vline(xintercept = max_spec_fl$.threshold[1],
             linetype = 2) +
  geom_vline(xintercept = max_j_90_fl$.threshold[1],
             linetype = 3) +
  facet_zoom(x = .threshold > 0.5,
             y = .estimate > 0.75,
             zoom.size = 1) +
  xlab("Threshold Value") +
  ylab("Metric Estimate") +
  scale_color_discrete(name = "Metric") +
  #geom_vline(xintercept = threshold_flower_spec$.threshold[1]) +
  theme_bw() +
  theme(legend.position = "bottom",
        legend.direction = "vertical")

###### Pub version ############

flower_pic <- svgparser::read_svg("data/icons/noun-flowers-6819381.svg")

pal <- c("Negative Predictive Value" = "#008080",
         "True Positive Rate (sensitivity)" = "#DAA520",
         "Positive Predictive Value (precision)" = "#FF7F50",
         "True Negative Rate (specificity)" = "#4682B4")

threshold_ppv_90 <- thresholds_fruit |>
  filter(.metric == "precision") |>
  filter(.estimate >= 0.9)


p1 <- ggplot(thresholds_flower |>
         filter(!is.na(label)) |>
         filter(.metric %in% c("sens", "spec", "precision", "npv")),
       aes(.threshold, .estimate)) +
  geom_path(aes(colour = label), linewidth = 1.5, alpha = 0.85) +
  geom_vline(xintercept = max_j_fl$.threshold[1],
             linetype = 1) +
  geom_vline(xintercept = max_spec_fl$.threshold[1],
             linetype = 2) +
  geom_vline(xintercept = max_j_90_fl$.threshold[1],
             linetype = 3) +
  annotation_custom(flower_pic, xmin = 0, xmax = 0.1,
                    ymin = 0, ymax = 0.1) +
  facet_zoom(x = .threshold > 0.63,
             y = .estimate > 0.75,
             zoom.size = 1, horizontal = FALSE) +
  xlab("Threshold Value") +
  ylab("Metric Estimate") +
  scale_color_manual(name = "Metric", values = pal) +
  #geom_vline(xintercept = threshold_flower_spec$.threshold[1]) +
  theme_light() +
  theme(legend.position = c(0.36, 0.70),
        legend.direction = "vertical",
        legend.box.background = element_rect(colour = "black",
                                             linewidth = 1.25),
        legend.text = element_text(size = 6),
        legend.title = element_text(size = 8))

p1

p2 <- ggplot(thresholds_fruit |>
         filter(!is.na(label)) |>
         filter(.metric %in% c("sens", "spec", "precision", "npv")),
       aes(.threshold, .estimate)) +
  geom_path(aes(colour = label), linewidth = 1.5, alpha = 0.85) +
  geom_vline(xintercept = max_j_fr$.threshold[1],
             linetype = 2) +
  geom_vline(xintercept = max_spec_fr$.threshold[1],
             linetype = 1) +
  geom_vline(xintercept = threshold_ppv_90$.threshold[1],
             linetype = 3) +
  facet_zoom(x = .threshold < 0.65,
             y = .estimate > 0.55,
             zoom.size = 1, horizontal = FALSE) +
  xlab("Threshold Value") +
  ylab("") +
  scale_color_manual(name = "Metric", values = pal) +
  #geom_vline(xintercept = threshold_flower_spec$.threshold[1]) +
  theme_light() +
  theme(legend.position = 'none')


ragg::agg_png("output/Metrics_plot.png", width = 2560, height = 1600,
              scaling = 4)

plot(p1 + p2)

dev.off()


## choice dotted line 0.84: max_j_90_fl$.threshold[1]

thresholds_prec_bal_fr <- thresholds_fruit |>
  filter(.metric %in% c("precision", "sens")) |>
  select(.threshold, .metric, .estimate) |>
  pivot_wider(names_from = .metric,
              values_from = .estimate) |>
  mutate(sum = precision + sens, diff = precision - sens)

theshold_ppv_90 <- thresholds_fruit |>
  filter(.metric == "precision") |>
  filter(.estimate >= 0.9)

ggplot(thresholds_fruit |>
         filter(.metric %in% c("sens", "spec", "precision", "j_index", "npv", "ppv+npv")),
       aes(.threshold, .estimate)) +
  geom_path(aes(colour = label), size = 1.25, alpha = 0.75) +
#  geom_vline(xintercept = max_j_fr$.threshold[1]) +
#  geom_vline(xintercept = max_spec_fr$.threshold[1],
#             linetype = 2) +
#  geom_vline(xintercept = max_j_80_fr$.threshold[1],
#             linetype = 3) +
  facet_zoom(x = .threshold < 0.4,
             y = .estimate > 0.60,
             zoom.size = 1) +
  xlab("Threshold Value") +
  ylab("Metric Estimate") +
  scale_color_discrete(name = "Metric") +
  #geom_vline(xintercept = threshold_flower_spec$.threshold[1]) +
  theme_bw() +
  theme(legend.position = "bottom",
        legend.direction = "vertical")

## choice fruit: 0.175, max_j_80_fr$.threshold[1]
## new choice fruit: 0.53, goes for PPV of 0.9

ggplot(thresholds_fruit,
       aes(.threshold, .estimate)) +
  geom_path(aes(colour = .metric)) +
  geom_vline(xintercept = threshold_fruit_spec$.threshold[1]) +
  theme_minimal()

test_dat <- test_dat |>
  mutate(
    .pred_fl_max = make_two_class_pred(
      estimate = .pred_flower,
      levels = levels(flower),
      threshold = threshold_flower_spec$.threshold[1],
      buffer = 0.025
    ),
    .pred_fr_max = make_two_class_pred(
      estimate = .pred_fruit,
      levels = levels(fruit),
      threshold = threshold_fruit_spec$.threshold[1],
      buffer = 0.025
    )
  )

test_acc_fl <- accuracy(test_dat, flower, .pred_fl_max)
test_acc_fr <- accuracy(test_dat, fruit, .pred_fr_max)

spec(test_dat, flower, .pred_fl_max)
sens(test_dat, flower, .pred_fl_max)

spec(test_dat, fruit, .pred_fr_max)
sens(test_dat, fruit, .pred_fr_max)

conf_mat_fl <- conf_mat(test_dat,
                        flower,
                        .pred_fl_max)

conf_mat_fr <- conf_mat(test_dat,
                        fruit,
                        .pred_fr_max)
