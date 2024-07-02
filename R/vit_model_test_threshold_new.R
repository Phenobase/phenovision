library(tidyverse)
library(tidymodels)
library(probably)
library(ggforce)
library(santoku)
library(wesanderson)

test_dat <- read_rds("output/epoch_4_testing_data.rds")

val_dat <- test_dat |> filter(partition == "validation")
test_dat <- test_dat |> filter(partition == "testing")

test_dat <- bind_rows(val_dat, test_dat) |>
  mutate(.class_flower = make_two_class_pred(.pred_flower,
                                             levels(flower),
                                             threshold = 0.84),
         .class_fruit = make_two_class_pred(.pred_fruit,
                                             levels(fruit),
                                             threshold = 0.53)) |>
  mutate(.cut_flower = chop_evenly(.pred_flower, 100,
                                   labels = lbl_midpoints()),
         .cut_fruit = chop_evenly(.pred_fruit, 100,
                                   labels = lbl_midpoints()))

flower_acc <- test_dat |>
  group_by(.cut_flower) |>
  accuracy(flower, .class_flower) |>
  mutate(type = "flower", value = as.numeric(as.character(.cut_flower))) |>
  ungroup()

fruit_acc <- test_dat |>
  group_by(.cut_fruit) |>
  accuracy(fruit, .class_fruit) |>
  mutate(type = "fruit", value = as.numeric(as.character(.cut_fruit))) |>
  ungroup()

flower_cut_count <- test_dat |>
  group_by(.cut_flower) |>
  summarise(count = n()) |>
  mutate(type = "flower", value = as.numeric(as.character(.cut_flower))) |>
  ungroup()

fruit_cut_count <- test_dat |>
  group_by(.cut_fruit) |>
  summarise(count = n()) |>
  mutate(type = "fruit", value = as.numeric(as.character(.cut_fruit))) |>
  ungroup()

accs <- bind_rows(flower_acc,
                  fruit_acc) |>
  left_join(bind_rows(flower_cut_count, fruit_cut_count)) |>
  group_by(type) |>
  mutate(prop = count / sum(count))

sums <- accs |>
  group_by(type) |>
  summarise(max = max(count))

pal <- wes_palette("FantasticFox1")[c(3, 5)]
names(pal) <- c("fruit", "flower")

ragg::agg_png("output/equivocal_zone_plot_new.png",
              width = 800, height = 640,
              scaling = 2)

ggplot(accs, aes(value, .estimate)) +
  geom_area(aes(value, count / max(sums$max), fill = type)) +
  geom_path(aes(colour = type), linewidth = 0.75) +
  geom_vline(xintercept = 0.53, linewidth = 1, linetype = 2) +
  geom_vline(xintercept = 0.84, linewidth = 1, linetype = 2) +
  xlab("Model Output") +
  ylab("Accuracy") +
  scale_y_continuous(sec.axis = sec_axis(trans = ~.*max(sums$max),
                                         name = "Image Count",
                                         labels = scales::label_comma())) +
  scale_colour_manual(values = pal) +
  scale_fill_manual(values = pal) +
  theme_minimal() +
  theme(legend.position = c(0.5, 0.9))

dev.off()

acc_cutoff <- 0.75
accs_bad_fl <- accs |>
  filter(type == "flower", .metric == "accuracy", .estimate <= acc_cutoff)

accs_bad_fr <- accs |>
  filter(type == "fruit", .metric == "accuracy", .estimate <= acc_cutoff)


test_new_fl <-  test_dat |>
  filter(split == "test") |>
  mutate(new_pred_flower = make_two_class_pred(.pred_flower, 0.84,
                                               levels = levels(flower),
                                               buffer = c(0.56, 0.01)))

test_new_fl |>
  accuracy(flower, new_pred_flower)

test_new_fl |>
  spec(flower, new_pred_flower)

test_new_fl |>
  sens(flower, new_pred_flower)

test_new_fl |>
  ppv(flower, new_pred_flower)

test_new_fl |>
  npv(flower, new_pred_flower)

test_new_fr <-  test_dat |>
  filter(split == "test") |>
  mutate(new_pred_fruit = make_two_class_pred(.pred_fruit, 0.53,
                                               levels = levels(flower),
                                               buffer = c(0.3, 0.22)))

test_new_fr |>
  accuracy(fruit, new_pred_fruit)

test_new_fr |>
  spec(fruit, new_pred_fruit)

test_new_fr |>
  sens(fruit, new_pred_fruit)

test_new_fr |>
  ppv(fruit, new_pred_fruit)

test_new_fr |>
  npv(fruit, new_pred_fruit)

###### Erin's annotations
erin_dat <- read_csv("output/erin_annotations_w_meta.csv")

erin_fl <- erin_dat |>
  filter(type == "fl") |>
  mutate(new_pred = make_two_class_pred(pred, 0.84,
                                        levels = c("1", "0"),
                                        buffer = c(0.56, 0.01)),
         new_pred_noeq = make_two_class_pred(pred, 0.84,
                                        levels = c("1", "0")))

erin_fr <- erin_dat |>
  filter(type == "fr") |>
  mutate(new_pred = make_two_class_pred(pred, 0.53,
                                        levels = c("1", "0"),
                                        buffer = c(0.3, 0.22)),
         new_pred_noeq = make_two_class_pred(pred, 0.53,
                                        levels = c("1", "0")))
