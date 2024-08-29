library(tidyverse)
library(arrow)
library(tidymodels)
library(probably)

inf_dat <- read_csv("output/model_04_13_2024/inference_results.csv")

erin_dat <- bind_rows(
  read_csv("data/annotations/Erin/dr_det_un_ant.csv") |>
    mutate(type = "fr", detected_orig = 1, equivocal = 0, focal_plant = as.character(focal_plant)) |>
    rename(detected_erin = fruits),
  read_csv("data/annotations/Erin/fl_det_eq_ant.csv") |>
    mutate(type = "fl", detected_orig = 1, equivocal = 1, focal_plant = as.character(focal_plant)) |>
    rename(detected_erin = flowers),
  read_csv("data/annotations/Erin/fl_det_un_ant.csv") |>
    mutate(type = "fl", detected_orig = 1, equivocal = 0, focal_plant = as.character(focal_plant)) |>
    rename(detected_erin = flowers),
  read_csv("data/annotations/Erin/fl_nt_eq_ant.csv") |>
    mutate(type = "fl", detected_orig = 0, equivocal = 1, focal_plant = as.character(focal_plant)) |>
    rename(detected_erin = flowers),
  read_csv("data/annotations/Erin/fl_nt_un_ant.csv") |>
    mutate(type = "fl", detected_orig = 0, equivocal = 0, focal_plant = as.character(focal_plant)) |>
    rename(detected_erin = flowers),
  read_csv("data/annotations/Erin/fr_det_un_ant2.csv") |>
    mutate(type = "fr", detected_orig = 1, equivocal = 0, focal_plant = as.character(focal_plant)) |>
    rename(detected_erin = fruits),
  read_csv("data/annotations/Erin/fr_nt_un_ant.csv") |>
    mutate(type = "fr", detected_orig = 0, equivocal = 0, focal_plant = as.character(focal_plant)) |>
    rename(detected_erin = fruits)
)

inf_dat <- inf_dat |>
  filter(file_name %in% erin_dat$file)

inf_dat2 <- inf_dat |>
  select(file_name, .class_flower, .class_fruit) |>
  pivot_longer(c(.class_flower, .class_fruit), names_to = "type", values_to = "detected") |>
  mutate(detected = ifelse(detected == "Not Detected", 0, 1),
         type = ifelse(type == ".class_flower", "fl", "fr")) |>
  rename(file = file_name)

inf_dat3 <- inf_dat |>
  select(file_name, .equivocal_flower, .equivocal_fruit) |>
  pivot_longer(c(.equivocal_flower, .equivocal_fruit), names_to = "type", values_to = "eq") |>
  mutate(eq = ifelse(eq == "Equivocal", 1, 0),
         type = ifelse(type == ".equivocal_flower", "fl", "fr")) |>
  rename(file = file_name)

inf_dat4 <- inf_dat |>
  select(file_name, .pred_flower, .pred_fruit) |>
  pivot_longer(c(.pred_flower, .pred_fruit), names_to = "type", values_to = "pred") |>
  mutate(type = ifelse(type == ".pred_flower", "fl", "fr")) |>
  rename(file = file_name)

erin_dat <- erin_dat |>
  left_join(inf_dat2) |>
  left_join(inf_dat3) |>
  left_join(inf_dat4)

angio_meta <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/angio_photos")
taxa_meta <- open_dataset("/blue/guralnick/share/phenobase_inat_data/metadata/taxa/part-0.parquet")

ranks <- taxa_meta |> pull(rank)
unique(ranks)

families <- taxa_meta |>
  filter(rank == "family") |>
  collect()

erin_dat <- erin_dat |>
  mutate(photo_id = as.integer(fs::path_ext_remove(file)))

erin_dat <- erin_dat |>
  left_join(angio_meta |> select(photo_id, taxon_id) |> filter(photo_id %in% erin_dat$photo_id), copy = TRUE)

erin_dat <- erin_dat |>
  left_join(taxa_meta |> select(taxon_id, ancestry) |> filter(taxon_id %in% unique(erin_dat$taxon_id)), copy = TRUE)

erin_dat <- erin_dat |>
  mutate(ancestry = str_split(ancestry, "/"))

erin_dat <- erin_dat |>
  mutate(family = map(ancestry, ~ match(.x, families$taxon_id)))

## looks like family is (always?) position 7
erin_dat <- erin_dat |>
  mutate(family_id = as.integer(map_chr(ancestry, 7))) |>
  left_join(families |> select(family_id = taxon_id, name))

fam_acc <- erin_dat |>
  mutate(detected_erin = factor(ifelse(detected_erin == "U", NA, detected_erin), levels = c("1", "0")),
         detected = factor(detected, levels = c("1", "0"))) |>
  group_by(name) |>
  accuracy(detected_erin, detected)

counts <- erin_dat |>
  group_by(name) |>
  summarise(count = n())

fam_acc <- fam_acc |>
  left_join(counts)

write_csv(erin_dat, "output/erin_annotations_w_meta.csv")
write_csv(fam_acc, "output/erin_annotations_fam_accuracy.csv")

######## do rethresholding ############

erin_dat <- read_csv("output/erin_annotations_w_meta.csv")

erin_fl <- erin_dat |>
  filter(type == "fl") |>
  mutate(.truth = factor(detected_erin, levels = c("1", "0")),
         .class_eq = make_two_class_pred(pred, levels(.truth),
                                             threshold = 0.84,
                                             buffer = c(0.56, 0.025)),
         .class = make_two_class_pred(pred, levels(.truth),
                                             threshold = 0.84),
         .equivocal = is.na(as.character(.class_eq)))

fl_accs <- erin_fl |>
  group_by(.class_fl, .equivocal_fl, detected_erin) |>
  summarise(count = n()) |>
  group_by(.class_fl, .equivocal_fl) |>
  mutate(prop = count / sum(count))

fl_accs_orig <- erin_fl |>
  group_by(detected_orig, equivocal, detected_erin) |>
  summarise(count = n()) |>
  group_by(detected_orig, equivocal) |>
  mutate(prop = count / sum(count))


erin_fr <- erin_dat |>
  filter(type == "fr") |>
  mutate(.truth = factor(detected_erin, levels = c("1", "0")),
         .class_eq = make_two_class_pred(pred, levels(.truth),
                                            threshold = 0.84,
                                            buffer = c(0.56, 0.025)),
         .class = make_two_class_pred(pred, levels(.truth),
                                         threshold = 0.84),
         .equivocal = is.na(as.character(.class_eq)))

fr_accs <- erin_fr |>
  group_by(.class_fr, .equivocal_fr, detected_erin) |>
  summarise(count = n()) |>
  group_by(.class_fr, .equivocal_fr) |>
  mutate(prop = count / sum(count))

fr_accs_orig <- erin_fr |>
  group_by(detected_orig, equivocal, detected_erin) |>
  summarise(count = n()) |>
  group_by(detected_orig, equivocal) |>
  mutate(prop = count / sum(count))

erin_dat_updated <- bind_rows(
  erin_fl, erin_fr
) |>
  select(file, detected_erin,
         detected_model = .class,
         equivocal = .equivocal,
         repro, focal_plant,
         time, type, pred,
         taxon_id, family_id, family = name)

write_csv(erin_dat_updated, "output/erin_annotations_with_updated_model_predictions_07-08-2024.csv")
