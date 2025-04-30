library(reticulate)
library(tidyverse)
library(tidymodels)
library(probably)
library(unglue)
library(wesanderson)
library(colorspace)
library(directlabels)

mae_log <- read_lines("output/mae_2025_03_09/pheno_vit_mae-60645826.out")
plantclef_log <- read_lines("output/plantclef_2025_03_09/pheno_vit_plantclef-60645825.out")

mae_epochs <- str_subset(mae_log, "Epoch:") |>
  unglue_data("Epoch: [{epoch}]  [{batch}/2400]  eta: {eta}  lr: {lr}  loss: {loss} ({loss2})  time: {time}  data: {data}  max mem: {max_mem}",
              convert = TRUE)

plantclef_epochs <- str_subset(plantclef_log, "Epoch:") |>
  unglue_data("Epoch: [{epoch}]  [{batch}/2400]  eta: {eta}  lr: {lr}  loss: {loss} ({loss2})  time: {time}  data: {data}  max mem: {max_mem}",
              convert = TRUE)

mae_test <- which(str_detect(mae_log, "Test:")) |>
  map_chr(~paste(mae_log[.x:(.x+5)], collapse = " ")) |>
  unglue_data("Epoch  {epoch}  Test:  flower acc:  {acc_flower}  fruit acc:  {acc_fruit}  flower j-index:  {jind_flower}  fruit j-index:  {jind_fruit}  loss:  {loss} ",
              convert = TRUE)

plantclef_test <- which(str_detect(plantclef_log, "Test:")) |>
  map_chr(~paste(plantclef_log[.x:(.x+5)], collapse = " ")) |>
  unglue_data("Epoch  {epoch}  Test:  flower acc:  {acc_flower}  fruit acc:  {acc_fruit}  flower j-index:  {jind_flower}  fruit j-index:  {jind_fruit}  loss:  {loss} ",
              convert = TRUE)

test_df <- mae_test |>
  mutate(model = "Imagenet") |>
  bind_rows(plantclef_test |>
              mutate(model = "PlantCLEF")) |>
  mutate(acc_mean = (acc_flower + acc_fruit) / 2,
         jind_mean = (jind_flower + jind_fruit) / 2)

acc_df <- test_df |>
  select(epoch, model, starts_with("acc")) |>
  pivot_longer(cols = starts_with("acc"), names_to = "type", values_to = "accuracy") |>
  filter(epoch > 0 & epoch <= 42)

ggplot(acc_df, aes(epoch, accuracy)) +
  geom_path(aes(colour = model)) +
  facet_grid(cols = vars(type)) +
  coord_cartesian(ylim = c(0.87, 1.0)) +
  theme_minimal()

jind_df <- test_df |>
  select(epoch, model, starts_with("jind")) |>
  pivot_longer(cols = starts_with("jind"), names_to = "type", values_to = "j-index") |>
  filter(epoch > 0 & epoch <= 42)

ggplot(jind_df, aes(epoch, `j-index`)) +
  geom_path(aes(colour = model)) +
  facet_grid(cols = vars(type)) +
  coord_cartesian(ylim = c(0.70, 1.0)) +
  theme_minimal()

pal <- wes_palette("FantasticFox1")[c(3, 5)]
pal <- c(pal, "grey40", darken(pal, 0.4), "grey20")
names(pal) <- c("PlantCLEF (fruit)", "PlantCLEF (flower)", "PlantCLEF (mean)",
                "Imagenet (fruit)", "Imagenet (flower)", "Imagenet (mean)")

ragg::agg_png("figures/pretraining_comparison.png", width = 1024, height = 800, scaling = 3)
p <- ggplot(jind_df |> #filter(type != "jind_mean") |>
         mutate(`j-index` = ifelse(is.nan(`j-index`), 0.5, `j-index`)) |>
         mutate(combo = case_when(model == "Imagenet" & type == "jind_fruit" ~ "Imagenet (fruit)",
                                  model == "Imagenet" & type == "jind_flower" ~ "Imagenet (flower)",
                                  model == "Imagenet" & type == "jind_mean" ~ "Imagenet (mean)",
                                  model == "PlantCLEF" & type == "jind_fruit" ~ "PlantCLEF (fruit)",
                                  model == "PlantCLEF" & type == "jind_flower" ~ "PlantCLEF (flower)",
                                  model == "PlantCLEF" & type == "jind_mean" ~ "PlantCLEF (mean)")),
       aes(epoch, `j-index`)) +
  geom_path(aes(colour = combo)) +
  geom_vline(xintercept = 4, colour = "grey40", linetype = 2) +
  geom_vline(xintercept = 11, colour = "grey20", linetype = 2) +
  annotate("text", x = 4.5, y = 0.97, label = "PlantCLEF Max TSS", colour = "grey40", hjust = 0) +
  annotate("text", x = 11.5, y = 0.94, label = "Imagenet Max TSS", colour = "grey20", hjust = 0) +
  #facet_grid(cols = vars(type)) +
  coord_cartesian(ylim = c(0.70, 1.0)) +
  scale_colour_manual(values = pal) +
  scale_x_continuous(expand = expansion(c(0.02, 0.65)), name = "Training Epoch", breaks = c(0, 10, 20, 30, 40, 50)) +
  ylab("True Skill Statistic (TSS)") +
  theme_minimal()
plot(direct.label(p, list("last.qp", cex = 0.9,
                     fontface="bold")))
dev.off()

model_bests <- jind_df |>
  left_join(acc_df |> mutate(type = str_replace_all(type, "acc_", "jind_"))) |>
  filter(type == "jind_mean") |>
  group_by(model) |>
  slice_max(`j-index`) |>
  left_join(jind_df |> select(epoch, model, type2 = type, jind2 = `j-index`)) |>
  left_join(acc_df |> mutate(type = str_replace_all(type, "acc_", "jind_")) |>
              select(epoch, model, type2 = type, acc2 = `accuracy`))

torch <- import("torch")
timm <- import("timm")
PIL <- import("PIL")
ds <- import_from_path("custom_dataset", "py")
del <- import_from_path("remove_bad_images", "py")
engine <- import_from_path("engine_finetune", "PlantCLEF2022")
lrd <- import_from_path("lr_decay", "PlantCLEF2022/util")
misc <- import_from_path("misc", "PlantCLEF2022/util")
eval <- import_from_path("evaluate", "py")
NativeScaler <- misc$NativeScalerWithGradNormCount
types <- import("types")
