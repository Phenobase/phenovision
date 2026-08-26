#!/usr/bin/env Rscript
# =============================================================================
# Pretraining as Preadaptation: Fitness Landscape Figures
# =============================================================================
#
# Compares "Virtual Taxonomist" (PlantCLEF) vs "ImageNet Generalist" (MAE)
# pretraining strategies, framed through evolutionary biology concepts.
#
# Fitness is measured as exp(-validation_loss) — performance on held-out data,
# analogous to "fitness in the wild" rather than "fitness in the lab."
#
# Generates:
#   - Figure 1: Validation fitness vs epoch (the primary story)
#   - Figure 2: Validation fitness gradient (rate of adaptation)
#   - Figure 3: Training fitness vs step (batch-level, for context)
#   - Figure 4: Training fitness gradient (early epochs)
#
# Saves intermediate data to xAI/output/ for downstream analyses.
# =============================================================================

library(tidyverse)
library(unglue)
library(wesanderson)
library(colorspace)
library(zoo)
library(ragg)

# =============================================================================
# 1. Parse training logs
# =============================================================================

mae_log <- read_lines("output/mae_2025_03_09/pheno_vit_mae-60645826.out")
plantclef_log <- read_lines("output/plantclef_2025_03_09/pheno_vit_plantclef-60645825.out")

# --- Batch-level training data ---
parse_batch_data <- function(log_lines) {
  str_subset(log_lines, "Epoch:") |>
    str_subset("Total time", negate = TRUE) |>
    unglue_data(
      "Epoch: [{epoch}]  [{batch}/2400]  eta: {eta}  lr: {lr}  loss: {loss} ({loss_running})  time: {time}  data: {data}  max mem: {max_mem}",
      convert = TRUE
    ) |>
    select(epoch, batch, loss, loss_running, lr)
}

mae_batch <- parse_batch_data(mae_log)
plantclef_batch <- parse_batch_data(plantclef_log)

# --- Per-epoch validation data ---
parse_test_data <- function(log_lines) {
  test_idx <- which(str_detect(log_lines, "Test:"))
  test_idx |>
    map_chr(~ paste(log_lines[.x:(.x + 5)], collapse = " ")) |>
    unglue_data(
      "Epoch  {epoch}  Test:  flower acc:  {acc_flower}  fruit acc:  {acc_fruit}  flower j-index:  {jind_flower}  fruit j-index:  {jind_fruit}  loss:  {loss} ",
      convert = TRUE
    )
}

mae_test <- parse_test_data(mae_log)
plantclef_test <- parse_test_data(plantclef_log)

# Handle NaN/NA in epoch 1
plantclef_test <- plantclef_test |>
  mutate(
    across(starts_with("acc_"), ~ ifelse(is.nan(.x), NA_real_, .x)),
    across(starts_with("jind_"), ~ replace_na(.x, 0))
  )
mae_test <- mae_test |>
  mutate(across(starts_with("jind_"), ~ replace_na(.x, 0)))

# =============================================================================
# 2. Compute fitness
# =============================================================================

# --- Batch-level (training fitness = "lab fitness") ---
add_fitness <- function(batch_df, model_name) {
  batch_df |>
    mutate(
      model = model_name,
      global_step = (epoch - 1) * 2400 + batch,
      fitness = exp(-loss)
    ) |>
    arrange(global_step)
}

mae_fitness <- add_fitness(mae_batch, "ImageNet Generalist")
plantclef_fitness <- add_fitness(plantclef_batch, "Virtual Taxonomist")

combined_batch <- bind_rows(plantclef_fitness, mae_fitness)

smooth_window <- 151
gradient_window <- 251

combined_smooth <- combined_batch |>
  group_by(model) |>
  arrange(global_step) |>
  mutate(
    fitness_smooth = zoo::rollmean(fitness, k = smooth_window, fill = NA, align = "center"),
    gradient_raw = c(NA, diff(fitness_smooth)),
    gradient_smooth = zoo::rollmean(gradient_raw, k = gradient_window, fill = NA, align = "center")
  ) |>
  ungroup()

# --- Epoch-level (validation fitness = "field fitness") ---
test_combined <- bind_rows(
  plantclef_test |> mutate(model = "Virtual Taxonomist"),
  mae_test |> mutate(model = "ImageNet Generalist")
) |>
  mutate(
    jind_mean = (jind_flower + jind_fruit) / 2,
    val_fitness = exp(-loss)
  )

# Validation fitness gradient
test_combined <- test_combined |>
  group_by(model) |>
  arrange(epoch) |>
  mutate(
    val_fitness_delta = c(NA, diff(val_fitness))
  ) |>
  ungroup()

# =============================================================================
# 3. Save intermediate data
# =============================================================================

combined_smooth |>
  select(model, epoch, batch, global_step, loss, fitness, fitness_smooth, gradient_smooth, lr) |>
  write_csv("xAI/output/batch_fitness_combined.csv")

test_combined |>
  write_csv("xAI/output/epoch_validation_combined.csv")

cat("Intermediate data saved to xAI/output/\n")

# =============================================================================
# 4. Theme and palette setup
# =============================================================================

pal2 <- wes_palette("FantasticFox1")[c(3, 5)]
names(pal2) <- c("Virtual Taxonomist", "ImageNet Generalist")

# Portrait-oriented figures for talks (taller than wide, emphasizes y-axis changes)
# Using 4:5 aspect ratio
fig_width <- 1400
fig_height <- 1750

theme_evo <- function(base_size = 16) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = rel(1.2)),
      plot.subtitle = element_text(color = "grey40", size = rel(0.9)),
      axis.title = element_text(face = "bold"),
      legend.position = "top",
      legend.title = element_blank(),
      legend.text = element_text(size = rel(1.0)),
      panel.grid.minor = element_blank(),
      plot.margin = margin(15, 20, 10, 15)
    )
}

save_fig <- function(plot, filename, width = fig_width, height = fig_height) {
  path <- file.path("xAI/figures", filename)
  ragg::agg_png(path, width = width, height = height, res = 150, scaling = 2)
  print(plot)
  dev.off()
  cat("Saved:", path, "\n")
}

# =============================================================================
# 5. Figure 1: Validation Fitness vs Epoch (THE primary figure)
# =============================================================================
# This is "fitness in the wild" — performance on data the model has never seen.
# Peaks then declines = overspecialization, a biologically real phenomenon.

# Find peak epochs
peaks <- test_combined |>
  filter(epoch > 1) |>
  group_by(model) |>
  slice_max(val_fitness) |>
  slice(1)

fig1 <- ggplot(
  test_combined |> filter(epoch > 0),
  aes(epoch, val_fitness, color = model)
) +
  geom_line(linewidth = 1.3) +
  geom_point(size = 2.5) +
  # Peak markers
  geom_vline(
    xintercept = peaks$epoch[peaks$model == "Virtual Taxonomist"],
    color = pal2["Virtual Taxonomist"], linetype = 2, alpha = 0.5
  ) +
  geom_vline(
    xintercept = peaks$epoch[peaks$model == "ImageNet Generalist"],
    color = pal2["ImageNet Generalist"], linetype = 2, alpha = 0.5
  ) +
  annotate("text",
    x = peaks$epoch[peaks$model == "Virtual Taxonomist"] + 0.8,
    y = peaks$val_fitness[peaks$model == "Virtual Taxonomist"] + 0.003,
    label = paste0("Peak (epoch ", peaks$epoch[peaks$model == "Virtual Taxonomist"], ")"),
    color = darken(pal2["Virtual Taxonomist"], 0.2), hjust = 0, size = 4, fontface = "bold"
  ) +
  annotate("text",
    x = peaks$epoch[peaks$model == "ImageNet Generalist"] + 0.8,
    y = peaks$val_fitness[peaks$model == "ImageNet Generalist"] - 0.003,
    label = paste0("Peak (epoch ", peaks$epoch[peaks$model == "ImageNet Generalist"], ")"),
    color = darken(pal2["ImageNet Generalist"], 0.2), hjust = 0, size = 4, fontface = "bold"
  ) +
  # Random baseline
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey60") +
  annotate("text", x = 40, y = 0.505, label = "Random Baseline",
           color = "grey50", hjust = 0.5, size = 4) +
  scale_color_manual(values = pal2) +
  scale_x_continuous(name = "Epoch", breaks = seq(0, 45, 5)) +
  scale_y_continuous(
    name = expression(Fitness~~(e^{-L[val]})),
    breaks = seq(0.3, 0.7, 0.05)
  ) +
  coord_cartesian(ylim = c(0.3, 0.7)) +
  labs(
    title = "Fitness in the Wild",
    subtitle = "Performance on unseen data (validation set)"
  ) +
  theme_evo()

save_fig(fig1, "val_fitness_vs_epoch.png")

# --- Figure 1b: Zoomed to first 20 epochs ---
fig1b <- ggplot(
  test_combined |> filter(epoch > 0, epoch <= 20),
  aes(epoch, val_fitness, color = model)
) +
  geom_line(linewidth = 1.3) +
  geom_point(size = 3) +
  geom_vline(
    xintercept = peaks$epoch[peaks$model == "Virtual Taxonomist"],
    color = pal2["Virtual Taxonomist"], linetype = 2, alpha = 0.5
  ) +
  geom_vline(
    xintercept = peaks$epoch[peaks$model == "ImageNet Generalist"],
    color = pal2["ImageNet Generalist"], linetype = 2, alpha = 0.5
  ) +
  annotate("text",
    x = peaks$epoch[peaks$model == "Virtual Taxonomist"] + 0.5,
    y = peaks$val_fitness[peaks$model == "Virtual Taxonomist"] + 0.004,
    label = paste0("Peak (epoch ", peaks$epoch[peaks$model == "Virtual Taxonomist"], ")"),
    color = darken(pal2["Virtual Taxonomist"], 0.2), hjust = 0, size = 4.5, fontface = "bold"
  ) +
  annotate("text",
    x = peaks$epoch[peaks$model == "ImageNet Generalist"] + 0.5,
    y = peaks$val_fitness[peaks$model == "ImageNet Generalist"] - 0.004,
    label = paste0("Peak (epoch ", peaks$epoch[peaks$model == "ImageNet Generalist"], ")"),
    color = darken(pal2["ImageNet Generalist"], 0.2), hjust = 0, size = 4.5, fontface = "bold"
  ) +
  scale_color_manual(values = pal2) +
  scale_x_continuous(name = "Epoch", breaks = seq(0, 20, 2)) +
  scale_y_continuous(
    name = expression(Fitness~~(e^{-L[val]})),
    breaks = seq(0.3, 0.7, 0.02)
  ) +
  coord_cartesian(ylim = c(0.44, 0.60)) +
  labs(
    title = "Fitness in the Wild: Adaptation Phase",
    subtitle = "Peak fitness reached then decline = overspecialization"
  ) +
  theme_evo()

save_fig(fig1b, "val_fitness_vs_epoch_zoomed.png")

# =============================================================================
# 6. Figure 2: Validation Fitness Gradient (rate of adaptation)
# =============================================================================

fig2 <- ggplot(
  test_combined |> filter(epoch > 1, epoch <= 20),
  aes(epoch, val_fitness_delta, color = model)
) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
  annotate("text", x = 17, y = 0.002,
           label = "Improving", color = "grey50", size = 4, fontface = "italic") +
  annotate("text", x = 17, y = -0.002,
           label = "Declining", color = "grey50", size = 4, fontface = "italic") +
  scale_color_manual(values = pal2) +
  scale_x_continuous(name = "Epoch", breaks = seq(0, 20, 2)) +
  ylab(expression(Delta ~ Fitness / Epoch)) +
  labs(
    title = "Rate of Adaptation in the Wild",
    subtitle = "Marginal fitness change per epoch (validation set)"
  ) +
  theme_evo()

save_fig(fig2, "val_fitness_gradient.png")

# =============================================================================
# 7. Figure 3: Training Fitness vs Step (batch-level, contextual)
# =============================================================================
# This is "lab fitness" — how well the model fits the training data.
# Always increases (by definition). Shown for context, not as primary fitness.

fig3 <- ggplot(
  combined_smooth |> filter(!is.na(fitness_smooth)),
  aes(global_step, fitness_smooth, color = model)
) +
  geom_line(linewidth = 1.2) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey60") +
  annotate("text", x = 8000, y = 0.515, label = "Random Baseline",
           color = "grey50", hjust = 0, size = 4) +
  scale_color_manual(values = pal2) +
  scale_x_continuous(
    name = "Training Step",
    labels = scales::comma,
    sec.axis = sec_axis(~ . / 2400, name = "Epoch", breaks = seq(0, 45, 5))
  ) +
  scale_y_continuous(
    name = expression(Training~Fitness~~(e^{-L[train]})),
    limits = c(0.45, 1.0),
    breaks = seq(0.5, 1.0, 0.1)
  ) +
  labs(
    title = "Training Fitness (Lab Environment)",
    subtitle = "Adaptation to training data — always increases, but not true fitness"
  ) +
  theme_evo()

save_fig(fig3, "train_fitness_vs_step.png")

# =============================================================================
# 8. Figure 4: Training Fitness Gradient (early epochs)
# =============================================================================

fig4 <- ggplot(
  combined_smooth |>
    filter(!is.na(gradient_smooth), global_step <= 24000),
  aes(global_step, gradient_smooth, color = model)
) +
  geom_line(linewidth = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
  scale_color_manual(values = pal2) +
  scale_x_continuous(
    name = "Training Step",
    labels = scales::comma,
    sec.axis = sec_axis(~ . / 2400, name = "Epoch", breaks = seq(0, 10, 1))
  ) +
  ylab("Rate of Training Adaptation (dFitness / dStep)") +
  labs(
    title = "Adaptation Rate: Early Training",
    subtitle = "How fast each lineage adapts to the training environment"
  ) +
  theme_evo()

save_fig(fig4, "train_fitness_gradient_early.png")

cat("\nAll figures generated successfully.\n")
