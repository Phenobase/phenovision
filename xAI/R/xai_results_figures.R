#!/usr/bin/env Rscript
# =============================================================================
# xAI Results Figures: Pretraining as Preadaptation
# =============================================================================

library(tidyverse)
library(wesanderson)
library(colorspace)
library(zoo)
library(ragg)

# =============================================================================
# 1. Load data
# =============================================================================

cat("Loading data...\n")

plantclef_val <- read_csv("xAI/output/plantclef_run/metrics/validation_metrics.csv",
                           show_col_types = FALSE) |>
  mutate(model = "Virtual Taxonomist")

mae_val <- read_csv("xAI/output/mae_run/metrics/validation_metrics.csv",
                      show_col_types = FALSE) |>
  mutate(model = "ImageNet Generalist")

val_combined <- bind_rows(plantclef_val, mae_val)

plantclef_grad <- read_csv("xAI/output/plantclef_run/metrics/gradient_norms.csv",
                            show_col_types = FALSE) |>
  mutate(model = "Virtual Taxonomist")

mae_grad <- read_csv("xAI/output/mae_run/metrics/gradient_norms.csv",
                      show_col_types = FALSE) |>
  mutate(model = "ImageNet Generalist")

grad_combined <- bind_rows(plantclef_grad, mae_grad)

# Phase transition steps
plantclef_transition <- val_combined |>
  filter(model == "Virtual Taxonomist", phase == "frozen_final") |>
  pull(global_step) |> first()

mae_transition <- val_combined |>
  filter(model == "ImageNet Generalist", phase == "frozen_final") |>
  pull(global_step) |> first()

# Add Phase 2 batch count
val_combined <- val_combined |>
  mutate(
    phase2_batch = case_when(
      model == "Virtual Taxonomist" & phase == "unfrozen" ~ global_step - plantclef_transition,
      model == "ImageNet Generalist" & phase == "unfrozen" ~ global_step - mae_transition,
      TRUE ~ NA_real_
    )
  )

# Phase 1 plateau values
plantclef_p1_auc <- val_combined |>
  filter(model == "Virtual Taxonomist", phase == "frozen_final") |>
  pull(auc_roc_mean)
mae_p1_auc <- val_combined |>
  filter(model == "ImageNet Generalist", phase == "frozen_final") |>
  pull(auc_roc_mean)

# Detect asymptotic peak (rolling mean, find where improvement < threshold)
detect_asymptote <- function(df, window = 5, threshold = 0.001) {
  df <- df |> arrange(phase2_batch)
  df$auc_smooth <- zoo::rollmean(df$auc_roc_mean, k = window, fill = NA, align = "center")
  # Find the first point where the smoothed value is within threshold of the max smoothed
  max_smooth <- max(df$auc_smooth, na.rm = TRUE)
  asymptote_row <- df |>
    filter(!is.na(auc_smooth), auc_smooth >= max_smooth - threshold) |>
    slice(1)
  asymptote_row
}

plantclef_asymptote <- val_combined |>
  filter(model == "Virtual Taxonomist", phase == "unfrozen") |>
  detect_asymptote()

mae_asymptote <- val_combined |>
  filter(model == "ImageNet Generalist", phase == "unfrozen") |>
  detect_asymptote()

# Peak values (asymptote-based, not raw max)
plantclef_peak_auc <- plantclef_asymptote$auc_smooth
mae_peak_auc <- mae_asymptote$auc_smooth
plantclef_peak_batch <- plantclef_asymptote$phase2_batch
mae_peak_batch <- mae_asymptote$phase2_batch

cat("PlantCLEF: Phase 1 =", round(plantclef_p1_auc, 4),
    " Asymptote =", round(plantclef_peak_auc, 4),
    " at p2_batch =", plantclef_peak_batch, "\n")
cat("MAE: Phase 1 =", round(mae_p1_auc, 4),
    " Asymptote =", round(mae_peak_auc, 4),
    " at p2_batch =", mae_peak_batch, "\n")

# Rates
plantclef_rate <- (plantclef_peak_auc - plantclef_p1_auc) / plantclef_peak_batch
mae_rate <- (mae_peak_auc - mae_p1_auc) / mae_peak_batch

cat("PlantCLEF rate:", round(plantclef_rate * 1000, 4), "AUC/1000 batches\n")
cat("MAE rate:", round(mae_rate * 1000, 4), "AUC/1000 batches\n")

# Gradient-based weight distance (Phase 2)
grad_combined <- grad_combined |>
  mutate(
    phase2_batch = case_when(
      model == "Virtual Taxonomist" ~ global_step - plantclef_transition,
      model == "ImageNet Generalist" ~ global_step - mae_transition
    )
  )

weight_distance <- grad_combined |>
  filter(!is.na(grad_norm_total), is.finite(grad_norm_total)) |>
  group_by(model) |>
  summarise(
    total_path = sum(grad_norm_total, na.rm = TRUE),
    head_path = sum(grad_norm_head, na.rm = TRUE),
    last_block_path = sum(grad_norm_last_block, na.rm = TRUE),
    first_block_path = sum(grad_norm_first_block, na.rm = TRUE),
    n_steps = n(),
    .groups = "drop"
  )

cat("\nWeight space path length (cumulative gradient L2 norm):\n")
print(weight_distance)

cat("\nData loaded.\n\n")

# =============================================================================
# 2. Theme and palette
# =============================================================================

pal2 <- wes_palette("FantasticFox1")[c(3, 5)]
names(pal2) <- c("Virtual Taxonomist", "ImageNet Generalist")

# Wider dimensions to avoid text clipping
fig_width <- 1800
fig_height <- 1800

theme_evo <- function(base_size = 15) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = rel(1.1)),
      plot.subtitle = element_text(color = "grey40", size = rel(0.85)),
      axis.title = element_text(face = "bold"),
      legend.position = "top",
      legend.title = element_blank(),
      legend.text = element_text(size = rel(0.95)),
      panel.grid.minor = element_blank(),
      plot.margin = margin(20, 25, 15, 20)
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
# 3. Figure 1: Complete Fitness Trajectory (Hero Figure)
# =============================================================================

cat("Generating Figure 1: Hero trajectory...\n")

val_plot <- val_combined |> filter(phase != "frozen_final")

fig1 <- ggplot(val_plot, aes(global_step, auc_roc_mean, color = model)) +
  annotate("rect",
    xmin = -Inf, xmax = max(plantclef_transition, mae_transition),
    ymin = -Inf, ymax = Inf, fill = "grey90", alpha = 0.5
  ) +
  annotate("text", x = 200, y = 0.995,
           label = "Phase 1: Equalization", hjust = 0,
           color = "grey50", size = 4, fontface = "italic") +
  annotate("text", x = max(plantclef_transition, mae_transition) + 200, y = 0.995,
           label = "Phase 2: Evolution", hjust = 0,
           color = "grey50", size = 4, fontface = "italic") +
  geom_vline(xintercept = plantclef_transition,
             color = pal2["Virtual Taxonomist"], linetype = "dashed", alpha = 0.6) +
  geom_vline(xintercept = mae_transition,
             color = pal2["ImageNet Generalist"], linetype = "dashed", alpha = 0.6) +
  geom_hline(yintercept = plantclef_p1_auc,
             color = darken(pal2["Virtual Taxonomist"], 0.3),
             linetype = "dotted", alpha = 0.5) +
  geom_hline(yintercept = mae_p1_auc,
             color = darken(pal2["ImageNet Generalist"], 0.3),
             linetype = "dotted", alpha = 0.5) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 0.8, alpha = 0.5) +
  annotate("text", x = 100, y = plantclef_p1_auc + 0.007,
           label = paste0("Ecological fitting: ", round(plantclef_p1_auc, 3)),
           color = darken(pal2["Virtual Taxonomist"], 0.3),
           hjust = 0, size = 3.5) +
  annotate("text", x = 100, y = mae_p1_auc - 0.007,
           label = paste0("Ecological fitting: ", round(mae_p1_auc, 3)),
           color = darken(pal2["ImageNet Generalist"], 0.3),
           hjust = 0, size = 3.5) +
  scale_color_manual(values = pal2) +
  scale_x_continuous(name = "Training Step", labels = scales::comma) +
  scale_y_continuous(name = "Fitness (AUC-ROC)", breaks = seq(0.7, 1.0, 0.02)) +
  coord_cartesian(ylim = c(0.70, 1.0)) +
  labs(
    title = "Fitness Trajectory: Preadapted vs Naive Lineage",
    subtitle = "Two-phase experiment: equalization (head only) then evolution (full model)"
  ) +
  theme_evo()

save_fig(fig1, "xai_01_hero_trajectory.png")

# =============================================================================
# 4. Figure 2: Phase 1 Zoom (Ecological Fitting)
# =============================================================================

cat("Generating Figure 2: Phase 1 zoom...\n")

val_p1 <- val_combined |> filter(phase %in% c("frozen", "frozen_final"))

fig2 <- ggplot(val_p1, aes(global_step, auc_roc_mean, color = model)) +
  geom_line(linewidth = 1.3) +
  geom_point(size = 2) +
  geom_hline(yintercept = plantclef_p1_auc,
             color = pal2["Virtual Taxonomist"], linetype = "dotted", alpha = 0.6) +
  geom_hline(yintercept = mae_p1_auc,
             color = pal2["ImageNet Generalist"], linetype = "dotted", alpha = 0.6) +
  annotate("text",
    x = max(val_p1$global_step) * 0.55, y = plantclef_p1_auc + 0.005,
    label = round(plantclef_p1_auc, 3),
    color = darken(pal2["Virtual Taxonomist"], 0.2), fontface = "bold", size = 5
  ) +
  annotate("text",
    x = max(val_p1$global_step) * 0.55, y = mae_p1_auc - 0.005,
    label = round(mae_p1_auc, 3),
    color = darken(pal2["ImageNet Generalist"], 0.2), fontface = "bold", size = 5
  ) +
  annotate("segment",
    x = max(val_p1$global_step) * 0.82, xend = max(val_p1$global_step) * 0.82,
    y = mae_p1_auc, yend = plantclef_p1_auc,
    color = "grey30", linewidth = 0.8,
    arrow = arrow(ends = "both", length = unit(0.08, "inches"))
  ) +
  annotate("label",
    x = max(val_p1$global_step) * 0.82,
    y = (plantclef_p1_auc + mae_p1_auc) / 2,
    label = paste0("Gap = ", round(plantclef_p1_auc - mae_p1_auc, 3)),
    color = "grey30", size = 4, fontface = "bold", fill = "white", label.size = 0
  ) +
  scale_color_manual(values = pal2) +
  scale_x_continuous(name = "Training Step", labels = scales::comma) +
  ylab("Fitness (AUC-ROC)") +
  coord_cartesian(ylim = c(0.70, 0.95)) +
  labs(
    title = "Phase 1: Ecological Fitting",
    subtitle = "Frozen backbone, head-only training reveals representation quality"
  ) +
  theme_evo()

save_fig(fig2, "xai_02_phase1_ecological_fitting.png")

# =============================================================================
# 5. Figure 3: Phase 2 with rate shading
# =============================================================================

cat("Generating Figure 3: Phase 2 with rate shading...\n")

val_p2 <- val_combined |> filter(phase == "unfrozen", !is.na(phase2_batch))

# Build shading polygons (area between curve and the diagonal start-to-peak line)
make_rate_polygon <- function(df, p1_auc, peak_batch, peak_auc) {
  segment <- df |> filter(phase2_batch > 0, phase2_batch <= peak_batch)
  # Diagonal line from (0, p1_auc) to (peak_batch, peak_auc)
  slope <- (peak_auc - p1_auc) / peak_batch
  diagonal_y <- p1_auc + slope * segment$phase2_batch
  # Polygon: curve on top, diagonal on bottom (reversed)
  bind_rows(
    segment |> select(phase2_batch, y = auc_roc_mean),
    tibble(phase2_batch = rev(segment$phase2_batch), y = rev(diagonal_y))
  )
}

poly_plantclef <- val_p2 |>
  filter(model == "Virtual Taxonomist") |>
  make_rate_polygon(plantclef_p1_auc, plantclef_peak_batch, plantclef_peak_auc) |>
  mutate(model = "Virtual Taxonomist")

poly_mae <- val_p2 |>
  filter(model == "ImageNet Generalist") |>
  make_rate_polygon(mae_p1_auc, mae_peak_batch, mae_peak_auc) |>
  mutate(model = "ImageNet Generalist")

fig3 <- ggplot(val_p2, aes(phase2_batch, auc_roc_mean, color = model)) +
  # Shaded area = total fitness gain over time
  geom_polygon(data = poly_plantclef, aes(phase2_batch, y, fill = model),
               alpha = 0.15, color = NA) +
  geom_polygon(data = poly_mae, aes(phase2_batch, y, fill = model),
               alpha = 0.15, color = NA) +
  # Reference lines from Phase 1 plateau to asymptote
  geom_segment(aes(x = 0, xend = plantclef_peak_batch,
                   y = plantclef_p1_auc, yend = plantclef_peak_auc),
               color = pal2["Virtual Taxonomist"], linetype = "dashed", linewidth = 0.6,
               inherit.aes = FALSE) +
  geom_segment(aes(x = 0, xend = mae_peak_batch,
                   y = mae_p1_auc, yend = mae_peak_auc),
               color = pal2["ImageNet Generalist"], linetype = "dashed", linewidth = 0.6,
               inherit.aes = FALSE) +
  # Asymptote markers
  geom_vline(xintercept = plantclef_peak_batch,
             color = pal2["Virtual Taxonomist"], linetype = "dotted", alpha = 0.4) +
  geom_vline(xintercept = mae_peak_batch,
             color = pal2["ImageNet Generalist"], linetype = "dotted", alpha = 0.4) +
  # Main curves
  geom_line(linewidth = 1.1) +
  geom_point(size = 0.8, alpha = 0.4) +
  # Annotations
  annotate("text",
    x = plantclef_peak_batch, y = plantclef_peak_auc + 0.006,
    label = paste0("Asymptote: ", round(plantclef_peak_auc, 3)),
    color = darken(pal2["Virtual Taxonomist"], 0.2), fontface = "bold", size = 4, hjust = 0.5
  ) +
  annotate("text",
    x = mae_peak_batch, y = mae_peak_auc + 0.006,
    label = paste0("Asymptote: ", round(mae_peak_auc, 3)),
    color = darken(pal2["ImageNet Generalist"], 0.2), fontface = "bold", size = 4, hjust = 0.5
  ) +
  scale_color_manual(values = pal2) +
  scale_fill_manual(values = pal2, guide = "none") +
  scale_x_continuous(name = "Phase 2 Batches (from unfreeze point)", labels = scales::comma) +
  ylab("Fitness (AUC-ROC)") +
  coord_cartesian(ylim = c(0.80, 1.0)) +
  labs(
    title = "Phase 2: Evolutionary Adaptation",
    subtitle = "Shaded area = cumulative fitness gain; dashed line = average rate of gain"
  ) +
  theme_evo()

save_fig(fig3, "xai_03_phase2_evolution.png")

# =============================================================================
# 6. Figure 4: Decomposition Bar Chart
# =============================================================================

cat("Generating Figure 4: Decomposition...\n")

decomp <- tibble(
  model = rep(c("Virtual Taxonomist", "ImageNet Generalist"), each = 2),
  component = rep(c("Ecological Fitting\n(Phase 1)", "Evolutionary Gain\n(Phase 2)"), 2),
  value = c(
    plantclef_p1_auc, plantclef_peak_auc - plantclef_p1_auc,
    mae_p1_auc, mae_peak_auc - mae_p1_auc
  )
) |>
  mutate(
    component = factor(component,
      levels = c("Evolutionary Gain\n(Phase 2)", "Ecological Fitting\n(Phase 1)")
    ),
    model = factor(model, levels = c("Virtual Taxonomist", "ImageNet Generalist"))
  )

totals <- decomp |> group_by(model) |> summarise(total = sum(value), .groups = "drop")

fig4 <- ggplot(decomp, aes(x = model, y = value, fill = component)) +
  geom_col(width = 0.6) +
  geom_text(data = totals,
    aes(x = model, y = total, label = round(total, 3), fill = NULL),
    vjust = -0.5, fontface = "bold", size = 5
  ) +
  geom_text(aes(label = round(value, 3)),
    position = position_stack(vjust = 0.5),
    color = "white", fontface = "bold", size = 4.5
  ) +
  scale_fill_manual(values = c(
    "Ecological Fitting\n(Phase 1)" = "grey50",
    "Evolutionary Gain\n(Phase 2)" = "grey25"
  ), name = NULL) +
  scale_y_continuous(name = "Fitness (AUC-ROC)", limits = c(0, 1.05),
                     breaks = seq(0, 1.0, 0.1)) +
  xlab(NULL) +
  labs(
    title = "Fitness Decomposition",
    subtitle = "Ecological fitting (where you start) vs evolutionary gain (how much you improve)"
  ) +
  theme_evo() +
  theme(legend.position = "bottom")

save_fig(fig4, "xai_04_decomposition.png")

# =============================================================================
# 7. Figure 5: Summary Bar Chart (start, end, rate, time to peak)
# =============================================================================

cat("Generating Figure 5: Summary metrics...\n")

# Load weight displacement data for weight space rate
plantclef_disp <- read_csv("xAI/output/plantclef_weight_displacement.csv", show_col_types = FALSE)
mae_disp <- read_csv("xAI/output/mae_weight_displacement.csv", show_col_types = FALSE)

# Find displacement closest to each model's asymptote
plantclef_disp_p2 <- plantclef_disp |>
  filter(phase == "unfrozen") |>
  mutate(p2_batch = global_step - plantclef_transition)
mae_disp_p2 <- mae_disp |>
  filter(phase == "unfrozen") |>
  mutate(p2_batch = global_step - mae_transition)

plantclef_disp_at_peak <- plantclef_disp_p2 |>
  mutate(d = abs(p2_batch - plantclef_peak_batch)) |> slice_min(d) |> slice(1)
mae_disp_at_peak <- mae_disp_p2 |>
  mutate(d = abs(p2_batch - mae_peak_batch)) |> slice_min(d) |> slice(1)

plantclef_weight_rate <- plantclef_disp_at_peak$total_displacement / plantclef_peak_batch * 1000
mae_weight_rate <- mae_disp_at_peak$total_displacement / mae_peak_batch * 1000

summary_metrics <- tibble(
  model = factor(c("Virtual Taxonomist", "ImageNet Generalist"),
                 levels = c("Virtual Taxonomist", "ImageNet Generalist")),
  start = c(plantclef_p1_auc, mae_p1_auc),
  peak = c(plantclef_peak_auc, mae_peak_auc),
  gain = c(plantclef_peak_auc - plantclef_p1_auc, mae_peak_auc - mae_p1_auc),
  time_to_peak = c(plantclef_peak_batch, mae_peak_batch),
  rate = c(plantclef_rate * 1000, mae_rate * 1000),  # per 1000 batches
  weight_rate = c(plantclef_weight_rate, mae_weight_rate)  # displacement per 1000 batches
)

# Panel A: start and peak fitness
p5a <- summary_metrics |>
  select(model, `Starting Fitness` = start, `Peak Fitness` = peak) |>
  pivot_longer(-model, names_to = "metric", values_to = "value") |>
  ggplot(aes(model, value, fill = metric)) +
  geom_col(position = "dodge", width = 0.6) +
  geom_text(aes(label = round(value, 3)),
    position = position_dodge(width = 0.6), vjust = -0.5, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("Starting Fitness" = "grey65", "Peak Fitness" = "grey30")) +
  scale_y_continuous(limits = c(0, 1.08), breaks = seq(0, 1, 0.2)) +
  xlab(NULL) + ylab("AUC-ROC") +
  labs(title = "Starting vs Peak Fitness") +
  theme_evo() +
  theme(legend.position = "bottom", legend.title = element_blank())

# Panel B: time to peak
p5b <- ggplot(summary_metrics, aes(model, time_to_peak, fill = model)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = scales::comma(time_to_peak)), vjust = -0.5, fontface = "bold", size = 4) +
  scale_fill_manual(values = pal2, guide = "none") +
  scale_y_continuous(limits = c(0, max(summary_metrics$time_to_peak) * 1.15)) +
  xlab(NULL) + ylab("Batches to Asymptote") +
  labs(title = "Time to Peak") +
  theme_evo()

# Panel C: rate
p5c <- ggplot(summary_metrics, aes(model, rate, fill = model)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = round(rate, 2)), vjust = -0.5, fontface = "bold", size = 4) +
  scale_fill_manual(values = pal2, guide = "none") +
  scale_y_continuous(limits = c(0, max(summary_metrics$rate) * 1.15)) +
  xlab(NULL) + ylab(expression(Delta * "AUC / 1000 batches")) +
  labs(title = "Rate of Adaptation") +
  theme_evo()

# Panel D: weight space rate
p5d <- ggplot(summary_metrics, aes(model, weight_rate, fill = model)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = round(weight_rate, 1)), vjust = -0.5, fontface = "bold", size = 4) +
  scale_fill_manual(values = pal2, guide = "none") +
  scale_y_continuous(limits = c(0, max(summary_metrics$weight_rate) * 1.15)) +
  xlab(NULL) + ylab("L2 Displacement / 1000 batches") +
  labs(title = "Rate of Phenotypic Change") +
  theme_evo()

fig5 <- patchwork::wrap_plots(p5a, p5b, p5c, p5d, ncol = 1) +
  patchwork::plot_annotation(
    title = "Evolutionary Performance Summary",
    subtitle = "Fitness, convergence time, adaptation rate, and phenotypic rate of change",
    theme = theme(
      plot.title = element_text(face = "bold", size = 18),
      plot.subtitle = element_text(color = "grey40", size = 13)
    )
  )

save_fig(fig5, "xai_05_summary_metrics.png", height = 2800)

# =============================================================================
# 8. Figure 6: Per-Class AUC-ROC (Phase 2 only)
# =============================================================================

cat("Generating Figure 6: Per-class AUC-ROC (Phase 2)...\n")

val_class <- val_combined |>
  filter(phase == "unfrozen", !is.na(phase2_batch)) |>
  select(phase2_batch, model, auc_roc_flower, auc_roc_fruit) |>
  pivot_longer(
    cols = starts_with("auc_roc_"),
    names_to = "class", values_to = "auc_roc",
    names_prefix = "auc_roc_"
  ) |>
  mutate(class = str_to_title(class))

fig6 <- ggplot(val_class, aes(phase2_batch, auc_roc, color = model, linetype = class)) +
  geom_line(linewidth = 1.0) +
  scale_color_manual(values = pal2) +
  scale_linetype_manual(values = c("Flower" = "solid", "Fruit" = "dashed"), name = "Class") +
  scale_x_continuous(name = "Phase 2 Batches", labels = scales::comma) +
  ylab("AUC-ROC") +
  coord_cartesian(ylim = c(0.80, 1.0)) +
  labs(
    title = "Fitness by Phenological Class (Phase 2)",
    subtitle = "Fruit detection shows the largest pretraining advantage"
  ) +
  theme_evo() +
  guides(color = guide_legend(order = 1), linetype = guide_legend(order = 2))

save_fig(fig6, "xai_06_per_class_auc.png")

# =============================================================================
# 9. Figure 7: Gradient Norms by Layer
# =============================================================================

cat("Generating Figure 7: Gradient norms...\n")

smooth_k <- 51

grad_smooth <- grad_combined |>
  filter(!is.na(grad_norm_total), is.finite(grad_norm_total),
         is.finite(grad_norm_head), is.finite(grad_norm_last_block),
         is.finite(grad_norm_first_block)) |>
  group_by(model) |>
  arrange(phase2_batch) |>
  mutate(
    head = zoo::rollmean(grad_norm_head, k = smooth_k, fill = NA, align = "center"),
    last_block = zoo::rollmean(grad_norm_last_block, k = smooth_k, fill = NA, align = "center"),
    first_block = zoo::rollmean(grad_norm_first_block, k = smooth_k, fill = NA, align = "center")
  ) |>
  ungroup() |>
  select(model, phase2_batch, head, last_block, first_block) |>
  pivot_longer(cols = c(head, last_block, first_block),
    names_to = "layer", values_to = "grad_norm") |>
  filter(!is.na(grad_norm)) |>
  mutate(layer = factor(layer,
    levels = c("head", "last_block", "first_block"),
    labels = c("Classification Head", "Last Transformer Block", "First Transformer Block")
  ))

fig7 <- ggplot(grad_smooth, aes(phase2_batch, grad_norm, color = layer)) +
  geom_line(linewidth = 0.8, alpha = 0.8) +
  facet_wrap(~ model, ncol = 1) +
  scale_color_manual(values = c(
    "Classification Head" = "#E69F00",
    "Last Transformer Block" = "#56B4E9",
    "First Transformer Block" = "#009E73"
  ), name = "Layer") +
  scale_x_continuous(name = "Phase 2 Batches", labels = scales::comma) +
  ylab("Gradient L2 Norm (smoothed)") +
  coord_cartesian(ylim = c(0, 1.5)) +
  labs(
    title = "Where Adaptation Happens",
    subtitle = "Per-layer gradient magnitude during backbone evolution"
  ) +
  theme_evo() +
  theme(legend.position = "bottom",
        strip.text = element_text(face = "bold", size = rel(1.0)))

save_fig(fig7, "xai_07_gradient_norms.png")

# =============================================================================
# 10. Figure 8: Weight Space Distance (from gradients)
# =============================================================================

cat("Generating Figure 8: Weight space distance...\n")

# Cumulative gradient path length over Phase 2
grad_cumulative <- grad_combined |>
  filter(!is.na(grad_norm_total), is.finite(grad_norm_total),
         is.finite(grad_norm_head), is.finite(grad_norm_last_block),
         is.finite(grad_norm_first_block)) |>
  group_by(model) |>
  arrange(phase2_batch) |>
  mutate(
    cum_total = cumsum(grad_norm_total),
    cum_head = cumsum(grad_norm_head),
    cum_last = cumsum(grad_norm_last_block),
    cum_first = cumsum(grad_norm_first_block)
  ) |>
  ungroup() |>
  select(model, phase2_batch, cum_total, cum_head, cum_last, cum_first) |>
  pivot_longer(cols = starts_with("cum_"),
    names_to = "layer", values_to = "distance",
    names_prefix = "cum_") |>
  mutate(layer = factor(layer,
    levels = c("total", "head", "last", "first"),
    labels = c("Total", "Classification Head", "Last Block", "First Block")
  ))

fig8 <- ggplot(grad_cumulative |> filter(layer == "Total"),
               aes(phase2_batch, distance, color = model)) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = pal2) +
  scale_x_continuous(name = "Phase 2 Batches", labels = scales::comma) +
  ylab("Cumulative Gradient Path Length") +
  labs(
    title = "Evolutionary Distance in Weight Space",
    subtitle = "Total gradient L2 norm accumulated during backbone evolution"
  ) +
  theme_evo()

save_fig(fig8, "xai_08_weight_distance.png")

# =============================================================================
# 11. Figure 9: Rate of Adaptation
# =============================================================================

cat("Generating Figure 9: Rate of adaptation...\n")

val_p2_rate <- val_p2 |>
  group_by(model) |>
  arrange(phase2_batch) |>
  mutate(
    auc_smooth = zoo::rollmean(auc_roc_mean, k = 5, fill = NA, align = "center"),
    delta_auc = c(NA, diff(auc_smooth)),
    delta_batch = c(NA, diff(phase2_batch)),
    rate = delta_auc / delta_batch
  ) |>
  filter(!is.na(rate), delta_batch > 0) |>
  ungroup()

fig9 <- ggplot(val_p2_rate, aes(phase2_batch, rate * 1000, color = model)) +
  geom_line(linewidth = 0.8, alpha = 0.5) +
  geom_smooth(se = FALSE, span = 0.3, linewidth = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
  scale_color_manual(values = pal2) +
  scale_x_continuous(name = "Phase 2 Batches", labels = scales::comma) +
  ylab(expression(Rate ~ of ~ Adaptation ~ (Delta * AUC / 1000 ~ batches))) +
  labs(
    title = "Evolvability: Speed of Fitness Increase",
    subtitle = "Rate of AUC-ROC improvement during backbone evolution"
  ) +
  theme_evo()

save_fig(fig9, "xai_09_rate_of_adaptation.png")

# =============================================================================
# 12. Figure 10: Validation Loss Trajectory
# =============================================================================

cat("Generating Figure 10: Validation loss...\n")

fig10 <- ggplot(val_plot, aes(global_step, val_loss, color = model)) +
  annotate("rect",
    xmin = -Inf, xmax = max(plantclef_transition, mae_transition),
    ymin = -Inf, ymax = Inf, fill = "grey90", alpha = 0.5
  ) +
  annotate("text", x = 200, y = max(val_plot$val_loss) * 0.95,
           label = "Phase 1", hjust = 0,
           color = "grey50", size = 4, fontface = "italic") +
  annotate("text", x = max(plantclef_transition, mae_transition) + 200,
           y = max(val_plot$val_loss) * 0.95,
           label = "Phase 2", hjust = 0,
           color = "grey50", size = 4, fontface = "italic") +
  geom_vline(xintercept = plantclef_transition,
             color = pal2["Virtual Taxonomist"], linetype = "dashed", alpha = 0.6) +
  geom_vline(xintercept = mae_transition,
             color = pal2["ImageNet Generalist"], linetype = "dashed", alpha = 0.6) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 0.8, alpha = 0.5) +
  scale_color_manual(values = pal2) +
  scale_x_continuous(name = "Training Step", labels = scales::comma) +
  ylab("Validation Loss (BCE)") +
  labs(
    title = "Validation Loss Trajectory",
    subtitle = "Lower = better adapted to the environment"
  ) +
  theme_evo()

save_fig(fig10, "xai_10_val_loss_trajectory.png")

# =============================================================================
# 13. Figure 11: Validation Fitness (exp(-loss))
# =============================================================================

cat("Generating Figure 11: Validation fitness (exp(-loss))...\n")

val_plot_fitness <- val_plot |> mutate(fitness = exp(-val_loss))

fig11 <- ggplot(val_plot_fitness, aes(global_step, fitness, color = model)) +
  annotate("rect",
    xmin = -Inf, xmax = max(plantclef_transition, mae_transition),
    ymin = -Inf, ymax = Inf, fill = "grey90", alpha = 0.5
  ) +
  geom_vline(xintercept = plantclef_transition,
             color = pal2["Virtual Taxonomist"], linetype = "dashed", alpha = 0.6) +
  geom_vline(xintercept = mae_transition,
             color = pal2["ImageNet Generalist"], linetype = "dashed", alpha = 0.6) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 0.8, alpha = 0.5) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey60") +
  annotate("text", x = max(val_plot$global_step) * 0.75, y = 0.51,
           label = "Random Baseline", color = "grey50", size = 3.5) +
  scale_color_manual(values = pal2) +
  scale_x_continuous(name = "Training Step", labels = scales::comma) +
  scale_y_continuous(name = expression(Fitness~~(e^{-L[val]})),
                     breaks = seq(0.4, 0.9, 0.05)) +
  labs(
    title = "Validation Fitness (Survival Probability)",
    subtitle = expression(Fitness == e^{-loss} ~ ": higher = better adapted")
  ) +
  theme_evo()

save_fig(fig11, "xai_11_val_fitness.png")

# =============================================================================
# Summary
# =============================================================================

cat("\n=== All 11 figures generated ===\n")
list.files("xAI/figures", pattern = "xai_") |> cat(sep = "\n")
cat("\n")

# Save combined data
val_combined |> write_csv("xAI/output/xai_validation_combined.csv")
summary_metrics |> write_csv("xAI/output/xai_summary_metrics.csv")
weight_distance |> write_csv("xAI/output/xai_weight_distance.csv")
cat("Data saved to xAI/output/\n")
