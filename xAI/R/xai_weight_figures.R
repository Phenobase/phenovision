#!/usr/bin/env Rscript
# =============================================================================
# xAI Weight Displacement Figures
# =============================================================================
#
# Visualizes weight space displacement trajectories and rates.
# Uses data from xAI/py/weight_displacement.py output CSVs.
# =============================================================================

library(tidyverse)
library(wesanderson)
library(colorspace)
library(ragg)

# =============================================================================
# 1. Load data
# =============================================================================

cat("Loading data...\n")

plantclef_disp <- read_csv("xAI/output/plantclef_weight_displacement.csv",
                            show_col_types = FALSE) |>
  mutate(model = "Virtual Taxonomist")

mae_disp <- read_csv("xAI/output/mae_weight_displacement.csv",
                      show_col_types = FALSE) |>
  mutate(model = "ImageNet Generalist")

disp_combined <- bind_rows(plantclef_disp, mae_disp)

# Phase transition steps (from run summaries)
plantclef_transition <- 1400
mae_transition <- 2000

# Add phase2_batch
disp_combined <- disp_combined |>
  mutate(
    phase2_batch = case_when(
      model == "Virtual Taxonomist" ~ global_step - plantclef_transition,
      model == "ImageNet Generalist" ~ global_step - mae_transition
    ),
    is_phase2 = phase == "unfrozen"
  )

# Load validation data for asymptote info
val_combined <- read_csv("xAI/output/xai_validation_combined.csv", show_col_types = FALSE)

# Asymptote points (from previous analysis)
plantclef_peak_batch <- 1200
mae_peak_batch <- 3000

# Find displacement at asymptote for each model
# PlantCLEF: p2_batch 1200 = global_step ~2600
plantclef_at_peak <- disp_combined |>
  filter(model == "Virtual Taxonomist", is_phase2) |>
  mutate(dist_to_peak = abs(phase2_batch - plantclef_peak_batch)) |>
  slice_min(dist_to_peak) |>
  slice(1)

mae_at_peak <- disp_combined |>
  filter(model == "ImageNet Generalist", is_phase2) |>
  mutate(dist_to_peak = abs(phase2_batch - mae_peak_batch)) |>
  slice_min(dist_to_peak) |>
  slice(1)

cat("PlantCLEF at peak (p2_batch", plantclef_at_peak$phase2_batch,
    "): displacement =", round(plantclef_at_peak$total_displacement, 2), "\n")
cat("MAE at peak (p2_batch", mae_at_peak$phase2_batch,
    "): displacement =", round(mae_at_peak$total_displacement, 2), "\n")

# Rates of movement in weight space
plantclef_weight_rate <- plantclef_at_peak$total_displacement / plantclef_peak_batch
mae_weight_rate <- mae_at_peak$total_displacement / mae_peak_batch

cat("PlantCLEF weight rate:", round(plantclef_weight_rate * 1000, 2), "displacement/1000 batches\n")
cat("MAE weight rate:", round(mae_weight_rate * 1000, 2), "displacement/1000 batches\n")

# =============================================================================
# 2. Theme and palette
# =============================================================================

pal2 <- wes_palette("FantasticFox1")[c(3, 5)]
names(pal2) <- c("Virtual Taxonomist", "ImageNet Generalist")

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
# 3. Figure: Displacement Trajectory (Phase 2 only)
# =============================================================================

cat("Generating displacement trajectory figure...\n")

disp_p2 <- disp_combined |>
  filter(is_phase2) |>
  arrange(model, phase2_batch)

fig_traj <- ggplot(disp_p2, aes(phase2_batch, total_displacement, color = model)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2.5) +
  # Mark the asymptote points
  geom_point(data = plantclef_at_peak,
             aes(phase2_batch, total_displacement),
             color = pal2["Virtual Taxonomist"], size = 5, shape = 18) +
  geom_point(data = mae_at_peak,
             aes(phase2_batch, total_displacement),
             color = pal2["ImageNet Generalist"], size = 5, shape = 18) +
  # Dashed lines from origin to peak (the "rate" diagonal)
  annotate("segment",
    x = 0, xend = plantclef_peak_batch,
    y = 0, yend = plantclef_at_peak$total_displacement,
    color = pal2["Virtual Taxonomist"], linetype = "dashed", linewidth = 0.7
  ) +
  annotate("segment",
    x = 0, xend = mae_peak_batch,
    y = 0, yend = mae_at_peak$total_displacement,
    color = pal2["ImageNet Generalist"], linetype = "dashed", linewidth = 0.7
  ) +
  # Vertical lines at asymptotes
  geom_vline(xintercept = plantclef_peak_batch,
             color = pal2["Virtual Taxonomist"], linetype = "dotted", alpha = 0.4) +
  geom_vline(xintercept = mae_peak_batch,
             color = pal2["ImageNet Generalist"], linetype = "dotted", alpha = 0.4) +
  # Annotations
  annotate("label",
    x = plantclef_peak_batch + 100,
    y = plantclef_at_peak$total_displacement + 2,
    label = paste0("At peak: ", round(plantclef_at_peak$total_displacement, 1)),
    color = darken(pal2["Virtual Taxonomist"], 0.2),
    fontface = "bold", size = 4, fill = "white", label.size = 0, hjust = 0
  ) +
  annotate("label",
    x = mae_peak_batch + 100,
    y = mae_at_peak$total_displacement + 2,
    label = paste0("At peak: ", round(mae_at_peak$total_displacement, 1)),
    color = darken(pal2["ImageNet Generalist"], 0.2),
    fontface = "bold", size = 4, fill = "white", label.size = 0, hjust = 0
  ) +
  scale_color_manual(values = pal2) +
  scale_x_continuous(name = "Phase 2 Batches", labels = scales::comma) +
  ylab("Weight Displacement (L2 distance from start)") +
  labs(
    title = "Evolutionary Distance: Displacement in Weight Space",
    subtitle = "Straight-line distance from Phase 2 starting point; diamonds mark fitness peak"
  ) +
  theme_evo()

save_fig(fig_traj, "xai_12_weight_displacement_trajectory.png")

# =============================================================================
# 4. Figure: Weight Movement Rate Bar Chart
# =============================================================================

cat("Generating weight movement rate figure...\n")

rate_data <- tibble(
  model = factor(c("Virtual Taxonomist", "ImageNet Generalist"),
                 levels = c("Virtual Taxonomist", "ImageNet Generalist")),
  displacement_at_peak = c(plantclef_at_peak$total_displacement, mae_at_peak$total_displacement),
  batches_to_peak = c(plantclef_peak_batch, mae_peak_batch),
  weight_rate = displacement_at_peak / batches_to_peak * 1000
)

# Three-panel figure
p_disp <- ggplot(rate_data, aes(model, displacement_at_peak, fill = model)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = round(displacement_at_peak, 1)),
            vjust = -0.5, fontface = "bold", size = 5) +
  scale_fill_manual(values = pal2, guide = "none") +
  scale_y_continuous(limits = c(0, max(rate_data$displacement_at_peak) * 1.15)) +
  xlab(NULL) + ylab("L2 Displacement") +
  labs(title = "Distance Traveled to Peak") +
  theme_evo()

p_time <- ggplot(rate_data, aes(model, batches_to_peak, fill = model)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = scales::comma(batches_to_peak)),
            vjust = -0.5, fontface = "bold", size = 5) +
  scale_fill_manual(values = pal2, guide = "none") +
  scale_y_continuous(limits = c(0, max(rate_data$batches_to_peak) * 1.15)) +
  xlab(NULL) + ylab("Batches to Peak") +
  labs(title = "Time to Reach Peak") +
  theme_evo()

p_rate <- ggplot(rate_data, aes(model, weight_rate, fill = model)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = round(weight_rate, 1)),
            vjust = -0.5, fontface = "bold", size = 5) +
  scale_fill_manual(values = pal2, guide = "none") +
  scale_y_continuous(limits = c(0, max(rate_data$weight_rate) * 1.15)) +
  xlab(NULL) + ylab("L2 Displacement / 1000 batches") +
  labs(title = "Rate of Movement in Weight Space") +
  theme_evo()

fig_rates <- patchwork::wrap_plots(p_disp, p_time, p_rate, ncol = 1) +
  patchwork::plot_annotation(
    title = "Weight Space Dynamics: Distance, Time, and Rate",
    subtitle = "Similar distance to peak, but Virtual Taxonomist gets there 2.5x faster",
    theme = theme(
      plot.title = element_text(face = "bold", size = 18),
      plot.subtitle = element_text(color = "grey40", size = 13)
    )
  )

save_fig(fig_rates, "xai_13_weight_rate_summary.png", height = 2400)

# =============================================================================
# 5. Save summary data
# =============================================================================

rate_data |> write_csv("xAI/output/xai_weight_rate_summary.csv")

cat("\n=== Weight figures generated ===\n")
cat("PlantCLEF: displacement =", round(plantclef_at_peak$total_displacement, 1),
    "in", plantclef_peak_batch, "batches → rate =",
    round(plantclef_weight_rate * 1000, 1), "/1000 batches\n")
cat("MAE: displacement =", round(mae_at_peak$total_displacement, 1),
    "in", mae_peak_batch, "batches → rate =",
    round(mae_weight_rate * 1000, 1), "/1000 batches\n")
cat("PlantCLEF rate is", round(plantclef_weight_rate / mae_weight_rate, 1),
    "x faster\n")
