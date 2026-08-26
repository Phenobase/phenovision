#!/usr/bin/env Rscript
# =============================================================================
# Three Mechanisms of Preadaptation: Individual Figures
# =============================================================================
#
# Splits the summary figure into 3 standalone figures, one per mechanism:
#   1. Ecological Fitting   — difference in starting fitness (Phase 1 plateau)
#   2. Long-term Evolvability — accessible fitness peaks (asymptotic AUC-ROC)
#   3. Short-term Evolvability — rate of phenotypic change (weight displacement)
#
# Reads pre-computed data from xAI/output/xai_summary_metrics.csv
# =============================================================================

library(tidyverse)
library(wesanderson)
library(colorspace)
library(ragg)

# =============================================================================
# 1. Load data
# =============================================================================

summary_metrics <- read_csv("xAI/output/xai_summary_metrics.csv", show_col_types = FALSE) |>
  mutate(model = case_when(
    model == "ImageNet Generalist" ~ "Naive ImageNet",
    TRUE ~ model
  )) |>
  mutate(model = factor(model,
    levels = c("Virtual Taxonomist", "Naive ImageNet")
  ))

cat("Summary metrics:\n")
print(summary_metrics)

# =============================================================================
# 2. Theme and palette
# =============================================================================

pal2 <- wes_palette("FantasticFox1")[c(3, 5)]
names(pal2) <- c("Virtual Taxonomist", "Naive ImageNet")

fig_width <- 1800
fig_height <- 1800

theme_mechanism <- function(base_size = 20) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = rel(1.15), hjust = 0.5),
      axis.title = element_text(face = "bold"),
      axis.title.x = element_blank(),
      axis.text.x = element_text(size = rel(1.0), face = "bold",
                                  angle = 30, hjust = 1),
      legend.position = "none",
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(25, 30, 15, 30)
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
# 3. Figure A: Ecological Fitting — Starting Fitness Difference
# =============================================================================

cat("\nGenerating Figure A: Ecological Fitting...\n")

fig_a <- ggplot(summary_metrics, aes(model, start, fill = model)) +
  geom_col(width = 0.55) +
  geom_text(aes(label = round(start, 3)),
    vjust = -0.6, fontface = "bold", size = 7, color = "grey20") +
  scale_fill_manual(values = pal2) +
  scale_y_continuous(
    name = "Starting Fitness (AUC-ROC)",
    limits = c(0, 1.05),
    breaks = seq(0, 1.0, 0.2),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(title = "Ecological Fitting") +
  theme_mechanism()

save_fig(fig_a, "mechanism_1_ecological_fitting.png")

# =============================================================================
# 4. Figure B: Long-term Evolvability — Accessible Fitness Peaks
# =============================================================================

cat("Generating Figure B: Long-term Evolvability...\n")

fig_b <- ggplot(summary_metrics, aes(model, peak, fill = model)) +
  geom_col(width = 0.55) +
  geom_text(aes(label = round(peak, 3)),
    vjust = -0.6, fontface = "bold", size = 7, color = "grey20") +
  scale_fill_manual(values = pal2) +
  scale_y_continuous(
    name = "Peak Fitness (AUC-ROC)",
    limits = c(0, 1.05),
    breaks = seq(0, 1.0, 0.2),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(title = "Long-term Evolvability") +
  theme_mechanism()

save_fig(fig_b, "mechanism_2_longterm_evolvability.png")

# =============================================================================
# 5. Figure C: Short-term Evolvability — Rate of Phenotypic Change
# =============================================================================

cat("Generating Figure C: Short-term Evolvability...\n")

# True per-batch rate: total displacement / N_batches_to_peak
summary_metrics <- summary_metrics |>
  mutate(rate_per_batch = weight_rate / 1000)

fig_c <- ggplot(summary_metrics, aes(model, rate_per_batch, fill = model)) +
  geom_col(width = 0.55) +
  geom_text(aes(label = round(rate_per_batch, 4)),
    vjust = -0.6, fontface = "bold", size = 7, color = "grey20") +
  scale_fill_manual(values = pal2) +
  scale_y_continuous(
    name = expression(atop(bold("Rate of Phenotypic Change"), (L[2] ~ displacement / N[batches]))),
    limits = c(0, max(summary_metrics$rate_per_batch) * 1.2),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(title = "Short-term Evolvability") +
  theme_mechanism()

save_fig(fig_c, "mechanism_3_shortterm_evolvability.png")

# =============================================================================
# Summary
# =============================================================================

cat("\n=== Three mechanism figures generated ===\n")
cat("  mechanism_1_ecological_fitting.png\n")
cat("  mechanism_2_longterm_evolvability.png\n")
cat("  mechanism_3_shortterm_evolvability.png\n")
