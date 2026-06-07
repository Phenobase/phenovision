#!/usr/bin/env Rscript
# §4/§5 shared-prediction figure for the two_noise paper.
# Consumes tidy CSVs written by xAI/two_noise/figures/export_csv.py:
#   runs/figures/sde_curve.csv        (analytic alpha* vs effective sample size)
#   runs/figures/optimizer_panel.csv  (alpha* vs batch, real models)   [optional]
#   runs/figures/sim_panel.csv        (M-anisotropy vs N*, Sim B)      [optional]
# One phenomenon, two substrates, on a shared effective-sample-size axis.
#
# Run: Rscript xAI/R/two_noise_figures.R   (from repo root, in an env with ggplot2)

suppressPackageStartupMessages({
  library(ggplot2); library(readr); library(dplyr); library(patchwork)
})

fig_dir <- "xAI/two_noise/runs/figures"
out_dir <- "xAI/figures"; dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

theme_set(theme_minimal(base_size = 12) +
          theme(panel.grid.minor = element_blank(),
                plot.title = element_text(face = "bold")))

sde <- read_csv(file.path(fig_dir, "sde_curve.csv"), show_col_types = FALSE)

# LEFT panel: ML alpha* vs effective batch, with the SDE curve overlaid.
p_left <- ggplot() +
  geom_line(data = sde, aes(eff_sample_size, alpha_star),
            linetype = "dashed", linewidth = 1, colour = "grey30") +
  scale_x_log10() + ylim(0, 1) +
  labs(title = "Optimizer (ML)", x = "effective batch size  S",
       y = expression(alpha^"*"~"(precond_power)"),
       subtitle = "dashed = two-noise SDE prediction")
opt_f <- file.path(fig_dir, "optimizer_panel.csv")
if (file.exists(opt_f)) {
  opt <- read_csv(opt_f, show_col_types = FALSE)
  bcol <- intersect(c("batch_size", "eff_sample_size", "S"), names(opt))[1]
  acol <- intersect(c("alpha_star", "alpha_best"), names(opt))[1]
  if (!is.na(bcol) && !is.na(acol))
    p_left <- p_left + geom_point(data = opt, aes(.data[[bcol]], .data[[acol]]),
                                  size = 3, colour = "#D55E00")
}

# RIGHT panel: biology M-anisotropy vs N*, same SDE curve overlaid (mapped to alpha).
sim_f <- file.path(fig_dir, "sim_panel.csv")
if (file.exists(sim_f)) {
  sim <- read_csv(sim_f, show_col_types = FALSE)
  ncol <- intersect(c("N_star", "n_star", "Nstar"), names(sim))[1]
  ycol <- intersect(c("M_anisotropy", "anisotropy", "aniso_ratio"), names(sim))[1]
  p_right <- ggplot(sim, aes(.data[[ncol]], .data[[ycol]])) +
    geom_point(size = 2, alpha = .5, colour = "#0072B2") +
    geom_smooth(se = TRUE, colour = "#0072B2", method = "loess", formula = y ~ x) +
    scale_x_log10() +
    labs(title = "Evolution (biology)", x = expression(N^"*"~"(effective sample size)"),
         y = "M eigenvalue anisotropy")
} else {
  p_right <- ggplot() + annotate("text", 0, 0, label = "Sim B pending") +
    labs(title = "Evolution (biology)") + theme_void()
}

combined <- p_left + p_right +
  plot_annotation(title = "One prediction, two substrates: alpha* / G-A exponent vs effective sample size")
ggsave(file.path(out_dir, "two_noise_shared_prediction.png"), combined,
       width = 11, height = 4.5, dpi = 150)
cat("wrote", file.path(out_dir, "two_noise_shared_prediction.png"), "\n")
