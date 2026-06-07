#!/usr/bin/env Rscript
# §4/§5 shared-prediction figure for the two_noise paper — 3 panels telling the
# errors-in-variables story across substrates.
#   runs/figures/sde_curve.csv        analytic alpha* vs effective sample size
#   runs/figures/optimizer_panel.csv  alpha* vs batch, real models            [optional]
#   runs/figures/sim_panel.csv        G-anisotropy vs effective N* (Sim B incoherent)
#   runs/figures/alpha1_panel.csv     full-inverse ViT stability by condition  [optional]
# Run: Rscript xAI/R/two_noise_figures.R   (from repo root, env with ggplot2+patchwork)

suppressPackageStartupMessages({
  library(ggplot2); library(readr); library(dplyr); library(patchwork)
})

fig_dir <- "xAI/two_noise/runs/figures"
out_dir <- "xAI/figures"; dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
theme_set(theme_minimal(base_size = 12) +
          theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold")))
ok <- function(f) file.exists(file.path(fig_dir, f))
rd <- function(f) suppressWarnings(read_csv(file.path(fig_dir, f), show_col_types = FALSE))

# ---- Panel 1: ML optimizer — alpha* vs effective batch, SDE curve overlaid ----------------
sde <- rd("sde_curve.csv")
p1 <- ggplot() +
  geom_line(data = sde, aes(eff_sample_size, alpha_star),
            linetype = "dashed", linewidth = 1, colour = "grey30") +
  scale_x_log10() + ylim(0, 1) +
  labs(title = "Optimizer (ML)", x = "effective batch size  S",
       y = expression(alpha^"*"~"(precond_power)"), subtitle = "dashed = two-noise SDE")
if (ok("optimizer_panel.csv")) {
  opt <- rd("optimizer_panel.csv")
  bcol <- intersect(c("batch_size", "eff_sample_size", "S"), names(opt))[1]
  acol <- intersect(c("alpha_star", "alpha_best"), names(opt))[1]
  if (is.na(acol) && "is_alpha_star" %in% names(opt) && "alpha" %in% names(opt)) {
    opt <- opt %>% filter(is_alpha_star); acol <- "alpha"
  }
  if (!is.na(bcol) && !is.na(acol))
    p1 <- p1 + geom_point(data = opt, aes(.data[[bcol]], .data[[acol]]),
                          size = 3, colour = "#D55E00")
}

# ---- Panel 2: Biology — G-anisotropy vs effective N* (compression as N* falls) ------------
if (ok("sim_panel.csv")) {
  sim <- rd("sim_panel.csv")
  xcol <- intersect(c("eff_N_star", "n_star", "N_star"), names(sim))[1]
  ycol <- intersect(c("G_anisotropy", "m_anisotropy", "M_anisotropy"), names(sim))[1]
  p2 <- ggplot(sim, aes(.data[[xcol]], .data[[ycol]])) +
    geom_line(colour = "#0072B2", linewidth = 1) +
    geom_point(size = 3, colour = "#0072B2") +
    geom_hline(yintercept = 1, linetype = "dotted", colour = "grey50") +
    scale_x_log10() +
    labs(title = "Evolution (biology)", x = expression(N^"*"~"(effective sample size)"),
         y = "G eigenvalue anisotropy", subtitle = "compresses toward isotropy as N* falls")
} else {
  p2 <- ggplot() + labs(title = "Evolution (biology)") + theme_void()
}

# ---- Panel 3: the import — full-inverse (alpha=1) ViT stability by condition --------------
if (ok("alpha1_panel.csv")) {
  a1 <- rd("alpha1_panel.csv")
  ycol <- intersect(c("finite_fraction", "finite_frac"), names(a1))[1]
  lab <- if ("label" %in% names(a1)) a1$label else a1$condition
  a1$lab <- factor(lab, levels = lab[order(a1[[ycol]])])
  a1$stable <- ifelse(a1[[ycol]] > 0.9, "stable", "diverged")
  p3 <- ggplot(a1, aes(lab, .data[[ycol]], fill = stable)) +
    geom_col(width = .6) +
    scale_fill_manual(values = c(stable = "#009E73", diverged = "#D55E00"), guide = "none") +
    ylim(0, 1) + coord_flip() +
    labs(title = expression("Full-inverse ("*alpha*"=1) ViT @ batch 16"),
         x = NULL, y = "fraction of finite steps",
         subtitle = "true-Fisher curvature stabilizes; damping alone does not")
} else {
  p3 <- ggplot() + labs(title = "ViT alpha=1 stability") + theme_void()
}

combined <- (p1 | p2) / p3 +
  plot_layout(heights = c(1.4, 1)) +
  plot_annotation(
    title = "Errors-in-variables in the curvature/selection signal: one law across substrates",
    subtitle = "alpha*/anisotropy rise with effective sample size; correctly-specified (true-Fisher / high-h^2) curvature stabilizes the full inverse")
ggsave(file.path(out_dir, "two_noise_shared_prediction.png"), combined,
       width = 11, height = 8, dpi = 150)
cat("wrote", file.path(out_dir, "two_noise_shared_prediction.png"), "\n")
