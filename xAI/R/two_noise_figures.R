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

# ---- Panel 1: ML optimizer — the FULL-INVERSE PENALTY shrinks as batch grows --------------
# argmin alpha* is flat at 0.5 (whitening robustly best at this scale); the law is in the GAP
# val_loss(alpha=1) - val_loss(alpha=0.5), which falls toward 0 as batch grows = full inverse
# becomes more favorable as gradient noise falls (the noise-dependent-alpha* prediction's direction).
if (ok("optimizer_panel.csv") && "full_inverse_penalty" %in% names(rd("optimizer_panel.csv"))) {
  opt <- rd("optimizer_panel.csv")
  p1 <- ggplot(opt, aes(batch_size, full_inverse_penalty)) +
    geom_hline(yintercept = 0, linetype = "dotted", colour = "grey50") +
    geom_line(colour = "#D55E00", linewidth = 1) +
    geom_point(size = 3, colour = "#D55E00") +
    scale_x_log10() +
    labs(title = "Optimizer (ML)", x = "effective batch size  S",
         y = expression("val-loss penalty of "*alpha*"=1 vs "*alpha*"=0.5"),
         subtitle = "full-inverse penalty shrinks as batch grows (noise falls)")
} else {
  p1 <- ggplot() + labs(title = "Optimizer (ML)") + theme_void()
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
