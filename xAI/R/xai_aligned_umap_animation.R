#!/usr/bin/env Rscript
# =============================================================================
# Aligned UMAP Animation: Representation Space Evolution
# =============================================================================
#
# Creates two GIF animations (one per model) showing how the penultimate-layer
# representations restructure during Phase 2 training. Points are colored by
# phenological status (flowering = fill, fruiting = border), matching the
# existing PHATE figure style.
#
# Uses AlignedUMAP (Python umap-learn) via reticulate to solve the embedding
# identifiability problem across time slices.
#
# Also generates a static multi-panel comparison figure (15 panels per model,
# MAE subsampled to match PlantCLEF count).
#
# Outputs:
#   xAI/figures/aligned_umap_virtual_taxonomist.gif
#   xAI/figures/aligned_umap_naive_imagenet.gif
#   xAI/figures/aligned_umap_panels_virtual_taxonomist.png
#   xAI/figures/aligned_umap_panels_naive_imagenet.png
#   xAI/output/aligned_umap_plantclef_embeddings.rds  (cached)
#   xAI/output/aligned_umap_mae_embeddings.rds         (cached)
# =============================================================================

library(tidyverse)
library(wesanderson)
library(colorspace)
library(zoo)
library(ragg)
library(gifski)
library(patchwork)
library(reticulate)

use_condaenv("reticulate-gpu2", required = TRUE)

torch <- import("torch")
np <- import("numpy")
umap_module <- import("umap")

cat("Python packages loaded successfully.\n")

# =============================================================================
# 1. Load phenology labels for the 1000 subset images
# =============================================================================

cat("Loading phenology labels...\n")

# 0-based indices (no header), same for both models
repr_indices <- scan("xAI/output/plantclef_run/repr_subset_indices.csv",
                     integer(), quiet = TRUE)

val_data <- read_csv("data/inat/val_v1.1.0.csv", show_col_types = FALSE)
subset_meta <- val_data[repr_indices + 1L, ]  # +1 for R indexing

pheno_labels <- subset_meta |>
  select(flowering, fruiting) |>
  mutate(
    flowering = factor(flowering),
    fruiting = factor(fruiting)
  )

cat("  ", nrow(pheno_labels), "images with labels loaded.\n")
cat("  Flowering: ", table(pheno_labels$flowering), "\n")
cat("  Fruiting:  ", table(pheno_labels$fruiting), "\n")

# =============================================================================
# 2. Determine which representation steps to include
# =============================================================================

cat("\nDetermining asymptote steps...\n")

val_combined <- read_csv("xAI/output/xai_validation_combined.csv",
                          show_col_types = FALSE)

# Phase transition steps
plantclef_transition <- val_combined |>
  filter(model == "Virtual Taxonomist", phase == "frozen_final") |>
  pull(global_step) |> first()

mae_transition <- val_combined |>
  filter(model == "ImageNet Generalist", phase == "frozen_final") |>
  pull(global_step) |> first()

# Add phase2_batch
val_combined <- val_combined |>
  mutate(
    phase2_batch = case_when(
      model == "Virtual Taxonomist" & phase == "unfrozen" ~
        global_step - plantclef_transition,
      model == "ImageNet Generalist" & phase == "unfrozen" ~
        global_step - mae_transition,
      TRUE ~ NA_real_
    )
  )

# Detect asymptote (from xai_results_figures.R)
detect_asymptote <- function(df, window = 5, threshold = 0.001) {
  df <- df |> arrange(phase2_batch)
  df$auc_smooth <- zoo::rollmean(df$auc_roc_mean, k = window,
                                  fill = NA, align = "center")
  max_smooth <- max(df$auc_smooth, na.rm = TRUE)
  df |>
    filter(!is.na(auc_smooth), auc_smooth >= max_smooth - threshold) |>
    slice(1)
}

plantclef_asymptote <- val_combined |>
  filter(model == "Virtual Taxonomist", phase == "unfrozen") |>
  detect_asymptote()

mae_asymptote <- val_combined |>
  filter(model == "ImageNet Generalist", phase == "unfrozen") |>
  detect_asymptote()

# Convert to global steps
plantclef_peak_step <- plantclef_asymptote$global_step
mae_peak_step <- mae_asymptote$global_step

cat("  PlantCLEF transition:", plantclef_transition,
    " asymptote step:", plantclef_peak_step,
    " (phase2_batch:", plantclef_asymptote$phase2_batch, ")\n")
cat("  MAE transition:", mae_transition,
    " asymptote step:", mae_peak_step,
    " (phase2_batch:", mae_asymptote$phase2_batch, ")\n")

# =============================================================================
# 3. Load representation tensors
# =============================================================================

load_representations <- function(repr_dir, max_step = Inf) {
  files <- list.files(repr_dir, pattern = "^features_step.*\\.pt$",
                      full.names = FALSE)
  # Extract step numbers and sort
  steps <- as.integer(str_extract(files, "\\d+"))
  ord <- order(steps)
  files <- files[ord]
  steps <- steps[ord]

  # Filter to steps up to ~1-2 beyond peak
  keep <- steps <= max_step
  files <- files[keep]
  steps <- steps[keep]

  cat("  Loading", length(files), "representation files...\n")

  matrices <- vector("list", length(files))
  for (i in seq_along(files)) {
    path <- file.path(repr_dir, files[i])
    tensor <- torch$load(path, map_location = "cpu", weights_only = TRUE)
    matrices[[i]] <- as.matrix(tensor$numpy())
  }

  list(matrices = matrices, steps = steps)
}

cat("\nLoading PlantCLEF representations...\n")
plantclef_repr <- load_representations(
  "xAI/output/plantclef_run/representations/",
  max_step = plantclef_peak_step + 1000
)

cat("Loading MAE representations...\n")
mae_repr <- load_representations(
  "xAI/output/mae_run/representations/",
  max_step = mae_peak_step + 1000
)

cat("  PlantCLEF: steps", paste(plantclef_repr$steps, collapse = ", "), "\n")
cat("  MAE: steps", paste(mae_repr$steps, collapse = ", "), "\n")

# =============================================================================
# 4. Run AlignedUMAP via reticulate
# =============================================================================

run_aligned_umap <- function(matrices_list, cache_path, n_neighbors = 15L,
                              min_dist = 0.1, alignment_regularisation = 1e-3,
                              alignment_window_size = 3L, random_state = 42L) {
  # Check cache
  if (file.exists(cache_path)) {
    cat("  Loading cached AlignedUMAP embeddings from", cache_path, "\n")
    return(readRDS(cache_path))
  }

  cat("  Running AlignedUMAP on", length(matrices_list), "slices...\n")

  n_samples <- nrow(matrices_list[[1]])
  n_slices <- length(matrices_list)

  # Pass matrices to Python as a list of numpy arrays
  py$slices_r <- lapply(matrices_list, function(m) np$array(m))

  # Run entire AlignedUMAP in Python to avoid reticulate type conversion issues
  py_run_string(sprintf("
import numpy as np
from umap import AlignedUMAP

# Convert from R list to Python list of numpy arrays
slices = [np.array(s) for s in slices_r]
n_samples = slices[0].shape[0]
n_slices = len(slices)

# Identity relations (same samples across all timesteps)
identity_rel = {i: i for i in range(n_samples)}
relation_dicts = [identity_rel] * (n_slices - 1)

# Fit AlignedUMAP
mapper = AlignedUMAP(
    n_neighbors=%d,
    min_dist=%f,
    alignment_regularisation=%f,
    alignment_window_size=%d,
    random_state=%d
)
mapper.fit(slices, relations=relation_dicts)

# Extract embeddings as list of numpy arrays
aligned_embeddings = [e.copy() for e in mapper.embeddings_]
", n_neighbors, min_dist, alignment_regularisation,
   alignment_window_size, random_state))

  # Pull results back to R
  py_embeddings <- py$aligned_embeddings
  embeddings <- lapply(py_embeddings, as.matrix)

  # Cache
  saveRDS(embeddings, cache_path)
  cat("  Cached to", cache_path, "\n")

  embeddings
}

cat("\nRunning AlignedUMAP for PlantCLEF...\n")
plantclef_embeddings <- run_aligned_umap(
  plantclef_repr$matrices,
  "xAI/output/aligned_umap_plantclef_embeddings.rds"
)

cat("Running AlignedUMAP for MAE...\n")
mae_embeddings <- run_aligned_umap(
  mae_repr$matrices,
  "xAI/output/aligned_umap_mae_embeddings.rds"
)

# =============================================================================
# 5. Interpolate between frames (proportional to batch gap)
# =============================================================================

# Target: constant time per batch across the animation.
# More interpolated frames where keyframes are further apart in batch space.

interpolate_embeddings <- function(embeddings, steps, target_fps = 12,
                                    target_duration_sec = 15,
                                    min_interp = 4) {
  n_key <- length(embeddings)
  total_batches <- steps[n_key] - steps[1]
  total_frames <- target_fps * target_duration_sec

  all_frames <- list()
  all_steps <- numeric()
  is_keyframe <- logical()
  frame_idx <- 0

  for (i in seq_len(n_key)) {
    # Add the real keyframe
    frame_idx <- frame_idx + 1
    all_frames[[frame_idx]] <- embeddings[[i]]
    all_steps[frame_idx] <- steps[i]
    is_keyframe[frame_idx] <- TRUE

    # Interpolated frames: proportional to batch gap, with a minimum
    if (i < n_key) {
      gap <- steps[i + 1] - steps[i]
      n_proportional <- max(1, round(gap / total_batches * total_frames)) - 1
      n_interp <- max(min_interp, n_proportional)

      for (j in seq_len(n_interp)) {
        t <- j / (n_interp + 1)
        frame_idx <- frame_idx + 1
        all_frames[[frame_idx]] <- (1 - t) * embeddings[[i]] +
          t * embeddings[[i + 1]]
        all_steps[frame_idx] <- round(steps[i] + t * gap)
        is_keyframe[frame_idx] <- FALSE
      }
    }
  }

  cat("    Total frames:", frame_idx,
      " (", n_key, "keyframes +", frame_idx - n_key, "interpolated)\n")

  list(frames = all_frames, steps = all_steps, is_keyframe = is_keyframe)
}

cat("\nInterpolating frames...\n")

cat("  PlantCLEF:\n")
plantclef_interp <- interpolate_embeddings(plantclef_embeddings,
                                            plantclef_repr$steps)
cat("  MAE:\n")
mae_interp <- interpolate_embeddings(mae_embeddings,
                                      mae_repr$steps)

# =============================================================================
# 6. Build animation dataframes (with randomized point order)
# =============================================================================

build_animation_df <- function(interp_result, transition_step) {
  n_pts <- nrow(interp_result$frames[[1]])
  dfs <- vector("list", length(interp_result$frames))

  for (i in seq_along(interp_result$frames)) {
    emb <- interp_result$frames[[i]]
    step <- interp_result$steps[i]

    # Randomize plotting order per frame so no point is permanently covered
    ord <- sample(n_pts)

    dfs[[i]] <- tibble(
      UMAP1 = emb[ord, 1],
      UMAP2 = emb[ord, 2],
      flowering = pheno_labels$flowering[ord],
      fruiting = pheno_labels$fruiting[ord],
      step = step,
      phase2_batch = step - transition_step,
      frame = i
    )
  }
  bind_rows(dfs)
}

set.seed(42)
plantclef_df <- build_animation_df(plantclef_interp, plantclef_transition)
mae_df <- build_animation_df(mae_interp, mae_transition)

# =============================================================================
# 7. Render GIF with gifski
# =============================================================================

# PHATE-style color palette
pal_fill <- c("0" = "#B7E1A1", "1" = "#8A6B9E")
pal_border <- c("0" = "white", "1" = "black")

# Shared ggplot for a single UMAP frame
make_umap_plot <- function(frame_data, model_name, xlim, ylim,
                            show_legend = TRUE, base_size = 18,
                            point_size = 3.6) {
  p2b <- frame_data$phase2_batch[1]

  p <- ggplot(frame_data, aes(UMAP1, UMAP2)) +
    geom_point(
      aes(fill = flowering, colour = fruiting),
      shape = 21, size = point_size, alpha = 0.7, stroke = 0.6
    ) +
    scale_fill_manual(
      values = pal_fill,
      labels = c("0" = "No Flower", "1" = "Flower"),
      name = "Flowering"
    ) +
    scale_colour_manual(
      values = pal_border,
      labels = c("0" = "No Fruit", "1" = "Fruit"),
      name = "Fruiting"
    ) +
    coord_cartesian(xlim = xlim, ylim = ylim) +
    labs(
      title = model_name,
      subtitle = paste0("Phase 2 batch: ", p2b),
      x = "Aligned UMAP 1",
      y = "Aligned UMAP 2"
    ) +
    theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = rel(1.2), hjust = 0.5),
      plot.subtitle = element_text(color = "grey40", size = rel(0.95),
                                    hjust = 0.5),
      axis.title = element_text(face = "bold", size = rel(1.0)),
      axis.text = element_text(size = rel(0.9)),
      legend.position = if (show_legend) "bottom" else "none",
      legend.text = element_text(size = rel(0.95)),
      legend.title = element_text(face = "bold", size = rel(1.0)),
      panel.grid.minor = element_blank(),
      plot.margin = margin(15, 20, 10, 15)
    ) +
    guides(
      fill = guide_legend(override.aes = list(size = 5)),
      colour = guide_legend(override.aes = list(size = 5))
    )

  p
}

render_animation <- function(anim_df, model_name, output_gif, fps = 12,
                              pause_start_sec = 5, pause_end_sec = 10) {
  frames <- sort(unique(anim_df$frame))
  n_frames <- length(frames)

  # Fixed axis limits across all frames (with 5% padding)
  x_range <- range(anim_df$UMAP1)
  y_range <- range(anim_df$UMAP2)
  x_pad <- diff(x_range) * 0.05
  y_pad <- diff(y_range) * 0.05
  xlim <- c(x_range[1] - x_pad, x_range[2] + x_pad)
  ylim <- c(y_range[1] - y_pad, y_range[2] + y_pad)

  # Temp directory for frames
  frame_dir <- file.path(tempdir(), paste0("umap_frames_",
                          tolower(gsub(" ", "_", model_name))))
  dir.create(frame_dir, showWarnings = FALSE, recursive = TRUE)

  cat("  Rendering", n_frames, "frames to", frame_dir, "...\n")

  png_files <- character(n_frames)
  img_w <- 3200
  img_h <- 2800

  for (idx in seq_along(frames)) {
    fr <- frames[idx]
    frame_data <- anim_df |> filter(frame == fr)

    p <- make_umap_plot(frame_data, model_name, xlim, ylim,
                         show_legend = TRUE, base_size = 16, point_size = 3)

    png_path <- file.path(frame_dir, sprintf("frame_%04d.png", idx))
    ragg::agg_png(png_path, width = img_w, height = img_h, res = 150,
                  scaling = 2)
    print(p)
    dev.off()

    png_files[idx] <- png_path

    if (idx %% 25 == 0) cat("    Frame", idx, "/", n_frames, "\n")
  }

  # Pauses: duplicate first/last frame PNGs (gifski only accepts scalar delay)
  n_start_dupes <- round(pause_start_sec * fps)
  n_end_dupes <- round(pause_end_sec * fps)

  gif_files <- c(
    rep(png_files[1], n_start_dupes),
    png_files,
    rep(png_files[n_frames], n_end_dupes)
  )

  # Stitch into GIF
  cat("  Stitching GIF (", length(gif_files), "total frames incl.",
      n_start_dupes, "start +", n_end_dupes, "end pause frames)...\n")
  # Encode at half resolution for smaller file size (source PNGs are high-res)
  gif_w <- img_w %/% 2
  gif_h <- img_h %/% 2

  gifski::gifski(
    gif_files,
    gif_file = output_gif,
    delay = 1 / fps,
    width = gif_w,
    height = gif_h
  )

  # Optimize with gifsicle if available
  gifsicle_path <- Sys.which("gifsicle")
  if (nchar(gifsicle_path) > 0) {
    cat("  Optimizing with gifsicle...\n")
    system2("gifsicle", c("-O3", "--lossy=80", "--colors", "128",
                          output_gif, "-o", output_gif))
  }

  # Clean up temp frames
  unlink(frame_dir, recursive = TRUE)

  cat("  Saved:", output_gif, "\n")
  cat("  Size:", round(file.size(output_gif) / 1e6, 1), "MB\n")
}

# =============================================================================
# 7b. Render combined side-by-side GIF (Naive ImageNet left, VT right)
# =============================================================================

render_combined_animation <- function(mae_df, plantclef_df, output_gif,
                                       fps = 12, pause_start_sec = 5,
                                       pause_end_sec = 10) {
  # Align frame counts: pad shorter animation by holding last frame
  n_mae <- max(mae_df$frame)
  n_plantclef <- max(plantclef_df$frame)
  n_frames <- max(n_mae, n_plantclef)

  # Compute axis limits for each model independently
  mae_xlim <- range(mae_df$UMAP1) + c(-1, 1) * diff(range(mae_df$UMAP1)) * 0.05
  mae_ylim <- range(mae_df$UMAP2) + c(-1, 1) * diff(range(mae_df$UMAP2)) * 0.05
  pc_xlim <- range(plantclef_df$UMAP1) + c(-1, 1) * diff(range(plantclef_df$UMAP1)) * 0.05
  pc_ylim <- range(plantclef_df$UMAP2) + c(-1, 1) * diff(range(plantclef_df$UMAP2)) * 0.05

  frame_dir <- file.path(tempdir(), "umap_frames_combined")
  dir.create(frame_dir, showWarnings = FALSE, recursive = TRUE)

  cat("  Rendering", n_frames, "combined frames...\n")

  png_files <- character(n_frames)
  img_w <- 4800  # wide landscape for side-by-side
  img_h <- 2400

  for (idx in seq_len(n_frames)) {
    # Get frame data, clamping to last available frame if one model is shorter
    mae_frame <- min(idx, n_mae)
    pc_frame <- min(idx, n_plantclef)

    mae_data <- mae_df |> filter(frame == mae_frame)
    pc_data <- plantclef_df |> filter(frame == pc_frame)

    # Left panel: Naive ImageNet (no legend — shared below)
    p_left <- make_umap_plot(mae_data, "Naive ImageNet",
                              mae_xlim, mae_ylim,
                              show_legend = FALSE)

    # Right panel: Virtual Taxonomist (no legend)
    p_right <- make_umap_plot(pc_data, "Virtual Taxonomist",
                               pc_xlim, pc_ylim,
                               show_legend = FALSE)

    # Combine with patchwork, shared legend at bottom
    p_combined <- p_left + p_right +
      plot_layout(guides = "collect") &
      theme(legend.position = "bottom")

    # Need to re-enable legend on one panel for collection to work
    p_left_leg <- make_umap_plot(mae_data, "Naive ImageNet",
                                  mae_xlim, mae_ylim,
                                  show_legend = TRUE)
    p_combined <- p_left_leg + p_right +
      plot_layout(guides = "collect") &
      theme(legend.position = "bottom")

    png_path <- file.path(frame_dir, sprintf("frame_%04d.png", idx))
    ragg::agg_png(png_path, width = img_w, height = img_h, res = 150,
                  scaling = 2)
    print(p_combined)
    dev.off()

    png_files[idx] <- png_path

    if (idx %% 25 == 0) cat("    Frame", idx, "/", n_frames, "\n")
  }

  # Pauses via frame duplication
  n_start_dupes <- round(pause_start_sec * fps)
  n_end_dupes <- round(pause_end_sec * fps)

  gif_files <- c(
    rep(png_files[1], n_start_dupes),
    png_files,
    rep(png_files[n_frames], n_end_dupes)
  )

  # Encode at half resolution
  gif_w <- img_w %/% 2
  gif_h <- img_h %/% 2

  cat("  Stitching GIF (", length(gif_files), "total frames,",
      gif_w, "x", gif_h, ")...\n")
  gifski::gifski(
    gif_files,
    gif_file = output_gif,
    delay = 1 / fps,
    width = gif_w,
    height = gif_h
  )

  # Optimize with gifsicle
  gifsicle_path <- Sys.which("gifsicle")
  if (nchar(gifsicle_path) > 0) {
    cat("  Optimizing with gifsicle...\n")
    system2("gifsicle", c("-O3", "--lossy=80", "--colors", "128",
                          output_gif, "-o", output_gif))
  }

  unlink(frame_dir, recursive = TRUE)

  cat("  Saved:", output_gif, "\n")
  cat("  Size:", round(file.size(output_gif) / 1e6, 1), "MB\n")
}

cat("\nRendering combined side-by-side animation...\n")
render_combined_animation(
  mae_df, plantclef_df,
  "xAI/figures/aligned_umap_combined.gif"
)

# Also render individual animations
cat("\nRendering Virtual Taxonomist animation...\n")
render_animation(
  plantclef_df,
  "Virtual Taxonomist",
  "xAI/figures/aligned_umap_virtual_taxonomist.gif"
)

cat("Rendering Naive ImageNet animation...\n")
render_animation(
  mae_df,
  "Naive ImageNet",
  "xAI/figures/aligned_umap_naive_imagenet.gif"
)

# =============================================================================
# 8. Multi-panel static figure (15 keyframes per model)
# =============================================================================

cat("\nGenerating multi-panel static figures...\n")

render_panel_figure <- function(embeddings, steps, transition_step,
                                 model_name, output_path, n_panels = 15) {
  n_key <- length(embeddings)

  # Subsample to n_panels keyframes (evenly spaced indices)
  if (n_key > n_panels) {
    panel_idx <- round(seq(1, n_key, length.out = n_panels))
  } else {
    panel_idx <- seq_len(n_key)
  }

  n_pts <- nrow(embeddings[[1]])

  # Build dataframe for all panels with randomized order
  set.seed(123)
  dfs <- list()
  for (i in seq_along(panel_idx)) {
    ki <- panel_idx[i]
    emb <- embeddings[[ki]]
    step <- steps[ki]
    p2b <- step - transition_step
    ord <- sample(n_pts)

    dfs[[i]] <- tibble(
      UMAP1 = emb[ord, 1],
      UMAP2 = emb[ord, 2],
      flowering = pheno_labels$flowering[ord],
      fruiting = pheno_labels$fruiting[ord],
      phase2_batch = p2b,
      panel_label = paste0("Batch ", p2b)
    )
  }
  panel_df <- bind_rows(dfs)

  # Ordered factor for facets
  batch_levels <- unique(panel_df$phase2_batch)
  panel_df <- panel_df |>
    mutate(panel_label = factor(panel_label,
      levels = paste0("Batch ", batch_levels)
    ))

  # Fixed axis limits
  x_range <- range(panel_df$UMAP1)
  y_range <- range(panel_df$UMAP2)
  x_pad <- diff(x_range) * 0.05
  y_pad <- diff(y_range) * 0.05
  xlim <- c(x_range[1] - x_pad, x_range[2] + x_pad)
  ylim <- c(y_range[1] - y_pad, y_range[2] + y_pad)

  n_actual <- length(panel_idx)
  ncol <- 5
  nrow <- ceiling(n_actual / ncol)

  p <- ggplot(panel_df, aes(UMAP1, UMAP2)) +
    geom_point(
      aes(fill = flowering, colour = fruiting),
      shape = 21, size = 1.2, alpha = 0.6, stroke = 0.3
    ) +
    scale_fill_manual(
      values = pal_fill,
      labels = c("0" = "No Flower", "1" = "Flower"),
      name = "Flowering"
    ) +
    scale_colour_manual(
      values = pal_border,
      labels = c("0" = "No Fruit", "1" = "Fruit"),
      name = "Fruiting"
    ) +
    coord_cartesian(xlim = xlim, ylim = ylim) +
    facet_wrap(~ panel_label, ncol = ncol) +
    labs(
      title = paste0(model_name, ": Representation Space Over Training"),
      x = "Aligned UMAP 1",
      y = "Aligned UMAP 2"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(size = 7),
      strip.text = element_text(face = "bold", size = 10),
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      plot.margin = margin(15, 15, 10, 15)
    ) +
    guides(
      fill = guide_legend(override.aes = list(size = 3)),
      colour = guide_legend(override.aes = list(size = 3))
    )

  fig_w <- 2400
  fig_h <- 600 * nrow + 200  # scale height by number of rows

  ragg::agg_png(output_path, width = fig_w, height = fig_h, res = 150,
                scaling = 2)
  print(p)
  dev.off()

  cat("  Saved:", output_path, "(", n_actual, "panels,",
      ncol, "x", nrow, ")\n")
}

render_panel_figure(
  plantclef_embeddings, plantclef_repr$steps, plantclef_transition,
  "Virtual Taxonomist",
  "xAI/figures/aligned_umap_panels_virtual_taxonomist.png",
  n_panels = 15
)

render_panel_figure(
  mae_embeddings, mae_repr$steps, mae_transition,
  "Naive ImageNet",
  "xAI/figures/aligned_umap_panels_naive_imagenet.png",
  n_panels = 15
)

cat("\n=== Done ===\n")
cat("  xAI/figures/aligned_umap_combined.gif\n")
cat("  xAI/figures/aligned_umap_virtual_taxonomist.gif\n")
cat("  xAI/figures/aligned_umap_naive_imagenet.gif\n")
cat("  xAI/figures/aligned_umap_panels_virtual_taxonomist.png\n")
cat("  xAI/figures/aligned_umap_panels_naive_imagenet.png\n")
