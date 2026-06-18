#!/usr/bin/env Rscript
# =============================================================================
# 3D Fitness Landscape: Three Mechanisms of Preadaptation
# =============================================================================
#
# A single topological landscape that demonstrates all three preadaptation
# mechanisms by placing starting points at different locations:
#
#   1. Ecological Fitting — different starting heights
#   2. Long-term Evolvability — accessible peaks separated by uncrossable chasm
#   3. Short-term Evolvability — smooth vs rough approach to same peak
#
# Uses rgl for 3D rendering (black/white wireframe mesh), webshot2 for
# headless PNG capture. Evolvability plots include gradient ascent paths.
#
# Outputs:
#   xAI/figures/landscape_ecological_fitting.png
#   xAI/figures/landscape_longterm_evolvability.png
#   xAI/figures/landscape_shortterm_evolvability.png
#   xAI/figures/landscape_mechanism_tour.gif (animation)
# =============================================================================

options(rgl.useNULL = TRUE)

library(rgl)
library(wesanderson)
library(gifski)

cat("Packages loaded.\n")

# =============================================================================
# 1. Landscape function
# =============================================================================

landscape <- function(x, y) {
  z <- rep(0, length(x))

  # Far-side tall peak: centered at (0, 2.5), height ~9
  z <- z + 9 * exp(-(x^2 / 2.0 + (y - 2.5)^2 / 1.5))

  # Near-side moderate peak: centered at (0, -2.5), height ~5.5
  z <- z + 5.5 * exp(-(x^2 / 1.0 + (y + 2.5)^2 / 1.0))

  # Deep chasm at y ~ 0
  z <- z - 5 * exp(-y^2 / (2 * 0.3^2))

  # Left-side roughness (x < 0, y < 0): sinusoidal bumps + localized peaks
  roughness <- 1.0 * sin(x * 3.5) * cos(y * 2.5) *
    exp(-((x + 2)^2 / 5 + (y + 2.5)^2 / 4))
  roughness <- roughness + 0.7 * cos(x * 5) * sin(y * 3) *
    exp(-((x + 1.5)^2 / 4 + (y + 2)^2 / 4))
  # Localized small peaks (saddle traps)
  roughness <- roughness + 1.8 * exp(-((x + 2.5)^2 / 0.35 + (y + 2.0)^2 / 0.35))
  roughness <- roughness + 1.2 * exp(-((x + 1.5)^2 / 0.25 + (y + 3.5)^2 / 0.25))
  roughness <- roughness + 0.9 * exp(-((x + 3.0)^2 / 0.30 + (y + 3.0)^2 / 0.30))
  z <- z + roughness

  # Right-side smooth swell
  z <- z + 1.0 * exp(-((x - 3.5)^2 / 2 + (y + 2.5)^2 / 2))

  # Floor
  z <- z + 1.0

  z
}

# =============================================================================
# 2. Generate surface grid (coarser for clean wireframe)
# =============================================================================

n <- 80  # coarser grid for visible mesh lines
x_seq <- seq(-5, 5, length.out = n)
y_seq <- seq(-5, 5, length.out = n)
grid <- expand.grid(x = x_seq, y = y_seq)
grid$z <- landscape(grid$x, grid$y)
z_mat <- matrix(grid$z, nrow = n, ncol = n)

cat("Landscape generated.\n")
cat("  Far peak:", round(max(z_mat[, y_seq > 1]), 2), "\n")
cat("  Near peak:", round(max(z_mat[, y_seq < -1]), 2), "\n")
cat("  Chasm min:", round(min(z_mat), 2), "\n")

# =============================================================================
# 3. Gradient ascent for evolution paths
# =============================================================================

# Numerical gradient of the landscape
landscape_grad <- function(x, y, eps = 0.01) {
  dz_dx <- (landscape(x + eps, y) - landscape(x - eps, y)) / (2 * eps)
  dz_dy <- (landscape(x, y + eps) - landscape(x, y - eps)) / (2 * eps)
  c(dz_dx, dz_dy)
}

# Gradient ascent from a starting point
gradient_ascent <- function(x0, y0, step_size = 0.05, max_steps = 500,
                             tol = 1e-4) {
  path_x <- x0
  path_y <- y0
  path_z <- landscape(x0, y0)

  x <- x0
  y <- y0

  for (i in seq_len(max_steps)) {
    g <- landscape_grad(x, y)
    gnorm <- sqrt(sum(g^2))
    if (gnorm < tol) break

    # Normalized step
    x <- x + step_size * g[1] / gnorm
    y <- y + step_size * g[2] / gnorm

    # Clamp to grid bounds
    x <- max(-5, min(5, x))
    y <- max(-5, min(5, y))

    path_x <- c(path_x, x)
    path_y <- c(path_y, y)
    path_z <- c(path_z, landscape(x, y))
  }

  list(x = path_x, y = path_y, z = path_z)
}

# =============================================================================
# 4. Marker colors
# =============================================================================

pal2 <- wes_palette("FantasticFox1")[c(3, 5)]
col_vt <- pal2[1]   # Virtual Taxonomist (teal)
col_ni <- pal2[2]   # Naive ImageNet (red)

# =============================================================================
# 5. Helper: render landscape with markers and paths
# =============================================================================

render_landscape <- function(markers, title = "",
                              paths = NULL,
                              theta = 40, phi = 25,
                              output_path = NULL,
                              width = 1600, height = 1200) {
  # Clear scene
  while (rgl.cur() > 0) close3d()
  open3d(windowRect = c(0, 0, width, height))

  bg3d(color = "white")

  # Draw wireframe mesh: white fill + black grid lines
  surface3d(x_seq, y_seq, z_mat,
            color = "white",
            front = "filled",
            back = "filled",
            alpha = 0.9,
            smooth = TRUE,
            lit = TRUE)

  # Overlay grid lines
  surface3d(x_seq, y_seq, z_mat + 0.01,
            color = "black",
            front = "lines",
            back = "lines",
            alpha = 0.4,
            lit = FALSE)

  # Draw evolution paths
  if (!is.null(paths)) {
    for (p in paths) {
      lines3d(p$path$x, p$path$y, p$path$z + 0.15,
              color = p$color, lwd = 4, alpha = 0.9)
    }
  }

  # Add markers (spheres above surface)
  for (m in markers) {
    z_surface <- landscape(m$x, m$y)
    spheres3d(m$x, m$y, z_surface + 0.5,
              radius = 0.3,
              color = m$color,
              lit = TRUE)
    # Label
    if (!is.null(m$label)) {
      text3d(m$x, m$y, z_surface + 1.3,
             text = m$label,
             color = m$color,
             cex = 1.8,
             font = 2)
    }
  }

  # Title
  if (nchar(title) > 0) {
    bgplot3d({
      par(mar = c(0, 0, 0, 0))
      plot.new()
      text(0.5, 0.95, title, cex = 3, font = 2)
    })
  }

  # Set viewpoint — theta rotates around z-axis, phi is elevation
  view3d(theta = theta, phi = phi, fov = 30, zoom = 0.75)

  # Capture
  if (!is.null(output_path)) {
    snapshot3d(output_path, webshot = TRUE)
    cat("Saved:", output_path, "\n")
  }
}

# =============================================================================
# 6. Compute evolution paths
# =============================================================================

cat("\nComputing gradient ascent paths...\n")

# Long-term evolvability paths
path_vt_long <- gradient_ascent(1, 1.5)    # far side → tall peak
path_ni_long <- gradient_ascent(-1, -1.5)  # near side → moderate peak

cat("  VT long-term: starts at z=", round(landscape(1, 1.5), 2),
    " ends at z=", round(tail(path_vt_long$z, 1), 2), "\n")
cat("  NI long-term: starts at z=", round(landscape(-1, -1.5), 2),
    " ends at z=", round(tail(path_ni_long$z, 1), 2), "\n")

# Short-term evolvability paths
path_vt_short <- gradient_ascent(3, -3.5)   # smooth right approach
path_ni_short <- gradient_ascent(-3, -3.5)  # rough left approach

cat("  VT short-term: starts at z=", round(landscape(3, -3.5), 2),
    " ends at z=", round(tail(path_vt_short$z, 1), 2),
    " steps=", length(path_vt_short$x), "\n")
cat("  NI short-term: starts at z=", round(landscape(-3, -3.5), 2),
    " ends at z=", round(tail(path_ni_short$z, 1), 2),
    " steps=", length(path_ni_short$x), "\n")

# =============================================================================
# 7. Figure A: Ecological Fitting
# =============================================================================

cat("\nRendering Figure A: Ecological Fitting...\n")

markers_eco <- list(
  list(x = 0, y = 2, color = col_vt, label = "VT"),
  list(x = 2, y = -4, color = col_ni, label = "NI")
)

render_landscape(markers_eco,
                 title = "Ecological Fitting",
                 output_path = "xAI/figures/landscape_ecological_fitting.png")

# =============================================================================
# 8. Figure B: Long-term Evolvability (with paths)
# =============================================================================

cat("Rendering Figure B: Long-term Evolvability...\n")

markers_long <- list(
  list(x = 1, y = 1.5, color = col_vt, label = "VT"),
  list(x = -1, y = -1.5, color = col_ni, label = "NI")
)

paths_long <- list(
  list(path = path_vt_long, color = col_vt),
  list(path = path_ni_long, color = col_ni)
)

render_landscape(markers_long,
                 title = "Long-term Evolvability",
                 paths = paths_long,
                 output_path = "xAI/figures/landscape_longterm_evolvability.png")

# =============================================================================
# 9. Figure C: Short-term Evolvability (with paths)
# =============================================================================

cat("Rendering Figure C: Short-term Evolvability...\n")

markers_short <- list(
  list(x = 3, y = -3.5, color = col_vt, label = "VT"),
  list(x = -3, y = -3.5, color = col_ni, label = "NI")
)

paths_short <- list(
  list(path = path_vt_short, color = col_vt),
  list(path = path_ni_short, color = col_ni)
)

render_landscape(markers_short,
                 title = "Short-term Evolvability",
                 paths = paths_short,
                 output_path = "xAI/figures/landscape_shortterm_evolvability.png")

# =============================================================================
# 10. Three-part mechanism tour animation
# =============================================================================

cat("\nRendering mechanism tour animation...\n")

fps <- 12
hold_sec <- 5
transition_sec <- 2
hold_frames <- hold_sec * fps
transition_frames <- transition_sec * fps

all_markers <- list(markers_eco, markers_long, markers_short)
all_paths <- list(NULL, paths_long, paths_short)
titles <- c("Ecological Fitting", "Long-term Evolvability", "Short-term Evolvability")

frame_dir <- file.path(tempdir(), "landscape_tour_frames")
dir.create(frame_dir, showWarnings = FALSE, recursive = TRUE)

# Render one frame per hold section + transitions (duplicate for pauses)
# Only render unique frames to save time, then duplicate for holds

unique_frames <- list()
unique_paths_list <- list()
frame_idx <- 0

for (part in seq_along(all_markers)) {
  # One unique frame for the hold
  frame_idx <- frame_idx + 1
  unique_frames[[frame_idx]] <- list(
    markers = all_markers[[part]],
    paths = all_paths[[part]],
    title = titles[part]
  )

  # Transition frames (interpolated markers, no paths during transition)
  if (part < length(all_markers)) {
    curr <- all_markers[[part]]
    nxt <- all_markers[[part + 1]]

    for (f in seq_len(transition_frames)) {
      t <- f / (transition_frames + 1)
      frame_idx <- frame_idx + 1
      unique_frames[[frame_idx]] <- list(
        markers = list(
          list(
            x = (1 - t) * curr[[1]]$x + t * nxt[[1]]$x,
            y = (1 - t) * curr[[1]]$y + t * nxt[[1]]$y,
            color = col_vt, label = "VT"
          ),
          list(
            x = (1 - t) * curr[[2]]$x + t * nxt[[2]]$x,
            y = (1 - t) * curr[[2]]$y + t * nxt[[2]]$y,
            color = col_ni, label = "NI"
          )
        ),
        paths = NULL,
        title = ""
      )
    }
  }
}

cat("  Rendering", length(unique_frames), "unique frames...\n")

png_paths <- character(length(unique_frames))
for (i in seq_along(unique_frames)) {
  uf <- unique_frames[[i]]
  png_path <- file.path(frame_dir, sprintf("frame_%04d.png", i))

  render_landscape(uf$markers,
                   title = uf$title,
                   paths = uf$paths,
                   output_path = png_path)
  png_paths[i] <- png_path
}

# Build final frame sequence with holds via duplication
# Frame 1 = eco hold, then transition frames, then long hold, etc.
gif_files <- character()

ptr <- 1
for (part in seq_along(all_markers)) {
  # Hold: repeat this frame
  gif_files <- c(gif_files, rep(png_paths[ptr], hold_frames))
  ptr <- ptr + 1

  # Transition frames
  if (part < length(all_markers)) {
    for (f in seq_len(transition_frames)) {
      gif_files <- c(gif_files, png_paths[ptr])
      ptr <- ptr + 1
    }
  }
}

# End pause
n_end_pause <- 10 * fps
gif_files <- c(gif_files, rep(gif_files[length(gif_files)], n_end_pause))

cat("  Stitching GIF (", length(gif_files), "total frames)...\n")

gif_path <- "xAI/figures/landscape_mechanism_tour.gif"
gifski::gifski(
  gif_files,
  gif_file = gif_path,
  delay = 1 / fps,
  width = 1600,
  height = 1200
)

# Optimize with gifsicle
gifsicle_path <- Sys.which("gifsicle")
if (nchar(gifsicle_path) > 0) {
  cat("  Optimizing with gifsicle...\n")
  system2("gifsicle", c("-O3", "--lossy=80", "--colors", "128",
                        gif_path, "-o", gif_path))
}

unlink(frame_dir, recursive = TRUE)

cat("  Saved:", gif_path, "\n")
cat("  Size:", round(file.size(gif_path) / 1e6, 1), "MB\n")

# =============================================================================
# Summary
# =============================================================================

cat("\n=== Done ===\n")
cat("  xAI/figures/landscape_ecological_fitting.png\n")
cat("  xAI/figures/landscape_longterm_evolvability.png\n")
cat("  xAI/figures/landscape_shortterm_evolvability.png\n")
cat("  xAI/figures/landscape_mechanism_tour.gif\n")
