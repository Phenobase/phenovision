#!/usr/bin/env Rscript
# =============================================================================
# rewrite_csv_extensions_to_webp.R -- ONE-TIME: after the image store is converted to WebP, rewrite
# the `file_name` column's image extension (.jpg/.jpeg/.png -> .webp) in the training/inference CSVs
# so the baked-in paths point at the real on-disk files. The `extension` column (original format,
# used only for the S3 fetch URL) is LEFT UNCHANGED. Each file is backed up to <file>.prewebp.bak.
#
# Usage (run from the repo root, after the bulk conversion):
#   Rscript scripts/rewrite_csv_extensions_to_webp.R \
#       data/inat/train_v1.1.0.csv data/inat/val_v1.1.0.csv \
#       data/leaves/train.csv data/leaves/validation.csv <...repro_annotations_full_*.csv...>
# With no args it uses the defaults below (edit as needed). Idempotent: rows already ending in
# .webp are untouched.
# =============================================================================
suppressMessages({library(readr); library(stringr)})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  args <- c(
    "data/inat/train_v1.1.0.csv", "data/inat/val_v1.1.0.csv",
    "data/leaves/train.csv", "data/leaves/validation.csv",
    Sys.glob("data/**/repro_annotations_full_*.csv"),
    Sys.glob("data/leaves/*test*.csv")
  )
}
args <- unique(args[nzchar(args)])

pat <- regex("\\.(jpe?g|png)$", ignore_case = TRUE)
for (f in args) {
  if (!file.exists(f)) { message("skip (missing): ", f); next }
  df <- readr::read_csv(f, show_col_types = FALSE)          # readr auto-handles .gz
  if (!"file_name" %in% names(df)) { message("skip (no file_name col): ", f); next }
  n_hit <- sum(str_detect(df$file_name, pat), na.rm = TRUE)
  if (n_hit == 0L) { message(sprintf("%s : 0 to rewrite (already .webp?)", f)); next }
  bak <- paste0(f, ".prewebp.bak")
  if (!file.exists(bak)) file.copy(f, bak)
  df$file_name <- str_replace(df$file_name, pat, ".webp")
  readr::write_csv(df, f)
  message(sprintf("%s : rewrote %d file_name extensions -> .webp (backup: %s)", f, n_hit, bak))
}
