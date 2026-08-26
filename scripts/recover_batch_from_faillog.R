#!/usr/bin/env Rscript
# =============================================================================
# recover_batch_from_faillog.R <batch_number>
# One-time RECOVERY of images the WebP bulk conversion deleted (see the 2026-07-01 incident:
# a transient /blue read error was misclassified as corruption and the originals were removed).
# Every deleted photo_id was logged to .webp_convert/failed/batch_<N>.failed. This refetches those
# exact photo_ids from the iNaturalist open-data S3 bucket and converts them on ingest to <id>.webp
# (the hardened converter never deletes on failure). Idempotent: any <id>.webp already present is
# skipped by download_batch_from_list()'s webp-aware resume.
# =============================================================================
suppressMessages({library(readr); library(stringr); library(curl); library(dplyr)})
source("R/download_images_new_download.R")
reticulate::use_condaenv("reticulate-gpu2", required = TRUE)

batch    <- commandArgs(trailingOnly = TRUE)[1]
stopifnot(nzchar(batch))
faillog  <- sprintf("/blue/guralnick/share/r.dinnage/Projects/phenovision/.webp_convert/failed/batch_%s.failed", batch)
statusd  <- sprintf("/blue/guralnick/share/r.dinnage/Projects/phenovision/.webp_convert/recover_status")
dir.create(statusd, showWarnings = FALSE, recursive = TRUE)
root     <- "data/phenobase_inat_data/images/medium"
bdir     <- file.path(root, paste0("batch_", batch))

lines <- readLines(faillog)
lines <- lines[nzchar(lines)]
base  <- basename(lines)
pid   <- sub("\\.[^.]+$", "", base)
ext   <- tolower(sub("^.*\\.", "", base))         # lowercase for the S3 key (iNat keys are lowercase)
# de-dup on photo_id (a photo can appear once per fail-log line; guard anyway)
keep  <- !duplicated(pid)
pid <- pid[keep]; ext <- ext[keep]
cat(sprintf("[recover batch_%s] %d photo_ids from fail-log\n", batch, length(pid)))

batch_data <- list(
  batch_id   = as.integer(batch),
  photo_ids  = pid,
  urls       = sprintf("https://inaturalist-open-data.s3.amazonaws.com/photos/%s/medium.%s", pid, ext),
  dest_files = file.path(bdir, paste0(pid, ".", ext))
)

res <- download_batch_from_list(batch_data, status_dir = statusd)
n_webp <- sum(file.exists(file.path(bdir, paste0(pid, ".webp"))))
cat(sprintf("[recover batch_%s] DONE: %d/%d now present as .webp\n", batch, n_webp, length(pid)))
