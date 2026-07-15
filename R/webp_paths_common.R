#' Build the on-disk image path for an iNaturalist photo (WebP store).
#'
#' The "medium" image store was converted to WebP q82 (see notes/image_compression_report.md):
#' on-disk files are `<photo_id>.webp` regardless of the photo's original `extension`. The
#' `extension` column is retained ONLY for the iNat S3 fetch URL (`.../medium.<extension>`); it must
#' NOT be used to build on-disk filenames. Use these helpers everywhere a per-image path is built.
#'
#' @param images_root images-store root, e.g. "/blue/.../images/medium" or
#'   "data/phenobase_inat_data/images/medium".
#' @param batch_j integer/character batch number; @param photo_id the photo id.
#' @return character path `<images_root>/batch_<batch_j>/<photo_id>.webp`.
webp_image_path <- function(images_root, batch_j, photo_id) {
  file.path(images_root, paste0("batch_", batch_j), paste0(photo_id, ".webp"))
}

#' Flat variant (no batch subdir) -- `<images_root>/<photo_id>.webp`.
webp_image_path_flat <- function(images_root, photo_id) {
  file.path(images_root, paste0(photo_id, ".webp"))
}
