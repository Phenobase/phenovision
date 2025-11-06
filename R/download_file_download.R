#' Download File from URL
#'
#' Downloads a file from a URL to a specified destination. Handles large files
#' with progress reporting.
#'
#' @param url Character. URL to download from
#' @param dest_file Character. Destination file path
#' @param quiet Logical. If FALSE, show progress (default: FALSE)
#' @param mode Character. Download mode - "wb" for binary (default: "wb")
#'
#' @return Character. Path to downloaded file (for targets format = "file")
#'
#' @details
#' This function uses base R download.file() with appropriate settings for
#' large files. It creates the destination directory if it doesn't exist.
#'
#' For parallel downloads of multiple files, use download_files_parallel().
#'
#' @examples
#' \dontrun{
#' # Download a single file
#' dwc_path <- download_file(
#'   "https://www.inaturalist.org/observations/phenobase-observations-dwca.zip",
#'   "data/inat/phenobase-observations-dwca.zip"
#' )
#' }
#'
#' @export
download_file <- function(url, dest_file, quiet = FALSE, mode = "wb") {

  # Create destination directory if needed
  dest_dir <- dirname(dest_file)
  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)

  # Download file
  message(sprintf("Downloading: %s", basename(url)))
  message(sprintf("         to: %s", dest_file))

  download.file(
    url = url,
    destfile = dest_file,
    mode = mode,
    quiet = quiet,
    method = "auto"  # Auto-detect best method (curl, wget, etc.)
  )

  # Verify download
  if (!file.exists(dest_file)) {
    stop(sprintf("Download failed: %s", dest_file))
  }

  file_size_mb <- file.size(dest_file) / 1024^2
  message(sprintf("Downloaded: %.2f MB", file_size_mb))

  # Return path (for targets format = "file")
  return(dest_file)
}


#' Download Multiple Files in Parallel
#'
#' Downloads multiple files concurrently using future/furrr for speed.
#'
#' @param urls Character vector. URLs to download
#' @param dest_files Character vector. Destination paths (same length as urls)
#' @param workers Integer. Number of parallel workers (default: 2)
#' @param quiet Logical. If FALSE, show progress (default: FALSE)
#'
#' @return Character vector. Paths to downloaded files
#'
#' @details
#' Uses furrr::future_map2() for parallel downloads. Set up parallel backend
#' first with future::plan(multisession, workers = N).
#'
#' Limited to 2 workers by default to avoid overwhelming servers.
#'
#' @examples
#' \dontrun{
#' # Download multiple files
#' urls <- c(
#'   "https://example.com/file1.zip",
#'   "https://example.com/file2.tar.gz"
#' )
#' dest_files <- c("data/file1.zip", "data/file2.tar.gz")
#'
#' paths <- download_files_parallel(urls, dest_files, workers = 2)
#' }
#'
#' @export
download_files_parallel <- function(urls, dest_files, workers = 2, quiet = FALSE) {

  # Validate inputs
  if (length(urls) != length(dest_files)) {
    stop("urls and dest_files must have the same length")
  }

  # Setup parallel backend
  future::plan(future::multisession, workers = workers)
  on.exit(future::plan(future::sequential), add = TRUE)

  # Download files in parallel
  message(sprintf("Downloading %d files with %d workers", length(urls), workers))

  paths <- furrr::future_map2_chr(
    urls,
    dest_files,
    download_file,
    quiet = quiet,
    .progress = !quiet,
    .options = furrr::furrr_options(seed = TRUE)
  )

  return(paths)
}
