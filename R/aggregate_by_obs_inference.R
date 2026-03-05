#' Aggregate Image-Level Annotations to Observation Level
#'
#' Aggregates predictions from multiple images of the same observation,
#' taking the median prediction and adding taxonomy and family-level statistics.
#'
#' @param annotations_long_leaves Data frame with long-format annotations
#' @param taxonomy Data frame with taxon ID to ancestry mapping
#' @param families Data frame with family-level taxonomic info
#' @param genera Data frame with genus-level taxonomic info
#' @param fam_dat_long_leaves Data frame with family-level accuracy statistics (long format)
#' @param meta_images Arrow dataset with image metadata
#' @param meta_taxa Arrow dataset with taxonomic metadata
#'
#' @return Data frame with one row per observation-trait-detection-certainty combination:
#'   - `observation_uuid`: iNaturalist observation ID
#'   - `trait`: Phenological trait (e.g., "green leaves", "colored leaves")
#'   - `detected`: Detection class
#'   - `certainty`: High or Low (based on equivocal status)
#'   - `pred_med`: Median prediction probability across images
#'   - `count_images`: Number of images for this observation-trait
#'   - `n_images`: Total number of images for this observation
#'   - Geographic, taxonomic, and family statistics metadata
#'
#' @details
#' This function:
#' 1. Joins geographic and observer metadata from images
#' 2. Groups by observation_uuid and trait
#' 3. Computes median prediction across all images
#' 4. Adds family and genus names from taxonomy
#' 5. Joins family-level accuracy statistics
#' 6. Adds scientific name from taxa metadata
#' 7. Formats output for Phenobase ingestion
#' 8. Creates iNaturalist URLs for observation and image
#'
#' **Trait Name Mapping (Leaves):**
#' - `gr` → "green leaves"
#' - `cl` → "colored leaves"
#' - `bb` → "breaking buds"
#' - `no` → "no live leaves"
#'
#' **Trait Name Mapping (Reproductive):**
#' - `fl` → "flower"
#' - `fr` → "fruit"
#'
#' @examples
#' \dontrun{
#' # Aggregate to observation level
#' obs_annotations <- aggregate_by_obs(
#'   annotations_long,
#'   taxonomy,
#'   families,
#'   genera,
#'   fam_stats_long,
#'   meta_images = arrow::open_dataset("path/to/images"),
#'   meta_taxa = arrow::open_dataset("path/to/taxa")
#' )
#' }
#'
#' @export
aggregate_by_obs <- function(annotations_long_leaves,
                              taxonomy,
                              families,
                              genera,
                              fam_dat_long_leaves,
                              meta_images,
                              meta_taxa) {

  # Open Arrow datasets
  meta <- arrow::open_dataset(meta_images)
  taxa <- arrow::open_dataset(meta_taxa)

  # Join geographic and observer metadata
  annotations_long_leaves <- annotations_long_leaves |>
    dplyr::left_join(
      meta |>
        dplyr::filter(batch_j %in% annotations_long_leaves$batch_j) |>
        dplyr::select(
          photo_id,
          observer_id,
          latitude,
          longitude,
          positional_accuracy,
          observed_on
        ),
      copy = TRUE
    )

  # Aggregate by observation and trait
  obs_df <- annotations_long_leaves |>
    dplyr::group_by(observation_uuid) |>
    dplyr::mutate(n_images = dplyr::n_distinct(photo_id, na.rm = TRUE)) |>
    dplyr::group_by(observation_uuid, trait, detected, equivocal) |>
    dplyr::mutate(samp = sample.int(dplyr::n()) == 1) |>
    dplyr::summarize(
      count = dplyr::n(),
      photo_id_samp = photo_id[which(samp)],
      pred_med = median(preds, na.rm = TRUE),
      batch_j = batch_j[which(samp)],
      extension = extension[which(samp)],
      path = path[which(samp)],
      run_name = run_name[which(samp)],
      taxon_id = taxon_id[which(samp)],
      model_version = model_version[which(samp)],
      n_images = n_images[which(samp)],
      observer_id = observer_id[which(samp)],
      latitude = latitude[which(samp)],
      longitude = longitude[which(samp)],
      positional_accuracy = positional_accuracy[which(samp)],
      observed_on = observed_on[which(samp)],
      photo_ids = list(photo_id)
    ) |>
    dplyr::ungroup()

  # Extract genus names from taxonomy
  gen_tax <- taxonomy |>
    dplyr::filter(taxa_ids %in% genera$taxon_id) |>
    dplyr::left_join(genera, by = c(taxa_ids = "taxon_id"))

  # Extract family names from taxonomy
  fam_tax <- taxonomy |>
    dplyr::filter(taxa_ids %in% families$taxon_id) |>
    dplyr::left_join(families, by = c(taxa_ids = "taxon_id"))

  # Join genus and family names
  obs_df <- obs_df |>
    dplyr::left_join(gen_tax |> dplyr::select(taxon_id, genus = name)) |>
    dplyr::left_join(fam_tax |> dplyr::select(taxon_id, family = name)) |>
    dplyr::left_join(fam_dat_long_leaves)

  # Join scientific name
  obs_df <- obs_df |>
    dplyr::left_join(
      taxa |>
        dplyr::select(taxon_id, scientific_name = name) |>
        dplyr::filter(taxon_id %in% unique(obs_df$taxon_id)),
      copy = TRUE
    )

  # Format for output
  obs_df <- obs_df |>
    dplyr::select(
      verbatim_date = observed_on,
      latitude,
      longitude,
      coordinate_uncertainty_meters = positional_accuracy,
      family,
      genus,
      scientific_name,
      observed_image_guid = photo_id_samp,
      count_images = count,
      dplyr::everything()
    ) |>
    dplyr::mutate(
      datasource = "iNaturalist",
      day_of_year = lubridate::yday(verbatim_date),
      year = lubridate::year(verbatim_date),
      certainty = ifelse(equivocal == "Equivocal", "Low", "High"),
      model_uri = model_version,
      observed_metadata_url = paste0("https://www.inaturalist.org/observations/", observation_uuid),
      observed_image_url = paste0("https://www.inaturalist.org/photos/", observed_image_guid),
      prediction_probability = pred_med,
      prediction_class = detected,
      trait = dplyr::case_match(
        trait,
        # Leaf phenology
        "gr" ~ "green leaves",
        "cl" ~ "colored leaves",
        "bb" ~ "breaking buds",
        "no" ~ "no live leaves",
        # Reproductive structures
        "fl" ~ "flower",
        "fr" ~ "fruit",
        .default = ""
      )
    )

  obs_df
}
