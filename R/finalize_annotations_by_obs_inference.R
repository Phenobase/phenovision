#' Finalize Annotations for Phenobase Ingestion
#'
#' Final formatting step to map internal field names to Phenobase schema and
#' prepare annotations for database ingestion.
#'
#' @param annotations_by_obs_leaves Data frame with observation-level annotations
#' @param fields Character vector of fields to retain (currently unused)
#' @param field_map Data frame mapping old_field to new_field names
#'
#' @return Data frame with:
#'   - Columns renamed according to field_map
#'   - `certainty`: Mapped from equivocal status (Unequivocal → High, Equivocal → Low)
#'   - `recorded_by`: Mapped from observer_id
#'   - `verbatimTrait`: Original trait code
#'   - `trait`: Formatted trait with " present" suffix
#'   - `annotation_method`: Set to "machine"
#'
#' @details
#' This function performs the final transformation to match Phenobase's
#' expected schema. It:
#'
#' 1. Maps certainty levels (Unequivocal/Equivocal → High/Low)
#' 2. Adds recorded_by field
#' 3. Selects only fields present in both data and field_map
#' 4. Renames columns according to field_map
#' 5. Adds verbatimTrait (original code) and formatted trait
#' 6. Marks all as "machine" annotation method
#'
#' **Expected field_map structure:**
#' ```
#' old_field          | new_field
#' -------------------|--------------------
#' observation_uuid   | observationID
#' latitude           | decimalLatitude
#' ...                | ...
#' ```
#'
#' @examples
#' \dontrun{
#' # Define field mapping
#' field_map <- tibble(
#'   old_field = c("observation_uuid", "latitude", "longitude"),
#'   new_field = c("observationID", "decimalLatitude", "decimalLongitude")
#' )
#'
#' # Finalize for ingestion
#' final_annotations <- finalize_annotations_by_obs(
#'   annotations_by_obs,
#'   fields = NULL,
#'   field_map = field_map
#' )
#' }
#'
#' @export
finalize_annotations_by_obs <- function(annotations_by_obs_leaves,
                                         fields,
                                         field_map) {

  # Create named vector for column renaming
  field_map_v <- field_map$new_field
  names(field_map_v) <- field_map$old_field

  # Map certainty and add recorded_by
  annotations_by_obs_leaves <- annotations_by_obs_leaves |>
    dplyr::mutate(
      certainty = ifelse(equivocal == "Unequivocal", "High", "Low"),
      recorded_by = observer_id
    )

  # Select only fields that exist in both data and field_map
  leaf_data_new <- annotations_by_obs_leaves[
    ,
    intersect(names(field_map_v), colnames(annotations_by_obs_leaves))
  ]

  # Rename columns according to field_map
  colnames(leaf_data_new) <- field_map_v[colnames(leaf_data_new)]

  # Add verbatim and formatted trait, annotation method
  leaf_data_new <- leaf_data_new |>
    dplyr::mutate(
      verbatimTrait = trait,
      trait = paste0(trait, " present"),
      annotation_method = "machine"
    )

  leaf_data_new
}
