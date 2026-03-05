#' Convert Wide Annotations to Long Format
#'
#' Converts wide-format annotation data (one row per image, multiple trait columns)
#' to long format (one row per image-trait combination). This facilitates
#' aggregation by observation and trait.
#'
#' @param annotations_thresholded_leaves Data frame with thresholded annotations
#' @param trait Character. Type of predictions: "flower/fruit" or "leaves"
#'
#' @return Data frame in long format with columns:
#'   - `photo_id`: Image identifier
#'   - `trait`: Trait code (gr, cl, bb, no for leaves; fl, fr for flowers/fruits)
#'   - `detected`: Detection class (Detected, Not Detected)
#'   - `equivocal`: Certainty (Equivocal, Unequivocal)
#'   - `preds`: Prediction probability
#'   - All other metadata columns from input
#'
#' @details
#' **Trait Codes (leaves):**
#' - `gr`: Green leaves
#' - `cl`: Colored leaves
#' - `bb`: Breaking buds
#' - `no`: No live leaves
#'
#' **Trait Codes (flower/fruit):**
#' - `fl`: Flowers
#' - `fr`: Fruits
#'
#' **Note:** Typo fixed from original - was `breaking_nuds`, now correctly `breaking_buds`
#'
#' The function:
#' 1. Filters out corrupt images
#' 2. Pivots `.class_*` columns to `detected`
#' 3. Pivots `.equivocal_*` columns to `equivocal`
#' 4. Pivots `.pred_*` columns to `preds`
#' 5. Joins all three pivots together
#' 6. Retains all other metadata columns
#'
#' @examples
#' \dontrun{
#' # Convert leaf annotations to long format
#' annotations_long <- convert_to_long(
#'   annotations_thresholded,
#'   trait = "leaves"
#' )
#'
#' # Result has one row per image-trait combination
#' table(annotations_long$trait)
#' # gr  cl  bb  no
#' # 100 100 100 100  (for 100 images)
#' }
#'
#' @export
convert_to_long <- function(annotations_thresholded_leaves,
                             trait = c("flower/fruit", "leaves")) {

  trait <- match.arg(trait)

  # Remove corrupt images
  annotations_thresholded_leaves <- annotations_thresholded_leaves |>
    dplyr::filter(!corrupt) |>
    dplyr::select(-corrupt)

  if (trait == "leaves") {
    # Pivot classification columns
    inf_class <- annotations_thresholded_leaves |>
      dplyr::select(photo_id, starts_with(".class_")) |>
      dplyr::mutate(
        .class_gr = as.character(as.factor(.class_gr)),
        .class_cl = as.character(as.factor(.class_cl)),
        .class_bb = as.character(as.factor(.class_bb))
      ) |>
      tidyr::pivot_longer(
        -photo_id,
        names_to = "trait",
        names_prefix = ".class_",
        values_to = "detected"
      )

    # Pivot equivocal columns
    inf_equiv <- annotations_thresholded_leaves |>
      dplyr::select(photo_id, starts_with(".equivocal_")) |>
      tidyr::pivot_longer(
        -photo_id,
        names_to = "trait",
        names_prefix = ".equivocal_",
        values_to = "equivocal"
      )

    # Pivot prediction columns
    inf_pred <- annotations_thresholded_leaves |>
      dplyr::select(photo_id, starts_with(".pred_leaves_")) |>
      tidyr::pivot_longer(
        -photo_id,
        names_to = "trait",
        names_prefix = ".pred_leaves_",
        values_to = "preds"
      ) |>
      dplyr::mutate(
        trait = dplyr::case_match(
          trait,
          "green" ~ "gr",
          "colored" ~ "cl",
          "breaking_buds" ~ "bb",  # TYPO FIXED HERE
          .default = ""
        )
      )

    # Join all pivoted data
    inf_df <- inf_class |>
      dplyr::left_join(inf_equiv, by = c("photo_id", "trait")) |>
      dplyr::left_join(inf_pred, by = c("photo_id", "trait")) |>
      dplyr::left_join(
        annotations_thresholded_leaves |>
          dplyr::select(
            photo_id, batch_j, extension, path, run_name,
            taxon_id, model_version, observation_uuid
          ),
        by = "photo_id"
      )
  }

  if (trait == "flower/fruit") {
    # Pivot classification columns
    inf_class <- annotations_thresholded_leaves |>
      dplyr::select(photo_id, starts_with(".class_")) |>
      dplyr::mutate(
        .class_fl = as.character(as.factor(.class_fl)),
        .class_fr = as.character(as.factor(.class_fr))
      ) |>
      tidyr::pivot_longer(
        -photo_id,
        names_to = "trait",
        names_prefix = ".class_",
        values_to = "detected"
      )

    # Pivot equivocal columns
    inf_equiv <- annotations_thresholded_leaves |>
      dplyr::select(photo_id, starts_with(".equivocal_")) |>
      tidyr::pivot_longer(
        -photo_id,
        names_to = "trait",
        names_prefix = ".equivocal_",
        values_to = "equivocal"
      )

    # Pivot prediction columns
    inf_pred <- annotations_thresholded_leaves |>
      dplyr::select(photo_id, .pred_flower, .pred_fruit) |>
      tidyr::pivot_longer(
        -photo_id,
        names_to = "trait",
        names_prefix = ".pred_",
        values_to = "preds"
      ) |>
      dplyr::mutate(
        trait = dplyr::case_match(
          trait,
          "flower" ~ "fl",
          "fruit" ~ "fr",
          .default = ""
        )
      )

    # Join all pivoted data
    inf_df <- inf_class |>
      dplyr::left_join(inf_equiv, by = c("photo_id", "trait")) |>
      dplyr::left_join(inf_pred, by = c("photo_id", "trait")) |>
      dplyr::left_join(
        annotations_thresholded_leaves |>
          dplyr::select(
            photo_id, batch_j, extension, path, run_name,
            taxon_id, model_version, observation_uuid
          ),
        by = "photo_id"
      )
  }

  inf_df
}
