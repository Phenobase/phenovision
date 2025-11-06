#' Apply Learned Thresholds to Model Predictions
#'
#' Applies learned decision thresholds with uncertainty buffers to convert
#' continuous model predictions into three-class predictions: Detected,
#' Not Detected, or Equivocal (uncertain).
#'
#' @param annotations Data frame with model predictions (from annotate_batch)
#' @param thresholds Data frame or list with threshold values for each trait
#' @param trait Character. Type of predictions: "flower/fruit" or "leaves"
#' @param meta_images Arrow dataset with image metadata (for joining observation_uuid)
#'
#' @return Data frame with original predictions plus:
#'   - `.class_*`: Three-level factor (Detected, Not Detected, [NA] for Equivocal)
#'   - `.equivocal_*`: Two-level factor (Equivocal, Unequivocal)
#'   - `observation_uuid`: Joined from meta_images
#'
#' @details
#' **Threshold Structure:**
#' Each trait has three threshold values: [decision, lower_buffer, upper_buffer]
#' - Below `decision - lower_buffer`: Not Detected (Unequivocal)
#' - Within buffer zones: Equivocal (uncertain, marked as NA)
#' - Above `decision + upper_buffer`: Detected (Unequivocal)
#'
#' **For flower/fruit:**
#' - `thresholds$fl`: Flower thresholds
#' - `thresholds$fr`: Fruit thresholds
#'
#' **For leaves:**
#' - `thresholds$gr`: Green leaves
#' - `thresholds$cl`: Colored leaves
#' - `thresholds$bb`: Breaking buds
#' - Derives `no`: No live leaves (all three absent)
#'
#' **Note:** Typo fixed from original - was `breaking_nuds`, now correctly `breaking_buds`
#'
#' Uses `probably::make_two_class_pred()` for threshold application with buffers.
#'
#' @examples
#' \dontrun{
#' # Apply thresholds to leaf predictions
#' thresholds <- list(
#'   gr = c(0.5, 0.1, 0.1),
#'   cl = c(0.6, 0.15, 0.15),
#'   bb = c(0.55, 0.12, 0.12)
#' )
#'
#' results_thresh <- threshold_annotations(
#'   annotations_leaves,
#'   thresholds,
#'   trait = "leaves",
#'   meta_images = arrow::open_dataset("path/to/images")
#' )
#' }
#'
#' @export
threshold_annotations <- function(annotations,
                                  thresholds,
                                  trait = c("flower/fruit", "leaves"),
                                  meta_images) {

  trait <- match.arg(trait)

  if (trait == "leaves") {
    # Apply thresholds to each leaf trait
    inf_df <- annotations |>
      dplyr::filter(!corrupt) |>
      dplyr::mutate(
        # Apply thresholds with buffers
        .class_gr = probably::make_two_class_pred(
          estimate = .pred_leaves_green,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$gr[1],
          buffer = thresholds$gr[-1]
        ),
        .class_cl = probably::make_two_class_pred(
          estimate = .pred_leaves_colored,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$cl[1],
          buffer = thresholds$cl[-1]
        ),
        .class_bb = probably::make_two_class_pred(
          estimate = .pred_leaves_breaking_buds,  # TYPO FIXED HERE
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$bb[1],
          buffer = thresholds$bb[-1]
        ),
        # Mark equivocal predictions
        .equivocal_gr = ifelse(
          probably::is_equivocal(.class_gr),
          "Equivocal",
          "Unequivocal"
        ),
        .equivocal_cl = ifelse(
          probably::is_equivocal(.class_cl),
          "Equivocal",
          "Unequivocal"
        ),
        .equivocal_bb = ifelse(
          probably::is_equivocal(.class_bb),
          "Equivocal",
          "Unequivocal"
        )
      ) |>
      # Re-apply without buffers to get final class (NA → Not Detected)
      dplyr::mutate(
        .class_gr = probably::make_two_class_pred(
          estimate = .pred_leaves_green,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$gr[1]
        ),
        .class_cl = probably::make_two_class_pred(
          estimate = .pred_leaves_colored,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$cl[1]
        ),
        .class_bb = probably::make_two_class_pred(
          estimate = .pred_leaves_breaking_buds,  # TYPO FIXED HERE
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$bb[1]
        )
      )

    # Derive "no live leaves" class
    # (all three leaf types are "Not Detected")
    inf_df <- inf_df |>
      dplyr::mutate(
        .class_no = ifelse(
          .class_gr == "Not Detected" &
            .class_cl == "Not Detected" &
            .class_bb == "Not Detected",
          "Detected",
          "Not Detected"
        ),
        .equivocal_no = ifelse(
          .equivocal_gr == "Unequivocal" &
            .equivocal_cl == "Unequivocal" &
            .equivocal_bb == "Unequivocal",
          "Unequivocal",
          "Equivocal"
        )
      )
  }

  if (trait == "flower/fruit") {
    # Apply thresholds to flower and fruit
    inf_df <- annotations |>
      dplyr::filter(!corrupt) |>
      dplyr::mutate(
        # Apply thresholds with buffers
        .class_fl = probably::make_two_class_pred(
          estimate = .pred_flower,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$fl[1],
          buffer = thresholds$fl[-1]
        ),
        .class_fr = probably::make_two_class_pred(
          estimate = .pred_fruit,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$fr[1],
          buffer = thresholds$fr[-1]
        ),
        # Mark equivocal predictions
        .equivocal_fl = ifelse(
          probably::is_equivocal(.class_fl),
          "Equivocal",
          "Unequivocal"
        ),
        .equivocal_fr = ifelse(
          probably::is_equivocal(.class_fr),
          "Equivocal",
          "Unequivocal"
        )
      ) |>
      # Re-apply without buffers to get final class
      dplyr::mutate(
        .class_fl = probably::make_two_class_pred(
          estimate = .pred_flower,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$fl[1]
        ),
        .class_fr = probably::make_two_class_pred(
          estimate = .pred_fruit,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$fr[1]
        )
      )
  }

  # Join observation UUID from image metadata
  inf_df <- inf_df |>
    dplyr::left_join(
      arrow::open_dataset(meta_images) |>
        dplyr::select(photo_id, observation_uuid) |>
        dplyr::filter(photo_id %in% inf_df$photo_id),
      copy = TRUE
    )

  inf_df
}
