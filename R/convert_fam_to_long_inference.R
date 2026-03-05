#' Convert Family Statistics to Long Format
#'
#' Converts wide-format family-level accuracy statistics to long format
#' for easier joining with observation-level data.
#'
#' @param fam_dat Data frame with family-level statistics (wide format)
#' @param trait Character. Type of predictions: "flower/fruit" or "leaves"
#'
#' @return Data frame in long format with:
#'   - `family`: Family name
#'   - `trait`: Trait code (gr, cl, bb for leaves)
#'   - `proportion_certainty_family`: Proportion of predictions with high certainty (1 - equiv_prop)
#'   - `accuracy_family`: Accuracy including equivocal as errors
#'   - `accuracy_excluding_certainty_family`: Accuracy excluding equivocal
#'   - `count_family`: Number of test images for this family
#'
#' @details
#' This function pivots three sets of columns:
#' 1. `equiv_prop_*` → `proportion_certainty_family` (inverted: 1 - equiv_prop)
#' 2. `.accuracy_family_leaves_*_incl_equiv` → `accuracy_family`
#' 3. `.accuracy_family_leaves_*` (excluding _incl_equiv) → `accuracy_excluding_certainty_family`
#'
#' **Trait Name Mapping:**
#' - `green_incl_equiv` / `green` → `gr`
#' - `colored_incl_equiv` / `colored` → `cl`
#' - `breaking_buds_incl_equiv` / `breaking_buds` → `bb`
#'
#' Only implements "leaves" trait currently.
#'
#' @examples
#' \dontrun{
#' # Convert family statistics to long format
#' fam_stats_long <- convert_fam_to_long(fam_stats, trait = "leaves")
#'
#' # Join with observation data
#' obs_data <- obs_data |>
#'   left_join(fam_stats_long, by = c("family", "trait"))
#' }
#'
#' @export
convert_fam_to_long <- function(fam_dat, trait = c("flower/fruit", "leaves")) {

  trait <- match.arg(trait)

  if (trait == "leaves") {
    # Pivot proportion of equivocal (uncertainty) columns
    fam_equiv <- fam_dat |>
      dplyr::select(family, starts_with("equiv_prop_")) |>
      tidyr::pivot_longer(
        -family,
        names_to = "trait",
        names_prefix = "equiv_prop_",
        values_to = "proportion_certainty_family"
      ) |>
      # Invert: high proportion equivocal = low proportion certain
      dplyr::mutate(proportion_certainty_family = 1 - proportion_certainty_family)

    # Pivot accuracy INCLUDING equivocal as errors
    fam_acc <- fam_dat |>
      dplyr::select(family, ends_with("_incl_equiv")) |>
      tidyr::pivot_longer(
        -family,
        names_to = "trait",
        names_prefix = ".accuracy_family_leaves_",
        values_to = "accuracy_family"
      ) |>
      dplyr::mutate(
        trait = dplyr::case_match(
          trait,
          "green_incl_equiv" ~ "gr",
          "colored_incl_equiv" ~ "cl",
          "breaking_buds_incl_equiv" ~ "bb",
          .default = ""
        )
      )

    # Pivot accuracy EXCLUDING equivocal (only high certainty)
    fam_acc_cert <- fam_dat |>
      dplyr::select(
        family,
        starts_with(".accuracy_family_leaves"),
        -ends_with("_incl_equiv")
      ) |>
      tidyr::pivot_longer(
        -family,
        names_to = "trait",
        names_prefix = ".accuracy_family_leaves_",
        values_to = "accuracy_excluding_certainty_family"
      ) |>
      dplyr::mutate(
        trait = dplyr::case_match(
          trait,
          "green" ~ "gr",
          "colored" ~ "cl",
          "breaking_buds" ~ "bb",
          .default = ""
        )
      )

    # Join all pivoted statistics
    fam_stats <- fam_acc |>
      dplyr::left_join(fam_acc_cert, by = c("family", "trait")) |>
      dplyr::left_join(fam_equiv, by = c("family", "trait")) |>
      dplyr::left_join(
        fam_dat |> dplyr::select(family, count_family = count),
        by = "family"
      )
  }

  if (trait == "flower/fruit") {
    # Pivot proportion of equivocal (uncertainty) columns
    fam_equiv <- fam_dat |>
      dplyr::select(family, starts_with("equiv_prop_")) |>
      tidyr::pivot_longer(
        -family,
        names_to = "trait",
        names_prefix = "equiv_prop_",
        values_to = "proportion_certainty_family"
      ) |>
      # Invert: high proportion equivocal = low proportion certain
      dplyr::mutate(proportion_certainty_family = 1 - proportion_certainty_family)

    # Pivot accuracy INCLUDING equivocal as errors
    fam_acc <- fam_dat |>
      dplyr::select(family, ends_with("_incl_equiv")) |>
      tidyr::pivot_longer(
        -family,
        names_to = "trait",
        names_prefix = ".accuracy_family_",
        values_to = "accuracy_family"
      ) |>
      dplyr::mutate(
        trait = dplyr::case_match(
          trait,
          "flower_incl_equiv" ~ "fl",
          "fruit_incl_equiv" ~ "fr",
          .default = ""
        )
      )

    # Pivot accuracy EXCLUDING equivocal (only high certainty)
    fam_acc_cert <- fam_dat |>
      dplyr::select(
        family,
        starts_with(".accuracy_family_"),
        -ends_with("_incl_equiv")
      ) |>
      tidyr::pivot_longer(
        -family,
        names_to = "trait",
        names_prefix = ".accuracy_family_",
        values_to = "accuracy_excluding_certainty_family"
      ) |>
      dplyr::mutate(
        trait = dplyr::case_match(
          trait,
          "flower" ~ "fl",
          "fruit" ~ "fr",
          .default = ""
        )
      )

    # Join all pivoted statistics
    fam_stats <- fam_acc |>
      dplyr::left_join(fam_acc_cert, by = c("family", "trait")) |>
      dplyr::left_join(fam_equiv, by = c("family", "trait")) |>
      dplyr::left_join(
        fam_dat |> dplyr::select(family, count_family = count),
        by = "family"
      )
  }

  fam_stats
}
