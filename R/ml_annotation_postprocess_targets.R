require(tidyverse)
require(tidymodels)
require(probably)
require(arrow)
require(uuid)
require(cli)

threshold_model_output <- function(results, flower_thresholds = c(0.28, 0.84, 0.85),
                                   fruit_thresholds = c(0.23, 0.53, 0.75)) {

    fruit_range <- c(fruit_thresholds[2] - fruit_thresholds[1],
                    fruit_thresholds[3] - fruit_thresholds[2])

    flower_range <- c(flower_thresholds[2] - flower_thresholds[1],
                    flower_thresholds[3] - flower_thresholds[2])

    results <- results |>
      mutate(.class_flower = make_two_class_pred(.pred_flower, c("Detected", "Not Detected"),
                                                 threshold = flower_thresholds[2],
                                                 buffer = flower_range),
             .class_fruit = make_two_class_pred(.pred_fruit, c("Detected", "Not Detected"),
                                                threshold = fruit_thresholds[2],
                                                buffer = fruit_range),
             .equivocal_flower = ifelse(is_equivocal(.class_flower), "Equivocal", "Unequivocal"),
             .equivocal_fruit = ifelse(is_equivocal(.class_fruit), "Equivocal", "Unequivocal")) |>
      mutate(.class_flower = make_two_class_pred(.pred_flower, c("Detected", "Not Detected"),
                                                 threshold = flower_thresholds[2]),
             .class_fruit = make_two_class_pred(.pred_fruit, c("Detected", "Not Detected"),
                                                 threshold = fruit_thresholds[2]))

    results
}

convert_to_ingestion <- function(dat, meta_images, meta_obs, meta_taxa, fam_stats, filter_detected = TRUE, filter_low_certainty = TRUE, sample = NULL, datasource = "iNaturalist") {

    cli_alert_info("Preparing model output: ")
    cli_progress_step("Adding observation metadata..")
    sample_dataset <- dat |>
        distinct(file_name, .keep_all = TRUE) |>
        mutate(photo_id = fs::path_ext_remove(file_name),
               extension = fs::path_ext(file_name))

    sample_dataset <- sample_dataset |>
        left_join(meta_images |>
                    select(photo_id, observation_uuid) |>
                    filter(photo_id %in% sample_dataset$photo_id),
                    by = "photo_id",
                    copy = TRUE)

    sample_dataset <- sample_dataset |>
        mutate(inat_URL = paste0("https://www.inaturalist.org/observations/", observation_uuid),
               observed_image_url = paste0("https://www.inaturalist.org/photos/", photo_id))

    sample_dataset <- sample_dataset |>
        left_join(meta_obs |>
                    select(observation_uuid, latitude, longitude, positional_accuracy, taxon_id, quality_grade, observed_on) |>
                    filter(observation_uuid %in% sample_dataset$observation_uuid),
                    by = "observation_uuid",
                    copy = TRUE) |>
        mutate(observed_image_guid = file.path("https://inaturalist-open-data.s3.amazonaws.com/photos", photo_id, paste0("small.", extension)))

    cli_progress_step("Adding taxa metadata..")

    sample_dataset <- add_taxonomy(sample_dataset, taxa_meta)

    cli_progress_step("Adding family-level statistics..")
    ### add family level stats
    sample_dataset <- sample_dataset |>
        left_join(fam_stats |>
                    filter(test == "test") |>
                    select(family, .equivpropfamily_flower = equiv_prop_fl,
                            .equivpropfamily_fruit = equiv_prop_fr, .count_family = count,
                            .accuracyfamily_flower = .accuracy_family_flower,
                            .accuracyfamily_fruit = .accuracy_family_fruit,
                            .accuracyfamilyinclequiv_flower = .accuracy_family_flower_incl_equiv,
                            .accuracyfamilyinclequiv_fruit = .accuracy_family_fruit_incl_equiv))

    cli_progress_step("Reformating to ingestion-style format..")

    sample_dataset2 <- sample_dataset |>
        select(-.logit_flower, -.logit_fruit) |>
        pivot_longer(cols = c(.pred_flower, .pred_fruit,
                                .class_flower, .class_fruit,
                                .equivocal_flower, .equivocal_fruit,
                                .equivpropfamily_flower, .equivpropfamily_fruit,
                                .accuracyfamily_flower, .accuracyfamily_fruit,
                                .accuracyfamilyinclequiv_flower, .accuracyfamilyinclequiv_fruit),
                    names_to = c(".value", ".trait"),
                    names_sep = "_"
                    )

    ## filter out low certainty and 'not detected' records
    if(filter_detected) {
        cli_progress_step("Filtering out 'Not Detected' results..")
        sample_dataset2 <- sample_dataset2 |>
            filter(as.character(.class) == "Detected")
    }
    if(filter_low_certainty) {
        cli_progress_step("Filtering out low certainty results..")
        sample_dataset2 <- sample_dataset2 |>
            filter(.equivocal == "Unequivocal")
    }


    ## sample again if necessary
    if(!is.null(sample)) {
        cli_progress_step("Sampling results..")
        sample_dataset2 <- sample_dataset2 |>
            slice_sample(n = sample)
    }

    sample_dataset2 <- sample_dataset2 |>
        mutate(machine_learning_annotation_id = UUIDgenerate(n = n()),
                datasource = datasource,
                day_of_year = yday(observed_on),
                year = year(observed_on),
                certainty = ifelse(.equivocal == "Equivocal", "Low", "High"),
                model_uri = "10.57967/hf/2763",
                basis_of_record = "MachineObservation") |>
        select(machine_learning_annotation_id,
                datasource,
                verbatim_date = observed_on,
                day_of_year,
                year,
                latitude,
                longitude,
                coordinate_uncertainty_meters = positional_accuracy,
                family,
                count_family = .count_family,
                genus,
                scientific_name = name,
                taxon_rank = rank,
                basis_of_record,
                trait = .trait,
                observed_image_guid,
                observed_image_url,
                observed_metadata_url = inat_URL,
                certainty,
                model_uri,
                prediction_probability = .pred,
                prediction_class = .class,
                proportion_low_certainty_family = .equivpropfamily,
                accuracy_excluding_low_certainty_family = .accuracyfamily,
                accuracy_family = .accuracyfamilyinclequiv)

    sample_dataset2
}

add_taxonomy <- function(sample_dataset, meta_taxa) {
  sample_dataset <- sample_dataset |>
    left_join(meta_taxa |>
                select(taxon_id, name, ancestry, rank_level, rank) |>
                filter(taxon_id %in% sample_dataset$taxon_id),
              by = "taxon_id",
              copy = TRUE)

  families <- meta_taxa |>
    filter(rank == "family") |>
    collect()

  genera <- meta_taxa |>
    filter(rank == "genus") |>
    collect()

  taxonomy <- sample_dataset |>
    select(photo_id, ancestry) |>
    mutate(taxa_ids = str_split(ancestry, "/")) |>
    select(-ancestry) |>
    unnest_longer(taxa_ids, transform = as.integer)

  fams <- taxonomy |>
    left_join(families |> select(taxon_id, family = name), by = c(taxa_ids = "taxon_id")) |>
    drop_na()

  gens <- taxonomy |>
    left_join(genera |> select(taxon_id, genus = name), by = c(taxa_ids = "taxon_id")) |>
    drop_na()

  sample_dataset <- sample_dataset |>
    left_join(fams |> select(photo_id, family)) |>
    left_join(gens |> select(photo_id, genus))

  sample_dataset
}

threshold_annotations <- function(annotations,
                                  thresholds,
                                  trait = c("flower/fruit", "leaves"),
                                  meta_images) {

  trait <- match.arg(trait)

  if(trait == "leaves") {

    inf_df <- annotations |>
      filter(!corrupt) |>
      mutate(
        .class_gr = make_two_class_pred(
          estimate = .pred_leaves_green,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$gr[1],
          buffer = thresholds$gr[-1]
        ),
        .class_cl = make_two_class_pred(
          estimate = .pred_leaves_colored,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$cl[1],
          buffer = thresholds$cl[-1]
        ),
        .class_bb = make_two_class_pred(
          estimate = .pred_leaves_breaking_nuds,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$bb[1],
          buffer = thresholds$bb[-1]
        ),
        .equivocal_gr = ifelse(is_equivocal(.class_gr), "Equivocal", "Unequivocal"),
        .equivocal_cl = ifelse(is_equivocal(.class_cl), "Equivocal", "Unequivocal"),
        .equivocal_bb = ifelse(is_equivocal(.class_bb), "Equivocal", "Unequivocal")
      ) |>
      mutate(
        .class_gr = make_two_class_pred(
          estimate = .pred_leaves_green,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$gr[1]
        ),
        .class_cl = make_two_class_pred(
          estimate = .pred_leaves_colored,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$cl[1]
        ),
        .class_bb = make_two_class_pred(
          estimate = .pred_leaves_breaking_nuds,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$bb[1]
        ))

    inf_df <- inf_df |>
      mutate(.class_no = ifelse(.class_gr == "Not Detected" & .class_cl == "Not Detected" & .class_bb == "Not Detected",
                                "Detected", "Not Detected"),
             .equivocal_no = ifelse(.equivocal_gr == "Unequivocal" & .equivocal_cl == "Unequivocal" & .equivocal_bb == "Unequivocal",
                                    "Unequivocal", "Equivocal"))
  }

  if(trait == "flower/fruit") {
    inf_df <- annotations |>
      filter(!corrupt) |>
      mutate(
        .class_fl = make_two_class_pred(
          estimate = .pred_flower,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$fl[1],
          buffer = thresholds$fl[-1]
        ),
        .class_fr = make_two_class_pred(
          estimate = .pred_fruit,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$fr[1],
          buffer = thresholds$fr[-1]
        ),
        .equivocal_fl = ifelse(is_equivocal(.class_fl), "Equivocal", "Unequivocal"),
        .equivocal_fr = ifelse(is_equivocal(.class_fr), "Equivocal", "Unequivocal")
      ) |>
      mutate(
        .class_fl = make_two_class_pred(
          estimate = .pred_flower,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$fl[1]
        ),
        .class_fr = make_two_class_pred(
          estimate = .pred_fruit,
          levels = c("Detected", "Not Detected"),
          threshold = thresholds$fr[1]
        ))
  }

  #browser()

  inf_df <- inf_df |>
    left_join(open_dataset(meta_images) |>
                select(photo_id, observation_uuid) |>
                filter(photo_id %in% inf_df$photo_id),
              copy = TRUE)
  inf_df

}

add_metadata <- function(annotations_thresholded_leaves,
                         taxonomy, families, genera,
                         fam_dat_leaves,
                         meta_obs,
                         meta_observer,
                         model_doi_leaves) {

  gen_tax <- taxonomy |>
    filter(taxa_ids %in% genera$taxon_id) |>
    left_join(genera, by = c(taxa_ids = "taxon_id"))

  fam_tax <- taxonomy |>
    filter(taxa_ids %in% families$taxon_id) |>
    left_join(families, by = c(taxa_ids = "taxon_id"))

  new_annos <- annotations_thresholded_leaves |>
    left_join(gen_tax |> select(taxon_id, genus = name)) |>
    left_join(fam_tax |> select(taxon_id, family = name))

}

convert_to_long <- function(annotations_thresholded_leaves, trait = c("flower/fruit", "leaves")) {

  trait <- match.arg(trait)

  annotations_thresholded_leaves <- annotations_thresholded_leaves |>
    filter(!corrupt) |>
    select(-corrupt)

  if(trait == "leaves") {
    inf_class <- annotations_thresholded_leaves |>
      select(photo_id, starts_with(".class_")) |>
      mutate(.class_gr = as.character(as.factor(.class_gr)),
             .class_cl = as.character(as.factor(.class_cl)),
             .class_bb = as.character(as.factor(.class_bb))) |>
      pivot_longer(-photo_id, names_to = "trait", names_prefix = ".class_", values_to = "detected")

    inf_equiv <- annotations_thresholded_leaves  |>
      select(photo_id, starts_with(".equivocal_")) |>
      pivot_longer(-photo_id, names_to = "trait", names_prefix = ".equivocal_", values_to = "equivocal")

    inf_pred <- annotations_thresholded_leaves  |>
      select(photo_id, starts_with(".pred_leaves_")) |>
      pivot_longer(-photo_id, names_to = "trait", names_prefix = ".pred_leaves_", values_to = "preds") |>
      mutate(trait = case_match(trait,
                                "green" ~ "gr",
                                "colored" ~ "cl",
                                "breaking_nuds" ~ "bb",
                                .default = ""))

    inf_df <- inf_class |>
      left_join(inf_equiv) |>
      left_join(inf_pred) |>
      left_join(annotations_thresholded_leaves |>
                  select(photo_id, batch_j, extension, path, run_name, taxon_id, model_version,
                         observation_uuid))

  }

  inf_df
}

aggregate_by_obs <- function(annotations_long_leaves,
                             taxonomy, families, genera,
                             fam_dat_long_leaves,
                             meta_images,
                             meta_taxa) {
  #browser()
  meta <- open_dataset(meta_images)
  taxa <- open_dataset(meta_taxa)

  annotations_long_leaves <- annotations_long_leaves |>
    left_join(meta |>
                filter(batch_j %in% annotations_long_leaves$batch_j) |>
                select(photo_id,
                       observer_id,
                       latitude,
                       longitude,
                       positional_accuracy,
                       observed_on),
              copy = TRUE)

  obs_df <- annotations_long_leaves |>
    group_by(observation_uuid) |>
    mutate(n_images = n_distinct(photo_id, na.rm = TRUE)) |>
    group_by(observation_uuid, trait, detected, equivocal) |>
    mutate(samp = sample.int(n()) == 1) |>
    summarize(count = n(),
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
              photo_ids = list(photo_id)) |>
    ungroup()

  gen_tax <- taxonomy |>
    filter(taxa_ids %in% genera$taxon_id) |>
    left_join(genera, by = c(taxa_ids = "taxon_id"))

  gen_tax <- taxonomy |>
    filter(taxa_ids %in% genera$taxon_id) |>
    left_join(genera, by = c(taxa_ids = "taxon_id"))

  fam_tax <- taxonomy |>
    filter(taxa_ids %in% families$taxon_id) |>
    left_join(families, by = c(taxa_ids = "taxon_id"))

  obs_df <- obs_df |>
    left_join(gen_tax |> select(taxon_id, genus = name)) |>
    left_join(fam_tax |> select(taxon_id, family = name)) |>
    left_join(fam_dat_long_leaves)

  obs_df <- obs_df |>
    left_join(taxa |> select(taxon_id, scientific_name = name) |>
                filter(taxon_id %in% unique(obs_df$taxon_id)),
              copy = TRUE)

  obs_df <- obs_df |>
    select(verbatim_date = observed_on,
           latitude,
           longitude,
           coordinate_uncertainty_meters = positional_accuracy,
           family,
           genus,
           scientific_name,
           observed_image_guid = photo_id_samp,
           count_images = count,
           everything()) |>
    mutate(datasource = "iNaturalist",
           day_of_year = yday(verbatim_date),
           year = year(verbatim_date),
           certainty = ifelse(equivocal == "Equivocal", "Low", "High"),
           model_uri = model_version,
           observed_metadata_url = paste0("https://www.inaturalist.org/observations/", observation_uuid),
           observed_image_url = paste0("https://www.inaturalist.org/photos/", observed_image_guid),
           prediction_probability = pred_med,
           prediction_class = detected,
           trait = case_match(trait,
                              "gr" ~ "green leaves",
                              "cl" ~ "colored leaves",
                              "bb" ~ "breaking buds",
                              "no" ~ "no live leaves",
                              .default = ""))

  obs_df


}

convert_fam_to_long <- function(fam_dat, trait = c("flower/fruit", "leaves")) {

  trait <- match.arg(trait)

  if(trait == "leaves") {

    fam_equiv <- fam_dat |>
      select(family, starts_with("equiv_prop_")) |>
      pivot_longer(-family, names_to = "trait", names_prefix = "equiv_prop_", values_to = "proportion_certainty_family") |>
      mutate(proportion_certainty_family = 1 - proportion_certainty_family)

    fam_acc <- fam_dat |>
      select(family, ends_with("_incl_equiv")) |>
      pivot_longer(-family, names_to = "trait", names_prefix = ".accuracy_family_leaves_", values_to = "accuracy_family") |>
      mutate(trait = case_match(trait,
                                "green_incl_equiv" ~ "gr",
                                "colored_incl_equiv" ~ "cl",
                                "breaking_buds_incl_equiv" ~ "bb",
                                .default = ""))

    fam_acc_cert <- fam_dat |>
      select(family, starts_with(".accuracy_family_leaves"), -ends_with("_incl_equiv")) |>
      pivot_longer(-family, names_to = "trait", names_prefix = ".accuracy_family_leaves_", values_to = "accuracy_excluding_certainty_family") |>
      mutate(trait = case_match(trait,
                                "green" ~ "gr",
                                "colored" ~ "cl",
                                "breaking_buds" ~ "bb",
                                .default = ""))

    fam_stats <- fam_acc |>
      left_join(fam_acc_cert) |>
      left_join(fam_equiv) |>
      left_join(fam_dat |>
                  select(family, count_family = count))
  }

  fam_stats

}

finalize_annotations_by_obs <- function(annotations_by_obs_leaves,
                                         fields,
                                         field_map) {

  field_map_v <- field_map$new_field
  names(field_map_v) <- field_map$old_field

  annotations_by_obs_leaves <- annotations_by_obs_leaves |>
    mutate(certainty = ifelse(equivocal == "Unequivocal", "High", "Low"),
           recorded_by = observer_id)

  leaf_data_new <- annotations_by_obs_leaves[ , intersect(names(field_map_v),
                                                          colnames(annotations_by_obs_leaves))]
  colnames(leaf_data_new) <- field_map_v[colnames(leaf_data_new)]

  leaf_data_new <- leaf_data_new |>
    mutate(verbatimTrait = trait, trait = paste0(trait, " present"),
           annotation_method = "machine")

  leaf_data_new



}

concatenate_csvs <- function(annotations, csv_path) {
  write_csv(annotations, csv_path, append = file.exists(csv_path))
  csv_path
}
