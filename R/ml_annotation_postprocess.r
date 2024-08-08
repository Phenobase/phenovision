require(tidyverse)
require(tidymodels)
require(probably)
require(arrow)
require(uuid)

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

convert_to_ingestion <- function(dat, meta_images, meta_obs, meta_taxa, fam_stats, filter_detected = TRUE, filter_low_certainty = TRUE, datasource = "iNaturalist") {

    sample_dataset <- dat |>
        distinct(file_name, .keep_all = TRUE) |>
        mutate(photo_id = fs::path_ext_remove(file_name))

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
        left_join(metadat_obs |>
                    select(observation_uuid, latitude, longitude, positional_accuracy, taxon_id, quality_grade, observed_on) |>
                    filter(observation_uuid %in% sample_dataset$observation_uuid),
                    by = "observation_uuid",
                    copy = TRUE)

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
        select(file_name, ancestry) |>
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
        left_join(fams |> select(file_name, family)) |>
        left_join(gens |> select(file_name, genus))

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
        sample_dataset2 <- sample_dataset2 |>
            filter(as.character(.class) == "Detected")
    }
    if(filter_low_certainty) {
        sample_dataset2 <- sample_dataset2 |>
            filter(.equivocal == "Unequivocal")
    }
    

    ## sample again if necessary
    if(!is.null(sample)) {
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
                observed_image_guid = photo_id,
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