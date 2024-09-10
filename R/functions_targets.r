annotate_images <- function(images_batch_split, model_doi, model_vers, inat_folder,
                            meta_images, meta_obs, meta_taxa, fam_stats) {

    images_batch_split <- images_batch_split[[1]]

    ## see if there are any completed annotations already

    complete_folder <- file.path(inat_folder, model_doi)
    if(dir.exists(complete_folder)) {
        files <- list.files(complete_folder, full.names = TRUE)
        done <- purrr::map(files, read_csv) |>
                    list_rbind() |>
                    pull(photo_id)
    }

    anno_dat <- annotate_batch(images_batch_split$path, model_doi = model_doi)
    corrupt_images <- anno_dat |>
        dplyr::filter(corrupt) |>
        mutate(path = file.path(inat_folder, "images", "medium", file_name)) |>
        select(path)

    anno_dat <- anno_dat |>
        dplyr::filter(!corrupt)

    inf_dat <- threshold_model_output(anno_dat)

    meta_images <- open_dataset(meta_images)
    meta_obs <- open_dataset(meta_obs)
    meta_taxa <- open_dataset(meta_taxa)

    fam_stats <- read_csv(fam_stats)

    inf_dat <- convert_to_ingestion(inf_dat, meta_images = meta_images, meta_obs = meta_obs,
                                    meta_taxa = meta_taxa, fam_stats = fam_stats)

    inf_dat <- inf_dat |>
        drop_na(latitude, longitude)

    vers_path <- file.path(inat_folder, "annotations", model_vers)
    if(!dir.exists(vers_path)) {
        dir.create(vers_path)
    }

    run_name <- as.character(images_batch_split$run_name)[1]
    csv_path <- file.path(vers_path, paste0("inference_results_", model_vers, "_run_", run_name, ".csv"))
    corrupted_path <- file.path(inat_folder, "corrupted_image_list", "corrupted_image_list.csv")

    write_csv(inf_dat, csv_path)
    write_csv(corrupt_images, corrupted_path, append = TRUE)

    return(csv_path)

}
