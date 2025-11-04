library(tidyverse)

fields <- read_csv("data/Phenobase_fields.csv")

leaf_data <- read_csv("output/leaves/phenovision-init_model2_04_11_2025/leaves_sample_inferences_ingest_format.csv")

ml_fields <- fields$Field[fields$`iNat ML` != "Not Applicable" & fields$`User downloads` == "YES"] |>
  na.omit()

colnames(leaf_data)[!colnames(leaf_data) %in% ml_fields]

field_map <- tribble(~new_field, ~old_field,
                     "dataSource", "datasource",
                     "scientificName", "scientific_name",
                     "trait", "trait",
                     "family", "family",
                     "year", "year",
                     "dayOfYear", "day_of_year",
                     "latitude", "latitude",
                     "longitude", "longitude",
                     "observedMetadataUrl", "observed_metadata_url",
                     "annotationID", "machine_learning_annotation_id",
                     "annotationMethod", "annotation_method",
                     "occurrenceID", "observation_uuid",
                     "basisOfRecord", "basis_of_record",
                     "genus", "genus",
                     "taxonRank", "taxon_rank",
                     "date", "verbatim_date",
                     "recordedBy", "recorded_by",
                     "coordinateUncertaintyInMeters", "coordinate_uncertainty_meters",
                     "verbatimTrait", "trait",
                     "modelUri", "model_uri",
                     "accuracyExcludingUncertainFamily", "accuracy_excluding_certainty_family",
                     "ObservedImageUrl", "observed_image_url",
                     "mappedTraits", NA)

field_map_v <- field_map$new_field
names(field_map_v) <- field_map$old_field

leaf_data_new <- leaf_data[ , intersect(names(field_map_v), colnames(leaf_data))]
colnames(leaf_data_new) <- field_map_v[colnames(leaf_data_new)]

leaf_data_new <- leaf_data_new |>
  mutate(verbatimTrait = trait, trait = paste0(trait, " present"),
         annotation_method = "machine")

write_csv(leaf_data_new, "output/leaves/phenovision-init_model2_04_11_2025/leaves_sample_inferences_ingest_format_final_fields.csv")

colnames(leaf_data)[!colnames(leaf_data) %in% field_map$old_field]
