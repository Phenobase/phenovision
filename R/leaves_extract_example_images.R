library(tidyverse)

leaves_train <- read_csv("data/leaves/train.csv")

green_leaves_yes <- leaves_train |>
  filter(genus != "Oxalis" & genus != "Viola" & leaves_green == 1) |>
  slice_sample(n = 250)

green_leaves_no <- leaves_train |>
  filter(genus != "Oxalis" & genus != "Viola" & leaves_green == 0) |>
  slice_sample(n = 250)

colored_leaves_yes <- leaves_train |>
  filter(genus != "Oxalis" & genus != "Viola" & leaves_colored == 1) |>
  slice_sample(n = 250)

colored_leaves_no <- leaves_train |>
  filter(genus != "Oxalis" & genus != "Viola" & leaves_colored == 0) |>
  slice_sample(n = 250)

no_leaves_yes <- leaves_train |>
  filter(genus != "Oxalis" & genus != "Viola" & leaves_green == 0 & leaves_colored == 0 & leaves_breaking_buds == 0) |>
  slice_sample(n = 250)

no_leaves_no <- leaves_train |>
  filter(genus != "Oxalis" & genus != "Viola" & !(leaves_green == 0 & leaves_colored == 0 & leaves_breaking_buds == 0)) |>
  slice_sample(n = 250)

walk(green_leaves_yes$file_name,
      ~ file.copy(.x, file.path("output/leaves/image_user_annotation_test/green/Detected", basename(.x)),
                  copy.mode = FALSE, copy.date = TRUE))

walk(green_leaves_no$file_name,
     ~ file.copy(.x, file.path("output/leaves/image_user_annotation_test/green/Not Detected", basename(.x)),
                 copy.mode = FALSE, copy.date = TRUE))

walk(colored_leaves_yes$file_name,
     ~ file.copy(.x, file.path("output/leaves/image_user_annotation_test/colored/Detected", basename(.x)),
                 copy.mode = FALSE, copy.date = TRUE))

walk(colored_leaves_no$file_name,
     ~ file.copy(.x, file.path("output/leaves/image_user_annotation_test/colored/Not Detected", basename(.x)),
                 copy.mode = FALSE, copy.date = TRUE))

walk(no_leaves_yes$file_name,
     ~ file.copy(.x, file.path("output/leaves/image_user_annotation_test/none/Detected", basename(.x)),
                 copy.mode = FALSE, copy.date = TRUE))

walk(no_leaves_no$file_name,
     ~ file.copy(.x, file.path("output/leaves/image_user_annotation_test/none/Not Detected", basename(.x)),
                 copy.mode = FALSE, copy.date = TRUE))

## extra

colored_leaves_no <- colored_leaves_no |>
  slice_sample(n = 100)

colored_leaves_yes <- colored_leaves_yes |>
  slice_sample(n = 100)

walk(colored_leaves_yes$file_name,
     ~ file.copy(.x, file.path("output/leaves/image_user_annotation_test_extra_col/Detected", basename(.x)),
                 copy.mode = FALSE, copy.date = TRUE))

walk(colored_leaves_no$file_name,
     ~ file.copy(.x, file.path("output/leaves/image_user_annotation_test_extra_col/Not Detected", basename(.x)),
                 copy.mode = FALSE, copy.date = TRUE))

################# do another set #############
## get list of previous images so as not to double-dip
old <- list.files("output/leaves/image_user_annotation_test", recursive = TRUE)

leaves_train <- read_csv("data/leaves/train.csv")
leaves_train <- leaves_train |>
  filter(!basename(file_name) %in% basename(old))

green_leaves_yes <- leaves_train |>
  filter(genus != "Oxalis" & genus != "Viola" & leaves_green == 1) |>
  slice_sample(n = 250)

green_leaves_no <- leaves_train |>
  filter(genus != "Oxalis" & genus != "Viola" & leaves_green == 0) |>
  slice_sample(n = 250)

colored_leaves_yes <- leaves_train |>
  filter(genus != "Oxalis" & genus != "Viola" & leaves_colored == 1) |>
  slice_sample(n = 250)

colored_leaves_no <- leaves_train |>
  filter(genus != "Oxalis" & genus != "Viola" & leaves_colored == 0) |>
  slice_sample(n = 250)

walk(green_leaves_yes$file_name,
     ~ file.copy(.x, file.path("output/leaves/image_user_annotation_test/green/Detected", basename(.x)),
                 copy.mode = FALSE, copy.date = TRUE))

walk(green_leaves_no$file_name,
     ~ file.copy(.x, file.path("output/leaves/image_user_annotation_test/green/Not Detected", basename(.x)),
                 copy.mode = FALSE, copy.date = TRUE))

walk(colored_leaves_yes$file_name,
     ~ file.copy(.x, file.path("output/leaves/image_user_annotation_test/colored/Detected", basename(.x)),
                 copy.mode = FALSE, copy.date = TRUE))

walk(colored_leaves_no$file_name,
     ~ file.copy(.x, file.path("output/leaves/image_user_annotation_test/colored/Not Detected", basename(.x)),
                 copy.mode = FALSE, copy.date = TRUE))
