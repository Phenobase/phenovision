## Load your packages, e.g. library(targets).
library(targets)
library(crew)
library(crew.cluster)
source("./packages.R")

## Load your R files
lapply(list.files("./R", full.names = TRUE, pattern = "_targets"), source)

tar_option_set(
  controller = crew_controller_slurm()
)

## tar_plan supports drake-style targets and also tar_target()
tar_plan(

# target = function_to_make(arg), ## drake style

  tar_target(model_doi, "10.57967/hf/2763"),

  tar_target(images_to_annotate, get_images(meta_images, model_doi)) ## targets style

)
