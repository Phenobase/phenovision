library(tidyverse)
library(reticulate)
library(glue)

system("python phenobase/download_inat_data.py --output-dir data/inat")
system("python phenobase/extract_inat_data.py --dwc-file data/inat/phenobase-observations-dwca.zip --metadata-file data/inat/inaturalist-open-data-latest.tar.gz --out-parquet data/inat/inaturalist-phenobase-data.parquet")
system("python phenobase/split_data.py --observations data/inat/inaturalist-phenobase-data.parquet")
system("python phenobase/download_images.py --observations data/inat/inaturalist-phenobase-data.parquet --image-dir data/inat/images --max-workers 14")
