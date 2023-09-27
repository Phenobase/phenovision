library(torch)
library(torchvision)

image_files <- list.files("data/inat/images", full.names = TRUE)

for(i in image_files) {
  im <- base_loader(i)
  if(dim(im)[3] != 3) {
    print(i)
  }
}

## looks like base_loader doesn't deal with pngs well
