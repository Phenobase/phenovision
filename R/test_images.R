library(torch)
library(torchvision)
library(imager)

image_files <- list.files("data/inat/images", full.names = TRUE)

res <- list()
j <- 0

image_files <- image_files[which(image_files == "data/inat/images/2759396_medium.PNG"):length(image_files)]

for(i in image_files) {
  im <- try(base_loader(i))
  if(inherits(im, "try-error")) {
    print("error!")
    print(i)
    next
  }
  if(dim(im)[3] != 3) {
    im <- load.image(i)
    im <- flatten.alpha(im)
    ext <- fs::path_ext(i)
    if(toupper(ext) == ext) {
      new_i <- fs::path_ext_set(i, tolower(ext))
      save.image(im, new_i)
      fs::file_move(new_i, i)
    } else {
      save.image(im, i)
    }
    message("resaved image ", i)
  }
  j <- j + 1
  res[[j]] <- dim(im)
  print(j)
}

## looks like base_loader doesn't deal with pngs well
