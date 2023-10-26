library(torch)
library(torchvision)

## load model
model <- model_resnet18(pretrained = TRUE)
## freeze weights
model$parameters |> purrr::walk(function(param) param$requires_grad_(FALSE))
## get test image
logo_file <- tempfile(fileext = ".png")
download.file("https://www.r-project.org/logo/Rlogo.png",  logo_file, mode = "wb")
im <- base_loader(logo_file)[ , , 1:3] |>
  transform_to_tensor() |>
  transform_center_crop(c(224, 224))

## this works
model(im$unsqueeze(1))

## this crashes
model$cuda()(im$unsqueeze(1)$cuda())


library(torch)
library(safetensors)
x <- torch_rand(1000, 100)

fn <- tempfile()
safe_save_file(x, fn)
y <- safe_load_file(fn)

ser <- safe_serialize(x)

########## bechmarking image loading #########
logo_file <- tempfile(fileext = ".png")
download.file("https://www.r-project.org/logo/Rlogo.png",  logo_file, mode = "wb")
tests <- bench::mark(base_loader(logo_file),
                     magick_loader(logo_file),
                     imager::load.image(logo_file),
                     check = FALSE)

test1 <- base_loader(logo_file)
test2 <- magick_loader(logo_file)


library(torch)
options(torch.serialization_version = 2)

x <- torch_rand(1000, 50)
x$requires_grad <- TRUE

torch_save(x, "test1.pt")

lobstr::obj_size(x)

for(i in 1:10000) {
samp <- torch_tensor(sample.int(1000, 10))

loss <- torch_sum(x[samp, ]^2)
loss$backward()
}

lobstr::obj_size(loss)

torch_save(x, "test2.pt")


x <- torch_rand(1000, 50)
x$requires_grad <- TRUE
loss <- torch_sum(x^2)
loss$backward()


