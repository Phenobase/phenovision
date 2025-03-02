library(torch)
library(torchvision)
library(tidyverse)
library(zeallot)
library(hfhub)

ACT2FN <- list(
  gelu = nn_gelu,
  gelu_10 = list("ClippedGELUActivation", c(min = -10, max = 10)),
  gelu_fast = "FastGELUActivation",
  gelu_new = "NewGELUActivation",
  gelu_python = list("GELUActivation", c(use_gelu_python = TRUE)),
  gelu_pytorch_tanh = "PytorchGELUTanh",
  gelu_accurate = "AccurateGELUActivation",
  laplace = "LaplaceActivation",
  leaky_relu = nn_leaky_relu,
  linear = "LinearActivation",
  mish = "MishActivation",
  quick_gelu = "QuickGELUActivation",
  relu = nn_relu,
  relu2 = "ReLUSquaredActivation",
  relu6 = nn_relu6,
  sigmoid = nn_sigmoid,
  silu = nn_silu,
  swish = nn_silu,
  tanh = nn_tanh,
  prelu = nn_prelu
)

train_transforms <- function(img) {
  img %>%
    # first convert image to tensor
    transform_to_tensor() %>%
    #(function(x) x$to(device = "cuda")) %>%
    # data augmentation
    transform_resize(size = c(224, 224)) %>%
    # data augmentation
    transform_color_jitter() %>%
    # data augmentation
    transform_random_horizontal_flip() %>%
    # normalize according to what is expected by resnet
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)) #%>%
  #(function(x) x$to(device = "cpu"))
}

test_images <- cifar10_dataset("test", transform = train_transforms)
test_images_load <- dataloader(test_images)
test_images_load$.iter()$.next()

vit_mod <- read_rds("output/model_04_13_2024/R/vit_finetuned_epoch4.rds")

names(vit_mod$weights)

#' @description
#' This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
#' `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
#' Transformer.
vit_patch_embeddings <- nn_module("ViTPatchEmbeddings",
                                  initialize = function(config) {
                                    image_size <- config$image_size
                                    patch_size <- config$patch_size
                                    num_channels <- config$num_channels
                                    hidden_size <- config$hidden_size
                                    image_size <- c(image_size, image_size)
                                    patch_size <- c(patch_size, patch_size)
                                    num_patches <- floor(image_size[2] / patch_size[2]) * floor(image_size[1] / patch_size[1])
                                    self$image_size <- image_size
                                    self$patch_size <- patch_size
                                    self$num_channels <- num_channels
                                    self$num_patches <- num_patches
                                    self$projection = nn_conv2d(num_channels, hidden_size, kernel_size = patch_size, stride = patch_size)
                                  },
                                  forward = function(pixel_values, interpolate_pos_encoding = FALSE) {
                                    c(batch_size, num_channels, height, width) %<-% pixel_values$shape
                                    if(num_channels != self$num_channels) {
                                      rlang::abort("Make sure that the channel dimension of the pixel values match with the one set in the configuration.
                                                   Expected {self.num_channels} but got {num_channels}.",
                                                   use_cli_format = TRUE)
                                    }
                                    if(!interpolate_pos_encoding) {
                                      if(height != self$image_size[1] | width != self$image_size[2]) {
                                        rlang::abort("Input image size ({height}*{width}) doesn't match model",
                                                     "({self$image_size[1]}*{self$image_size[2]}).",
                                                     use_cli_format = TRUE)
                                      }
                                    }

                                    embeddings <- self$projection(pixel_values)$flatten(start_dim = 3L)$transpose(2, 3)
                                    return(embeddings)
                                  })


#' @description
#' Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
vit_embeddings <- nn_module("ViTEmbeddings",
                            initialize = function(config, use_mask_token = FALSE) {
                              self$cls_token <- nn_parameter(torch_randn(1, 1, config$hidden_size))
                              if(use_mask_token) {
                                self$mask_token <- nn_parameter(torch_randn(1, 1, config$hidden_size))
                              } else {
                                self$mask_token <- NULL
                              }
                              self$patch_embeddings <- vit_patch_embeddings(config)
                              num_patches <- self$patch_embeddings$num_patches
                              self$position_embeddings <- nn_parameter(torch_randn(1, num_patches + 1, config$hidden_size))
                              self$dropout <- nn_dropout(config$hidden_dropout_prob)
                              self$patch_size <- config$patch_size
                              self$config <- config

                            },
                            #' @description
                            #' This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
                            #' images. This method is also adapted to support torch.jit tracing.
                            #' Adapted from:
                            #'               - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
                            #'               - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
                            interpolate_pos_encoding = function(embeddings, height, width) {
                              num_patches <- embeddings$shape[2] - 1L
                              num_positions <- self$position_embeddings$shape[2] - 1
                              if(num_patches == num_positions & height == width) {
                                return(self$position_embeddings)
                              }
                              class_pos_embed <- self$position_embeddings[ , 1, drop = FALSE]
                              patch_pos_embed <- self$position_embeddings[ , 2:N, drop = FALSE]
                              dim <- embeddings$shape[length(embeddings$shape)]
                              new_height <- floor(height / self$patch_size)
                              new_width <- floor(width / self$patch_size)
                              sqrt_num_positions <- torch_int(num_positions**0.5)
                              patch_pos_embed <- patch_pos_embed$reshape(c(1, sqrt_num_positions, sqrt_num_positions, dim)) # Python: patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
                              patch_pos_embed <- patch_pos_embed$permute(c(1, 4, 2, 3)) #Python: patch_pos_embed.permute(0, 3, 1, 2)
                              patch_pos_embed <- nnf_interpolate(patch_pos_embed,
                                                                 size = c(new_height, new_width),
                                                                 mode = 'bicubic',
                                                                 align_corners = FALSE)
                              patch_pos_embed <- patch_pos_embed$permute(c(1, 3, 4, 2))$view(c(1, -1, dim)) #Python: patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
                              return(torch_cat(list(class_pos_embed, patch_pos_embed), dim = 2))
                            },
                            forward = function(pixel_values, bool_masked_pos = NULL, interpolate_pos_encoding = FALSE) {
                              #browser()
                              c(batch_size, num_channels, height, width) %<-% pixel_values$shape
                              embeddings <- self$patch_embeddings(pixel_values, interpolate_pos_encoding = interpolate_pos_encoding)
                              if(!is.null(bool_masked_pos)) {
                                seq_length <- embeddings$shape[2]
                                seq_length <- self$mask_token$expand(c(batch_size, seq_length, -1))
                                mask <- bool_masked_pos$unsqueeze(-1)$type_as(mask_tokens)
                                embeddings <- embeddings * (1.0 - mask) + mask_tokens * mask

                              }

                              # add the [CLS] token to the embedded patch tokens
                              cls_tokens <- self$cls_token$expand(c(batch_size, -1, -1))
                              embeddings <- torch_cat(list(cls_tokens, embeddings), dim = 2)
                              # add positional encoding to each token
                              if(interpolate_pos_encoding) {
                                embeddings <- embeddings + self$interpolate_pos_encoding(embeddings, height, width)
                              } else {
                                embeddings <- embeddings + self$position_embeddings
                              }

                              embeddings <- self$dropout(embeddings)

                              return(embeddings)

                            })


test_embed <- vit_embeddings(vit_mod$config)
im <- test_images_load$.iter()$.next()
test_embed(im$x)


vit_self_attention <- nn_module("ViTSelfAttention",
                                     initialize = function(config) {

                                       if((config$hidden_size %% config$num_attention_heads != 0) & !hasName(config, "embedding_size")) {
                                         rlang::abort(
                                           "The hidden size {config$hidden_size} is not a multiple of the number of attention ",
                                           "heads {config$num_attention_heads}."
                                         )
                                       }
                                       self$num_attention_heads <- config$num_attention_heads
                                       self$attention_head_size <- as.integer(config$hidden_size / config$num_attention_heads)
                                       self$all_head_size <- self$num_attention_heads * self$attention_head_size

                                       self$query <- nn_linear(config$hidden_size, self$all_head_size, bias = config$qkv_bias)
                                       self$key <- nn_linear(config$hidden_size, self$all_head_size, bias = config$qkv_bias)
                                       self$value <- nn_linear(config$hidden_size, self$all_head_size, bias = config$qkv_bias)

                                       self$dropout = nn_dropout(config$attention_probs_dropout_prob)


                                     },
                                     transpose_for_scores = function(x) {
                                       new_x_shape <- c(x$size()[1:(length(x$size())-1)], self$num_attention_heads, self$attention_head_size)
                                       x <- x$view(new_x_shape)
                                       return(x$permute(c(1, 3, 2, 4)))
                                     },
                                     forward = function(hidden_states, head_mask = NULL, output_attentions = FALSE) {
                                       browser()
                                       mixed_query_layer <- self$query(hidden_states)
                                       key_layer <- self$transpose_for_scores(self$key(hidden_states))
                                       value_layer <- self$transpose_for_scores(self$value(hidden_states))
                                       query_layer <- self$transpose_for_scores(mixed_query_layer)
                                       # Take the dot product between "query" and "key" to get the raw attention scores.
                                       attention_scores <- torch_matmul(query_layer, key_layer$transpose(-1, -2))

                                       attention_scores <- attention_scores / sqrt(self$attention_head_size)

                                       # Normalize the attention scores to probabilities.
                                       attention_probs <- nnf_softmax(attention_scores, dim = -1)

                                       # This is actually dropping out entire tokens to attend to, which might
                                       # seem a bit unusual, but is taken from the original Transformer paper.
                                       attention_probs <- self$dropout(attention_probs)

                                       # Mask heads if we want to
                                       if(!is.null(head_mask)) {
                                         attention_probs <- attention_probs * head_mask
                                       }

                                       context_layer <- torch_matmul(attention_probs, value_layer)

                                       context_layer <- context_layer$permute(c(1, 3, 2, 4))$contiguous()
                                       new_context_layer_shape <- c(context_layer$size()[1:(length(context_layer$size())-2)], self$all_head_size)
                                       context_layer <- context_layer$view(new_context_layer_shape)

                                       if(output_attentions) {
                                         outputs <- list(context_layer, attention_probs)
                                       } else {
                                         outputs <- list(context_layer)
                                       }

                                       return(outputs)

                                     })

vit_sdpa_self_attention <- nn_module("ViTSdpaSelfAttention",
                                     inherit = vit_self_attention,
                                     initialize = function(config) {
                                       super$initialize(config)
                                       self$attention_probs_dropout_prob <- config$attention_probs_dropout_prob
                                     },
                                     forward = function(hidden_states, head_mask = NULL, output_attentions = FALSE) {
                                       #browser()
                                       if(output_attentions | !is.null(head_mask)) {
                                         rlang::warn("`ViTSdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support ",
                                                     "`output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but ",
                                                     "specifying the manual implementation will be required from Transformers version v5.0.0 onwards. ",
                                                     'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.')
                                         return(super$forward(hidden_states = hidden_states, head_mask = head_mask, output_attentions = output_attentions))
                                       }

                                       mixed_query_layer <- self$query(hidden_states)

                                       key_layer <- self$transpose_for_scores(self$key(hidden_states))
                                       value_layer <- self$transpose_for_scores(self$value(hidden_states))
                                       query_layer <- self$transpose_for_scores(mixed_query_layer)

                                       context_layer <- call_torch_function("torch__scaled_dot_product_attention",
                                                                            query_layer,
                                                                            key_layer,
                                                                            value_layer,
                                                                            head_mask,
                                                                            if(self$training) 0.0 else self$attention_probs_dropout_prob,
                                                                            is_causal = FALSE)

                                       context_layer <- context_layer[[1]]$permute(c(1, 3, 2, 4))$contiguous()
                                       new_context_layer_shape <- c(context_layer$size()[1:(length(context_layer$size())-2)], self$all_head_size)
                                       context_layer <- context_layer$view(new_context_layer_shape)

                                       return(list(context_layer, NULL))

                                     })

test_att <- vit_sdpa_self_attention(vit_mod$config)

#' @description
#' The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
#' layernorm applied before each block.
vit_self_output <- nn_module("ViTSelfOutput",
                             initialize = function(config) {
                               self$dense <- nn_linear(config$hidden_size, config$hidden_size)
                               self$dropout <- nn_dropout(config$hidden_dropout_prob)
                             },
                             forward = function(hidden_states, input_tensor) {
                               hidden_states <- self$dense(hidden_states)
                               hidden_states <- self$dropout(hidden_states)
                               return(hidden_states)
                             })

#' Finds the heads and their indices taking already pruned heads into account
#'
#' @param heads List of integers. The indices of heads to prune.
#' @param n_heads Integer. The number of heads in the model.
#' @param head_size Integer. The size of each head.
#' @param already_pruned_heads Set of integers. A set of already pruned heads.
#'
#' @return A list containing:
#'   \item{heads_to_prune}{Set of integers. The indices of heads to prune taking already_pruned_heads into account}
#'   \item{index}{torch_tensor. The indices of rows/columns to keep in the layer weight}
find_pruneable_heads_and_indices <- function(heads, n_heads, head_size, already_pruned_heads) {
  mask <- torch_ones(n_heads, head_size)
  heads <- setdiff(unique(heads), already_pruned_heads)  # Convert to set and remove already pruned heads
  for(head in heads) {
    # Compute how many pruned heads are before the head and move the index accordingly
    head <- head - sum(already_pruned_heads < head)
    mask[head, ] <- 0
  }
  mask <- mask$view(-1)$contiguous()$eq(1)
  index <- torch_arange(length(mask))[mask]$to(torch_long())

  return(list(heads, index))

}

prune_linear_layer <- function(layer, index, dim = 1) {
  # Move index to same device as layer
  index <- index$to(device = layer$weight$device)

  # Select weights using index
  W <- layer$weight$index_select(dim, index)$clone()$detach()

  # Handle bias if it exists
  if (!is.null(layer$bias)) {
    if (dim == 2) {
      b <- layer$bias$clone()$detach()
    } else {
      b <- layer$bias[index]$clone()$detach()
    }
  }

  # Get new size
  new_size <- layer$weight$size()
  new_size[dim] <- index$size(1)

  # Create new layer
  new_layer <- nn_linear(new_size[2], new_size[1], bias = !is.null(layer$bias))

  # Move to same device
  new_layer <- new_layer$to(device = layer$weight$device)

  # Copy weights
  new_layer$weight$requires_grad_(FALSE)
  new_layer$weight$copy_(W$contiguous())
  new_layer$weight$requires_grad_(TRUE)

  # Copy bias if it exists
  if (!is.null(layer$bias)) {
    new_layer$bias$requires_grad_(FALSE)
    new_layer$bias$copy_(b$contiguous())
    new_layer$bias$requires_grad_(TRUE)
  }

  new_layer
}

vit_attention <- nn_module("ViTAttention",
                           initialize = function(config) {
                             self$attention <- vit_self_attention(config)
                             self$output = vit_self_output(config)
                             self$pruned_heads = integer()
                           },
                           prune_heads = function(heads) {
                             if(length(head) == 0) {
                               return(invisible(NULL))
                             }
                             c(heads, index) %<-% find_pruneable_heads_and_indices(
                               heads, self$attention.num_attention_heads,
                               self$attention$attention_head_size, self$pruned_heads
                             )
                             # Prune linear layers
                             self$attention$query <- prune_linear_layer(self$attention$query, index)
                             self$attention$key <- prune_linear_layer(self$attention$key, index)
                             self$attention$value <- prune_linear_layer(self$attention$value, index)
                             self$output$dense <- prune_linear_layer(self$output$dense, index, dim = 2)
                             # Update hyper params and store pruned heads
                             self$attention$num_attention_heads <- self$attention$num_attention_heads - length(heads)
                             self$attention$all_head_size <- self$attention$attention_head_size * self$attention$num_attention_heads
                             self$pruned_heads = union(self$pruned_heads, heads)
                           },
                           forward = function(hidden_states, head_mask = NULL, output_attentions = FALSE) {
                             self_outputs <- self$attention(hidden_states, head_mask, output_attentions)
                             attention_output <- self$output(self_outputs[[1]], hidden_states)

                             outputs <- c(list(attention_output), self_outputs[-1]) # add attentions if we output them
                             return(outputs)

                           })

test_att <- vit_attention(vit_mod$config)

vit_sdpa_attention <- nn_module("ViTSdpaAttention",
                                inherit = vit_attention,
                                initialize = function(config) {
                                  super$initialize(config)
                                  self$attention <- vit_sdpa_self_attention(config)
                                })

vit_intermediate <- nn_module("ViTIntermediate",
                              initialize = function(config) {
                                self$dense = nn_linear(config$hidden_size, config$intermediate_size)
                                if(is.character(config$hidden_act)) {
                                  self$intermediate_act_fn <- ACT2FN[[config$hidden_act]]()
                                } else {
                                  self$intermediate_act_fn <- config$hidden_act()
                                }
                              },
                              forward = function(hidden_states) {
                                hidden_states <- self$dense(hidden_states)
                                hidden_states <- self$intermediate_act_fn(hidden_states)

                                return(hidden_states)
                              })

vit_output <- nn_module("ViTOutput",
                        initialize = function(config) {
                          self$dense <- nn_linear(config$intermediate_size, config$hidden_size)
                          self$dropout <- nn_dropout(config$hidden_dropout_prob)
                        },
                        forward = function(hidden_states, input_tensor) {
                          hidden_states <- self$dense(hidden_states)
                          hidden_states <- self$dropout(hidden_states)

                          hidden_states <- hidden_states + input_tensor

                          return(hidden_states)

                        })


VIT_ATTENTION_CLASSES <- list(
  eager = vit_attention,
  sdpa = vit_sdpa_attention
)

#' @description
#' This corresponds to the Block class in the timm implementation."
vit_layer <- nn_module("ViTLayer",
                       initialize = function(config) {
                         self$chunk_size_feed_forward <- config$chunk_size_feed_forward
                         self$seq_len_dim <- 1
                         if(hasName(config, "_attn_implementation")) {
                          self$attention <- VIT_ATTENTION_CLASSES[config$`_attn_implementation`](config)
                         } else {
                           self$attention <- vit_sdpa_attention(config)
                         }
                         self$intermediate <- vit_intermediate(config)
                         self$output <- vit_output(config)
                         self$layernorm_before <- nn_layer_norm(config$hidden_size, eps = config$layer_norm_eps)
                         self$layernorm_after <- nn_layer_norm(config$hidden_size, eps = config$layer_norm_eps)
                       },
                       forward = function(hidden_states, head_mask = NULL, output_attentions = FALSE) {
                         self_attention_outputs <- self$attention(
                           self$layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
                           head_mask,
                           output_attentions = output_attentions
                         )
                         attention_output <- self_attention_outputs[[1]]
                         outputs <- self_attention_outputs[-1]  # add self attentions if we output attention weights

                         # first residual connection
                         hidden_states = attention_output + hidden_states

                         # in ViT, layernorm is also applied after self-attention
                         layer_output <- self$layernorm_after(hidden_states)
                         layer_output <- self$intermediate(layer_output)

                         # second residual connection is done here
                         layer_output <- self$output(layer_output, hidden_states)

                         outputs = list(layer_output, outputs)

                         return(outputs)

                       })

test_vit <- vit_layer(vit_mod$config)

vit_encoder <- nn_module("ViTEncoder",
                         initialize = function(config) {
                           self$config <- config
                           self$layer <- nn_module_list(map(seq_len(config$num_hidden_layer),
                                                            ~ vit_layer(config)))
                           self$gradient_checkpointing <- FALSE
                         },
                         forward = function(hidden_states, head_mask = NULL, output_attentions = FALSE,
                                            output_hidden_states = FALSE) {
                           all_hidden_states <- if(output_hidden_states) list() else NULL
                           all_self_attentions <- if(output_attentions) list() else NULL
                           for(i in length(self$layer)) {
                             if(output_hidden_states) {
                               all_hidden_states <- c(all_hidden_states, list(hidden_states))
                             }

                             layer_head_mask <- if(!is.null(head_mask)) head_mask[[i]] else NULL

                             if(self$gradient_checkpointing & self$training) {
                                 layer_outputs <- self$`._gradient_checkpointing_func`(
                                   self$layer[[i]]$`__call__`,
                                   hidden_states,
                                   layer_head_mask,
                                   output_attentions
                                 )
                             } else {
                               layer_outputs <- self$layer[[i]](hidden_states, layer_head_mask, output_attentions)
                             }
                             hidden_states <- layer_outputs[[1]]

                             if(output_attentions) {
                               all_self_attentions <- c(all_self_attentions, list(layer_outputs[[2]]))
                             }
                           }

                           if(output_hidden_states) {
                             all_hidden_states <- c(all_hidden_states, list(hidden_states))
                           }

                           return(c(list(hidden_states), all_hidden_states, all_self_attentions))


                         })

vit_pretrained_model <- nn_module("ViTPretrainedModel",
                              initialize = function(config) {
                                self$config <- config
                              },
                              .init_weights = function(module) {
                                with_no_grad({
                                if(inherits(module, c("nn_linear", "nn_conv2d"))) {
                                  module$weight <- nn_init_trunc_normal_(module$weight$to(torch_float32()),
                                                                         mean = 0.0, std = self$config$initializer_range)$to(module$weight$dtype)
                                  if(!is.null(module$bias)) {
                                    module$bias$zero_()
                                  }
                                }
                                if(inherits(module, "nn_layer_norm")) {
                                  module$bias$zero_()
                                  module$weight$fill_(1.0)
                                }
                                if(inherits(module, "vit_embeddings")) {
                                  module$position_embeddings <- nn_init_trunc_normal_(
                                    module$position_embeddings$to(torch_float32),
                                    mean = 0.0,
                                    std = self$config$initializer_range,
                                  )$to(module$position_embeddings$dtype)

                                  module$cls_token <- nn_init_trunc_normal_(
                                    module$cls_token$to(torch_float32),
                                    mean = 0.0,
                                    std = self$config$initializer_range,
                                  )$to(module$cls_token$dtype)
                                }
                                })

                              },
                              #' @description
                              #' If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
                              #' initialization logic in `.init_weights
                              init_weights = function() {
                                # Prune heads if needed
                                if(length(self$config$pruned_heads) > 0) {
                                  self$prune_heads(self$prune_heads(self$config$pruned_heads))
                                }
                                # Initialize weights
                                self$apply(self$.init_weights)
                              },
                              #' @description
                              #' Prepare the head mask if needed.
                              #' Args:
                              #'   head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                              #'   The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
                              #' num_hidden_layers (`int`):
                              #'   The number of hidden layers in the model.
                              #' is_attention_chunked (`bool`, *optional*, defaults to `False`):
                              #'   Whether or not the attentions scores are computed by chunks or not.
                              #'
                              #' Returns:
                              #'   `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
                              #' `[None]` for each layer.
                              get_head_mask = function(head_mask, num_hidden_layers, is_attention_chunked = FALSE) {
                                if(!is.null(head_mask)) {
                                  head_mask <- self$.convert_head_mask_to_5d(head_mask, num_hidden_layers)
                                  if(is_attention_chunked) {
                                    head_mask <- head_mask$unsqueeze(-1)
                                  }
                                } else {
                                  head_mask <- rep(list(NULL), num_hidden_layers)
                                }
                                return(head_mask)

                              },
                              .convert_head_mask_to_5d = function(self, head_mask, num_hidden_layers) {
                                ##-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]
                                if(head_mask$dim() == 1) {
                                  head_mask <- head_mask$unsqueeze(1)$unsqueeze(1)$unsqueeze(-1)$unsqueeze(-1)
                                  head_mask <- head_mask$expand(c(num_hidden_layers, -1, -1, -1, -1))
                                }
                                if(head_mask.dim() == 2) {
                                  head_mask <- head_mask$unsqueeze(2)$unsqueeze(-1)$unsqueeze(-1)  # We can specify head_mask for each layer
                                }
                                if(head_mask$dim() != 5) rlang::abort("head_mask$dim != 5, instead {head_mask$dim()}")
                                head_mask <- head_mask$to(dtype = self$dtype)  # switch to float if need + fp16 compatibility
                                return(head_mask)
                              })

vit_pooler <- nn_module("ViTPooler",
                        initialize = function(config) {
                          self$dense <- nn_linear(config$hidden_size, config$hidden_size)
                          self$activation <- nn_tanh()
                        },
                        forward = function(hidden_states) {
                          # We "pool" the model by simply taking the hidden state corresponding
                          # to the first token.
                          first_token_tensor <- hidden_states[ , 1]
                          pooled_output <- self$dense(first_token_tensor)
                          pooled_output <- self$activation(pooled_output)
                          return(pooled_output)
                        })

vit_model <- nn_module("ViTModel",
                       inherit = vit_pretrained_model,
                       initialize = function(config, add_pooling_layer = TRUE,
                                             use_mask_token = FALSE) {
                         self$config <- config

                         self$embeddings <- vit_embeddings(config, use_mask_token = use_mask_token)
                         self$encoder <- vit_encoder(config)
                         self$layernorm <- nn_layer_norm(config$hidden_size, eps = config$layer_norm_eps)
                         self$pooler <- if(add_pooling_layer) vit_pooler(config) else NULL
                         self$init_weights()
                       },
                       get_input_embeddings = function() {
                         return(self$embeddings$patch_embeddings)
                       },
                       prune_heads = function(heads_to_prune = list()) {
                         for(i in heads_to_prune) {
                           for(j in heads_to_prune[[i]]) {
                            self$encoder$layer[[i]]$attention$prune_heads(j)
                           }
                         }
                       },
                       forward = function(pixel_values, bool_masked_pos = NULL, head_mask = NULL,
                                          output_attentions = FALSE, output_hidden_states = FALSE,
                                          interpolate_pos_encoding = FALSE) {
                         output_attentions <- if(!is.null(output_attentions)) output_attentions else self$config$output_attentions
                         output_hidden_states <- if(!is.null(output_hidden_states)) output_hidden_states else config$output_hidden_states

                         # Prepare head mask if needed
                         # 1.0 in head_mask indicate we keep the head
                         # attention_probs has shape bsz x n_heads x N x N
                         # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
                         # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
                         head_mask <- self$get_head_mask(head_mask, self$config$num_hidden_layers)

                         expected_dtype <- self$embeddings$patch_embeddings$projection$weight$dtype

                         if(pixel_values$dtype != expected_dtype) {
                           pixel_values <- pixel_values$to(expected_dtype)
                         }
                         embedding_output <- self$embeddings(
                           pixel_values, bool_masked_pos = bool_masked_pos, interpolate_pos_encoding = interpolate_pos_encoding
                         )
                         encoder_outputs <- self$encoder(
                           embedding_output,
                           head_mask = head_mask,
                           output_attentions = output_attentions,
                           output_hidden_states = output_hidden_states
                         )

                         sequence_output <- encoder_outputs[[1]]
                         sequence_output <- self$layernorm(sequence_output)
                         pooled_output <- if(!is.null(self$pooler)) self$pooler(sequence_output) else NULL

                         return(list(sequence_output, pooled_output, encoder_outputs[-1]))

                       })

vit_for_image_classification <- nn_module("ViTForImageClassification",
                                          inherit = vit_pretrained_model,
                                          initialize = function() {

                                          },
                                          forward = function() {

                                          })


class ViTForImageClassification(ViTPreTrainedModel):
  def __init__(self, config: ViTConfig) -> None:
  super().__init__(config)

self.num_labels = config.num_labels
self.vit = ViTModel(config, add_pooling_layer=False)

# Classifier head
self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

# Initialize weights and apply final processing
self.post_init()

@add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
@add_code_sample_docstrings(
  checkpoint=_IMAGE_CLASS_CHECKPOINT,
  output_type=ImageClassifierOutput,
  config_class=_CONFIG_FOR_DOC,
  expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
)
def forward(
  self,
  pixel_values: Optional[torch.Tensor] = None,
  head_mask: Optional[torch.Tensor] = None,
  labels: Optional[torch.Tensor] = None,
  output_attentions: Optional[bool] = None,
  output_hidden_states: Optional[bool] = None,
  interpolate_pos_encoding: Optional[bool] = None,
  return_dict: Optional[bool] = None,
) -> Union[tuple, ImageClassifierOutput]:
  r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
return_dict = return_dict if return_dict is not None else self.config.use_return_dict

outputs = self.vit(
  pixel_values,
  head_mask=head_mask,
  output_attentions=output_attentions,
  output_hidden_states=output_hidden_states,
  interpolate_pos_encoding=interpolate_pos_encoding,
  return_dict=return_dict,
)

sequence_output = outputs[0]

logits = self.classifier(sequence_output[:, 0, :])

loss = None
if labels is not None:
  # move labels to correct device to enable model parallelism
  labels = labels.to(logits.device)
if self.config.problem_type is None:
  if self.num_labels == 1:
  self.config.problem_type = "regression"
elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
  self.config.problem_type = "single_label_classification"
else:
  self.config.problem_type = "multi_label_classification"

if self.config.problem_type == "regression":
  loss_fct = MSELoss()
if self.num_labels == 1:
  loss = loss_fct(logits.squeeze(), labels.squeeze())
else:
  loss = loss_fct(logits, labels)
elif self.config.problem_type == "single_label_classification":
  loss_fct = CrossEntropyLoss()
loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
elif self.config.problem_type == "multi_label_classification":
  loss_fct = BCEWithLogitsLoss()
loss = loss_fct(logits, labels)

if not return_dict:
  output = (logits,) + outputs[1:]
return ((loss,) + output) if loss is not None else output

return ImageClassifierOutput(
  loss=loss,
  logits=logits,
  hidden_states=outputs.hidden_states,
  attentions=outputs.attentions,
)


vit_mod2 <- vit_model(vit_mod$config)

im <- test_images_load$.iter()$.next()
tt <- vit_mod2(im[[1]])



phenovision <- hfhub::hub_download("phenobase/phenovision", "model.safetensors")
mod <- torch_load(phenovision)
