#wget -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg
#wget -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg
#wget -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg
#wget -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg
#wget -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg
#wget -P /tmp/nst/ https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg

library(keras)
use_implementation("tensorflow")
library(tensorflow)
#tfe_enable_eager_execution(device_policy = "warn")
tfe_enable_eager_execution(device_policy = "silent")

library(tfdatasets)
library(purrr)
library(glue)

tf$set_random_seed(7777)
img_shape <- c(128, 128, 3)
content_path <- "/tmp/nst/Green_Sea_Turtle_grazing_seagrass.jpg"
style_path <- "/tmp/nst/The_Great_Wave_off_Kanagawa.jpg"

content_image <-
  image_load(content_path, target_size = img_shape[1:2])
# content_image %>% image_to_array() %>%
#   `/`(., 255) %>%
#   as.raster() %>%  plot()

style_image <-
  image_load(style_path, target_size = img_shape[1:2])
# style_image %>% image_to_array() %>%
#   `/`(., 255) %>%
#   as.raster() %>%  plot()


load_and_process_image <- function(path) {
  img <- image_load(path, target_size = img_shape[1:2]) %>%
    image_to_array() %>%
    k_expand_dims(axis = 1) %>%
    imagenet_preprocess_input()
}

deprocess_image <- function(x) {
  x <- x[1, , , ]
  # Remove zero-center by mean pixel
  x[, , 1] <- x[, , 1] + 103.939
  x[, , 2] <- x[, , 2] + 116.779
  x[, , 3] <- x[, , 3] + 123.68
  # 'BGR'->'RGB'
  x <- x[, , c(3, 2, 1)]
  x[x > 255] <- 255
  x[x < 0] <- 0
  x[] <- as.integer(x) / 255
  x
}

content_layers <- c("block5_conv2")
style_layers = c("block1_conv1",
                 "block2_conv1",
                 "block3_conv1",
                 "block4_conv1",
                 "block5_conv1")
num_content_layers <- length(content_layers)
num_style_layers <- length(style_layers)

get_model <- function() {
  vgg <- application_vgg19(include_top = FALSE, weights = "imagenet")
  vgg$trainable <- FALSE
  style_outputs <-
    purrr::map(style_layers, function(layer)
      vgg$get_layer(layer)$output)
  content_outputs <-
    purrr::map(content_layers, function(layer)
      vgg$get_layer(layer)$output)
  model_outputs <- c(style_outputs, content_outputs)
  keras_model(vgg$input, model_outputs)
}

content_loss <- function(content_image, target) {
  k_sum(k_square(target - content_image))
}

gram_matrix <- function(x) {
  features <- k_batch_flatten(k_permute_dimensions(x, c(3, 1, 2)))
  gram <- k_dot(features, k_transpose(features))
  gram
}

style_loss <- function(gram_target, combination) {
  gram_comb <- gram_matrix(combination)
  k_sum(k_square(gram_target - gram_comb)) / (4 * (img_shape[3]^2) * (img_shape[1] * img_shape[2])^2)
}

total_variation_loss <- function(x) {
  y_ij  <- x[, 1:(img_nrows - 1L), 1:(img_ncols - 1L),]
  y_i1j <- x[, 2:(img_nrows), 1:(img_ncols - 1L),]
  y_ij1 <- x[, 1:(img_nrows - 1L), 2:(img_ncols),]
  a <- k_square(y_ij - y_i1j)
  b <- k_square(y_ij - y_ij1)
  k_sum(k_pow(a + b, 1.25))
}

get_feature_representations <-
  function(model, content_path, style_path) {
    style_image <- load_and_process_image(style_path) %>%
      k_cast_to_floatx()
    content_image <- load_and_process_image(content_path) %>%
      k_cast_to_floatx()
    stack_images <-
      k_concatenate(list(style_image, content_image), axis = 1)
    
    model_outputs <- model(stack_images)
    style_features <- model_outputs[1:num_style_layers] %>%
      map(function(batch)
        batch[1, , ,])
    content_features <-
      model_outputs[(num_style_layers + 1):(num_style_layers + num_content_layers)] %>%
      map(function(batch)
        batch[2, , ,])
    list(style_features, content_features)
  }

compute_loss <-
  function(model,
           loss_weights,
           init_image,
           gram_style_features,
           content_features) {
    c(style_weight, content_weight) %<-% loss_weights
    model_outputs <- model(init_image)
    # dim(style_output_features[[1]]) == 1 512 512  64
    style_output_features <- model_outputs[1:num_style_layers]
    # dim(content_output_features[[1]]) == 1  32  32 512
    content_output_features <-
      model_outputs[(num_style_layers + 1):(num_style_layers + num_content_layers)]
    
    weight_per_style_layer <- 1 / num_style_layers
    style_score <- 0
    style_zip <-
      transpose(list(gram_style_features, style_output_features))
    for (l in 1:length(style_zip)) {
      # dim(target_style) == 64 64
      # dim(comb_style) == 1 512 512  64
      c(target_style, comb_style) %<-% style_zip[[l]]
      style_score <-
        style_score + weight_per_style_layer * style_loss(target_style, comb_style[1, , ,])
    }
    
    weight_per_content_layer <- 1 / num_content_layers
    content_score <- 0
    content_zip <-
      transpose(list(content_features, content_output_features))
    for (l in 1:length(content_zip)) {
      # dim(comb_content) ==  1 32  32 512
      # dim(target_content) == 32  32 512
      c(target_content, comb_content) %<-% content_zip[[l]]
      content_score <-
        content_score + weight_per_content_layer * content_loss(comb_content[1, , ,], target_content)
    }
    
    style_score <- style_score * style_weight
    content_score <- content_score * content_weight
    
    loss <- style_score + content_score
    list(loss, style_score, content_score)
  }

compute_grads <-
  function(model,
           loss_weights,
           init_image,
           gram_style_features,
           content_features) {
    with(tf$GradientTape() %as% tape, {
      scores <-
        compute_loss(model,
                     loss_weights,
                     init_image,
                     gram_style_features,
                     content_features)
    })
    total_loss <- scores[[1]]
    list(tape$gradient(total_loss, init_image), scores)
  }

run_style_transfer <- function(content_path,
                               style_path,
                               num_iterations = 2000,
                               content_weight = 1e2,
                               style_weight = 1e-2) {
  model <- get_model()
  walk(model$layers, function(layer)
    layer$trainable = FALSE)
  
  # dim(style_features[[1]]) == 512 512  64
  # dim(content_features[[1]]) == 32  32 512
  c(style_features, content_features) %<-% get_feature_representations(model, content_path, style_path)
  # dim(gram_style_features[[1]]) == 64 64
  gram_style_features <-
    map(style_features, function(feature)
      gram_matrix(feature))
  
  init_image <- load_and_process_image(content_path)
  init_image <-
    tf$contrib$eager$Variable(init_image, dtype = "float32")
  
  optimizer <-
    tf$train$AdamOptimizer(learning_rate = 1,
                           beta1 = 0.99,
                           epsilon = 1e-1)
  
  c(best_loss, best_image) %<-% list(Inf, NULL)
  loss_weights <- list(style_weight, content_weight)
  
  start_time <- Sys.time()
  global_start <- Sys.time()
  
  norm_means <- c(103.939, 116.779, 123.68)
  min_vals <- -norm_means
  max_vals <- 255 - norm_means
  
  for (i in seq_len(num_iterations)) {
    c(grads, all_losses) %<-% compute_grads(model,
                                            loss_weights,
                                            init_image,
                                            gram_style_features,
                                            content_features)
    c(loss, style_score, content_score) %<-% all_losses
    # grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer$apply_gradients(list(tuple(grads, init_image)))
    clipped <- tf$clip_by_value(init_image, min_vals, max_vals)
    init_image$assign(clipped)
    
    end_time <- Sys.time()
    
    if (k_cast_to_floatx(loss) < best_loss) {
      best_loss <- k_cast_to_floatx(loss)
      best_image <- init_image
    }
    
    if (i %% 50 == 0) {
      glue("Iteration: {i}") %>% print()
      glue(
        "Total loss: {loss$numpy()}, style loss: {style_score$numpy()}, content loss: {content_score$numpy()}, time: {Sys.time() - start_time} seconds"
      ) %>% print()
      
      if (i %% 100 == 0) {
        png(paste0("style_epoch_", i, ".png"))
        plot_image <- best_image$numpy()
        plot_image <- deprocess_image(plot_image)
        plot(as.raster(plot_image), main = glue("Iteration {i}"))
        dev.off()
      }
    }
  }
  
  glue("Total time: {Sys.time() - global_start} seconds") %>% print()
  list(best_image, best_loss)
}

c(best_image, best_loss) %<-% run_style_transfer(content_path,
                                                 style_path)
