library(keras)
use_implementation("tensorflow")
library(tensorflow)
#tfe_enable_eager_execution(device_policy = "warn")
tfe_enable_eager_execution(device_policy = "silent")

np <- import("numpy")

library(reticulate)
library(tfdatasets)
library(purrr)
library(stringr)
library(glue)
library(rjson)

tf$set_random_seed(7777)

annotation_file <- "train2014/annotations/captions_train2014.json"
image_path <- "train2014/train2014"

annotations <- fromJSON(file = annotation_file)

annot_captions <- annotations[[4]]
num_captions <- length(annot_captions)

all_captions <- vector(mode = "list", length = num_captions)
all_img_names <- vector(mode = "list", length = num_captions)

for (i in seq_len(num_captions)) {
  caption <-
    paste0("<start> ", annot_captions[[i]][["caption"]], " <end>")
  image_id <- annot_captions[[i]][["image_id"]]
  full_coco_image_path <-
    sprintf("train2014/train2014/COCO_train2014_%012d.jpg", image_id)
  all_img_names[[i]] <- full_coco_image_path
  all_captions[[i]] <- caption
}

set.seed(7777)
#num_examples <- 30000
num_examples <- 10
random_sample <- sample(1:num_captions, size = num_examples)
sample_captions <- all_captions[random_sample]
sample_images <- all_img_names[random_sample]

load_image <- function(image_path) {
  img <- tf$read_file(image_path) %>%
    tf$image$decode_jpeg(channels = 3) %>%
    tf$image$resize_images(c(299L, 299L)) %>%
    tf$keras$applications$inception_v3$preprocess_input()
  list(img, image_path)
}

image_model <- application_inception_v3(include_top = FALSE,
                                        weights = "imagenet")
new_input <- image_model$input
hidden_layer <- image_model$get_layer("mixed10")$output

image_features_extract_model <- keras_model(new_input, hidden_layer)

preencode <- unique(sample_images) %>% unlist() %>% sort()

save_batch_size <- 4
image_dataset <- tensor_slices_dataset(preencode) %>%
  dataset_map(load_image) %>%
  dataset_batch(batch_size)

# save_iter <- make_iterator_one_shot(image_dataset)
# until_out_of_range({
#   batch <- save_iter$get_next()
#   img <- batch[[1]]
#   path <- batch[[2]]
#   #shape=(bs, 8, 8, 2048)
#   batch_features <- image_features_extract_model(img)
#   batch_features <- tf$reshape(batch_features,
#                                list(batch_features$shape[0],-1L, batch_features$shape[3]))
#   for (i in 1:batch_size) {
#     np$save(path[i]$numpy()$decode("utf-8"), batch_features[i, ,]$numpy())
#   }
# })

top_k <- 5000
tokenizer <- text_tokenizer(num_words = top_k,
                            oov_token = "<unk>",
                            filters = '!"#$%&()*+.,-/:;=?@[\\]^_`{|}~ ')
tokenizer$fit_on_texts(sample_captions)
sample_captions <- tokenizer %>% texts_to_sequences(sample_captions)
tokenizer$word_index

tokenizer$word_index["<unk>"]

tokenizer$word_index["<pad>"] <- length(tokenizer$word_index) + 1
tokenizer$word_index["<pad>"]

#py_run_string("index_word = {value:key for key, value in r.tokenizer.word_index.items()}")
#index2word <- py$index_word

word_index_df <- data.frame(
  word = tokenizer$word_index %>% names(),
  index = tokenizer$word_index %>% unlist(use.names = FALSE),
  stringsAsFactors = FALSE
)

decode_caption <- function(text) {
  paste(map(text, function(number)
    word_index_df %>%
      filter(idx == number) %>%
      select(word) %>%
      pull()),
    collapse = " ")
}

sample_captions <- pad_sequences(sample_captions, padding = "post")
max_length <- map(sample_captions, length) %>% unlist() %>% max()

length(sample_images)
nrow(sample_captions)

train_indices <-
  sample(length(sample_images), size = length(sample_images) * 0.8)

validation_indices <-
  setdiff(1:length(sample_images), train_indices)

img_train <- sample_images[train_indices] 
cap_train <- sample_captions[train_indices, ]

img_val <- sample_images[validation_indices]
cap_val <- sample_captions[validation_indices, ]

check_img_train <- img_train[1:3, ]
check_cap_train <- cap_train[1:3, ]

check_img_val <- img_val[1:2, ]
check_cap_val <- cap_val[1:2, ]



batch_size <- 2
buffer_size <- 1000
embedding_dim <- 64
gru_units <- 128
vocab_size <- length(tokenizer$word_index)
features_shape <- 2048
attention_features_shape <- 64


#######################################################################
#######################################################################

map_func <- function(img_name, cap) {
  # 64 2048
  print(img_name)
  # https://github.com/tensorflow/tensorflow/issues/14732
  img_tensor <- np$load(paste0(as.character(img_name) %>% str_sub(start = 3, end = -2), ".npy"))
  list(img_tensor, cap)
}

train_dataset <-
  tensor_slices_dataset(list(img_train, cap_train)) %>%
  # https://www.tensorflow.org/api_docs/python/tf/py_func
  dataset_map(function(item1, item2)
    tf$py_func(map_func, list(item1, item2), list(tf$float32, tf$int32)),
    num_parallel_calls = 4) %>%
  # this cannot work as we need to eagerly access the image path in map_func
  # numpy() does not work as we don't get an eager tensor from datasets API
  #dataset_map(map_func) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_batch(batch_size) %>%
  dataset_prefetch(1)


######  this works
i <- img_train[[1]]
c <- cap_train[[1]]
i
c
tf$py_func(map_func, list(tf$constant(i), tf$constant(c)), list(tf$float32, tf$int32))
###### 

######  this works
my_py_func <- function(x) {
  x <- tf$matmul(x, x)  # You can use tf ops
  print(x)  # but it's eager!
  x
}
x <- tf$constant(2, shape = shape(1, 1))
pf <- tf$py_func(my_py_func, list(x), tf$float32)
###### 

train_iter <- make_iterator_one_shot(train_dataset)
# Error: C stack usage  284672054196 is too close to the limit
batch <- iterator_get_next(train_iter)
img_tensor <- batch[[1]]
target_caption <- batch[[2]]
img_tensor
target_caption

#######################################################################
#######################################################################

cnn_encoder <-
  function(embedding_dim,
           name = NULL) {
    keras_model_custom(name = name, function(self) {
      # shape after fc == (batch_size, 64, embedding_dim)
      self$fc <-
        layer_dense(units = embedding_dim, activation = "relu")
      function(x, mask = NULL) {
        self$fc(x)
      }
    })
  }

attention_module <-
  function(gru_units,
           name = NULL) {
    keras_model_custom(name = name, function(self) {
      self$W1 = layer_dense(units = gru_units)
      self$W2 = layer_dense(units = gru_units)
      self$V = layer_dense(units = 1)
      
      function(inputs, mask = NULL) {
        features <- inputs[[1]]
        hidden <- inputs[[2]]
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, gru_units)
        # hidden_with_time_axis shape == (batch_size, 1, gru_units)
        hidden_with_time_axis <- k_expand_dims(hidden, axis = 2)
        # score shape == (batch_size, 64, 1)
        score <-
          self$V(k_tanh(self$W1(features) + self$W2(hidden_with_time_axis)))
        # attention_weights shape == (batch_size, 64, 1)
        attention_weights <- k_softmax(score, axis = 2)
        # context_vector shape after sum == (batch_size, embedding_dim)
        ### tbd check if this really is embedding_dim
        context_vector <-
          k_sum(attention_weights * features, axis = 2)
        list(context_vector, attention_weights)
      }
    })
  }

rnn_decoder <-
  function(embedding_dim,
           gru_units,
           vocab_size,
           name = NULL) {
    keras_model_custom(name = name, function(self) {
      self$gru_units <- gru_units
      self$embedding <-
        layer_embedding(input_dim = vocab_size, output_dim = embedding_dim)
      self$gru <- if (tf$test$is_gpu_available()) {
        layer_cudnn_gru(
          units = gru_units,
          return_sequences = TRUE,
          return_state = TRUE,
          recurrent_initializer = 'glorot_uniform'
        )
      } else {
        layer_gru(
          units = gru_units,
          return_sequences = TRUE,
          return_state = TRUE,
          recurrent_initializer = 'glorot_uniform'
        )
      }
      
      self$fc1 <- layer_dense(units = self$gru_units)
      self$fc2 <- layer_dense(units = vocab_size)
      
      self$attention <- attention_module(self$gru_units)
      
      function(inputs, mask = NULL) {
        x <- inputs[[1]]
        features <- inputs[[2]]
        hidden <- inputs[[3]]
        
        c(context_vector, attention_weights) %<-% self$attention(features, hidden)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x <- self$embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + gru_units)
        x <-
          k_concatenate(list(k_expand_dims(context_vector, 2), x))
        
        # passing the concatenated vector to the GRU
        c(output, state) %<-% self$gru(x)
        
        # shape == (batch_size, max_length (= timesteps), gru_units)
        x <- self$fc1(output)
        
        # x shape == (batch_size * max_length, gru_units)
        x <- k_reshape(x, c(-1, x$shape[3]))
        
        # output shape == (batch_size * max_length (= timesteps), vocab_size)
        x <- self$fc2(x)
        
        list(x, state, attention_weights)
        
      }
    })
  }


encoder <- cnn_encoder(embedding_dim)
decoder <- rnn_decoder(embedding_dim, gru_units, vocab_size)

optimizer = tf$train$AdamOptimizer()

# We are masking the loss calculated for padding
cx_loss <- function(real, pred) {
  mask <- 1 - k_cast(y_true == 0L, dtype = "float32")
  loss <-
    tf$nn$sparse_softmax_cross_entropy_with_logits(labels = y_true, logits =
                                                     y_pred) * mask
  tf$reduce_mean(loss)
}

evaluate <-
  function(image) {
    attention_matrix <-
      matrix(0, nrow = max_length, ncol = attention_features_shape)
    temp_input <- K_expand_dims(load_image(image)[1], 1)
    img_tensor_val <- image_features_extract_model(temp_input)
    img_tensor_val <- k_reshape(img_tensor_val,
                                tuple(img_tensor_val.shape[1],-1, img_tensor_val.shape[4]))
    
    features <- encoder(img_tensor_val)
    
    hidden <- k_zeros(c(1, gru_units))
    dec_input <-
      k_expand_dims(list(tokenizer$word_index["<start>"]))
    
    result <- ""
    
    for (t in seq_len(target_maxlen - 1)) {
      c(preds, dec_hidden, attention_weights) %<-%
        decoder(list(dec_input, features, hidden))
      attention_weights <- k_reshape(attention_weights, c(-1))
      attention_matrix[t,] <- attention_weights %>% as.double()
      
      pred_idx <-
        tf$multinomial(k_exp(preds), num_samples = 1)[1, 1] %>% as.double()
      pred_word <- word_index_df %>% filter(index = pred_idx)
      
      if (pred_word == '<stop>') {
        result <-
          paste0(result, pred_word)
        attention_matrix <- attention_matrix[1:length(result),]
        return (list(result, attention_matrix))
      } else {
        result <-
          paste0(result, pred_word, " ")
        dec_input <- k_expand_dims(list(pred_idx))
      }
    }
    
    list(str_trim(result), attention_matrix)
  }

plot_attention <-
  function(attention_matrix,
           image,
           result) {
    # temp_image = np.array(Image.open(image))
    #
    # fig = plt.figure(figsize=(10, 10))
    #
    # len_result = len(result)
    # for l in range(len_result):
    #   temp_att = np.resize(attention_plot[l], (8, 8))
    # ax = fig.add_subplot(len_result//2, len_result//2, l+1)
    # ax.set_title(result[l])
    # img = ax.imshow(temp_image)
    # ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    #
    # plt.tight_layout()
    # plt.show()
  }


check_captions <- function() {
  walk2(check_images_train, check_captions_train, function(image, caption) {
    c(result, attention_matrix) %<-% evaluate(image)
    generated_caption <- decode_caption(result)
    real_caption <- decode_caption(caption)
    print(paste0("Real caption: ",  real_caption))
    print(paste0("Predicted caption: ", generated_caption))
  })
}


num_epochs <- 1

train_iter <- make_iterator_one_shot(train_dataset)
batch <- iterator_get_next(train_iter)
loss <- 0
img_tensor <- batch[[1]]
target_caption <- batch[[2]]
img_tensor
target_caption

for (epoch in seq_len(num_epochs)) {
  total_loss <- 0
  train_iter <- make_iterator_one_shot(train_dataset)
  
  until_out_of_range({
    batch <- iterator_get_next(train_iter)
    loss <- 0
    img_tensor <- batch[[1]]
    target_caption <- batch[[2]]
    
    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    decoder_init_hidden <- k_zeros(c(batch_size, gru_units))
    
    dec_input <-
      k_expand_dims(rep(list(word_index_df[word_index_df$word == "<start>", "index"], batch_size)))
    
    with(tf$GradientTape() %as% tape, {
      
      # bs 299 299 64
      features %<-% encoder(img_tensor)
      
      for (t in seq_len(dim(target_caption)[2])) {
        c(preds, dec_hidden, weights) %<-%
          decoder(list(dec_input, features, decoder_init_hidden))
        loss <- loss + cx_loss(target_caption[, t], preds)
        dec_input <- k_expand_dims(target_caption[, t])
      }
      
    })
    total_loss <-
      total_loss + loss / k_cast_to_floatx(dim(target_caption)[2])
    
    variables <- c(encoder$variables, decoder$variables)
    gradients <- tape$gradient(loss, variables)
    
    optimizer$apply_gradients(purrr::transpose(list(gradients, variables)),
                              global_step = tf$train$get_or_create_global_step())
  })
  cat(paste0(
    "Total loss (epoch): ",
    epoch,
    ": ",
    (total_loss / k_cast_to_floatx(buffer_size)) %>% as.double() %>% round(4),
    "\n"
  ))
}


