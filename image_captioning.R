library(keras)
use_implementation("tensorflow")
library(tensorflow)
#tfe_enable_eager_execution(device_policy = "warn")
tfe_enable_eager_execution(device_policy = "silent")

np <- import("numpy")

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
all_img_name_vector <- vector(mode = "list", length = num_captions)

for (i in seq_len(num_captions)) {
  caption <-
    paste0("<start> ", annot_captions[[i]][["caption"]], " <end>")
  image_id <- annot_captions[[i]][["image_id"]]
  full_coco_image_path <-
    sprintf("train2014/train2014/COCO_train2014_%012d.jpg", image_id)
  all_img_name_vector[[i]] <- full_coco_image_path
  all_captions[[i]] <- caption
}

set.seed(7777)
num_examples <- 30000
random_sample <- sample(1:num_captions, size = 30000)
train_captions <- all_captions[random_sample]
img_name_vector <- all_img_name_vector[random_sample]

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

encode_train <- unique(img_name_vector) %>% unlist() %>% sort()

# batch_size <- 4
# image_dataset <- tensor_slices_dataset(encode_train) %>%
#   dataset_map(load_image) %>%
#   dataset_batch(batch_size)
#
# iter <- make_iterator_one_shot(image_dataset)
# until_out_of_range( {
#   batch <- iter$get_next()
#   img <- batch[[1]]
#   path <- batch[[2]]
#   #shape=(16, 8, 8, 2048)
#   batch_features <- image_features_extract_model(img)
#   batch_features <- tf$reshape(batch_features,
#                               list(batch_features$shape[0], -1L, batch_features$shape[3]))
#   for (i in 1:batch_size) {
#     np$save(path[i]$numpy()$decode("utf-8"),batch_features[i, , ]$numpy())
#   }
# })

top_k <- 5000
tokenizer <- text_tokenizer(num_words = top_k,
                            oov_token = "<unk>",
                            filters = '!"#$%&()*+.,-/:;=?@[\\]^_`{|}~ ')
tokenizer$fit_on_texts(train_captions)
train_seqs <- tokenizer %>% texts_to_sequences(train_captions)
tokenizer$word_index
tokenizer$word_index["<unk>"]

tokenizer$word_index["<pad>"] <- length(tokenizer$word_index) + 1
tokenizer$word_index["<pad>"]

py_run_string("index_word = {value:key for key, value in r.tokenizer.word_index.items()}")
index_word <- py$index_word

cap_vector <- pad_sequences(train_seqs, padding = "post")
max_length <- map(train_seqs, length) %>% unlist() %>% max()

length(img_name_vector)
nrow(cap_vector)

train_indices <-
  sample(length(img_name_vector), size = length(img_name_vector) * 0.8)

validation_indices <-
  setdiff(1:length(img_name_vector), train_indices)

img_name_train <- img_name_vector[train_indices]
cap_train <- cap_vector[train_indices,]

img_name_val <- img_name_vector[validation_indices]
cap_val <- cap_vector[validation_indices,]

batch_size <- 16
buffer_size <- 1000
embedding_dim <- 64
num_units <- 128
vocab_size <- length(tokenizer$word_index)
features_shape <- 2048
attention_features_shape <- 64

map_func <- function(img_name, cap) {
  img_tensor <- np$load(img_name$decode('utf-8') + '.npy')
  list(img_tensor, cap)
}

dataset <-
  tensor_slices_dataset(list(img_name_train, cap_train)) %>%
  
  # using map to load the numpy files in parallel
  # NOTE: Be sure to set num_parallel_calls to the number of CPU cores you have
  # https://www.tensorflow.org/api_docs/python/tf/py_func
  dataset_map(function(item1, item2)
    tf$py_func(map_func, list(item1, item2), list(tf$float32, tf$int32)),
    num_parallel_calls = 4) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_batch(batch_size) %>%
  dataset_prefetch(1)


cnn_encoder <- 
  function(embedding_dim,
           name = NULL) {
    keras_model_custom(name = name, function(self) {
      # shape after fc == (batch_size, 64, embedding_dim)
      self$fc <- layer_dense(units = embedding_dim, activation = "relu")
      function(x, mask = NULL) {
        self$fc(x)
      }
    })
  }

attention_module <- 
  function(units,
           name = NULL) {
    keras_model_custom(name = name, function(self) {
      self$W1 = layer_dense(units = units)
      self$W2 = layer_dense(units = units)
      self$V = layer_dense(units = 1)
      function(x, mask = NULL) {
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis <- k_expand_dims(hidden, axis = 2)
        # score shape == (batch_size, 64, 1)
        score <- self$V(k_tanh(self$W1(features) + self$W2(hidden_with_time_axis)))
        # attention_weights shape == (batch_size, 64, 1)
        attention_weights <- k_softmax(score, axis = 2)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector <- k_sum(attention_weights * features, axis = 2)
        list(context_vector, attention_weights)
      }
    })
  }

rnn_decoder <- 
  function(embedding_dim,
           units,
           vocab_size,
           name = NULL) {
    keras_model_custom(name = name, function(self) {
      self$units <- units
      self$embedding <- layer_embedding(vocab_size, embedding_dim)
      self$gru <-
        layer_gru(
          units = units,
          return_sequences = TRUE,
          return_state = TRUE,
          recurrent_initializer='glorot_uniform'
        )
      self$fc1 <- layer_dense(self$units)
      self$fc2 <- layer_dense(vocab_size)
      
      self$attention <- attention_module(self$units)
      function(x, mask = NULL) {
        ###
      }
    })
  }


