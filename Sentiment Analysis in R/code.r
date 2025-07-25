library(tm)
library(e1071)
library(caret)
library(text2vec)
library(glmnet)
library(SnowballC)
library(wordcloud)
library(ggplot2)
library(dplyr)

# Load Data
load_data <- function(file_path = NULL) {
  if (!is.null(file_path) && file.exists(file_path)) {
    df <- read.csv(file_path, stringsAsFactors = FALSE)
    colnames(df) <- tolower(colnames(df))
    df <- df[, c("review", "sentiment")]
  } else {
    review <- c(
      "Great movie, I loved it!",
      "Worst movie ever.",
      "Fantastic acting!",
      "Terrible plot.",
      "Amazing experience.",
      "Boring and slow.",
      "Highly recommended!",
      "Not worth it.",
      "Beautiful scenes.",
      "Bad script.",
      "I enjoyed the movie a lot.",
      "A complete disaster and waste of time.",
      "Heart-touching story!",
      "Bad direction ruined it.",
      "Incredible cinematography.",
      "Plot had too many holes.",
      "Absolutely thrilling!",
      "Dialogue delivery was weak."
    )
    sentiment <- c("positive", "negative", "positive", "negative", "positive",
                   "negative", "positive", "negative", "positive", "negative",
                   "positive", "negative", "positive", "negative", "positive",
                   "negative", "positive", "negative")
    df <- data.frame(review, sentiment, stringsAsFactors = FALSE)
  }
  return(df)
}

# Text Cleaning
clean_text <- function(text) {
  text <- tolower(text)
  text <- removePunctuation(text)
  text <- removeNumbers(text)
  text <- removeWords(text, stopwords("en"))
  text <- wordStem(text)
  text <- stripWhitespace(text)
  return(text)
}

# Preprocess Data
preprocess <- function(df) {
  df$clean <- sapply(df$review, clean_text)
  return(df)
}

# Word Cloud by Sentiment
generate_wordclouds <- function(df) {
  pos_words <- paste(df$clean[df$sentiment == "positive"], collapse = " ")
  neg_words <- paste(df$clean[df$sentiment == "negative"], collapse = " ")
  
  par(mfrow = c(1,2))
  wordcloud(pos_words, max.words = 100, colors = brewer.pal(8, "Greens"), main = "Positive")
  wordcloud(neg_words, max.words = 100, colors = brewer.pal(8, "Reds"), main = "Negative")
}

# Create TF-IDF matrix
vectorize_text <- function(texts) {
  tok <- word_tokenizer(texts)
  it <- itoken(tok, progressbar = FALSE)
  vocab <- create_vocabulary(it)
  vec <- vocab_vectorizer(vocab)
  dtm <- create_dtm(it, vec)
  tfidf <- TfIdf$new()
  tfidf_mat <- fit_transform(dtm, tfidf)
  return(list(tfidf = tfidf_mat, vocab = vocab, vectorizer = vec, tfidf_obj = tfidf))
}

# Train Models
train_models <- function(tfidf_mat, labels) {
  set.seed(123)
  idx <- createDataPartition(labels, p = 0.8, list = FALSE)
  train_x <- tfidf_mat[idx, ]
  test_x <- tfidf_mat[-idx, ]
  train_y <- labels[idx]
  test_y <- labels[-idx]
  
  nb <- naiveBayes(as.matrix(train_x), as.factor(train_y))
  lr <- cv.glmnet(train_x, as.factor(train_y), family = "binomial", alpha = 0)
  
  nb_pred <- predict(nb, as.matrix(test_x))
  lr_pred <- predict(lr, test_x, s = "lambda.min", type = "class")
  
  cat("\n📊 Naive Bayes:\n")
  print(confusionMatrix(nb_pred, as.factor(test_y)))
  
  cat("\n📊 Logistic Regression:\n")
  print(confusionMatrix(as.factor(lr_pred), as.factor(test_y)))
  
  return(list(nb = nb, lr = lr, test_x = test_x, test_y = test_y))
}

# Predict New Reviews
predict_reviews <- function(model_nb, model_lr, vec, tfidf_obj, texts) {
  clean_texts <- sapply(texts, clean_text)
  tok <- word_tokenizer(clean_texts)
  it <- itoken(tok)
  dtm <- create_dtm(it, vec)
  tfidf_new <- transform(dtm, tfidf_obj)
  
  nb_pred <- predict(model_nb, as.matrix(tfidf_new))
  lr_pred <- predict(model_lr, tfidf_new, s = "lambda.min", type = "class")
  
  result <- data.frame(Review = texts, NB = nb_pred, LR = lr_pred)
  print(result)
  write.csv(result, "predicted_reviews.csv", row.names = FALSE)
}

# Main Execution
main <- function() {
  cat("🎬 MOVIE SENTIMENT ANALYSIS PROJECT IN R 🎬\n")
  df <- load_data()             # Load built-in or custom data
  df <- preprocess(df)          # Clean text
  generate_wordclouds(df)       # EDA: word clouds
  
  # Vectorize
  vec_data <- vectorize_text(df$clean)
  tfidf_mat <- vec_data$tfidf
  vocab <- vec_data$vocab
  vec <- vec_data$vectorizer
  tfidf_obj <- vec_data$tfidf_obj
  
  # Train models
  models <- train_models(tfidf_mat, df$sentiment)
  
  # Predict new
  new_reviews <- c(
    "This movie was a masterpiece!",
    "I can't believe I wasted 2 hours.",
    "Mediocre story but good visuals."
  )
  predict_reviews(models$nb, models$lr, vec, tfidf_obj, new_reviews)
}

main()
