# 🎮 Movie Review Sentiment Analysis in R

This project is a beginner-to-intermediate level **Natural Language Processing (NLP)** and **Machine Learning (ML)** application in **R** that classifies movie reviews as either **positive** or **negative**.

It uses basic NLP techniques (cleaning, tokenization, TF-IDF) and implements two ML models: **Naive Bayes** and **Logistic Regression**.

---

## 📌 Features

* ✅ Load sample or external movie review dataset
* ✅ Clean and preprocess review text (lowercase, punctuation, stemming, stopword removal)
* ✅ Convert text into TF-IDF features
* ✅ Train Naive Bayes and Logistic Regression models
* ✅ Evaluate model performance (confusion matrix, accuracy, precision, recall)
* ✅ Visualize word clouds for positive and negative reviews
* ✅ Predict sentiment for new reviews
* ✅ Export predictions to CSV

---

## 📁 Dataset

You can either:

* Use the default built-in sample dataset (18 labeled reviews), or
* Load your own dataset in `.csv` format with two columns:

  * `review` (text of the review)
  * `sentiment` (either `positive` or `negative`)

---

## ⚙️ Requirements

Install required R packages:

```r
install.packages(c("tm", "e1071", "caret", "text2vec", "glmnet",
                   "wordcloud", "SnowballC", "ggplot2", "dplyr"))
```

---

## ▶️ How to Run

1. Clone the repo or copy the script into an R file (e.g., `movie_sentiment.R`)
2. Open in **RStudio**
3. Run the `main()` function
4. Optionally modify to load your own dataset:

```r
df <- load_data("your_file.csv")
```

---

## 📊 Example Output

* Accuracy, precision, recall for both models
* Word clouds of frequent positive/negative words
* Prediction results for new reviews like:

```
Review: "This movie was a masterpiece!"
NB: positive, LR: positive
```

---

## 📂 Output Files

* `predicted_reviews.csv`: Predictions of new reviews using both models

---

## 🔍 Next Steps

* Use a larger dataset (like IMDB reviews)
* Build a GUI using R Shiny
* Add more classifiers (Random Forest, XGBoost)
* Deploy as a Shiny web app

---

## 👨‍💼 Author

**Akash Raj**
🎓 Student @ Lovely Professional University
💼 Full-Stack Developer & Data Scientist
🌐 [GitHub](https://github.com/) | [LinkedIn](https://www.linkedin.com/)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
