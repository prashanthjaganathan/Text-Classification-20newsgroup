# Text Classification with TF-IDF and Word2Vec

This project focuses on text classification using different text representations and machine learning models. The objective was to process a dataset and implement various classification models with different feature representations, including TF-IDF and Word2Vec, to compare their performance in classifying documents into predefined categories.

## Dataset

The dataset used in this project is the **20 Newsgroups dataset**. The dataset consists of approximately 18,774 documents spread across 20 different newsgroups. Each document is labeled with one of the 20 categories, and the goal is to classify the documents into the correct category.

## Methodology

### Data Preprocessing

1. **Text Cleaning**
   - All text was converted to lowercase.
   - Emails, URLs, and special characters were removed.
   - Common stopwords (e.g., "forward", "reply") and header information (e.g., "From", "Subject") were eliminated.

2. **Tokenization & Normalization**
   - Words were lemmatized to their base form.
   - Empty or single-word documents were removed.
   - The text was tokenized into words and stored in separate train and test sets.

### Word Representation

Two different methods were used for representing words in the dataset:

1. **TF-IDF Vectorization**
   - Text data was converted into numerical features using term frequency-inverse document frequency (TF-IDF) with the help of the sklearn library.

2. **Word2Vec Embeddings**
   - 100-dimensional word vectors were generated to capture word meanings using the gensim library.
   
3. **Pre-trained Word2Vec Embeddings**
   - Google’s pre-trained Word2Vec embeddings (GoogleNews-vectors-negative300.bin) were used to represent words.

### Models Implemented

Several classification models were trained and evaluated using both TF-IDF and Word2Vec features:

1. **Naive Bayes Classifier** (with TF-IDF)
2. **Logistic Regression** (with TF-IDF and Word2Vec)
3. **Convolutional Neural Network (CNN)** 
   - CNN was trained using TF-IDF, Word2Vec, and Pre-trained Word2Vec features. The CNN included Dropout, MaxPooling, and GlobalMaxPooling layers.

## Results

The performance of each model was evaluated using accuracy and F1-score metrics:

- **Naive Bayes with TF-IDF:**
  - Average Accuracy: 80.17%
  - Macro avg F1-score: 0.778

- **Logistic Regression with TF-IDF:**
  - Average Accuracy: 82.7%
  - Macro avg F1-score: 0.818

- **Logistic Regression with Word2Vec:**
  - Average Accuracy: 53.7%
  - Macro avg F1-score: 0.526

- **Logistic Regression with Pre-trained Word2Vec:**
  - Average Accuracy: 69.87%
  - Macro avg F1-score: 0.68

- **CNN with TF-IDF:**
  - Epochs: 5
  - Test Loss: 2.99
  - Test Accuracy: 0.05

- **CNN with Word2Vec:**
  - Epochs: 30
  - Test Loss: 1.80
  - Test Accuracy: 0.47

- **CNN with Pre-trained Word2Vec:**
  - Epochs: 30
  - Test Loss: 1.68 
  - Test Accuracy: 0.58


## Repository Structure
```bash
/Text-Classification-20newsgroup
│
├── text_classification.ipynb   # Main notebook containing implementation
└── README.md                   # Project documentation
```
