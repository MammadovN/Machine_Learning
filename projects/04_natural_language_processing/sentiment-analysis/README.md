# 🎥 Movie Review Sentiment Analysis

A Python project that implements both a TF-IDF + Logistic Regression baseline and a fine-tuned DistilBERT model to classify IMDB movie reviews as positive or negative. Includes end-to-end training, evaluation, and inference pipelines.

---

## 🛠 Technologies Used

- **Python 3.8+**: Core scripting language  
- **transformers**: Hugging Face library for DistilBERT  
- **datasets**: Hugging Face library to load IMDB dataset  
- **PyTorch**: Deep learning framework for model training  
- **scikit-learn**: TF-IDF vectorization, Logistic Regression, and evaluation metrics  
- **NumPy**: Numerical operations and array handling  

---

## ▶️ How to Run the Project

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/movie-review-sentiment.git
   cd movie-review-sentiment
2. **Install requirements**
   ```bash
   pip install -r requirements.txt
3. **Run the main script**
   ```bash
   jupyter notebook movie sentiment analyzer.ipynb

---

## 📂 Project Description

This project is structured into three main stages:

1. **Baseline Model**
   - **TF-IDF Vectorization**: Converts each review into a high-dimensional sparse feature vector (`max_features=20000`, bi-grams, English stop words removed).  
   - **Logistic Regression**: Trained on TF-IDF features (`max_iter=1000`).  
   - **Evaluation**: Reports accuracy, precision, recall, and F1-scores on the IMDB test split.  

2. **Transformer Fine-Tuning**
   - **Tokenization**: Uses `DistilBertTokenizerFast` with truncation/padding to length 512.  
   - **Dataset Wrapping**: Builds `datasets.Dataset` objects with `input_ids`, `attention_mask`, and `labels`.  
   - **Model**: `DistilBertForSequenceClassification` (pre-trained `distilbert-base-uncased`, 2-class head).  
   - **TrainingArguments**:  
     - Output to `./results`  
     - 2 epochs, batch sizes 16/32 for train/eval  
     - Evaluation and checkpointing at each epoch  
     - `report_to="none"` (no W&B or other logging)  
   - **Trainer**: Computes accuracy on validation; saves best model locally.  

3. **Inference**
   - Loads the saved `./sentiment_model` (model + tokenizer).  
   - Uses the Hugging Face `pipeline("sentiment-analysis")` API for single-sentence predictions.  

---

## ✅ Steps for Analysis

1. **Data Loading**  
   - Load IMDB via `datasets.load_dataset('imdb')`.  
   - Extract train/test texts and labels.  

2. **Preprocessing**  
   - **Baseline**: TF-IDF vectorization (unigrams, bigrams).  
   - **Transformer**: Tokenization (`max_length=512`, truncation, padding).  

3. **Model Training**  
   - Train Logistic Regression on TF-IDF features.  
   - Fine-tune DistilBERT with `Trainer`, monitoring accuracy.  

4. **Evaluation**  
   - Print baseline accuracy and classification report.  
   - Print transformer evaluation metrics after each epoch.  

5. **Model Saving & Inference**  
   - Persist fine-tuned model and tokenizer under `./sentiment_model`.  
   - Load and run sample inference through a simple function call.  

---

## 🧠 Model Architecture

```text
TF-IDF + Logistic Regression
└─ Input: raw review text
   → TfidfVectorizer (max_features=20k, ngram_range=(1,2))
   → LogisticRegression (max_iter=1000)
   → Output: class probabilities → argmax → {negative, positive}

DistilBERT Fine-Tuning
└─ Input: raw review text
   → DistilBertTokenizerFast
     └─ input_ids (512), attention_mask (512)
   → DistilBertForSequenceClassification
     └─ Pre-trained DistilBERT encoder
     └─ Classification head (pre_classifier + classifier)
   → Trainer
     └─ Loss: cross-entropy
     └─ Metric: accuracy
   → Output: class logits → softmax → {negative, positive}
