# 📄 WordEmbedding: Word Embedding & Sentiment Analysis

This project trains custom Word2Vec and FastText embeddings on the NLTK Gutenberg corpus and evaluates them both intrinsically and extrinsically via an IMDb sentiment classification task.

---

## 🛠️ Technologies Used

- **Python** – Core language
- **Gensim** – Word2Vec & FastText embedding training
- **NLTK** – Gutenberg corpus access & basic text preprocessing
- **scikit-learn** – t-SNE visualization & Logistic Regression classifier
- **TensorFlow / Keras** – IMDb dataset loading
- **Matplotlib** – Plotting t-SNE projections
- **NumPy & pandas** – Data manipulation

---

## ▶️ How to Run the Project

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/word_embedding.git
   cd word_embedding
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Launch the notebook**
   ```bash
   jupyter notebook word_embedding.ipynb

---

## 📂 Project Description

### 01. Data Preparation
- Loads raw texts from **NTLK Gutenberg** corpus.  
- Cleans lowercase, removes non-alphabetic characters, splits on sentences via punctuation.  
- Tokenizes by simple `str.split()` to avoid external downloads.

### 02. Embedding Training
- Trains **Word2Vec** and **FastText** models with `vector_size=100`, `window=5`, `min_count=5`, `sg=1`, and `epochs=5`.  
- Saves models as `word2vec.model` and `fasttext.model`.

### 03. Intrinsic Evaluation
- Loads saved Word2Vec embeddings.  
- Performs analogy (`king - man + woman`) and similarity (`king, queen`) queries.  
- Visualizes a small word set via **t-SNE** with `perplexity=5`.

### 04. Extrinsic Evaluation (IMDb Sentiment)
- Loads **IMDb** dataset from Keras.  
- Decodes integer-encoded reviews to text.  
- Converts each review to a fixed-length embedding via mean pooling.  
- Trains **Logistic Regression** on embedded vectors.  
- Reports **Accuracy** and **F1-score** on the test set.

---

## ✅ Steps for Analysis

| Step                  | Description                                                                                         |
|-----------------------|-----------------------------------------------------------------------------------------------------|
| **00. Setup**         | Install & patch dependencies (`scipy`, `gensim`, etc.)                                               |
| **01. Prep**          | Clean & tokenize Gutenberg texts into sentence lists                                                  |
| **02. Train**         | Train Word2Vec & FastText; save models                                                               |
| **03. Intrinsic**     | Query analogies, compute similarity, visualize t-SNE                                                   |
| **04. Decode IMDb**   | Map IMDb indices back to words; prepare train/test texts                                             |
| **05. Embed Reviews** | Mean-pool word embeddings to fixed vectors                                                           |
| **06. Classify**      | Fit Logistic Regression; evaluate accuracy & F1                                                        |

---

## 🧠 Architecture Overview

   ```text
   Gutenberg Texts └─► Clean & Split Sentences └─► Token Lists ├─► Word2Vec & FastText Training │ └─► Saved .model Files ├─► Intrinsic Eval: │ ├─ Analogy & Similarity Queries │ └─ t-SNE Visualization └─► Extrinsic Eval (IMDb): ├─ Decode Reviews to Words ├─ Embed via Mean Pooling └─ Logistic Regression Classification
