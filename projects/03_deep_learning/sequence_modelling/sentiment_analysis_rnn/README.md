# 💬 SentimentRNN: Text Sentiment Classification with Bidirectional LSTM (PyTorch)

This project trains a **Bidirectional LSTM-based RNN** to predict the **sentiment** (positive / negative) of short texts such as tweets, reviews, or comments.  
The example workflow below uses the openly available **Sentiment140** dataset, fetched in one line with 🤗 *datasets*.

---

## 🛠 Technologies Used

- **Python** – Programming language  
- **PyTorch** – Deep-learning framework  
- **torchvision** – (optional) utilities for embeddings  
- **🤗 datasets** – Instant access to Sentiment140 / IMDb / Yelp, etc.  
- **NLTK** – Tokenisation, stop-words, stemming  
- **scikit-learn** – Data split & evaluation metrics  
- **pandas & NumPy** – Data manipulation  
- **matplotlib** – Training curves  
- **tqdm** – Progress bars  

---

## ▶️ How to Run the Project

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/sentiment_analysis_rnn.git
   cd SentimentRNN
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run the training script**
   ```bash
   jupyter sentiment_analysis_rnn.ipynb

## 📂 Project Description  

The pipeline converts raw text into integer sequences, feeds them into a bidirectional LSTM, and outputs a **binary sentiment score**.

1. **Download & split Sentiment140** → train / validation / test  
2. **Pre-processing** – lowercase → remove punctuation/digits → tokenise → remove stop-words → stemming  
3. **Build vocabulary** of the *N* most frequent tokens (default: 10 000)  
4. **Encode & pad** each text to a fixed length (default: 100 tokens)  
5. **Model** – Embedding → 2-layer **Bi-LSTM** → Dropout → Fully-Connected → Sigmoid  
6. **Train** with `BCEWithLogitsLoss` + **Adam**, model checkpointing on best val-loss  
7. **Evaluate** accuracy, classification report & confusion matrix  
8. **Predict** sentiment for any new sentence via a helper function  

---

## ✅ Steps for Analysis  

| Step | Description |
|------|-------------|
| **Data Loading** | `load_dataset("sentiment140")` pulls tweets in *(text, sentiment)* format. |
| **Tokenisation & Cleaning** | Custom `preprocess_text()` uses NLTK. |
| **Vocabulary** | `build_vocab()` assigns indices and adds `<PAD>` / `<UNK>`. |
| **Dataset / DataLoader** | `SentimentDataset` returns `(tensor, label)` pairs. |
| **Model Architecture** | See diagram below. |
| **Training Loop** | `train_model()` tracks loss & accuracy each epoch and saves best weights. |
| **Validation** | `evaluate_model()` computes metrics on the hold-out set. |
| **Testing** | `test_model()` prints accuracy, detailed report, confusion matrix. |
| **Inference** | `predict_sentiment()` returns sentiment & probability for raw sentences. |

---

## 🧠 Model Architecture Overview  

```text
Input (batch, seq_len) ─► Embedding (vocab_size × 100)
                         │
                         ▼
        ┌───────────────────────────────────────────┐
        │      2-layer Bidirectional LSTM           │
        │   hidden_dim = 256, dropout = 0.5         │
        └───────────────────────────────────────────┘
                         │
  Concatenate last fwd & bwd hidden states
                         │
                         ▼
                Dropout (0.5)
                         │
                         ▼
                Fully-Connected (1)
                         │
                         ▼
                 Sigmoid → p(Positive)

