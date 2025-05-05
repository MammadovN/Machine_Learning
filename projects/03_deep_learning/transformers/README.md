# 📝 TransformerIMDb: Text Classification via Custom Transformer (PyTorch)

This project implements a simple Transformer encoder from scratch and fine-tunes it on the **IMDb** movie review dataset to demonstrate transformer-based text classification.

---

## 🛠️ Technologies Used

- **Python** – Programming language  
- **PyTorch** – Deep learning framework  
- **torchtext / datasets** – Data loading and preprocessing  
- **transformers** – Tokenization (BERT tokenizer)  
- **NumPy & pandas** – Data manipulation  
- **tqdm** – Progress bars for training loops  

---

## ▶️ How to Run the Project

1. **Clone the repository**  
   ```bash
   git clone https://github.com/MammadovN/transformer_imdb_classification.git
   cd transformer_imdb_classification
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Launch training script**
   ```bash
   jupyter notebook transformer_imdb.ipynb

---

## 📂 Project Description

### Dataset
- Uses `datasets.load_dataset("imdb")` to fetch and split into train/test.  
- Samples a subset (1,000 train / 200 test) for quick experimentation.

### Tokenization & Preprocessing
- BERT’s `bert-base-uncased` tokenizer  
- Padding/truncation to length 128  
- Returns `input_ids` and `attention_mask` tensors

### DataLoader
- Batch size: 32  
- Shuffle training data, 2 worker processes  
- Uses PyTorch `DataLoader` for batching

### Model
- **Embedding Layer**: `nn.Embedding(vocab_size, d_model)`  
- **Positional Encoding**: Sine & cosine positional embeddings  
- **Encoder Layers**:  
  - Multi-Head Self-Attention (8 heads, `d_k = d_model/8`)  
  - Feed-Forward (linear → ReLU → linear)  
  - LayerNorm & residual connections  
- **Classifier Head**: Global average pooling → `nn.Linear(d_model, 2)`

### Training Loop
- Optimizer: `Adam(lr=1e-3)`  
- Loss: `CrossEntropyLoss`  
- 3 epochs, prints average loss per epoch

### Device
- Automatically uses GPU if available  

---

## ✅ Steps for Analysis

| Step                 | Description                                                                                      |
|----------------------|--------------------------------------------------------------------------------------------------|
| **Data Download**    | `load_dataset("imdb")` downloads and splits IMDb into train/test sets.                           |
| **Tokenization**     | BERT tokenizer pads/truncates to max length 128, returns `input_ids` & `attention_mask`.          |
| **DataLoader Setup** | Create `DataLoader` objects (batch_size=32, shuffle train, num_workers=2).                       |
| **Model Definition** | Build `SimpleTransformer`: embedding, positional encoding, encoder layers, and linear classifier.|
| **Attention**        | Implement Scaled Dot-Product attention with mask handling and dropout.                            |
| **Training**         | Loop over epochs: forward → loss → backward → optimizer step, track average loss.                 |
| **Evaluation**       | Compute accuracy on test subset after training completes.                                        |

---

## 🧠 Model Architecture Overview

```text
Input: [batch_size, seq_len] token IDs
  └─► Embedding Layer (vocab_size → d_model)
       └─► Positional Encoding added
            └─► N × [Multi-Head Self-Attention + Feed-Forward + LayerNorm]
                 └─► GlobalAveragePool (over seq_len)
                      └─► Linear(d_model → 2) logits

