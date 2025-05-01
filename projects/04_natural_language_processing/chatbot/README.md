# 🤖 Attention-Enhanced Seq2Seq Chatbot

A TensorFlow / Keras implementation of a sequence-to-sequence chatbot with an attention mechanism. This project trains on paired question-answer data and demonstrates both training and inference pipelines, including greedy decoding.

---

## 🛠 Technologies Used

- **Python**: Core scripting language  
- **TensorFlow / Keras**: Model definition, training, and inference  
- **NumPy**: Numerical operations and array handling  
- **NLTK**: Tokenization of text data  
- **faiss-cpu** (optional): For future retrieval-augmented extensions  

---

## ▶️ How to Run the Project

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/attention-seq2seq-chatbot.git
   cd attention-seq2seq-chatbot
2. **Install Required Libraries**:   
   ```bash
   pip install -r requirements.txt

3. **Run the Python script or notebook**:  
   ```bash
   jupyter notebook Seq2Seq Chatbot.ipynb

---
## 📂 Project Description

We implement a classic encoder–decoder LSTM architecture augmented with an **Attention** layer:

### Encoder
- Embeds input tokens to 256-dim vectors  
- Processes the sequence with an LSTM (256 units)  
- Outputs both hidden states (`state_h`, `state_c`) and full sequence outputs (`encoder_outputs`)

### Decoder
- Embeds target tokens to 256-dim vectors  
- Consumes previous decoder state and encoder states via LSTM  
- Applies an attention mechanism over the encoder outputs  
- Concatenates LSTM output and attention context (→ 512 dims)  
- Predicts next token with a softmax over the vocabulary  

### Inference Pipeline
- **`encoder_model`** produces both `encoder_outputs` and the final states  
- **`decoder_model`** takes one token + previous states + `encoder_outputs`, and returns next-token distribution + updated states  
- Greedy decoding loop runs until the `<end>` token is produced or maximum length is reached  

---

## ✅ Steps for Analysis

### 1. Data Loading
- A small sample of 10 question–answer pairs is hard-coded (replaceable with a larger CSV/JSON corpus)

### 2. Preprocessing
- Tokenize text with NLTK’s `punkt` tokenizer and Keras’s `Tokenizer`  
- Add `<start>` and `<end>` tokens  
- Pad all sequences to fixed lengths

### 3. Model Building
- Define encoder and decoder with shared embedding dimension (256)  
- Use Keras’s built-in `Attention` layer for soft-alignment  
- Concatenate context vector and decoder output before final prediction

### 4. Training
- Compile with **Adam** optimizer and **sparse_categorical_crossentropy** loss  
- Train for 50 epochs with batch size 16

### 5. Inference & Evaluation
- Load saved encoder/decoder inference models  
- Implement `chatbot_response()` for greedy decoding  
- Interactively converse and inspect qualitative performance

---

## 🧠 Model Architecture

```text
Encoder:
  Input (seq_len) → Embedding(256) → LSTM(256, return_sequences=True, return_state=True)
  ↓                               ↘︎─ state_h, state_c
  └─ encoder_outputs (seq_len, 256)

Decoder (training):
  Input (seq_len) → Embedding(256)
  → LSTM(256, return_sequences=True, initial_state=[state_h, state_c]) → decoder_outputs (seq_len,256)
  → Attention([decoder_outputs, encoder_outputs]) → context (seq_len,256)
  → Concatenate([decoder_outputs, context]) → (seq_len,512)
  → Dense(vocab_size, activation=softmax)

Inference decoding loop:
  target_token (1) + encoder_outputs + prev_states
  → decoder_pred (1, vocab_size) + new_states

