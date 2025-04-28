# 📚 TextGenRNN: Character-Level Text Generation with LSTM (PyTorch)

This project trains a **character-level LSTM-based RNN** to generate new text in the style of **Shakespeare**, based on a small dataset.  
The example workflow uses the openly available **Tiny Shakespeare** dataset.

---

## 🛠 Technologies Used

- **Python** – Programming language  
- **PyTorch** – Deep-learning framework  
- **Matplotlib** – Plotting training and validation loss  
- **Requests** – Dataset fetching from URL  
- **NumPy** – Data manipulation  
- **tqdm** – Progress bars during training  

---

## ▶️ How to Run the Project
1. **Clone the repository**  
   
bash
   git clone https://github.com/your-username/Text_Generation_Shakespeare.git
   cd SentimentRNN
2. **Install dependencies**
bash
   pip install -r requirements.txt
3. **Run the training script**
   
bash
   jupyter Text_Generation_Shakespeare.ipynb

# 📂 Project Description

The pipeline encodes Shakespeare’s text into integer sequences, trains a character-level LSTM model, and generates new text character-by-character.

- Download Tiny Shakespeare dataset (~1 MB, ~1 million characters)  
- Build character-level vocabulary mapping each unique character to an index  
- Encode the full text corpus into integer sequences  
- Prepare batches of input/output sequences for training  
- Model – Embedding → Single-layer LSTM → Fully-Connected → Softmax over characters  
- Train using CrossEntropyLoss and Adam optimizer  
- Generate novel text from a given seed string  

## ✅ Steps for Analysis

| Step               | Description                                                        |
|--------------------|--------------------------------------------------------------------|
| Data Loading       | Downloads Tiny Shakespeare via `requests.get()`.                   |
| Vocabulary Creation| Maps unique characters to indices (and vice-versa).                |
| Encoding           | Full text is converted into a tensor of integers.                  |
| Batch Preparation  | `get_batch()` samples random sequences for training.               |
| Model Architecture | Embedding → LSTM → Fully Connected output layer.                   |
| Training Loop      | Tracks training and validation loss.                               |
| Validation         | Evaluates loss every few steps during training.                    |
| Text Generation    | `generate_text()` produces novel text character-by-character.      |

## 🧠 Model Architecture Overview

```text
Input (batch_size, block_size) ─► Embedding (vocab_size × embed_dim)
                                  │
                                  ▼
                        Single-layer LSTM
                          hidden_dim = 512
                                  │
                                  ▼
                        Fully-Connected (vocab_size)
                                  │
                                  ▼
                       Softmax over next character
