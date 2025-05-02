# ✂️ Text Summarization

An end-to-end NLP project implementing abstractive text summarization on the CNN/DailyMail dataset. This Colab notebook demonstrates data loading, preprocessing, model fine-tuning with Hugging Face Transformers (T5), evaluation using ROUGE metrics, and a simple inference demo.

---

## 🛠 Technologies Used

- **Python 3.8+**: Core scripting language  
- **transformers**: Hugging Face library for T5 model and tokenization  
- **datasets**: Hugging Face library to load and preprocess CNN/DailyMail  
- **evaluate & rouge_score**: Compute ROUGE-1/2/L metrics  
- **PyTorch**: Backend for model training  
- **Google Colab**: Development and execution environment

---

## ▶️ How to Run the Project

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/text-summarization.git
   cd text-summarization
2. **Install requirements**
   ```bash
   pip install -r requirements.txt
3. **Run the main script**   
   ```bash
   jupyter notebook text-summarization.ipynb

---

## 📂 Project Description

This pipeline covers the full workflow for abstractive summarization:

1. **Data Loading & Preprocessing**  
   - Load CNN/DailyMail (`version=3.0.0`) via `datasets.load_dataset`.  
   - Tokenize articles and highlights with T5 tokenizer (`max_length=512` for inputs, `150` for labels) and prepare PyTorch-ready datasets.

2. **Model Fine-Tuning**  
   - Use `t5-small` from Hugging Face.  
   - Configure `Seq2SeqTrainingArguments` (2 epochs, batch size 4, evaluation every 500 steps, mixed precision if available).  
   - Train with `Seq2SeqTrainer.train()` and save best checkpoints.

3. **Evaluation**  
   - Compute ROUGE-1/2/L metrics on the validation split using `evaluate.load("rouge")`.  
   - Log and print evaluation results at the end of training.

4. **Inference Demo**  
   - Define a `summarize()` function wrapping `model.generate()`.  
   - Showcase on a sample test article with ground-truth vs. predicted summary outputs.

---

## ✅ Steps for Analysis

1. **Load Dataset**  
   - Load CNN/DailyMail via `datasets.load_dataset("cnn_dailymail", "3.0.0")`.  
   - Use the `article` field as input and `highlights` as reference summaries.  

2. **Preprocessing**  
   - Prefix each article with `"summarize: "` to form the T5 input prompt.  
   - Tokenize articles and highlights with the T5 tokenizer:  
     - `max_length=512`, truncation and padding for inputs  
     - `max_length=150`, truncation and padding for labels  

3. **Training**  
   - Fine-tune `t5-small` using the `Seq2SeqTrainer` API.  
   - Configure `Seq2SeqTrainingArguments` to monitor loss and ROUGE during training.  

4. **Evaluation**  
   - Load ROUGE metric via `evaluate.load("rouge")`.  
   - Compute and print ROUGE-1, ROUGE-2, and ROUGE-L scores as percentages.  

5. **Inference**  
   - Generate summaries with beam search (`num_beams=4`, `length_penalty=2.0`).  
   - Compare generated summaries against reference highlights.  

---

## 🧠 Model Architecture Overview
  ```text
  Abstractive Summarization with T5-small
  
  Input: raw article text
  └─ "summarize: " + article
     → T5 Tokenizer (input_ids, attention_mask)
       → T5-small Seq2Seq model
         ├─ Encoder: 6-layer transformer
         └─ Decoder: 6-layer transformer + cross-attention
     → generate(): beam search decoding (num_beams=4, length_penalty=2.0)
     → Output: abstractive summary text
