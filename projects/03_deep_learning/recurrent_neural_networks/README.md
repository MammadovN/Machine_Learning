# 🔬 ProtFuncPredictor: Protein Function Classification using Bidirectional LSTM

This project applies a **Bidirectional LSTM Neural Network** built with PyTorch to predict protein functions from amino acid sequences. The model can use synthetic dummy data or real sequences retrieved from UniProt.

---

## 🛠 Technologies Used

- **Python** – Programming language  
- **PyTorch** – Deep learning framework  
- **scikit-learn** – Preprocessing and evaluation  
- **pandas & NumPy** – Data manipulation  
- **matplotlib & seaborn** – Visualization  
- **requests** – HTTP requests for UniProt API  

---

## ▶️ How to Run the Project

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/your-username/ProtFuncPredictor.git
   cd ProtFuncPredictor
2. **Install Dependencies**:
   ```bash
    pip install -r requirements.txt
3. **Run the Main Script**:
   ```bash
   jupyter BIO_RNN.ipynb

---

## 📂 Project Description

The dataset consists of protein sequences and their corresponding function labels. Sequences are encoded numerically (one integer per amino acid), then padded or truncated to a uniform length. The goal is to classify each protein into its functional category using a Bidirectional LSTM network.

---

## ✅ Steps for Analysis

1. **Data Loading**  
   Generate dummy data for quick testing, or retrieve real data via the UniProt REST API.

2. **Data Preprocessing**  
   - Map amino acid letters to integers.  
   - Pad/truncate sequences to a fixed `max_seq_length`.  
   - Encode functional labels with `LabelEncoder`.  

3. **DataLoader Preparation**  
   - Wrap arrays in a custom `ProteinDataset` class.  
   - Use PyTorch `DataLoader` with batching and shuffling.  

4. **Model Architecture**  
   - **Bidirectional LSTM**:  
     - Input size: 1 (per amino acid)  
     - Hidden size: 128  
     - Layers: 2  
     - Dropout: 0.5  
   - **Classification Head**:  
     - Fully Connected (256) → ReLU → BatchNorm → Dropout  
     - Fully Connected → Softmax over `num_classes`  

5. **Training**  
   - Loss: `CrossEntropyLoss`  
   - Optimizer: `Adam`  
   - Tracks and plots training & validation loss and accuracy.  

6. **Evaluation**  
   - Compute test set accuracy and loss.  
   - Display confusion matrix and classification report.  
   - Visualize training curves.  

---

## 🧠 Model Architecture Overview

```text
Input: (batch_size, seq_length, 1)
→ Bidirectional LSTM (128 units, 2 layers, 0.5 dropout)
→ Extract last time-step output
→ FC (256) + ReLU + BatchNorm + Dropout
→ FC (num_classes)

