# ⚡ Anomaly Detection in ECG Signals using Autoencoder

This project showcases how to use a **deep autoencoder** in TensorFlow/Keras to detect anomalies in ECG (electrocardiogram) signals. The goal is to train the model on **normal heartbeats** and detect **abnormal** patterns using reconstruction error.

---

## 🛠 Technologies Used

- **Python**: Core language used for data processing and model development.
- **TensorFlow / Keras**: Framework for building and training the autoencoder model.
- **NumPy**: For efficient numerical computations.
- **scikit-learn**: Used for evaluation metrics and data preprocessing.
- **matplotlib**: For visualizing training loss and reconstruction error distributions.

---

## ▶️ How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MammadovN/ecg-autoencoder-anomaly-detection.git
   cd ecg-autoencoder-anomaly-detection
   ```

2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Python script or notebook**:
   ```bash
   python main.py
   ```

---

## 📂 Project Description

The ECG5000 dataset is used to train an autoencoder to reconstruct **normal heartbeats**. The idea is that abnormal signals will have a **higher reconstruction error** since the model has never seen them during training.

The reconstruction error is then compared to a threshold (determined via F1 score) to classify signals as **normal** or **abnormal**.

---

## ✅ Steps for Analysis

1. **Data Loading**:
   - ECG signals are loaded from a CSV file hosted online.
   - Labels are separated from input features.

2. **Preprocessing**:
   - Only normal signals are used for training.
   - The test set includes both normal and abnormal samples.
   - Features are standardized using `StandardScaler`.

3. **Model Building**:
   - A symmetrical deep autoencoder with three encoding and three decoding layers.
   - ReLU activations in hidden layers; sigmoid in output layer.

4. **Training**:
   - The model is trained with Mean Squared Error (MSE) loss and Adam optimizer.
   - Early stopping is used to prevent overfitting by monitoring validation loss.

5. **Reconstruction Error & Thresholding**:
   - Reconstruction error is calculated for the test set.
   - The optimal threshold is determined by maximizing the F1 score.

6. **Evaluation**:
   - Accuracy, precision, recall, and F1 score are reported.
   - Loss curves and error distributions are visualized for insight.

---

## 🧠 Model Architecture

```
Input: ECG signal vector
→ Dense(64) → ReLU  
→ Dense(32) → ReLU  
→ Dense(16) → ReLU (Latent space)  
→ Dense(32) → ReLU  
→ Dense(64) → ReLU  
→ Dense(original_input_dim) → Sigmoid
```
