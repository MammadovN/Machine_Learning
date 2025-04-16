# 🍷 Wine Quality Classification using Feedforward Neural Networks (FNN)

This project applies a **Feedforward Neural Network (FNN)** built with PyTorch to classify wine samples based on physicochemical features such as acidity, sugar, alcohol, and pH. The model is trained on the **Wine Quality Dataset** from the UCI Machine Learning Repository, aiming to predict wine quality scores.

---

## 🛠 Technologies Used

- **Python** – Programming language used
- **Jupyter Notebook / Google Colab** – Interactive development environment
- **PyTorch** – Deep learning framework used for building and training the neural network
- **scikit-learn** – For data preprocessing, splitting, scaling, and evaluation
- **Pandas & NumPy** – For data handling and numerical operations
- **Matplotlib & Seaborn** – For data visualization and analysis

---

## ▶️ How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/wine-quality-classification.git
   cd wine-quality-classification
2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
3. **Launch the Notebook**:
   ```bash
   jupyter notebook wine_quality_fnn.ipynb

## 📂 Project Description

The dataset includes various physicochemical features of red wines (such as acidity, sugar, pH, alcohol, etc.) along with their quality scores (ranging from 3 to 8). The goal is to classify each wine sample into its respective quality category using a feedforward neural network (FNN).

Since the target labels are not zero-based, they are encoded using `LabelEncoder` to make them compatible with PyTorch’s `CrossEntropyLoss`.

---

## ✅ Steps for Analysis

### 1. Data Loading
- Load the **Wine Quality (Red Wine)** dataset directly from the UCI Machine Learning Repository.

### 2. Data Preprocessing
- Normalize the features using `StandardScaler` to ensure all input features are on a similar scale.
- Encode the target labels using `LabelEncoder` to map quality scores (e.g., 3–8) to integer class labels (e.g., 0–5).

### 3. Model Architecture
- A **feedforward neural network** with:
  - 1 or 2 hidden layers
  - ReLU activation functions
  - Dropout for regularization to prevent overfitting

### 4. Training
- Loss Function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Includes **class weighting** to handle imbalance in quality label distribution

### 5. Evaluation
- Accuracy measured over training epochs
- Confusion matrix to visualize misclassifications
- Classification report showing precision, recall, and F1-score

### 6. Visualization
- Line plots of training loss and test accuracy
- Heatmap of the confusion matrix to evaluate the prediction quality

## 🧠 Model Architecture

The FNN model is composed of the following layers:

```text
Input: 11 features
→ Fully Connected (128) + ReLU + Dropout
→ Fully Connected (64) + ReLU + Dropout
→ Output Layer (6 classes)
