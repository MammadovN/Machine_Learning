# 🧼 Denoising Autoencoder with Fashion-MNIST

This project demonstrates how to build a **Denoising Autoencoder** using TensorFlow and Keras to remove noise from images in the **Fashion-MNIST** dataset. The model learns to reconstruct clean images from noisy versions, acting as a powerful tool for image denoising and unsupervised feature learning.

---

## 🛠 Technologies Used

- **Python**: Main programming language for model development.
- **TensorFlow & Keras**: Deep learning frameworks used to build and train the autoencoder.
- **NumPy**: For numerical computations.
- **Matplotlib**: For data visualization.
- **Google Colab / Jupyter Notebook**: Interactive environment for running and visualizing experiments.

---

## ▶️ How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/denoising-autoencoder.git
   cd denoising-autoencoder
2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
3. **Run the Notebook: Open the notebook in Jupyter or Google Colab**:
   ```bash
   denoising_autoencoder_fashion_mnist.ipynb

## 📂 Project Description

This project aims to denoise corrupted grayscale images using a symmetric autoencoder architecture. The autoencoder is trained on noisy inputs and learns to generate clean outputs.

We use the **Fashion-MNIST** dataset, which contains grayscale images of fashion items such as sneakers, t-shirts, trousers, etc. Gaussian noise is added to simulate real-world corrupted inputs.

---

## ✅ Steps for Analysis

### 🧪 Data Loading & Preprocessing:
- The Fashion-MNIST dataset is loaded and normalized to [0, 1].
- Input images are reshaped to `(28, 28, 1)`.
- Gaussian noise is added to simulate noisy inputs.

### 🏗️ Autoencoder Architecture:
- A symmetric fully-connected autoencoder is built using the Keras Functional API.
- The encoder compresses the image to a 32-dimensional latent space.
- The decoder reconstructs the image from the latent representation.

### 🏋️ Training:
- The model is trained using `MSE` loss and `Adam` optimizer.
- `EarlyStopping` is used to prevent overfitting based on validation loss.

### 📊 Evaluation:
- Denoised outputs are visualized along with original and noisy inputs.
- Results show that the model successfully removes noise and reconstructs the original fashion items.

---
## 🧠 Model Architecture
Input: 28x28x1
→ Flatten
→ Dense (128, ReLU)
→ Dense (64, ReLU)
→ Dense (32, ReLU) ← Latent space
→ Dense (64, ReLU)
→ Dense (128, ReLU)
→ Dense (784, Sigmoid)
→ Reshape (28x28x1)
