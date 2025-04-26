# 🧠 Image Classification using Convolutional Neural Networks (CNN)

This project demonstrates how to build and train a **Convolutional Neural Network (CNN)** using PyTorch to classify images from the **CIFAR-10 dataset**. The goal is to develop a model that can accurately predict the category of input images across 10 distinct classes such as airplanes, cars, cats, and more.

---

## 🛠 Technologies Used

- **Python**: Core programming language used for model development and experimentation.
- **Google Colab / Jupyter Notebook**: Interactive environment used for code execution and visualization.
- **PyTorch**: Deep learning framework for building and training neural networks.
- **torchvision**: Provides CIFAR-10 dataset and image transformation utilities.
- **matplotlib & seaborn**: For data visualization and analysis.
- **scikit-learn**: For classification metrics like confusion matrix and classification report.
- **NumPy**: For numerical computations.

---

## ▶️ How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/cnn-image-classifier.git
   cd cnn-image-classifier
2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
3. **Run the Notebook: Launch the Jupyter Notebook or Google Colab and open**:
   ```bash
   cnn_image_classifier_cifar10.ipynb

## 📂 Project Description

This project demonstrates how to perform image classification on the CIFAR-10 dataset using a custom-built Convolutional Neural Network (CNN) implemented in PyTorch.

The CIFAR-10 dataset contains 60,000 32x32 color images labeled across 10 categories:
- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

The goal is to train a deep learning model that can accurately identify the class of a given image based on learned features through convolutional layers.

---

## ✅ Steps for Analysis

1. **Data Loading**: The CIFAR-10 dataset is downloaded and loaded using `torchvision.datasets`.
2. **Data Preprocessing**: All images are normalized to have pixel values between -1 and 1, improving model convergence.
3. **Exploratory Visualization**: A batch of sample images is visualized along with their class labels to understand the data.
4. **CNN Model Building**:
   - A simple CNN is constructed with two convolutional layers followed by max-pooling and dropout.
   - Fully connected layers are added to map features to the output classes.
5. **Model Training**:
   - The model is trained using CrossEntropyLoss and the Adam optimizer.
   - Training loss and test accuracy are tracked over 10 epochs.
6. **Evaluation**:
   - The model is evaluated using a confusion matrix, classification report (precision, recall, F1-score), and accuracy metric.
   - Loss and accuracy curves are plotted to visualize learning.
7. **Sample Predictions**:
   - The model makes predictions on a batch of test images, which are visualized alongside true labels.
8. **Optional Improvements**:
   - Techniques such as data augmentation, batch normalization, or deeper networks can be explored to boost performance.

---

## 🧠 Model Architecture

The CNN model is composed of the following layers:

```text
Input: 32x32x3
→ Conv2D (32 filters, 3x3) → ReLU → MaxPooling (2x2)
→ Conv2D (64 filters, 3x3) → ReLU → MaxPooling (2x2)
→ Dropout (0.25)
→ Flatten
→ Fully Connected (512 units) → ReLU → Dropout (0.25)
→ Output Layer (10 units)
