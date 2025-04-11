# Fashion MNIST Model Comparison

This project focuses on training and evaluating various machine learning models on the **Fashion MNIST** dataset. It includes models such as traditional machine learning classifiers, like **K-Nearest Neighbors**, **Decision Tree**, **Random Forest**, **Gradient Boosting**, and **LightGBM**, as well as a **Deep Neural Network (ANN)** using Keras. The goal is to compare the performance of these models based on accuracy, precision, recall, F1-score, and training time.

## Technologies Used

- **Python**: The main programming language.
- **Scikit-learn**: For machine learning models, preprocessing, and evaluation.
- **TensorFlow/Keras**: For building and training the artificial neural network (ANN).
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **LightGBM**: For gradient boosting machine learning model.
- **Joblib**: For saving machine learning models.
- **PCA (Principal Component Analysis)**: For dimensionality reduction.

## Dataset

The **Fashion MNIST** dataset is a collection of 60,000 training images and 10,000 testing images of 28x28 grayscale images of 10 fashion categories. These categories are:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

The dataset is loaded from `tensorflow.keras.datasets.fashion_mnist`.

## How to Run the Project

### 1. Clone the Repository
Clone the repository using the following command:
    ```bash
    git clone https://github.com/your-username/your-repository-name.git

### 2. Install Required Libraries
Install all the required Python libraries using:
    ```bash
    pip install -r requirements.txt
### 3. Run the Jupyter Notebook
Open the project’s Jupyter notebook and run the analysis:
    ```bash
    jupyter notebook fashion_mnist_model_comparison.ipynb
## Project Description

### Data Preprocessing:

- **Normalization**: The pixel values of the images are normalized by dividing by 255.0 to bring the values between 0 and 1.

- **Flattening**: The 28x28 images are flattened into 1D arrays of 784 pixels to be used as input for machine learning models.

- **PCA (Principal Component Analysis)**: PCA is applied to reduce the dimensionality of the dataset while preserving 95% of the variance.

### Models Used:
The following machine learning models were trained and evaluated:

- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **LightGBM Classifier**
- **Artificial Neural Network (ANN) using TensorFlow/Keras**

### Model Evaluation:
The models are evaluated based on the following metrics:

- **Accuracy**: The percentage of correct predictions.
- **F1-Score**: Macro average F1-Score for multi-class classification.
- **Classification Report**: Precision, Recall, and F1-score for each class.
- **Confusion Matrix**: To show the number of correct and incorrect predictions for each class.

### Feature Importance:
For models like **Decision Tree**, **Random Forest**, **Gradient Boosting**, and **LightGBM**, feature importance is calculated and visualized to understand which features (PCA components) contribute the most to the model's predictions.

### Model Saving and Prediction:
- **Joblib** is used to save traditional machine learning models such as **KNN**, **Decision Tree**, and **Random Forest**.
- **Keras** is used to save the **ANN** model.

### Results:
A comparison of all models' performance is provided, including their accuracy, precision, recall, F1-score, and training time. All results are stored in a CSV file, and confusion matrices for each model are visualized.

### Steps for Analysis:

1. **Data Loading**: The Fashion MNIST dataset is loaded using TensorFlow's `fashion_mnist.load_data()`.
2. **Data Preprocessing**: The images are normalized and flattened. PCA is applied for dimensionality reduction.
3. **Model Training**: The models are trained using **Scikit-learn** and **Keras**.
4. **Model Evaluation**: The models' performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrix.
5. **Feature Importance**: For decision-tree-based models, the feature importance is calculated and visualized.
6. **Model Saving**: The best-performing models are saved using **joblib** (for traditional ML models) and **Keras** (for ANN).

## Files

- **fashion_mnist_model_comparison.ipynb**: The Jupyter notebook containing the full analysis, model training, and evaluation code.
- **fashion_mnist_results.csv**: A CSV file containing the comparison of model performance (accuracy, precision, recall, F1-score, and training time).
- **knn_fashion_mnist.joblib**: The saved **KNN** model.
- **decision_tree_fashion_mnist.joblib**: The saved **Decision Tree** model.
- **random_forest_fashion_mnist.joblib**: The saved **Random Forest** model.
- **gradient_boosting_fashion_mnist.joblib**: The saved **Gradient Boosting** model.
- **lightgbm_fashion_mnist.joblib**: The saved **LightGBM** model.
- **ann_fashion_mnist.h5**: The saved **Artificial Neural Network (ANN)** model.
