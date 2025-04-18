# Machine Learning Portfolio

Welcome to my Machine Learning portfolio! This repository contains a collection of projects showcasing my skills and experience in machine learning, data science, and deep learning. Each project demonstrates different techniques and models, and I am continuously adding new projects to improve my skills.

## Table of Contents
- [Overview](#overview)
- [Skills & Tools](#skills--tools)
- [Projects](#projects)
- [Installation Instructions](#installation-instructions)
- [Contact](#contact)

## Overview

This repository contains my personal Machine Learning projects that I have worked on over the past few months/years. They range from simple classification tasks to advanced deep learning and reinforcement learning applications. The projects are organized into categories based on the machine learning techniques used, such as supervised learning, unsupervised learning, deep learning, natural language processing (NLP), and more.

## Skills & Tools

Here are the main tools and technologies that I used throughout my projects:

- **Programming Languages**: Python
- **Libraries & Frameworks**: 
    - Scikit-learn
    - Pandas
    - Numpy
    - Matplotlib / Seaborn
    - TensorFlow / Keras
    - PyTorch
    - NLTK / SpaCy
    - OpenCV
- **Other Tools**: Jupyter Notebooks, Google Colab, Git, GitHub

## Projects

### Supervised Learning
- **[Credit Card Fraud Detection](projects/01_supervised/anomaly_detection/fraud_detection/README.md)**:Fraudulent credit card transactions based on features such as transaction time, amount, and anonymized user data. Key considerations include handling class imbalance with **SMOTE**, applying log transformation and scaling to features like `Amount`, and using models such as **Logistic Regression**, **Random Forest**, and **XGBoost**, which are evaluated based on metrics like **ROC-AUC** and **PR-AUC** due to the imbalanced dataset.
- **[Polynomical Curve Fitting](projects/01_supervised/regression/nonlinear_regression/polynomical_curve_fitting/README.md)**: A polynomial regression model is fitted using Random Forest after transforming the input features into polynomial terms for non-linear curve fitting.
- **[House Price Prediction](projects/01_supervised/regression/linear_regression/housing_price_prediction/README.md)**: Predicting house prices using linear regression.
- **[German Credit Risk Classification](projects/01_supervised/classification/credit_risk_classification/README.md)**: Predicting customer churn for a telecommunications company using various machine learning models.
- **[Customer Churn Prediction](projects/01_supervised/classification/customer_churn_prediction/README.md)**: Predicting credit risk for a financial institution using various machine learning models, with missing values and a mix of categorical (e.g., job, purpose) and numerical (e.g., age, credit amount) features.
- **[Text Classification Model for News Categories](projects/01_supervised/classification/text_classification/README.md)**: Building a text classification model to categorize news articles into predefined categories using machine learning and NLP techniques.
- **[Fashion MNIST Model Comparison](projects/01_supervised/classification/image_classification/README.md)**: Comparing different machine learning models on the Fashion MNIST dataset, including KNN, Decision Tree, Random Forest, Gradient Boosting, LightGBM, and Artificial Neural Networks (ANN).
### Unsupervised Learning
- **[Anomaly Detection](projects/02_unsupervised/anomaly_detection/fraud_detection/README.md)**: Detecting anomalies in data using multiple anomaly detection algorithms. The goal is to identify outliers or unusual behaviors in the dataset using three popular techniques: **Isolation Forest**, **Local Outlier Factor (LOF)**, and **One-Class SVM**.
- **[Customer Segmentation Clustering](projects/02_unsupervised/clustering/customer_segmentation/README.md)**: Using **K-Means**, **Hierarchical Clustering**, and **DBSCAN** algorithms to segment customers based on their demographic and spending behavior, helping businesses develop targeted marketing strategies and personalized offers.
- **[PCA & t-SNE Visualization](projects/02_unsupervised/dimensionality_reduction/pca_tsne_visualization/README.md)**: Using **Principal Component Analysis (PCA)** and **t-Distributed Stochastic Neighbor Embedding (t-SNE)** to reduce the dimensionality of the **Iris dataset** and visualize the results in 2D.

### Deep Learning
- **[Wine Quality Classifier - FNN](projects/03_deep_learning/feedforward_neural_networks/README.md)**: Wine Quality classification using Feedforward Neural Networks (FNN).
- **[Cifar10 Image Classifier - CNN](projects/03_deep_learning/convolutional_neural_networks/README.md)**: Image classification using Convolutional Neural Networks (CNN).


### NLP
- **[Chatbot - SEQ2SEQ](projects/04_natural_language_processing/chatbot/README.md)**: Chatbot development using Seq2Seq for context-aware responses.
- **[Sentiment Analysis - BERT](projects/04_natural_language_processing/sentiment-analysis/README.md)**: Advanced sentiment analysis using BERT.
- **[Text Summarization  - TRANSFORMER ](projects/04_natural_language_processing/text-summarization/README.md)**: Text summarization using Transformer models T5.

### Time-Series
- **[Stock Price Prediction - LSTM](projects/05_time-series/stock-price-prediction/README.md)**: Time-series forecasting using Long Short-Term Memory (LSTM) networks.

### Real World Applications
- **[Movie Recommendation System](projects/06_real-world-apps/movie-recommendation-system/README.md)**: Building a recommendation system for movies.
- **[Taxi Trip Duration Prediction](projects/06_real-world-apps/taxi-trip-duration-prediction/README.md)**: Predicting taxi trip durations based on historical data.

## Installation Instructions

To run the code in this repository, please follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/machine-learning-portfolio.git
   cd machine-learning-portfolio

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
3. Run the notebooks or scripts for each project.

## Contact
If you have any questions or would like to collaborate on a project, feel free to contact me!

### Email: **nadir.memmedov680@gmail.com**

### LinkedIn: **www.linkedin.com/in/nadir-mammadov-6b3729219**
