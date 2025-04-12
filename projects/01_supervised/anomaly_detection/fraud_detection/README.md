# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning. The goal is to predict whether a transaction is fraudulent or not based on various features such as transaction time, amount, and anonymized user data. The dataset contains real credit card transactions, with class labels indicating whether the transaction was fraudulent (1) or non-fraudulent (0).

## Technologies Used

- **Python**: The primary programming language used for data processing and model training.
- **Google Colab**: An interactive environment used for running Python code and machine learning models.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and transformations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning model building, evaluation, and hyperparameter tuning.
- **XGBoost**: For gradient boosting model implementation.
- **imbalanced-learn**: For oversampling techniques such as SMOTE.

## How to Run the Project

1. **Clone the Repository**: Clone the repository using the following command:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
2. **Install Required Libraries**: Install all the required Python libraries using:
   ```bash
   pip install -r requirements.txt
3. **Run the Jupyter Notebook**: Open the project’s Jupyter notebook and run the analysis:
   ```bash
   jupyter notebook credit_card_fraud_detection.ipynb

## Project Description

### Data Preprocessing:
- **Loading Data**: The dataset is loaded from a CSV file using **pandas**.
  
- **Feature Engineering**:
  - **Log Transformation** is applied to the `Amount` feature to handle its skewed distribution.
  - The `Time` and `Amount` features are scaled using **StandardScaler** to standardize their values.

- **Handling Imbalanced Data**: Since fraudulent transactions are much rarer than non-fraudulent ones, the data is balanced using **SMOTE (Synthetic Minority Over-sampling Technique)** to create synthetic samples for the minority class.

- **Data Splitting**: The dataset is split into **70% training** and **30% testing** sets.

### Models Used:
The following machine learning models are trained and evaluated:
- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Support Vector Machine (SVM)**
- **XGBoost Classifier**
- **K-Nearest Neighbors (KNN)**

### Model Evaluation:
The models are evaluated based on the following metrics:
- **Accuracy**: The percentage of correct predictions out of all predictions.
- **Precision**: The percentage of true positive predictions out of all positive predictions.
- **Recall**: The percentage of true positive predictions out of all actual positive instances.
- **F1-Score**: The harmonic mean of Precision and Recall.
- **ROC-AUC**: The area under the ROC curve, indicating the tradeoff between true positive rate and false positive rate.
- **PR-AUC**: The area under the Precision-Recall curve, particularly important for imbalanced datasets.

### Hyperparameter Tuning:
- **Random Forest Classifier** is optimized using **GridSearchCV** to find the best combination of hyperparameters, such as the number of estimators and maximum depth.

### Steps for Analysis:
1. **Data Loading**: The dataset is loaded using **pandas** and analyzed for missing values.
2. **Visualization**: The distribution of fraudulent vs non-fraudulent transactions is visualized using **Seaborn**'s `histplot` and `boxplot`.
3. **Log Transformation**: The `Amount` column is log-transformed to reduce skewness.
4. **Scaling**: The `Time` and transformed `Amount` columns are scaled using **StandardScaler**.
5. **Model Training**: Various machine learning models are trained on the balanced data using **Scikit-learn**.
6. **Model Evaluation**: The models are evaluated based on various metrics, including Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
7. **Hyperparameter Optimization**: **GridSearchCV** is used to tune the hyperparameters of the **Random Forest** model for improved performance.
