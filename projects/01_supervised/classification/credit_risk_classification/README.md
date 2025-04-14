# German Credit Risk Classification

This project involves building a machine learning model to predict whether a person is classified as having "good" or "bad" credit risk using the **German Credit** dataset. The dataset includes various features like age, job, credit amount, and more to determine creditworthiness. The aim is to develop a model that can predict credit risk based on these features.

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

### 1. Clone the Repository
**Clone the repository using the following command**:
    ```bash
    git clone https://github.com/your-username/german-credit-risk-classification.git
    cd german-credit-risk-classification
### 2. Install Required Libraries
**Install all the required Python libraries using**:
    ```bash
    pip install -r requirements.txt
### 3. Run the Jupyter Notebook
**To start the project and run the code, launch Jupyter Notebook**:
    ```bash
    jupyter notebook german_credit_risk_classification.ipynb
## Project Description

### Data Preprocessing:
- **Loading Data**: The dataset is loaded from a CSV file using pandas.
- **Feature Engineering**: Log transformations and scaling are applied to handle skewness and normalize the features.
- **Handling Missing Data**: Some columns simulate real-world scenarios with missing values that are handled by imputation.
- **Handling Imbalanced Data**: Since "bad" credit risk data is less frequent, SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the classes by creating synthetic samples for the minority class.
- **Data Splitting**: The dataset is split into 70% training and 30% testing sets using `train_test_split`.

### Models Used:
The following machine learning models are trained and evaluated:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)
- XGBoost Classifier
- K-Nearest Neighbors (KNN)

### Model Evaluation:
The models are evaluated based on the following metrics:
- **Accuracy**: The percentage of correct predictions out of all predictions.
- **Precision**: The percentage of true positive predictions out of all positive predictions.
- **Recall**: The percentage of true positive predictions out of all actual positive instances.
- **F1-Score**: The harmonic mean of Precision and Recall.
- **ROC-AUC**: The area under the ROC curve, indicating the tradeoff between true positive rate and false positive rate.
- **PR-AUC**: The area under the Precision-Recall curve, particularly important for imbalanced datasets.

### Hyperparameter Tuning:
- **Gradient Boosting** is optimized using **GridSearchCV** to find the best combination of hyperparameters, such as the number of estimators and maximum depth.

### Steps for Analysis:
1. **Data Loading**: The dataset is loaded using pandas and analyzed for missing values.
2. **Visualization**: The distribution of "good" vs "bad" credit risk classifications is visualized using Seaborn’s `countplot`.
3. **Log Transformation**: The `Amount` column is log-transformed to reduce skewness and improve model performance.
4. **Scaling**: The `Age` and `Credit Amount` columns are scaled using StandardScaler to standardize their values.
5. **Data Splitting**: The dataset is split into training and testing sets using a 70-30 ratio.
6. **Model Training**: Various machine learning models are trained on the balanced dataset using Scikit-learn.
7. **Model Evaluation**: The models are evaluated using metrics such as Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
8. **Hyperparameter Optimization**: The **Gradient Boosting** model is optimized using GridSearchCV for improved performance.
