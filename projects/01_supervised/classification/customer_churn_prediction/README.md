# Customer Churn Prediction

This project focuses on predicting customer churn for a telecommunications company. It uses various machine learning models to classify whether a customer will churn (leave) or stay with the service. The dataset includes various customer features such as subscription type, usage of services, and customer support interactions.

## Technologies Used

- **Python**: The main programming language.
- **Google Colab**: An interactive environment for running the code and working with Jupyter notebooks.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning model building, evaluation, and hyperparameter tuning.

## How to Run the Project

1. **Clone the Repository**: Clone the repository using the following command:
   ```bash
   git clone https://github.com/MammadovN/your-repository-name.git

2. **Install Required Libraries**: Install all the required Python libraries using:
   ```bash
   pip install -r requirements.txt

3. **Run the Jupyter Notebook**: Open the project’s Jupyter notebook and run the analysis:

   ```bash
   jupyter notebook churn_prediction.ipynb
## Project Description

### Data Preprocessing:
- The dataset is loaded from a CSV file.
- Categorical columns are encoded using **Label Encoding**.
- The target variable (**Churn**) is mapped to binary values (1 for churned, 0 for not churned).
- Data is split into training and testing sets using a **70/30 split**.

### Models Used:
The following machine learning models were trained and evaluated:
- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Support Vector Machine (SVM)**

### Model Evaluation:
The models are evaluated based on:
- **Accuracy**: The percentage of correct predictions.
- **ROC AUC**: The area under the Receiver Operating Characteristic curve.
- **Classification Report**: Precision, Recall, and F1-score for each class.
- **Confusion Matrix**: To show the number of correct and incorrect predictions.

### Hyperparameter Tuning:
- **Random Forest Classifier** was optimized using **GridSearchCV** to find the best hyperparameters for the model.

## Steps for Analysis:

1. **Data Loading**: The dataset is loaded using **pandas** and analyzed for any missing values.
2. **Visualization**: The distribution of the target variable (**Churn**) is visualized using **Seaborn**'s `countplot`.
3. **Model Training**: The models are trained using **Scikit-learn**, and performance metrics are calculated.
4. **Hyperparameter Optimization**: The best parameters for **Random Forest** are found using **GridSearchCV**.
