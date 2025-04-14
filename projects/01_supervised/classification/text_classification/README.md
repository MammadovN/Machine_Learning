# Text Classification Model for News Categories

This project focuses on building a text classification model to categorize news articles into predefined categories such as `sci.med`, `comp.graphics`, and `rec.sport.baseball`. The model uses machine learning algorithms and natural language processing (NLP) techniques to classify the text data.

## Technologies Used

- **Python**: The main programming language.
- **Scikit-learn**: For machine learning model building, evaluation, and hyperparameter tuning.
- **NLTK**: For text preprocessing, including stopwords removal and stemming.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **XGBoost**: For advanced boosting model.
- **Joblib & Pickle**: For saving and loading models and vectorizers.

## How to Run the Project

### 1. Clone the Repository
 **Clone the repository using the following command**:
    ```bash
    git clone https://github.com/your-username/your-repository-name.git

### 2. Install Required Libraries
 **Install all the required Python libraries using**:
    ```bash
    pip install -r requirements.txt

### 3. Run the Jupyter Notebook
  **Open the project’s Jupyter notebook and run the analysis**:
    ```bash
    jupyter notebook text_classification.ipynb

## Project Description

### Data Preprocessing:

- **Text Cleaning**: Text data is cleaned by converting to lowercase, removing punctuation, removing stopwords, and applying stemming.
  
- **TF-IDF Vectorization**: The cleaned text is converted into numerical features using TF-IDF vectorization.

- **Data Splitting**: The dataset is split into training and testing sets with a 70/30 ratio.

### Models Used:
The following machine learning models were trained and evaluated:

- **Naive Bayes**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **XGBoost**

### Model Evaluation:
The models are evaluated based on the following metrics:

- **Accuracy**: The percentage of correct predictions.
- **F1-Score**: Macro average F1-Score for multi-class classification.
- **Classification Report**: Precision, Recall, and F1-score for each class.
- **Confusion Matrix**: To show the number of correct and incorrect predictions.

### Hyperparameter Tuning:

- **SVM**: The Support Vector Machine (SVM) model was optimized using **GridSearchCV** to find the best hyperparameters.

### Model Saving and Prediction:

- **Joblib & Pickle** are used to save the best-performing model and the TF-IDF vectorizer.
- The saved model and vectorizer can be loaded to make predictions on new data.

## Steps for Analysis:

1. **Data Loading**: The dataset is loaded using the `fetch_20newsgroups` dataset from scikit-learn and analyzed for any missing values.
2. **Data Preprocessing**: The text data is preprocessed by cleaning and tokenizing the text, removing stopwords, and applying stemming.
3. **Model Training**: Multiple machine learning models are trained using **scikit-learn** and **XGBoost**.
4. **Model Evaluation**: The models' performance is evaluated using various metrics, including accuracy and F1-score.
5. **Hyperparameter Tuning**: Hyperparameters for **SVM** are optimized using **GridSearchCV**.
6. **Model Saving**: The best model is saved using **joblib**, and the TF-IDF vectorizer is saved using **pickle**.

## Files

- **text_classification.ipynb**: The Jupyter notebook containing the full analysis and model training code.
- **best_text_classifier.pkl**: The trained model saved for future use in making predictions.
- **tfidf_vectorizer.pkl**: The TF-IDF vectorizer used to transform text data before classification.
