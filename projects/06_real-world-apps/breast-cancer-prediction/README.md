# 🧬 Breast Cancer Gene Expression & Survival Prediction

This project analyzes breast cancer gene expression and clinical data to build predictive models for survival outcomes using statistical and machine learning methods.

---

## 📂 Project Structure

```
disease_prediction_gene_expression/
│
├── notebooks/ # Colab Notebooks for exploration & modeling
├── models/ # Saved models
└── app/ # App interface or dashboard
```


---

## 📊 Dataset

- **Source:** [Kaggle - METABRIC Breast Cancer Dataset](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric)
- **Files Used:**
  - `METABRIC_RNA_Mutation.csv` – Contains gene expression, survival time, and clinical attributes.

---

## 🔬 Analysis Overview

### 1. **Exploratory Data Analysis (EDA)**
- Missing values inspection
- Summary statistics
- Categorical feature distributions
- Correlation between features and death outcome

### 2. **Survival Analysis**
- Kaplan-Meier estimation of survival functions
- Group comparisons (e.g., ER status)
- Cox Proportional Hazards regression for multivariate survival modeling

### 3. **Dimensionality Reduction**
- PCA applied on gene expression data
- Visualization colored by clinical labels (e.g., Pam50 subtype)

### 4. **Feature Engineering**
- Categorical encoding (one-hot)
- Scaling gene expression features
- Merging clinical + gene expression features

---

## 🤖 Machine Learning Models

### Random Forest Classifier
- Trained on combined clinical + gene expression features
- Evaluated using:
  - Confusion Matrix
  - Classification Report
  - Accuracy
  - Feature Importance Plot (Top 20)

### XGBoost Classifier
- Tuned with:
  - `n_estimators=100`
  - `max_depth=5`
  - `learning_rate=0.1`
- Evaluation metrics:
  - Accuracy
  - ROC-AUC Score
  - ROC Curve visualization
  - Top 20 Feature Importance

---

## 📈 Key Visualizations

- Kaplan-Meier survival curves
- PCA scatter plots
- Stacked bar plots (Clinical vs. Death Outcome)
- Feature importance rankings
- ROC Curve

---

## ⚙️ How to Run

1. **Mount Google Drive (for Colab):**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')

2. **Download Dataset:**
    ```python
    import kagglehub
    kagglehub.dataset_download('raghadalharbi/breast-cancer-gene-expression-profiles-metabric')

3. **Install Dependencies:**
    ```python
    pip install lifelines xgboost

4. **Run Colab Notebooks in notebooks/ Folder**

---

## 🧪 Technologies Used

- Python (`pandas`, `numpy`, `seaborn`, `matplotlib`)
- Lifelines (Kaplan-Meier, Cox Regression)
- Scikit-learn
- XGBoost
- PCA for dimensionality reduction
- Google Colab for execution

---

## 📜 License

This project is for educational and research purposes only. Always cite the data source when using the METABRIC dataset.
