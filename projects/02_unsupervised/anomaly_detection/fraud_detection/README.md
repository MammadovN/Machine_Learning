# Anomaly Detection with Isolation Forest, LOF, and One-Class SVM

This project focuses on detecting anomalies in data using multiple anomaly detection algorithms. The goal is to identify outliers or unusual behaviors in the dataset using three popular techniques: **Isolation Forest**, **Local Outlier Factor (LOF)**, and **One-Class SVM**.

## Technologies Used

- **Python**: The primary programming language used for data processing and model training.
- **Google Colab**: An interactive environment used for running Python code and data analysis.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and transformations.
- **Matplotlib**: For creating visualizations of the data.
- **Seaborn**: For creating enhanced visualizations.
- **Scikit-learn**: For machine learning model building, including anomaly detection algorithms (Isolation Forest, LOF, One-Class SVM).

## How to Run the Project

### 1. Clone the Repository
Clone the repository using the following command:
    ```bash
    git clone https://github.com/your-username/anomaly-detection.git
    cd anomaly-detection
 
### 2. Install Required Libraries
Install all the required Python libraries using:
    ```bash
    pip install -r requirements.txt
  

### 3. Run the Jupyter Notebook
To start the project and run the code, launch Jupyter Notebook:
    ```bash
    jupyter notebook anomaly_detection.ipynb


## Project Description

### Data Preprocessing:
- **Loading Data**: A synthetic dataset is generated with features such as `Age`, `Annual Income`, and `Spending Score`. Some data points are intentionally introduced as anomalies (outliers).
- **Feature Transformation**: The features are selected and scaled using **StandardScaler** to standardize the data for anomaly detection models.

### Anomaly Detection Algorithms:
This project applies multiple anomaly detection algorithms to identify anomalies in the data:
- **Isolation Forest**: A tree-based anomaly detection method that isolates anomalies instead of profiling normal data points.
- **Local Outlier Factor (LOF)**: A density-based anomaly detection algorithm that evaluates the local density of data points compared to their neighbors.
- **One-Class SVM**: A support vector machine method that learns the boundary of normal data and identifies outliers as anomalies.

### Steps for Analysis:
1. **Data Loading**: The synthetic dataset is generated using **NumPy**, consisting of normal data and anomalies.
2. **Data Preprocessing**: The data is preprocessed by separating the features and labels, followed by scaling the features using **StandardScaler**.
3. **Anomaly Detection**: The anomaly detection algorithms (**Isolation Forest**, **LOF**, and **One-Class SVM**) are applied to the data.
4. **Model Evaluation**: The performance of the models is evaluated using metrics like **Silhouette Score** and **Calinski-Harabasz Index**.
5. **Visualization**: The results are visualized using **PCA** for dimensionality reduction and **boxplots** to display feature distributions for normal and anomalous data.

