
# Customer Segmentation using Anomaly Detection Algorithms

This project focuses on detecting anomalies in customer data using multiple anomaly detection algorithms. The goal is to segment customers into normal and anomalous groups, which can help businesses identify outliers or unusual behaviors in their customer base.

## Technologies Used

- **Python**: The primary programming language used for data processing and model training.
- **Google Colab**: An interactive environment used for running Python code and data analysis.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and transformations.
- **Matplotlib**: For creating visualizations of the data.
- **Seaborn**: For creating enhanced visualizations.
- **Scikit-learn**: For machine learning model building, including anomaly detection algorithms like Isolation Forest, Local Outlier Factor, and One-Class SVM.
- **Jupyter Notebook**: An interactive environment for running Python code and data analysis.

## How to Run the Project

### 1. Clone the Repository
Clone the repository using the following command:
    ```bash
    git clone https://github.com/your-username/customer-segmentation-anomaly-detection.git
    cd customer-segmentation-anomaly-detection
    ```

### 2. Install Required Libraries
Install all the required Python libraries using:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Run the Jupyter Notebook
To start the project and run the code, launch Jupyter Notebook:
    ```bash
    jupyter notebook anomaly_detection.ipynb
    ```

## Project Description

### Data Preprocessing:
- **Loading Data**: The dataset is generated with synthetic data, containing features such as `Age`, `Annual Income`, and `Spending Score`. Some data points are marked as anomalies (outliers).
- **Feature Transformation**: The features are selected, and scaling is applied using **StandardScaler** to standardize the data for anomaly detection models.

### Anomaly Detection Algorithms:
This project applies multiple anomaly detection algorithms to detect anomalies (outliers) in the data:
- **Isolation Forest**: A tree-based anomaly detection algorithm that isolates anomalies instead of profiling normal data points.
- **Local Outlier Factor (LOF)**: A density-based method that evaluates the local density deviation of data points with respect to their neighbors.
- **One-Class SVM**: A type of Support Vector Machine that learns the boundary of the normal data and detects outliers as anomalies.

### Steps for Analysis:
1. **Data Loading**: The synthetic dataset is generated using **NumPy**, with normal data and anomalies.
2. **Preprocessing**: The data is preprocessed by separating the features and labels and then scaling the features.
3. **Anomaly Detection**: The anomaly detection algorithms (**Isolation Forest**, **Local Outlier Factor**, and **One-Class SVM**) are applied to the preprocessed data.
4. **Evaluation**: The models are evaluated using various metrics like **Silhouette Score** and **Calinski-Harabasz Index** to assess clustering quality.
5. **Visualization**: The results are visualized using **PCA** for dimensionality reduction and **boxplots** to display feature distributions for normal and anomalous data.
