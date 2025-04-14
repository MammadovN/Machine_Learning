# Customer Segmentation using Clustering Algorithms
This project focuses on using **K-Means**, **Hierarchical Clustering**, and **DBSCAN** algorithms to segment customers based on their demographic and spending behavior. The goal is to classify customers into distinct groups, helping businesses develop targeted marketing strategies and personalized offers.

## Technologies Used

- **Python**: The primary programming language used for data processing and model training.
- **Google Colab**: An interactive environment used for running Python code and data analysis.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and transformations.
- **Matplotlib**: For creating visualizations of the data.
- **Seaborn**: For creating enhanced visualizations.
- **Scikit-learn**: For machine learning model building, including clustering algorithms (K-Means, DBSCAN, Hierarchical).
- **DBSCAN**: A density-based clustering method for identifying arbitrarily shaped clusters.

## How to Run the Project

1. **Clone the repository**:
Clone the repository using the following command:
    ```bash
    git clone https://github.com/your-username/customer-segmentation.git
    cd customer-segmentation


2. **Install Required Libraries**:
Install all the required Python libraries using:
    ```bash
    pip install -r requirements.txt


3. **Run the Jupyter Notebook**:
To start the project and run the code, launch Jupyter Notebook:
    ```bash
    jupyter notebook customer_segmentation.ipynb


## Project Description

### Data Preprocessing:
- **Loading Data**: The dataset (Mall Customer Segmentation) is loaded and cleaned using Pandas.
- **Feature Transformation**: Features such as `Age`, `Annual Income`, and `Spending Score` are selected for segmentation.
- **Scaling**: The features are standardized using **StandardScaler** to ensure they are on the same scale for clustering algorithms.

### Clustering Algorithms Used:
The following unsupervised learning techniques are used to perform customer segmentation:
- **K-Means Clustering**: A popular centroid-based clustering algorithm that assigns each point to the nearest cluster center.
- **Hierarchical Clustering**: Builds a tree of clusters and does not require the number of clusters to be specified in advance.
- **DBSCAN**: A density-based clustering algorithm that can find arbitrarily shaped clusters and also identifies noise points.

### Steps for Analysis:
1. **Data Loading**: The dataset is loaded from a CSV file using Pandas.
2. **Data Scaling**: The data is scaled using **StandardScaler** to standardize the features.
3. **Clustering Algorithms**:
   - **K-Means**: K-Means clustering is applied using the optimal number of clusters determined by the **Elbow Method**.
   - **DBSCAN**: A density-based clustering approach is applied using the **DBSCAN** algorithm.
   - **Hierarchical Clustering**: This method is used to group customers in a hierarchical manner.
4. **Model Evaluation**: The clustering results are evaluated using the **Silhouette Score** and **Calinski-Harabasz Index**.
5. **Visualization**: Results are visualized using **PCA** and **t-SNE** to project the high-dimensional data into 2D space for better visualization of the clusters.
