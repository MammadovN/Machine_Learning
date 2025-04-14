# Dimensionality Reduction and Visualization: PCA & t-SNE

This project focuses on using **Principal Component Analysis (PCA)** and **t-Distributed Stochastic Neighbor Embedding (t-SNE)** to reduce the dimensionality of the **Iris dataset** and visualize the results in 2D. The project demonstrates how these unsupervised learning techniques can be applied to gain insights into high-dimensional data.

## Technologies Used

- **Python**: The primary programming language used for data processing and model training.
- **Google Colab**: An interactive environment used for running Python code and data analysis.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and transformations.
- **Matplotlib**: For creating visualizations of the data.
- **Scikit-learn**: For machine learning model building, including PCA, t-SNE, and dataset loading.

## How to Run the Project

### 1. Clone the Repository
**Clone the repository using the following command**:
    ```bash
    git clone https://github.com/your-username/pca-tsne-visualization.git
    cd pca-tsne-visualization
### 2. Install Required Libraries
**Install all the required Python libraries using**:
    ```bash
    pip install -r requirements.txt
### 3. Run the Jupyter Notebook
**To start the project and run the code, launch Jupyter Notebook**:
    ```bash
    jupyter notebook pca_tsne_visualization.ipynb

## Project Description

### Data Preprocessing:
- **Loading Data**: The dataset (Iris) is loaded using Scikit-learn's `load_iris` function.
- **Feature Transformation**: PCA is applied to reduce the dataset into two principal components, while t-SNE is used to further visualize the data by projecting it into 2D space.
- **Scaling**: The features are automatically handled by PCA and t-SNE, but standardization and normalization techniques can be applied for different datasets.

### Dimensionality Reduction Techniques Used:
The following unsupervised learning techniques are used:
- **Principal Component Analysis (PCA)**: Reduces the high-dimensional data to its two most important components based on the variance in the data.
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: Reduces the data to two dimensions, focusing on preserving local relationships and structures in the dataset.

### Steps for Analysis:
1. **Data Loading**: The Iris dataset is loaded using Scikit-learn's `load_iris` function.
2. **PCA Application**: PCA is applied to reduce the dimensionality of the data from 4 dimensions to 2. The dataset is then visualized in 2D space.
3. **t-SNE Application**: t-SNE is applied to the same dataset, projecting it into a 2D space to reveal more meaningful clusters.
4. **Visualization**: Both PCA and t-SNE visualizations are displayed using Matplotlib. Data points are color-coded based on their target class (species of iris flowers).

