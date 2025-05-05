# Taxi Trip Duration Prediction

A real-world machine learning project that predicts the duration of New York City taxi trips using historical trip records and geographic features.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Methodology](#methodology)  
- [Results](#results)  
- [Requirements](#requirements)  
- [License](#license)  

---

## Project Overview

This notebook-based project explores and models the “taxi trip duration” problem using NYC taxi data. We perform:

1. **Exploratory Data Analysis** (EDA)  
2. **Feature Engineering** (time features, distance calculations)  
3. **Model Training & Tuning** (Random Forest, XGBoost)  
4. **Evaluation** (RMSE, MAE)  

The goal is to build a regression model that accurately estimates trip durations given pickup/dropoff coordinates and timestamps.

---

## Dataset

We use the publicly available NYC Yellow Taxi Trip Duration dataset (e.g., from Kaggle or the TLC SFTP). Each record includes:

- `pickup_datetime`, `dropoff_datetime`  
- `pickup_longitude`, `pickup_latitude`  
- `dropoff_longitude`, `dropoff_latitude`  
- `passenger_count`  
- `trip_duration` (target in seconds)  

Place your CSV files (e.g. `train.csv`, `test.csv`) in the `data/` folder before running the notebook.

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/MammadovN/Machine_Learning.git
   cd Machine_Learning/projects/06_real-world-apps/taxi-trip-duration-prediction

2. **Create & activate a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

---

## Usage

### Launch Jupyter Notebook

    ```bash
    jupyter notebook
Open taxi_trip_duration_prediction.ipynb and run all cells.
Inspect EDA outputs, model training logs, and evaluation metrics.
Adjust hyperparameters or try different models (e.g., LightGBM, CatBoost).

---

## Methodology

### Data Cleaning
- Remove outliers (e.g., durations > 3 hours, zero coordinates)  
- Handle missing or erroneous values  

### Feature Engineering
- Extract hour, day of week, month from `pickup_datetime`  
- Compute Haversine distance between pickup and dropoff  
- Calculate bearing and Manhattan distance  

### Modeling
- Baseline: Linear Regression  
- Tree-based models: Random Forest, XGBoost  
- Hyperparameter tuning with `GridSearchCV`  

### Evaluation
- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)  
- Feature importance analysis  

---

## Results

| Model             | RMSE (validation) | MAE (validation) |
|-------------------|-------------------|------------------|
| Linear Regression | 720.3 seconds     | 435.7 seconds    |
| Random Forest     | 523.8 seconds     | 298.4 seconds    |
| XGBoost           | **489.2 seconds** | **280.1 seconds** |

_The XGBoost model achieves the lowest error on the validation set and is recommended for production._

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.  
