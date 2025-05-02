# 📈 Stock Price Prediction

An end-to-end time-series forecasting project that builds and evaluates models to predict future stock prices. Includes data ingestion, feature engineering, model training (classical and deep learning), evaluation, and a simple backtest/demo.

---

## 🛠 Technologies Used

- **Python 3.8+**: Core scripting language  
- **pandas & NumPy**: Data manipulation  
- **scikit-learn**: Preprocessing, classical models, metrics  
- **statsmodels**: ARIMA/SARIMA forecasting  
- **TensorFlow 2 / PyTorch**: LSTM or Transformer-based models  
- **matplotlib**: Visualization  
- **yfinance**: Download historical stock data  

---

## ▶️ How to Run the Project

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/stock-price-prediction.git
   cd stock-price-prediction

2. **Install requirements**
   ```bash
   pip install -r requirements.txt

3. **Run the notebook**
   ```bash
   jupyter notebook notebooks/stock-price-prediction.ipynb

---

## 📂 Project Description

This project is structured into three main stages:

1. **Baseline Models**  
   - **ARIMA Forecasting**: Fit an ARIMA(p,d,q) model on historical closing prices (`statsmodels`).  
   - **Linear Regression**: Train a regression model on lagged and technical features (`scikit-learn`).  
   - **Evaluation**: Compute MAE and RMSE on the test split to quantify forecasting error.

2. **Deep Learning Model**  
   - **LSTM Network**: Build a two-layer LSTM with Dropout for sequential modeling (`TensorFlow 2`).  
   - **Data Preparation**: Scale features (Close, MA_10, MA_50, Return, RSI_14), create sliding windows of length 20.  
   - **Training**: Use early stopping on validation loss, monitor MSE during training.

3. **Inference & Deployment**  
   - **Model Saving**: Persist trained ARIMA, Linear Regression, and LSTM models to disk.  
   - **`predict_next()` API**: Load models and scaler, take the most recent window to forecast the next-day closing price.

---

## ✅ Steps for Analysis

1. **Data Loading**  
   - Download daily OHLCV data for AAPL from 2015-01-01 to 2025-01-01 via `yfinance`.  
   - Select only the “Close” column and drop any missing values.  

2. **Preprocessing & Feature Engineering**  
   - Compute 10-day and 50-day moving averages on the Close price.  
   - Calculate daily returns as percentage change.  
   - Derive a 14-period RSI indicator.  
   - Remove all rows with NaNs resulting from rolling/window calculations.  

3. **Model Training**  
   - **ARIMA**: Fit an ARIMA(5,1,0) model on the training Close series.  
   - **Linear Regression**: Train on lagged Close and engineered features.  
   - **LSTM**: Build a sequence model, train with validation split and early stopping (restore best weights).  

4. **Evaluation**  
   - Compute MAE and RMSE for each model’s predictions on the test set.  
   - Visualize actual vs. predicted price series to compare performance.  

5. **Model Saving & Inference**  
   - Persist the Linear Regression model, LSTM model file, and any scalers to disk.  
   - Load saved artifacts and run a helper function to forecast the next trading day’s price.  


---

## 🧠 Model Architecture

```text
ARIMA Forecasting
└─ Input: historical series (Close)
   → ARIMA(p,d,q) fit
   → forecast(steps=H)
   → Output: next-day price prediction

Linear Regression on Lagged Features
└─ Input: [Close_t-1, …, Close_t-k, MA_10, RSI_14, …]
   → StandardScaler → LinearRegression
   → Output: next-day price

LSTM Sequential Model
└─ Input: sequence of feature vectors (seq_length × n_features)
   → LSTM (50 units, return_sequences=True)
   → Dropout(0.2)
   → LSTM (50 units)
   → Dropout(0.2)
   → Dense(1)
   → Output: next-day closing price

