# 🌡️ TempForecastLSTM: Daily Minimum Temperature Forecasting with LSTM (PyTorch)

This project trains an **LSTM-based RNN** to forecast future **daily minimum temperatures (°C)** using the past 30 days of historical data from the publicly available Daily Minimum Temperatures dataset.

---

## 🛠 Technologies Used

- **Python** – Programming language  
- **PyTorch** – Deep-learning framework  
- **pandas & NumPy** – Data manipulation  
- **scikit-learn** – `StandardScaler` for normalization  
- **torch.utils.data** – `TensorDataset` & `DataLoader`  
- **tqdm** – Progress bars  
- **matplotlib** – Training & forecasting plots  

---

## ▶️ How to Run the Project

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/temp_forecast_lstm.git
   cd TempForecastLSTM
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run the training notebook/script**
   ```bash
   jupyter notebook temperature_forecasting.ipynb

---

## 📂 Project Description  

The pipeline converts raw daily temperature readings into supervised sequences, feeds them into a multi-layer LSTM, and outputs a forecasted temperature value.

1. **Download & split data** → train / validation / test  
2. **Normalize** using `StandardScaler` (fit on train only)  
3. **Windowing** – turn series into input/output pairs of length 30  
4. **Create DataLoaders** – batch size 256, shuffle train  
5. **Define Model** – 2-layer LSTM (input_size=1, hidden_dim=128, dropout=0.1) → Linear head  
6. **Train** with `MSELoss` + Adam, checkpointing best model by validation RMSE  
7. **Evaluate** RMSE on hold-out sets  
8. **Forecast** future temperatures (e.g., next 14 days) via recursive prediction  

---

## ✅ Steps for Analysis  

| Step               | Description                                                                     |
|--------------------|---------------------------------------------------------------------------------|
| **Data Loading**   | Fetch CSV from URL, parse `Date`, rename `Temp` → `temp_c`, sort by date.      |
| **Normalization**  | Fit `StandardScaler` on train temperatures → apply to full series.             |
| **Windowing**      | `windows()` function builds `(X, y)` arrays of shape `(n_samples, 30)` and `(n,)`. |
| **DataLoader**     | Wrap `(X, y)` in `TensorDataset` → `DataLoader` (batch_size=256, shuffle=train).|
| **Model Definition** | `class LSTM(nn.Module)` with `nn.LSTM(1, 128, 2, dropout=0.1)` + `nn.Linear(128,1)`. |
| **Training Loop**  | For each epoch: train on batches with `tqdm`, compute and backprop MSE loss.     |
| **Validation**     | Compute RMSE on validation set each epoch; save best model to `best.pth`.       |
| **Prediction**     | `predict_future(steps=14)` loads checkpoint, generates next values recursively.  |

---

## 🧠 Model Architecture Overview  

```text
Input (batch, seq_len=30, 1) ─► LSTM layer 1 (input_size=1) ──┐
                                │                             │
                                ▼                             │
                            LSTM layer 2                     │
                                │                             ▼
                                ▼                  Take last output timestep
                            Output sequence                (out[:, -1, :])
                                │                             │
                                ▼                             ▼
                       Fully-Connected (128 → 1) ──► Predicted normalized Δtemp
                                                              │
                                                              ▼
                                             Inverse-transform → °C forecast
