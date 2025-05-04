import polars as pl, joblib, math, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from preprocess import clean_data
from features   import build_features
from pathlib import Path

def train_and_save(raw_csv: str, model_dir: str):
    df = pl.read_csv(raw_csv)
    df = clean_data(df)
    df = build_features(df)

    X = df.drop("trip_duration").to_pandas()
    y = df["trip_duration"].to_pandas()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.1,
        num_leaves=64,
        seed=42,
    )

    dtrain = lgb.Dataset(X_tr, y_tr)
    dval   = lgb.Dataset(X_val, y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50)]      # ← NEW
    )

    rmsle = math.sqrt(mean_squared_log_error(y_val, model.predict(X_val)))
    print(f"Validation RMSLE: {rmsle:.4f}")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, f"{model_dir}/lgbm.pkl")

if __name__ == "__main__":
    train_and_save(
        "/content/drive/MyDrive/taxi-trip-duration/data/raw/train.csv",
        "/content/drive/MyDrive/taxi-trip-duration/models"
    )
