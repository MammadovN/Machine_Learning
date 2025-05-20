
import os
import warnings
import deepchem as dc
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

def train_rf_model():
    print("Loading data...")
    # Load the Tox21 dataset using ECFP (CircularFingerprint)
    tox21_tasks, datasets, _ = dc.molnet.load_tox21(
        featurizer='ECFP',   # CORRECT FEATURIZER
        splitter='random',
        reload=False
    )

    train_dataset, valid_dataset, test_dataset = datasets

    # Select a sample task (e.g., SR-HSE)
    task_index = tox21_tasks.index("SR-HSE")
    X_train, y_train = train_dataset.X, train_dataset.y[:, task_index]
    X_valid, y_valid = valid_dataset.X, valid_dataset.y[:, task_index]

    # Remove NaNs
    mask_train = ~np.isnan(y_train)
    mask_valid = ~np.isnan(y_valid)

    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_valid, y_valid = X_valid[mask_valid], y_valid[mask_valid]

    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of validation samples: {len(X_valid)}")

    # Random Forest model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate AUC score
    y_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_pred)
    print(f"Validation AUC: {auc:.3f}")

    # Save results
    results_df = pd.DataFrame({
        "y_true": y_valid,
        "y_pred": y_pred
    })

    results_path = "/content/drive/MyDrive/drug_discovery_ml/results/sr_hse_rf.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved: {results_path}")

if __name__ == "__main__":
    train_rf_model()
