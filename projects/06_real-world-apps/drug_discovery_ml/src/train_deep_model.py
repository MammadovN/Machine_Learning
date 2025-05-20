
import os
import warnings
import deepchem as dc
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import pandas as pd

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

def train_deep_model():
    print("Loading data...")
    tox21_tasks, datasets, _ = dc.molnet.load_tox21(
        featurizer='ECFP',
        splitter='random',
        reload=False
    )

    train_dataset, valid_dataset, test_dataset = datasets

    # Index for the SR-HSE task
    task_index = tox21_tasks.index("SR-HSE")
    X_train, y_train = train_dataset.X, train_dataset.y[:, task_index]
    X_valid, y_valid = valid_dataset.X, valid_dataset.y[:, task_index]

    # Remove NaNs
    mask_train = ~np.isnan(y_train)
    mask_valid = ~np.isnan(y_valid)

    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_valid, y_valid = X_valid[mask_valid], y_valid[mask_valid]

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_valid.shape}")

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC()])

    # Train the model
    print("Training model...")
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
              epochs=20, batch_size=64, verbose=1)

    # Predict and calculate AUC
    y_pred = model.predict(X_valid).flatten()
    auc = roc_auc_score(y_valid, y_pred)
    print(f"Validation AUC: {auc:.3f}")

    # Save the results
    results_df = pd.DataFrame({
        "y_true": y_valid,
        "y_pred": y_pred
    })

    results_path = "/content/drive/MyDrive/drug_discovery_ml/results/sr_hse_deep.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    train_deep_model()
