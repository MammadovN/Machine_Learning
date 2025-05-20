
import os
import warnings
import deepchem as dc

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

def load_tox21():
    # Use circular fingerprint featurization
    featurizer = dc.feat.CircularFingerprint(radius=2, size=1024)

    # Load the Tox21 dataset with a random split
    tox21_tasks, datasets, transformers = dc.molnet.load_tox21(
        featurizer=featurizer,
        splitter='random'
    )

    train_dataset, valid_dataset, test_dataset = datasets
    return tox21_tasks, train_dataset, valid_dataset, test_dataset

if __name__ == "__main__":
    tasks, train, valid, test = load_tox21()
    print(f"Train samples: {len(train)}")
    print(f"Validation samples: {len(valid)}")
    print(f"Test samples: {len(test)}")
