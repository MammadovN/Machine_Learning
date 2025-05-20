# 🧪 Drug Discovery with Machine Learning

This project demonstrates a basic machine learning pipeline for molecular property prediction using the [Tox21 dataset](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#tox21) from DeepChem. It includes data preprocessing, molecular descriptor analysis, and training both a classical Random Forest model and a deep learning model.

---

## 📁 Project Structure

```
drug_discovery_ml/
├── src/
│   ├── data_prep.py             # Load and preprocess the Tox21 dataset
│   ├── train_baseline.py        # Train a Random Forest baseline model
│   ├── molecule_analysis.py     # Extract RDKit molecular descriptors
│   └── deep_model.py            # Train a deep learning model with TensorFlow
├── results/
│   ├── sr_hse_rf.csv            # Validation results from the RF model
│   ├── sr_hse_deep.csv          # Validation results from the deep model
│   └── tox21_descriptors.csv    # Molecular descriptors dataset
```

---

## ⚙️ Installation & Environment Setup (Google Colab)

To run this project in Google Colab, use the following setup script:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install --quiet numpy==1.24.3 pandas scikit-learn rdkit-pypi deepchem==2.8.0
!pip install --quiet torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 torchdata==0.6.1
!pip install --quiet torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
!pip install --quiet tf-keras pytorch-lightning dm-haiku

# Clean up conflicting installations
!pip uninstall -y keras tensorflow keras-nightly keras-preprocessing
!pip install --upgrade "jax[cpu]" jaxlib

---

## ▶️ Running the Scripts

Make sure your working directory is in **Google Colab** and your `src/` folder is located under **Google Drive**.

### 1. Data Preparation
```bash
python /content/drive/MyDrive/drug_discovery_ml/src/data_prep.py
### 2. Analyze Molecular Descriptors
python /content/drive/MyDrive/drug_discovery_ml/src/molecule_analysis.py
### 3. Train Random Forest Baseline
python /content/drive/MyDrive/drug_discovery_ml/src/train_baseline.py
### 4. Train Deep Learning Model
python /content/drive/MyDrive/drug_discovery_ml/src/deep_model.py

---

## 📊 Output Files

- `sr_hse_rf.csv`: Validation predictions from the **Random Forest** model for the **SR-HSE** task.
- `sr_hse_deep.csv`: Validation predictions from the **Deep Learning** model for the **SR-HSE** task.
- `tox21_descriptors.csv`: Molecular descriptors extracted using **RDKit**.

---

## 🧠 Models Used

- **Random Forest** – implemented with `scikit-learn`
- **Feedforward Neural Network** – built using `TensorFlow/Keras`

---

## 🧬 Dataset

- **Tox21**: A benchmark dataset for toxicity prediction tasks.
  - Source: [`deepchem.molnet.load_tox21`](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#tox21)

---

## 📌 Requirements

- **Python** 3.8+
- **Google Colab** (recommended) or a local machine with GPU

### Required libraries:

- `deepchem`  
- `rdkit`  
- `torch`, `torch-geometric`  
- `tensorflow`, `keras`  
- `pandas`, `scikit-learn`  
- `jax`

---

## 📄 License

This project is intended for **educational and research purposes only**.  
No official license is included.
