# 🖼️ TransferLearningResNet50: Image Classification via Transfer Learning (PyTorch)

This project fine-tunes a pretrained **ResNet50** model on the **CIFAR-10** image dataset to demonstrate transfer learning for image classification.

---

## 🛠️ Technologies Used

- **Python** – Programming language  
- **PyTorch** – Deep learning framework  
- **torchvision** – Data transforms and built-in datasets  
- **NumPy & pandas** – Data manipulation  
- **tqdm** – Progress bars for training loops  
- **matplotlib** – Visualization of training and evaluation metrics  

---

## ▶️ How to Run the Project

1. **Clone the repository**  
   ```bash
   git clone https://github.com/MammadovN/transfer_learning_classification.git  
   cd transfer_learning_classification/projects/03_deep_learning/transfer_learning/transfer_learning_classification
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run training script**
The script automatically downloads CIFAR-10 and starts training:
   ```bash
   jupyter notebook transfer_learning_classification.ipynb

---

## 📂 Project Description

- **Dataset**  
  Uses `torchvision.datasets.CIFAR10` to download and split the dataset into training and validation/test sets.

- **Data Preprocessing**  
  - Resize images to 224×224  
  - Apply `RandomHorizontalFlip` and `RandomCrop` for training augmentation  
  - Center crop for validation/test  
  - Normalize using ImageNet mean and standard deviation  

- **DataLoader**  
  - Batch size: 32  
  - Shuffle training data, use 2 worker processes  

- **Model**  
  - Load pretrained `ResNet50`  
  - Freeze all convolutional layers when `feature_extract=True`  
  - Replace the final fully connected layer to output 10 classes  

- **Training Loop**  
  - Use `CrossEntropyLoss` and `Adam` optimizer  
  - Train for 10 epochs  
  - Track validation accuracy and save the best model weights  

- **Evaluation**  
  - Run evaluation on the test set and report final accuracy  

- **Saving**  
  - Save the best model checkpoint as `best_resnet50.pth`  

---

## ✅ Steps for Analysis

| Step                 | Description                                                                                     |
|----------------------|-------------------------------------------------------------------------------------------------|
| **Data Download**    | `CIFAR10(root='./data', download=True)` handles dataset download and split.                    |
| **Transforms**       | Apply `Resize(224)`, `RandomHorizontalFlip` + `RandomCrop` (train), `CenterCrop`, `Normalize`.  |
| **DataLoader Setup** | Create `DataLoader` objects (batch_size=32, shuffle train, num_workers=2).                      |
| **Model Definition** | Load pretrained ResNet50, freeze layers, replace `fc` layer for 10 output classes.             |
| **Training**         | Loop over epochs: compute train/val loss and accuracy, save best model.                         |
| **Testing**          | Evaluate final model on the test set using `evaluate_model`.                                    |
| **Checkpoint Saving**| Use `torch.save` to persist `best_resnet50.pth`.                                                |

---

## 🧠 Model Architecture Overview

```text
Input: 3×224×224 image  ─► ResNet50 Conv Blocks (frozen) ─► AdaptiveAvgPool ─► Fully Connected (2048→10) ─► Softmax logits
