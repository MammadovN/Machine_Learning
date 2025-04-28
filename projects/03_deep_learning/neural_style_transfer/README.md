# 🎨 NeuralStyleTransfer: Artistic Style Transfer with PyTorch & VGG-19

This project implements Neural Style Transfer in Google Colab using a pre-trained **VGG-19** network. It recreates a **content** image in the **style** of another image by optimising a generated image directly.

---

## 🛠 Technologies Used

- **Python** – Programming language  
- **PyTorch** – Deep learning framework  
- **torchvision** – Pre-trained VGG-19 & transforms  
- **NumPy** – Numeric operations  
- **Pillow (PIL)** – Image loading  
- **matplotlib** – Visualisation  
- **tqdm** – Progress bars (optional)  
- **Google Colab** – Runtime & file-upload helper  

---

## ▶️ How to Run the Project

1. **Open in Colab** (recommended)  
   ```bash
   # Click the “Open in Colab” badge or upload `neural_style_transfer.ipynb`
2. **Install Dependencies (Colab already includes most)**
   ```bash
   pip install torch torchvision pillow matplotlib tqdm
3. **Run the Notebook / Script**
    jupyter Style_Transfer.ipynb
    

## 📂 Project Description  
The algorithm combines the **high-level content representation** of one image with the **low-level texture representation** of another.  
A generated image is initialised as a copy of the content image and iteratively updated to minimise a weighted sum of:

- **Content loss** – difference between feature activations of generated and content images at a chosen layer.  
- **Style loss** – difference between **Gram matrices** of activations for generated and style images across several layers.

---

## ✅ Steps for Analysis  

### 1️⃣ Image Upload  
- File-upload widget auto-detects filenames containing `content` and `style`; otherwise asks the user.  

### 2️⃣ Pre-processing (`load_image`)  
- Resize to `max_size` (default = 400 px).  
- `ToTensor` & normalise with ImageNet mean/std.  

### 3️⃣ Feature Extraction  
- Uses **VGG-19** convolutional layers (`conv_1 … conv_5`).  
- The network is frozen; only the generated image has gradients.  

### 4️⃣ Loss Computation (`ContentStyleLoss`)  
- **Content layer**: `conv_4`.  
- **Style layers**: `conv_1–conv_5`.  
- Uses **MSE** for both content & style components.  

### 5️⃣ Optimisation Loop  
- Optimiser: `Adam` (`lr = 0.003`).  
- 2 000 iterations by default, with progress bar & intermediate visualisation every 200 steps.  

### 6️⃣ Output  
- Final image saved as `style_transfer_result.jpg`.  
- Automatically downloaded in Colab.  

---

## 🧠 Model / Algorithm Overview

```text
Content Image     Style Image
      │               │
      └──► Pre-processing (resize, normalise)
                      │
                      ▼
        ┌───────────────────────────┐
        │      VGG-19 (frozen)      │
        └───────────────────────────┘
                      │
           Feature Activations
        ┌───────┴────────┬───────────┐
        ▼                 ▼           ▼
   Content Loss   ──►  Style Loss   Gram Matrices
        └───────┬────────┴───────────┘
                ▼
        Weighted Total Loss
                ▼
      Optimise Generated Image
                ▼
        Stylised Output Image
