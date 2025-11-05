
# Dermoscopic Skin Lesion Classification using Deep Learning

This repository presents a deep learning framework for automated dermoscopic image analysis and skin lesion classification using the **HAM10000 dataset**. The project evaluates both a custom CNN and multiple transfer learning architectures, and introduces an **ensemble strategy** to maximize diagnostic accuracy.

---

## 🚀 Project Highlights

* Custom CNN baseline model
* Transfer learning using:

  * **DenseNet121**
  * **EfficientNetB0 / EfficientNetB1**
  * **InceptionV3**
  * (Additional experiments: ResNet50, Xception, VGG16)
* Advanced data augmentation pipeline
* End-to-end training and fine-tuning
* ROC-AUC, Confusion Matrix, F1-score evaluation
* Weighted ensemble model for best results

---

## 📂 Dataset: HAM10000

Seven skin lesion types:

| Label | Class                         |
| ----- | ----------------------------- |
| akiec | Actinic keratoses             |
| bcc   | Basal cell carcinoma          |
| bkl   | Benign keratosis-like lesions |
| df    | Dermatofibroma                |
| mel   | Melanoma                      |
| nv    | Melanocytic nevi              |
| vasc  | Vascular lesions              |

---

## 🧠 Model Architectures

### Custom CNN

* Conv2D + BatchNorm blocks
* AveragePooling
* Dropout regularization
* Dense layers + Softmax classifier

### Transfer Learning Models

* Pretrained on ImageNet
* GlobalAveragePooling + Dense layers
* Fine-tuned end-to-end

### Ensemble Method

* Simple averaging and weighted averaging of model predictions
* Best performance achieved by **weighted ensemble**

---

## 📊 Results

| Model                       | Micro-AUC   | Notes                         |
| --------------------------- | ----------- | ----------------------------- |
| DenseNet121                 | ~0.9637     | Best single model             |
| EfficientNetB1              | ~0.9007     | Strong precision              |
| InceptionV3                 | ~0.8957     | Struggled on minority classes |
| Custom CNN                  | ~0.70       | Baseline                      |
| **Weighted Ensemble**       | **~0.9588** | Best overall                  |
| **Melanoma AUC (Ensemble)** | **0.9334**  | Critical clinical class       |

---

## ⚙️ Training Setup

* Input size: `75 x 100 x 3`
* Optimizer: **Adam (lr = 0.001)**
* Loss: **Categorical Cross-entropy**
* Epochs: **50–100**
* Augmentations: rotation, flip, shift, zoom, brightness
* Early stopping + LR scheduler

---

## ✅ How to Run

```bash
git clone https://github.com/yourusername/skin-lesion-classification.git
cd skin-lesion-classification

pip install -r requirements.txt

python train.py
```

---

## 📎 Tech Stack

* Python 3.10
* TensorFlow / Keras
* NumPy, Pandas, Matplotlib
* CUDA / cuDNN (for GPU)
* TensorBoard

---

## 🔮 Future Enhancements

* Explainable AI (Grad-CAM, LIME, SHAP)
* Federated learning for privacy-safe medical AI
* Multi-modal models (metadata + image)
* Mobile deployment for real-time CAD tools

---

## 🙏 Acknowledgements

* HAM10000 dataset authors
* TensorFlow & open-source community
* Research guidance from **Dr. Arpan Garai**

---


