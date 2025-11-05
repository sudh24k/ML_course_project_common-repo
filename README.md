Dermoscopic Skin Lesion Classification using Deep Learning

This repository contains the implementation of a deep learning based framework for automated dermoscopic image classification. The project focuses on leveraging Convolutional Neural Networks and transfer learning to classify skin lesions into multiple categories using the HAM10000 dataset.

The system aims to assist dermatologists by providing accurate and robust AI-based diagnostic support for early detection of melanoma and other skin lesions.

Features

Multi-model deep learning pipeline

Custom CNN architecture

Transfer learning using:

DenseNet121

EfficientNetB0 & EfficientNetB1

InceptionV3

ResNet50 / Xception / VGG16 (extended experiments)

Data augmentation for improved generalization

Evaluation using confusion matrix, ROC-AUC, F1-score

Ensemble learning (simple & weighted averaging) for final prediction

HAM10000 dermoscopic dataset preprocessing + EDA

Dataset

Dataset: HAM10000 (Human Against Machine)
Source: Dermatoscopic images of pigmented skin lesions
Classes (7 total):

Melanoma (mel)

Melanocytic nevi (nv)

Basal cell carcinoma (bcc)

Actinic keratoses (akiec)

Benign keratosis-like lesions (bkl)

Dermatofibroma (df)

Vascular lesions (vasc)

Architecture Overview
Custom CNN

Multiple Conv2D + BatchNorm blocks

AveragePooling and Dropout for regularization

Dense classifier with Softmax

Pretrained CNNs (Transfer Learning)

Each pretrained model includes:

ImageNet weights

Frozen layers initially, then fine-tuned end-to-end

GlobalAveragePooling + Dense classifier + Dropout

Ensemble Strategy

Simple average of probabilities

Weighted average for best final performance


Tech Stack

Python 3.10

TensorFlow / Keras

CUDA 11.x / cuDNN (GPU training)

NumPy, Pandas, Matplotlib

TensorBoard (training logs)

Training Details

Image size: 75 x 100 x 3

Loss: Categorical Cross-Entropy

Optimizer: Adam (lr=0.001)

Epochs: 50-100

Data Augmentation: rotation, flip, zoom, brightness shifts

Early stopping + ReduceLROnPlateau

Future Work

Explainable AI (Grad-CAM, SHAP, LIME)

Temporal lesion-progression modeling

Federated learning for privacy-safe medical AI

Mobile deployment for point-of-care diagnostics

Multi-modal input (clinical metadata + images)

Acknowledgements

HAM10000 Dataset Authors

TensorFlow & Open-Source ML Community

Project Supervisor: Dr. Arpan Garai
