# Satellite-Based Earthquake Damage Detection using YOLOv8

This repository contains a deep learning pipeline designed to classify and detect earthquake damage on buildings using satellite/aerial imagery. Built on top of the **YOLOv8** classification architecture and optimized with the **Adam** optimizer, the model achieves a **99.6% validation accuracy** in distinguishing between damaged and undamaged structures.

Developed and verified native on **Apple Silicon M4 GPU** utilizing **Metal Performance Shaders (MPS)**.

---

## 🚀 Key Features
* **Automated Data Pipeline:** Cleans, shuffles with a fixed random seed (`seed=42`), and splits raw datasets into homogeneous 80% Train / 20% Val distributions.
* **Transfer Learning:** Leverages pretrained `yolov8m-cls` weights to achieve ultra-fast convergence.
* **Signal-Driven Architecture:** Utilizes deep Convolutional Neural Networks (CNN) acting as high-pass filters to dissect high-frequency signal changes (structural cracks, debris, edge deformations).
* **Dynamic Inference:** An automation-friendly inference pipeline that automatically loads the absolute best-performing weights (`best.pt`) from the latest training runs.

---

## 📊 Training Performance (M4 Native)
* **Hardware:** MacBook Air M4 (Unified Memory via PyTorch MPS backend)
* **Image Size (`imgsz`):** 224x224
* **Batch Size:** 32
* **Optimizer:** Adam
* **Epochs:** 50
* **Final Top-1 Accuracy:** **99.6%**
* **Memory Efficiency:** Heavily optimized ~2.46 GB GPU memory footprint via explicit runtime RAM caching controls.

---

## 📂 Project Structure
```text
earthquake-damage-yolov8/
├── data/
│   └── maras_dataset/       # 80/20 train/val split (Ignored by Git)
├── src/
│   ├── prepare_data.py      # Data ingestion and preprocessing pipeline
│   ├── train.py             # Model training & optimization script
│   └── inference.py         # Visual test & evaluation script
├── .gitignore               # Keeps deep learning artifacts and datasets out of remote repo
└── README.md                # Project documentation