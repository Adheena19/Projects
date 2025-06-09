# 🛡️ Intrusion Detection using Autoencoder on NSL-KDD Dataset

A machine learning project implementing an unsupervised deep autoencoder model to detect anomalies (intrusions) in the NSL-KDD dataset, with additional experiments using Isolation Forest and One-Class SVM.

---

## 📊 Project Highlights

* **Autoencoder** trained on normal data to learn typical network behavior.
* **Z-score thresholding** for anomaly detection.
* **Baseline comparison** with Isolation Forest and One-Class SVM.
* **Visualizations** of reconstruction error and ROC.

---

## 📂 Dataset

* **Name**: NSL-KDD
* **Source**: [HuggingFace: Mireu-Lab/NSL-KDD](https://huggingface.co/datasets/Mireu-Lab/NSL-KDD)
* Loaded using the `datasets` library.

---

## 📈 Results Summary

| Model            | Accuracy | F1 Score | ROC AUC |
| ---------------- | -------- | -------- | ------- |
| Autoencoder      | \~52%    | \~0.43   | 0.93    |
| Isolation Forest | \~66%    | \~0.67   | -       |
| One-Class SVM    | \~80%    | \~0.79   | -       |

---

## ⚖️ Model Overview

### Autoencoder

* 6 Dense layers
* Activation: ReLU, Output: Linear
* Loss: Mean Squared Error (MSE)

### Anomaly Detection

* Compute reconstruction error (MSE)
* Apply z-score based threshold to classify anomalies

### Alternatives Compared:

* Isolation Forest (tree-based unsupervised)
* One-Class SVM (kernel-based unsupervised)

---

## 🚀 Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

### Running the Project

Use the notebook or run the script:

```bash
python run.py
```

---

## 📅 Folder Structure

```
intrusion-detection-autoencoder/
├── notebooks/
│   └── intrusion_autoencoder.ipynb
├── src/
│   ├── model.py
│   └── utils.py
├── results/                  # Evaluation plots and saved models
├── data/                     # Optional local data if needed
├── run.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔎 Future Work

* Try **Variational Autoencoders (VAE)**
* Add **attention mechanisms** to encoder
* Use **GAN-based anomaly detection**
* Integrate **business-specific FP/FN cost analysis**

---

## 📄 License

[MIT License](LICENSE)

---

## 🚮 Disclaimer

This repository is for educational and experimental use. Results may vary depending on preprocessing and randomness in initialization.
