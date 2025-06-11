# 🔐 NSL-KDD Intrusion Detection using Autoencoder

This project implements an **unsupervised anomaly detection system** for identifying intrusions in network traffic using the **NSL-KDD dataset**. The core model is a **deep autoencoder**, complemented by traditional anomaly detection methods like **One-Class SVM** and **Isolation Forest** for performance comparison.

---

## 📦 Dataset

- **Source**: [NSL-KDD on Hugging Face](https://huggingface.co/datasets/Mireu-Lab/NSL-KDD)
- **Features**: 41 columns (numeric + categorical)
- **Label**: `class` — either `normal` or one of many intrusion types (e.g., DoS, Probe)

---

## 🧠 Model Architecture

### 🔷 Autoencoder
- Trained only on normal data
- Learns to reconstruct normal behavior
- Detects intrusions as samples with **high reconstruction error**

### 📌 Additional Methods
- **One-Class SVM**
- **Isolation Forest**

---

## 🚀 Workflow

1. **Load Dataset**
2. **Preprocess Data**
   - One-hot encode categorical features
   - Log-transform + scale numerical features
3. **Train Autoencoder** on normal data only
4. **Compute Reconstruction Error** on test set
5. **Tune Threshold** using z-scores and F1 curve
6. **Evaluate using**:
   - Confusion matrix
   - F1 score
   - ROC AUC

---

## 📊 Results

| Method            | Precision | Recall | F1-Score | ROC AUC |
|-------------------|-----------|--------|----------|---------|
| Autoencoder       | 0.66      | 1.00   | 0.79     | 0.95    |
| One-Class SVM     | 0.91      | 0.76   | 0.83     | —       |
| Isolation Forest  | 0.94      | 0.14   | 0.25     | —       |

> 🔎 **Conclusion**: One-Class SVM offers the best tradeoff between recall and false positives. Autoencoder is highly sensitive but aggressive — best used with tuned thresholds or in combination with other models.

---

## 📁 Project Structure

enhanced-autoencoder-nids/
├── enhanced_autoencoder.py # Main pipeline script
├── test_autoencoder.py # Unit tests
├── config.yaml # Model config (optional)
├── requirements.txt # Python dependencies
├── Makefile # CLI helpers
├── setup.py # Optional packaging script
├── .gitignore # Files to ignore
└── README.md # You're reading it!


---

## 🛠 Installation & Usage

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/enhanced-autoencoder-nids.git
cd enhanced-autoencoder-nids

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run main script
python enhanced_autoencoder.py


🧪 Running Tests
python test_autoencoder.py

Future Work
-Variational or Contractive Autoencoder
-Temporal LSTM Autoencoder
-Ensemble decision layer (AE + SVM)
-Streamlit frontend for inference demo

🙌 Acknowledgments
-Hugging Face for dataset access
-TensorFlow/Keras for deep learning
-Scikit-learn for classic ML models

