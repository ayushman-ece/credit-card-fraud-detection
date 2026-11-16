# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-orange)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-red?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

A Python-based Machine Learning project that identifies potentially fraudulent credit card transactions using a Random Forest Classifier.

The system analyzes transaction amount, time, online activity, international transactions, previous fraud history, and transaction frequency to classify transactions as legitimate or suspicious.

## 🎯 Objectives

- Analyze transaction data.
- Identify suspicious transaction patterns.
- Train a Random Forest classification model.
- Predict potential fraudulent transactions.
- Evaluate model performance using classification metrics.
- Export prediction results.

## ⚙️ Features

- Transaction dataset validation.
- Fraud and legitimate transaction analysis.
- Stratified train-test splitting.
- Random Forest classification.
- Class imbalance handling with balanced class weights.
- Accuracy, Precision, Recall and F1 Score.
- Confusion matrix visualization.
- New transaction prediction.
- CSV report generation.

## 🧠 Workflow

1. Load and validate transaction data.
2. Analyze fraud distribution.
3. Separate input features and target.
4. Split data into training and testing sets.
5. Train the Random Forest Classifier.
6. Evaluate predictions.
7. Predict a new transaction.
8. Export results.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Matplotlib | Visualization |
| Random Forest | Classification |

## 📊 Dataset Information

The sample dataset contains 40 synthetic transactions.

| Feature | Description |
|---|---|
| transaction_id | Unique transaction identifier |
| amount | Transaction amount |
| transaction_hour | Hour of transaction (0–23) |
| international | International transaction flag |
| online | Online transaction flag |
| previous_fraud | Previous fraud history flag |
| transactions_last_24h | Recent transaction count |
| fraud | Target label (0 or 1) |

**Target variable:**
- 0 = Legitimate
- 1 = Fraudulent

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

For fraud detection, recall is particularly important because missed fraudulent transactions can be costly. Precision is also important to avoid incorrectly flagging legitimate transactions.

## 📁 Generated Files

| File | Description |
|---|---|
| fraud_confusion_matrix.png | Classification visualization |
| fraud_predictions.csv | Actual and predicted results |

## 🚀 Installation and Execution

**1. Clone repository**

```bash
git clone https://github.com/ayushman-ece/credit-card-fraud-detection.git
```

**2. Open project**

```bash
cd credit-card-fraud-detection
```

**3. Create virtual environment**

```bash
python -m venv .venv
```

**4. Activate on Windows**

```bash
.venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run**

```bash
python main.py
```

## 📚 Key Concepts Learned

- Machine Learning Classification
- Fraud Detection
- Random Forest Classifier
- Imbalanced Data
- Class Weights
- Stratified Sampling
- Precision and Recall
- F1 Score
- Confusion Matrix
- Model Prediction

## ⚠️ Limitations

- Uses only 40 synthetic transactions.
- Does not represent real banking transaction patterns.
- Model performance cannot establish real-world fraud detection ability.
- Does not include customer identity, device, location history, or transaction sequences.
- Not suitable for real financial security decisions.

## 🔮 Future Improvements

- Use a larger real-world anonymized transaction dataset.
- Apply cross-validation.
- Tune classification thresholds.
- Compare Random Forest with XGBoost and Logistic Regression.
- Add transaction anomaly detection.
- Build a real-time fraud monitoring dashboard.

## 👨‍💻 Author

**Ayushman Tiwari**

GitHub: [@ayushman-ece](https://github.com/ayushman-ece)

---

⭐ If you find this project useful, consider giving the repository a star!