
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


FEATURES = [
    "amount",
    "transaction_hour",
    "international",
    "online",
    "previous_fraud",
    "transactions_last_24h"
]


def load_transactions(file_path):
    df = pd.read_csv(file_path)

    required = ["transaction_id"] + FEATURES + ["fraud"]

    if not all(col in df.columns for col in required):
        raise ValueError("Missing required columns.")

    if df.empty or df[required].isnull().any().any():
        raise ValueError("Dataset is empty or contains missing values.")

    if df["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found.")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if not np.isfinite(df[required].to_numpy()).all():
        raise ValueError("Invalid numeric values found.")

    if (df[FEATURES] < 0).any().any():
        raise ValueError("Transaction features cannot be negative.")

    if not df["transaction_hour"].between(0, 23).all():
        raise ValueError("Transaction hour must be between 0 and 23.")

    for col in ["international", "online", "previous_fraud", "fraud"]:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain only 0 or 1.")

    if df["fraud"].nunique() != 2:
        raise ValueError("Both fraud classes are required.")

    print(f"Loaded {len(df)} transactions.")
    return df


def analyze_transactions(df):
    print("\n========== TRANSACTION ANALYSIS ==========")

    print("Total transactions:", len(df))
    print("Fraud cases:", int(df["fraud"].sum()))
    print("Legitimate cases:", int((df["fraud"] == 0).sum()))

    print("\nAverage transaction amount:")
    print(df["amount"].mean())


def prepare_data(df):
    X = df[FEATURES]
    y = df["fraud"]

    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    print("\nFraud detection model trained successfully.")
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, predictions, zero_division=0):.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.title("Fraud Detection Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Legitimate", "Fraud"])
    plt.yticks([0, 1], ["Legitimate", "Fraud"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, matrix[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig("fraud_confusion_matrix.png")
    plt.close()

    return predictions


def predict_transaction(model, amount, hour, international,
                        online, previous_fraud, transactions_last_24h):

    transaction = pd.DataFrame([{
        "amount": amount,
        "transaction_hour": hour,
        "international": international,
        "online": online,
        "previous_fraud": previous_fraud,
        "transactions_last_24h": transactions_last_24h
    }])

    if not np.isfinite(transaction.to_numpy()).all():
        raise ValueError("Inputs must be finite numbers.")

    if (transaction < 0).any().any():
        raise ValueError("Inputs cannot be negative.")

    if hour > 23:
        raise ValueError("Hour must be between 0 and 23.")

    for col in ["international", "online", "previous_fraud"]:
        if transaction[col].iloc[0] not in [0, 1]:
            raise ValueError(f"{col} must be 0 or 1.")

    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0][
        list(model.classes_).index(1)
    ]

    print("\n========== NEW TRANSACTION ==========")
    print("Result:", "SUSPICIOUS" if prediction == 1 else "LEGITIMATE")
    print(f"Fraud probability: {probability:.2%}")

    return prediction, probability


def export_results(X_test, y_test, predictions):
    results = X_test.copy()
    results["actual_fraud"] = y_test.to_numpy()
    results["predicted_fraud"] = predictions

    results.to_csv("fraud_predictions.csv", index=False)
    print("\nResults exported to fraud_predictions.csv")
