"""
train_model.py
--------------
Trains a Random Forest Classifier on the crop dataset, evaluates it,
and saves the model + real evaluation metrics to the 'model' folder.

Run:  python train_model.py
"""
import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data_loader import load_dataset, preprocess, FEATURES

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def main():
    # 1. Load dataset
    df = load_dataset()
    print(f"Loaded {len(df)} records with {df['label'].nunique()} crop types.")

    # 2. Preprocess
    X, y = preprocess(df)

    # 3. Train/Test split (80% train, 20% test). stratify keeps class balance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train Random Forest (an ensemble of 100 decision trees)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate on unseen test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", cm)

    # Feature importance tells us which inputs matter most
    importances = dict(zip(FEATURES, np.round(model.feature_importances_, 4).tolist()))

    # 6. Save model and metrics (the website reads these - nothing is hardcoded)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "crop_model.pkl"))

    # Feature averages per crop - used for the rule-based explanation
    crop_means = df.groupby("label")[FEATURES].mean().round(2).to_dict(orient="index")

    metrics = {
        "accuracy": round(accuracy, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_records": len(df),
        "num_classes": len(labels),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "per_class": {c: report[c] for c in labels},
        "feature_importance": importances,
        "crop_means": crop_means,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel saved to model/crop_model.pkl and metrics to model/metrics.json")


if __name__ == "__main__":
    main()
