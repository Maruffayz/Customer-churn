"""Simple evaluation and demo predictions for the churn model.

Run as:

    python -m src.evaluate
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .data import load_data


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()

    model_path = Path("models") / "churn_model.joblib"
    if not model_path.exists():
        raise SystemExit(
            "Model file not found. Train the model first with 'python -m src.train'."
        )

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification report (test set):")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print(f"ROC-AUC (test set): {auc:.3f}")

    # Show a single demo prediction
    sample = X_test[0:1]
    proba = model.predict_proba(sample)[0, 1]
    print(f"Example sample churn probability (1 == positive class): {proba:.3f}")


if __name__ == "__main__":
    main()
