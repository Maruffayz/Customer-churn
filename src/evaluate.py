"""Simple evaluation and demo predictions for the churn model.

Run as:

    python -m src.evaluate
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report

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
    print("Classification report (test set):")
    print(classification_report(y_test, y_pred))

    # Show a single demo prediction
    sample = X_test[0:1]
    proba = model.predict_proba(sample)[0, 1]
    print(f"Example sample churn probability (1 == positive class): {proba:.3f}")


if __name__ == "__main__":
    main()
