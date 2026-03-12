"""Training script for the churn model.

Run as:

    python -m src.train
"""

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

from .data import load_data
from .model import build_model


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()

    model = build_model(n_features=X_train.shape[1])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Test accuracy: {acc:.3f}")
    print(f"Test ROC-AUC: {auc:.3f}")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "churn_model.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
