"""FastAPI app exposing the churn model for predictions.

Run with:

    uvicorn src.api:app --reload

All logic here is original and separate from any external projects.
"""

from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import get_feature_names, load_data


class ChurnRequest(BaseModel):
    """Request body for churn prediction.

    Clients send a flat list of numeric feature values ordered according to
    the feature names exposed by the /schema endpoint.
    """

    features: List[float] = Field(
        ..., description="Feature vector matching the documented order."
    )


class ChurnResponse(BaseModel):
    probability: float
    predicted_label: int


app = FastAPI(title="Maruf Churn From-Scratch API")


def _load_model():
    model_path = Path("models") / "churn_model.joblib"
    if not model_path.exists():
        # Train a fresh model if needed so the API is self-contained.
        X_train, X_test, y_train, y_test = load_data()
        from .model import build_model

        model = build_model(n_features=X_train.shape[1])
        model.fit(X_train, y_train)
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(model, model_path)
        return model

    return joblib.load(model_path)


MODEL = _load_model()
FEATURE_NAMES = get_feature_names()


@app.get("/schema")
def schema() -> dict:
    """Return the expected feature order and count.

    This helps API consumers build correct request payloads.
    """

    return {
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
    }


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest) -> ChurnResponse:
    if len(request.features) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid feature vector length. "
                f"Expected {len(FEATURE_NAMES)}, got {len(request.features)}."
            ),
        )

    import numpy as np

    X = np.array(request.features, dtype=float).reshape(1, -1)
    proba = float(MODEL.predict_proba(X)[0, 1])
    label = int(MODEL.predict(X)[0])
    return ChurnResponse(probability=proba, predicted_label=label)


def main() -> None:
    """Simple sanity check when running `python -m src.api`."""

    print("API module imported successfully. Run with:")
    print("  uvicorn src.api:app --reload")


if __name__ == "__main__":
    main()
