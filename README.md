# Bank Customer Churn – From Scratch

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
 [![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)](https://scikit-learn.org/)
 [![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)

This is an original machine-learning project by **Maruf Fayziev**.

The goal is to build and explain a compact end‑to‑end churn prediction
pipeline in pure Python using only open-source libraries and built-in
datasets (no copied notebooks or external project code).

## Features

- End‑to‑end binary classification pipeline (preprocessing + model)
- Train/test split with stratification for reliable evaluation
- Metrics: accuracy, ROC‑AUC, classification report, confusion matrix
- Model persistence with `joblib`
- FastAPI service with `/schema` and `/predict` endpoints

## Project structure

```text
maruf-churn-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── __init__.py
    ├── data.py          # data loading / splitting
    ├── model.py         # model and pipeline definition
    ├── train.py         # training script
    ├── evaluate.py      # evaluation and metrics
    └── api.py           # FastAPI app for online predictions
```

## Quick start

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python -m src.train
```

4. Evaluate and see example predictions:

```bash
python -m src.evaluate
```

5. Run the API server (optional):

```bash
uvicorn src.api:app --reload
```

Then open `http://127.0.0.1:8000/docs` to try the `/schema` and `/predict`
endpoints from the interactive Swagger UI.

All code in this repository was written from scratch for learning and
portfolio purposes.

## API usage example

Once the server is running (`uvicorn src.api:app --reload`), you can
query the model directly, for example with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
    -H "Content-Type: application/json" \
    -d "{\"features\": [0.2, 1.3, 2.1, 0.5, 1.0, 2.5, 0.1, 0.7, 1.8, 0.3,
             0.9, 1.2, 0.4, 0.6, 1.1, 0.8, 1.5, 0.2, 0.3, 0.7, 1.0, 0.4, 0.9,
             1.3, 0.5, 0.6, 0.2, 0.1, 0.4, 0.3] }"
```

Use `GET /schema` in the interactive docs (`/docs`) to see the exact
feature order and count that the model expects.
