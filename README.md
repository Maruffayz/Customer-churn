# Bank Customer Churn – From Scratch

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
 [![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)](https://scikit-learn.org/)
 [![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)

The goal is to build and explain a compact end‑to‑end churn prediction
pipeline in pure Python using only open-source libraries and built-in
datasets (no copied notebooks or external project code).

## Table of Contents

1. [Overview](#overview)
2. [Project Highlights](#project-highlights)
3. [Data Analysis and Insights](#data-analysis-and-insights)
4. [Features](#features)
5. [Project Structure](#project-structure)
6. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Running the API](#running-the-api)
7. [Usage](#usage)
8. [Technical Components](#technical-components)
9. [Model Performance](#model-performance)
10. [Example Scenario](#example-scenario)
11. [Contributing](#contributing)
12. [License](#license)

## Overview

Small, self-contained project that demonstrates how to build, train,
evaluate, and serve a bank customer churn model using classical
machine-learning techniques.

## Project Highlights

- Clean separation between data, model, training, evaluation, and API
- Zero external data files – uses a built-in dataset for simplicity
- Ready-to-use CLI scripts (`train`, `evaluate`) and HTTP API
- Written to be easy to read and extend for learning purposes

## Data Analysis and Insights

The focus is on understanding model behaviour rather than heavy
exploratory data analysis. The `src.evaluate` module prints:

- A classification report (precision, recall, F1-score)
- A confusion matrix to see correct vs incorrect predictions
- ROC‑AUC score based on predicted probabilities

You can extend this with additional plots (e.g., ROC curve, feature
importance) if you want to explore the data further.

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

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git (if you clone from GitHub)
- (Optional) a virtual environment manager such as `venv` or `conda`

### Installation

```bash
git clone https://github.com/Maruffayz/Customer-churn.git
cd Customer-churn
pip install -r requirements.txt
```

### Running the API

```bash
uvicorn src.api:app --reload
```

Then open `http://127.0.0.1:8000/docs` to try the `/schema` and
`/predict` endpoints from the interactive Swagger UI.

All code in this repository was written from scratch for learning and
portfolio purposes.

## Usage

Train the model:

```bash
python -m src.train
```

Evaluate and see detailed metrics:

```bash
python -m src.evaluate
```

Query the running API directly, for example with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
    -H "Content-Type: application/json" \
    -d "{\"features\": [0.2, 1.3, 2.1, 0.5, 1.0, 2.5, 0.1, 0.7, 1.8, 0.3,
             0.9, 1.2, 0.4, 0.6, 1.1, 0.8, 1.5, 0.2, 0.3, 0.7, 1.0, 0.4, 0.9,
             1.3, 0.5, 0.6, 0.2, 0.1, 0.4, 0.3] }"
```

Use `GET /schema` in the interactive docs (`/docs`) to see the exact
feature order and count that the model expects.

## Technical Components

- Language: Python 3.10+
- Core libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`
- Serving: `FastAPI` + `uvicorn`
- Model: `LogisticRegression` inside a preprocessing `Pipeline` with
    `StandardScaler` and `ColumnTransformer`

## Model Performance

On the default dataset, the model typically reaches:

- Accuracy above 97% on the test set
- ROC‑AUC around 0.99

Exact numbers can vary slightly depending on the train/test split.

## Example Scenario

A bank’s retention team can use this model to flag customers who are
likely to churn. High‑risk customers (with high predicted churn
probability) can be prioritised for personalised offers, follow‑up
calls, or loyalty programmes.

## Contributing

Suggestions and small improvements are welcome. Feel free to open an
issue or submit a pull request with a clear description of the change.

## License

All rights reserved by **Maruf Fayziev**. If you would like to use this
project in a commercial setting, please contact the author.
