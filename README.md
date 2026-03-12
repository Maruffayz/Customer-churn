# Bank Customer Churn – From Scratch

This is an original machine-learning project by **Maruf Fayziev**.

The goal is to build and explain a complete, but compact, churn prediction
pipeline in pure Python using only open-source libraries and built-in
datasets (no copied notebooks or external project code).

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
    └── evaluate.py      # evaluation and simple CLI
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

All code in this repository was written from scratch for learning and
portfolio purposes.
