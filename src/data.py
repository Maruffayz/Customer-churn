"""Data loading utilities for the churn project.

For simplicity and to keep the project self-contained, we use the
`sklearn.datasets.load_breast_cancer` dataset as a stand-in for a
binary churn / no-churn classification problem.
"""

from typing import List, Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data and return train/test splits.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def get_feature_names() -> List[str]:
    """Return the names of the numeric features.

    Useful for documentation and for clients of the prediction API so they
    know the expected order of the input vector.
    """

    dataset = load_breast_cancer()
    return list(dataset.feature_names)
