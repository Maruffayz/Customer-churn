"""Model and pipeline definition for churn prediction."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(n_features: int) -> Pipeline:
    """Create a simple preprocessing + logistic regression pipeline.

    The breast-cancer dataset is all numeric, so we standardize all
    features and feed them into logistic regression.
    """

    numeric_features = list(range(n_features))

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)]
    )

    clf = LogisticRegression(max_iter=500)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", clf),
        ]
    )

    return model
