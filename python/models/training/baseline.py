import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from models.evaluation.metrics import evaluate_regression

def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    stat: str = 'points'
) -> LinearRegression:
    """Train a linear regression baseline model."""
    print(f"\nTraining baseline model for: {stat}")
    print(f"  Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    evaluate_regression(y_test, preds, label=f'Baseline ({stat})')
    return model, preds