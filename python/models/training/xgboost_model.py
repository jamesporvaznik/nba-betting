import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from ..evaluation.metrics import evaluate_regression, evaluate_over_under, calibration_report

def train_xgboost(X_train, y_train, X_test, y_test, stat: str, save: bool = True):
    """
    Train XGBoost model for a given stat (points, rebounds, assists).
    Returns model and predictions.
    """
    print(f"\nTraining XGBoost model for: {stat}")
    print(f"  Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # 1. Standard regression metrics
    evaluate_regression(y_test, y_pred, label=f"XGBoost ({stat})")
    
    # 2. Over/Under accuracy (using season average as the line)
    evaluate_over_under(y_test, y_pred, label=f"XGBoost ({stat})")
    
    # 3. Calibration report (requires predicted probabilities)
    # For regression, we need to convert to binary classification
    # Using "over season average" as the binary target
    threshold = y_train.mean()  # Use training mean as baseline
    y_test_binary = (y_test > threshold).astype(int)
    
    # Get prediction probabilities using a proxy method
    # Method 1: Use prediction distance from threshold as confidence
    y_pred_proba = 1 / (1 + np.exp(-(y_pred - threshold)))  # Sigmoid transformation
    
    calibration_report(y_test_binary, y_pred_proba, n_buckets=5)
    
    # 4. Feature importance
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n── Feature Importance ({stat}) ──")
    print(importance_df.head(10).to_string(index=False))
    
    # Save model
    if save:
        artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        model_path = os.path.join(artifacts_dir, f'xgboost_{stat}.joblib')
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path}")
    
    return model, y_pred