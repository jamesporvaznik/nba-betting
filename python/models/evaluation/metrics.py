import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# MAE : On average, your model's prediction is off by 4.65 (or calculated MAE) points from the actual points scored.
#       For players with a lower points average the MAE is worse than for players with a higher points average.

# RMSE : RMSE tells you if your model occasionally makes wildly wrong predictions. 
#        If RMSE is much higher than MAE, you have outliers, if not its more consistent errors.

# Bias: The magnitude and direction your estimate is off from the actual value.

# Over/Under Direction Accuracy: Just splits up when you predict over vs under and get the accuracy for both sides.

# Calibration: Calibrating confidence on picks and the actual rate of that pick

# Feature Importance: How much each feature contributes to the model's prediction.

def evaluate_regression(y_true, y_pred, label: str = 'Model'):
    """Print regression evaluation metrics."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    bias = np.mean(y_pred - y_true)  # positive = model overpredicts

    print(f"\n── {label} ──────────────────")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  Bias: {bias:.3f}  ({'overpredicting' if bias > 0 else 'underpredicting'})")
    return {'mae': mae, 'rmse': rmse, 'bias': bias}

def evaluate_over_under(y_true, y_pred, label: str = 'Model'):
    """
    Evaluate over/under accuracy by confidence bucket.
    Uses the predicted value vs actual value to determine
    how often the model correctly identifies the direction.
    """
    results = pd.DataFrame({
        'actual':    y_true.values,
        'predicted': y_pred,
    })

    # Direction accuracy — did the model predict the right side?
    # Use season average as a proxy for the line
    results['pred_over'] = results['predicted'] > results['actual'].mean()
    results['was_over']  = results['actual']    > results['actual'].mean()
    results['correct']   = results['pred_over'] == results['was_over']

    accuracy = results['correct'].mean()
    print(f"\n── {label} Over/Under Direction Accuracy ──")
    print(f"  Accuracy: {accuracy:.1%}")
    return accuracy

def calibration_report(y_true, y_pred_proba, n_buckets: int = 5):
    """
    Check if confidence scores are well calibrated.
    Pass predicted probabilities (0-1) and actual binary outcomes.
    """
    df = pd.DataFrame({'prob': y_pred_proba, 'actual': y_true})
    df['bucket'] = pd.cut(df['prob'], bins=n_buckets)
    report = df.groupby('bucket').agg(
        count=('actual', 'count'),
        actual_rate=('actual', 'mean'),
        avg_confidence=('prob', 'mean')
    ).reset_index()
    print("\n── Calibration Report ──────────────────")
    print(report.to_string(index=False))
    return report