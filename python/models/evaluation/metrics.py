import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# MAE : On average, your model's prediction is off by 4.65 (or calculated MAE) points from the actual points scored.
#       For players with a lower points average the MAE is worse than for players with a higher points average.

# RMSE : RMSE tells you if your model occasionally makes wildly wrong predictions. 
#        If RMSE is much higher than MAE, you have outliers, if not its more consistent errors.

# Bias: The magnitude and direction your estimate is off from the actual value.

# Over/Under Direction Accuracy: Just splits up when you predict over vs under and get the accuracy for both sides.

# Calibration: Calibrating confidence on picks and the actual rate of that pick

# Feature Importance: How much each feature contributes to the model's prediction.

# evaulate stats and performance of model
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

# evaulate over under accuracy
def evaluate_over_under(y_true, y_pred, label: str = 'Model', *, lines):
    """
    Compare directions around explicit, pregame average-based proxy lines.
    Exclude actual ties (pushes) and prediction ties (abstentions), reporting
    coverage. This is not sportsbook accuracy or probability calibration.
    """
    actual, predicted, thresholds = [np.asarray(x, dtype=float) for x in (y_true, y_pred, lines)]
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.shape != thresholds.shape:
        raise ValueError('Actuals, predictions, and pregame lines must be matching vectors')
    if not all(np.isfinite(x).all() for x in (actual, predicted, thresholds)):
        raise ValueError('Proxy evaluation requires finite inputs')
    pushes = np.isclose(actual, thresholds, rtol=0, atol=1e-10)
    abstentions = np.isclose(predicted, thresholds, rtol=0, atol=1e-10)
    scored = ~pushes & ~abstentions
    accuracy = float(np.mean((predicted[scored] > thresholds[scored]) ==
                             (actual[scored] > thresholds[scored]))) if scored.any() else float('nan')
    print(f"\n── {label}: direction versus pregame season-average proxy ──")
    print(f"  Accuracy: {accuracy:.1%}; scored {scored.sum():,}/{len(actual):,}; "
          f"actual ties: {pushes.sum():,}; prediction ties excluding pushes: {(abstentions & ~pushes).sum():,}")
    return accuracy

# compare confidence and acutal results
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


def paired_date_interval(dates, baseline_errors, errors, repeats=1000):
    """Date-block bootstrap of paired MAE differences; negative favors new X."""
    daily = pd.DataFrame({'date': np.asarray(dates), 'difference': errors-baseline_errors}).groupby('date').difference.agg(['sum', 'count'])
    rng = np.random.default_rng(2026)
    indexes = rng.integers(0, len(daily), size=(repeats, len(daily)))
    estimates = daily['sum'].to_numpy()[indexes].sum(axis=1) / daily['count'].to_numpy()[indexes].sum(axis=1)
    return [float(v) for v in np.quantile(estimates, [.025, .975])]

def metrics(actual, predictions, lines):
    pushes = np.isclose(actual, lines, rtol=0, atol=1e-10)
    abstain = np.isclose(predictions, lines, rtol=0, atol=1e-10)
    scored = ~pushes & ~abstain
    return {'mae': float(mean_absolute_error(actual, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(actual, predictions))),
            'bias': float(np.mean(predictions-actual)),
            'proxy_accuracy': float(np.mean((predictions[scored] > lines[scored]) == (actual[scored] > lines[scored]))) if scored.any() else None,
            'proxy_scored': int(scored.sum()), 'actual_ties': int(pushes.sum()),
            'prediction_ties_excluding_pushes': int((abstain & ~pushes).sum())}
