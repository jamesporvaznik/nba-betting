"""Temporal calibration and evaluation shared by training and experiments."""

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, r2_score
from models.evaluation.metrics import metrics, paired_date_interval

KEYS = ['playerid', 'season', 'date', 'team', 'opponent']


def conformity_quantile(scores, level):
    """Finite-sample split-conformal order statistic; no interpolation."""
    scores = np.asarray(scores, dtype=float)
    if not len(scores) or not np.isfinite(scores).all() or (scores < 0).any() or not 0 < level < 1:
        raise ValueError('Invalid nonconformity scores or interval level')
    rank = math.ceil((len(scores) + 1) * level)
    return float(np.sort(scores)[rank-1]) if rank <= len(scores) else float('inf')

def margin(predictions, lines, scale):
    return ((np.asarray(predictions) - np.asarray(lines)) / np.asarray(scale)).reshape(-1, 1)

def confidence_buckets(actual, lines, probabilities):
    """Confidence for the chosen over/under direction, conditional on no push."""
    actual, lines, probabilities = map(np.asarray, (actual, lines, probabilities))
    valid = ~np.isclose(actual, lines, rtol=0, atol=1e-10)
    valid &= ~np.isclose(probabilities, .5, rtol=0, atol=1e-12)
    p, y = probabilities[valid], actual[valid] > lines[valid]
    confidence = np.maximum(p, 1-p)
    correct = (p > .5) == y
    bounds = [.5, .55, .60, .65, .70, .80, .90, 1.000001]
    buckets = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        mask = (confidence >= low) & (confidence < high)
        buckets.append({'confidence_range': f'{low:.0%}–{min(high, 1):.0%}',
                        'count': int(mask.sum()),
                        'mean_confidence': float(confidence[mask].mean()) if mask.any() else None,
                        'actual_direction_accuracy': float(correct[mask].mean()) if mask.any() else None})
    return buckets

def calibration_fit(frame):
    actual, prediction, lines, scale = [frame[c].to_numpy() for c in ['actual', 'prediction', 'line', 'scale']]
    valid = ~np.isclose(actual, lines, atol=1e-10, rtol=0)
    classifier = LogisticRegression(C=1e6, max_iter=1000)
    classifier.fit(margin(prediction, lines, scale)[valid], (actual[valid] > lines[valid]).astype(int))
    quantiles = {str(level): conformity_quantile(abs(actual - prediction) / scale, level) for level in [.5, .8, .9]}
    return classifier, quantiles, float(np.mean(actual[valid] > lines[valid]))

def assess(test, prediction, stat, classifier, quantiles, cal_rate):
    actual, lines, scale = test[stat].to_numpy(), test[f'{stat}_avg_season'].to_numpy(), test[f'{stat}_scale'].to_numpy()
    errors = abs(actual - prediction)
    valid = ~np.isclose(actual, lines, atol=1e-10, rtol=0)
    y = (actual[valid] > lines[valid]).astype(int)
    probabilities = classifier.predict_proba(margin(prediction, lines, scale))[:, 1]
    correct = ((probabilities[valid] > .5) == y).astype(float)
    result = {**metrics(actual, prediction, lines), 'r2': float(r2_score(actual, prediction)),
              'median_absolute_error': float(np.median(errors)),
              'mae_95pct_date_bootstrap': paired_date_interval(test.date, np.zeros(len(test)), errors),
              'negative_predictions': int((prediction < 0).sum()),
              'within_tolerance': {str(t): float(np.mean(errors <= t)) for t in ([3, 5, 10] if stat == 'points' else [1, 2, 3])},
              'intervals': [], 'direction_confidence': {
                  'accuracy': float(correct.mean()), 'nonpush_rows': int(valid.sum()),
                  'accuracy_95pct_date_bootstrap': paired_date_interval(test.loc[valid, 'date'], np.zeros(valid.sum()), correct),
                  'brier_score': float(brier_score_loss(y, probabilities[valid])),
                  'log_loss': float(log_loss(y, probabilities[valid], labels=[0, 1])),
                  'constant_calibration_rate': cal_rate,
                  'constant_direction_accuracy': float(np.mean((cal_rate > .5) == y)),
                  'constant_brier': float(brier_score_loss(y, np.full(len(y), cal_rate))),
                  'buckets': confidence_buckets(actual, lines, probabilities), 'thresholds': []}}
    output = test[KEYS + [stat]].copy()
    output['prediction'], output['proxy_line'], output['uncertainty_scale'] = prediction, lines, scale
    output['probability_over_proxy_nonpush'] = probabilities
    output['confidence_nonpush'] = np.maximum(probabilities, 1 - probabilities)
    for level, q in quantiles.items():
        nominal = float(level)
        lower, upper = np.maximum(0, prediction - q * scale), np.maximum(0, prediction + q * scale)
        result['intervals'].append({'nominal': nominal, 'observed_coverage': float(np.mean((actual >= lower) & (actual <= upper))),
                                    'mean_width': float(np.mean(upper - lower)), 'median_width': float(np.median(upper - lower)), 'score_quantile': q})
        output[f'lower_{int(nominal * 100)}'], output[f'upper_{int(nominal * 100)}'] = lower, upper
    for threshold in [.55, .6, .65, .7, .8]:
        mask = valid & (np.maximum(probabilities, 1 - probabilities) >= threshold)
        result['direction_confidence']['thresholds'].append({'minimum_confidence': threshold, 'rows': int(mask.sum()),
            'actual_accuracy': float(np.mean((probabilities[mask] > .5) == (actual[mask] > lines[mask]))) if mask.any() else None,
            'mean_confidence': float(np.maximum(probabilities[mask], 1 - probabilities[mask]).mean()) if mask.any() else None})
    return result, output
