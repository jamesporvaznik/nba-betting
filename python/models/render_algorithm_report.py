"""Render a completed algorithm comparison into a standalone HTML report."""

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = {'linear': 'Linear', 'xgb_current': 'Previous XGBoost', 'xgb_rmse': 'Tuned XGB squared',
          'xgb_mae': 'Tuned XGB MAE', 'cat_rmse': 'CatBoost squared',
          'cat_mae': 'CatBoost MAE', 'blend': 'Selected blend'}


def render(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    report = json.loads((folder/'report.json').read_text())
    summary = pd.read_csv(folder/'summary.csv')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, stat in enumerate(['points', 'rebounds', 'assists']):
        rows = summary[summary.stat.eq(stat)]
        baseline = float(rows[rows.model.eq('linear')].mae.iloc[0])
        improvement = 100 * (1 - rows.mae.to_numpy() / baseline)
        ax = axes[0, col]
        colors = ['#168978' if value >= 0 else '#bd695c' for value in improvement]
        ax.barh([LABELS[m] for m in rows.model], improvement, color=colors)
        ax.axvline(0, color='#9ca7b3', linewidth=1)
        ax.invert_yaxis()
        ax.set_title(stat.capitalize(), fontweight='bold')
        ax.set_xlabel('MAE reduction versus linear (%)')
        for i, (value, mae) in enumerate(zip(improvement, rows.mae)):
            ax.annotate(f'{value:+.2f}% | MAE {mae:.3f}', (max(value, 0), i),
                        xytext=(5, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=8)
        low, high = min(float(improvement.min()), 0), max(float(improvement.max()), 0)
        padding = max((high-low)*.55, .6)
        ax.set_xlim(low-padding, high+padding)
        ax.spines[['top', 'right']].set_visible(False)
        chosen = next(r for r in report['results'] if r['stat'] == stat and r['primary_development_choice'])
        linear = next(r for r in report['results'] if r['stat'] == stat and r['model'] == 'linear')
        ax = axes[1, col]
        ax.plot([.5,.9],[.5,.9],'--',color='#9ca7b3')
        for row, color, name in [(linear, '#7d91b1', 'Linear'), (chosen, '#168978', 'Development choice')]:
            buckets = [b for b in row['direction_confidence']['buckets'] if b['count'] >= 100]
            ax.plot([b['mean_confidence'] for b in buckets], [b['actual_direction_accuracy'] for b in buckets],
                    'o-', color=color, label=name)
        ax.set_xlim(.49,.9);ax.set_ylim(.48,.92)
        ax.set_xlabel('Mean stated direction confidence');ax.set_ylabel('Observed direction accuracy')
        ax.set_title('Buckets with at least 100 predictions',fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.legend(fontsize=8,loc='upper left');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Fixed-feature algorithm comparison: 2025–26',fontsize=16,fontweight='bold')
    fig.text(.5,.015,'Configurations and blends selected on earlier seasons. Accuracy uses average proxies and supplied historical absences.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.04,1,.95])
    fig.savefig(folder/'comparison.png',dpi=150)
    plt.close(fig)
    shown = summary.copy()
    for col in ['accuracy','coverage_90']:
        shown[col] = 100*shown[col]
    shown = shown.rename(columns={'accuracy':'accuracy (%)','coverage_90':'90% interval coverage (%)'})
    def table(df):
        return df.round(5).to_html(index=False,border=0,classes='metrics',escape=True)
    primary = shown[shown.development_choice.eq(True)]
    image = base64.b64encode((folder/'comparison.png').read_bytes()).decode()
    buckets = pd.read_csv(folder/'confidence_buckets.csv')
    thresholds = pd.read_csv(folder/'confidence_thresholds.csv')
    content = f'''<!doctype html><html><head><meta charset="utf-8"><title>Algorithm comparison</title><style>
body{{font:16px/1.55 system-ui;color:#1c2c3d;max-width:1350px;margin:30px auto;padding:0 22px}}h1,h2{{color:#153f51}}img{{width:100%}}.metrics{{border-collapse:collapse;font-size:13px;display:block;overflow:auto;margin:15px 0 30px}}th,td{{padding:8px 10px;border-bottom:1px solid #dbe4e9;text-align:right;white-space:nowrap}}th{{background:#e8f0f3}}tr:nth-child(even){{background:#f5f8fa}}summary{{cursor:pointer;font-weight:bold;padding:12px}}pre{{white-space:pre-wrap;background:#f5f8fa;padding:15px;font-size:13px}}</style></head><body>
<h1>Fixed-feature algorithm comparison</h1><p>Train: 2022–23 through 2024–25 (46,330 appearances). Evaluate: 2025–26 (15,572 appearances). Features remain fixed at 42/68/48 for points/rebounds/assists. Models use identical prediction rows.</p>
<p>Sixteen configurations per target were assessed on two expanding development folds. Each tree/loss family and blend was frozen before test scoring. Accuracy concerns exceeding a pregame season-average proxy, conditional on no push. Historical absences are supplied as known inputs; these are not sportsbook results.</p>
<img alt="Model error and confidence comparison" src="data:image/png;base64,{image}">
<h2>All final model results</h2>{table(shown)}
<h2>Choices selected before test evaluation</h2>{table(primary)}
<pre>{html.escape(json.dumps(report['frozen_comparison']['blends'],indent=2))}</pre>
<p>Blend tree_weight is the tree share; zero and one mean pure linear and pure tree respectively. Squared-error and MAE objectives need not have the same RMSE or probability performance. Bias is prediction minus actual; lower Brier/log loss is better.</p>
<details><summary>Confidence thresholds (proportions)</summary>{table(thresholds)}</details>
<details><summary>All confidence buckets (proportions)</summary>{table(buckets)}</details>
<details><summary>Development search results</summary>{table(pd.read_csv(folder/'development_results.csv'))}</details>
<details><summary>Exact frozen choices and feature lists</summary><pre>{html.escape(json.dumps(report['frozen_comparison'],indent=2))}</pre></details>
<h2>Interpretation limits</h2><ul>{''.join('<li>'+html.escape(item)+'</li>' for item in report['limitations'])}</ul>
<p>Complete results, date-block intervals, per-appearance forecasts, calibration rows, monthly scores and saved models are in the accompanying JSON/CSV/joblib files. No production models or database records were overwritten.</p></body></html>'''
    (folder/'results.html').write_text(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    render(parser.parse_args().folder)
