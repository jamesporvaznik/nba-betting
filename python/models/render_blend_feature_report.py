"""Render matched blend-feature/timing comparisons as a self-contained report."""

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def render(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((folder/'report.json').read_text())
    results=pd.read_csv(folder/'results.csv')
    labels={'availability_only':'Availability flags','player_profile':'Player profile',
            'opponent_profile':'Opponent profile','both_profiles':'Both profiles','interactions':'Interactions'}
    variants=list(labels)
    profile_changes=results[results.kind.isin(['playtype','shot_zone'])].mae_change_vs_baseline
    lower,upper=float(profile_changes.min()),float(profile_changes.max())
    padding=(upper-lower)*.1
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for i,kind in enumerate(['playtype','shot_zone']):
        for j,fold in enumerate(['late_2023-24','full_2024-25']):
            ax=axes[i,j]
            block=results[results.kind.eq(kind)&results.fold.eq(fold)]
            for offset,timing,color,label in [(-.18,'previous','#278b82','Previous season'),(.18,'completed_current','#d49339','Same season (retrospective)')]:
                values=[float(block[block.timing.eq(timing)&block.variant.eq(v)].mae_change_vs_baseline.iloc[0]) for v in variants]
                ax.bar(np.arange(len(variants))+offset,values,width=.35,color=color,label=label)
            ax.axhline(0,color='#8f9ba7',linewidth=1)
            ax.set_xticks(np.arange(len(variants)),[labels[v] for v in variants],rotation=18)
            ax.set_ylabel('Change in points MAE (lower is better)')
            ax.set_ylim(lower-padding,upper+padding)
            ax.set_title(f'{kind.replace("_"," ").title()} • {fold.replace("_"," ")}')
            ax.spines[['top','right']].set_visible(False)
    handles,legend=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='lower center',ncol=2,bbox_to_anchor=(.5,.015))
    fig.suptitle('Fixed-blend profile feature retest • development seasons only',fontsize=15,fontweight='bold')
    fig.tight_layout(rect=[0,.065,1,.95])
    fig.savefig(folder/'profile_comparison.png',dpi=150)
    plt.close(fig)
    def table(frame):
        return frame.round(5).to_html(index=False,border=0,classes='metrics',escape=True)
    keep=['kind','timing','stat','fold','variant','mae','rmse','accuracy','brier','mae_change_vs_baseline','paired_95pct_vs_baseline','incremental_control','incremental_mae_change']
    shown=results[keep].rename(columns={'accuracy':'proxy accuracy (proportion)'})
    image=base64.b64encode((folder/'profile_comparison.png').read_bytes()).decode()
    baseline=results[results.kind.eq('pace_defense')&results.variant.eq('baseline')][['stat','fold','mae','rmse']]
    content=f'''<!doctype html><html><head><meta charset="utf-8"><title>Blend feature retest</title><style>body{{font:16px/1.55 system-ui;color:#1c2c3d;max-width:1300px;margin:30px auto;padding:0 22px}}h1,h2{{color:#153f51}}img{{width:100%}}.metrics{{border-collapse:collapse;display:block;overflow:auto;font-size:13px;margin:15px 0 28px}}th,td{{padding:8px 10px;white-space:nowrap;border-bottom:1px solid #dbe4e9;text-align:right}}th{{background:#e8f0f3}}tr:nth-child(even){{background:#f5f8fa}}summary{{cursor:pointer;font-weight:bold;padding:12px}}pre{{white-space:pre-wrap;background:#f5f8fa;padding:15px;font-size:13px}}</style></head><body>
<h1>Fixed-blend feature retest</h1><p>25% linear regression + 75% MAE-trained CatBoost, with frozen settings and baseline features. Pace/opponent defense are tested for all three stats; playtypes and shot zones are tested for points with previous-season and completed-current-season timing.</p>
<p>Only the first three seasons are engineered/scored. Same-season profiles include future/scored games and are retrospective diagnostics. Profile comparisons have identical rows and baseline predictions across timing modes. Their smaller training splits differ from the full pace/defense folds.</p>
<h2>Pace/defense baseline scores</h2>{table(baseline)}<p>Selection requires lower MAE in both folds and at least 0.1% mean relative improvement.</p><pre>{html.escape(json.dumps(report['pace_selection'],indent=2))}</pre>
{table(shown[shown.kind.eq('pace_defense')])}<h2>Previous versus same-season profiles</h2><img alt="Matched profile error comparisons" src="data:image/png;base64,{image}">
<p>Negative changes favor the added feature. Availability-only controls distinguish profile content from knowing whether a player has a stored profile. Interactions are also compared against the additive both-profile control in report.json.</p>
{table(shown[~shown.kind.eq('pace_defense')])}<details><summary>Direct timing comparisons</summary>{table(pd.read_csv(folder/'timing_comparison.csv'))}</details>
<details><summary>Confidence buckets (proportions)</summary>{table(pd.read_csv(folder/'confidence_buckets.csv'))}</details><details><summary>Coverage and missing profiles</summary>{table(pd.DataFrame(report['coverage']))}</details>
<h2>Method and limits</h2><p>{html.escape(report['confidence_method'])}</p><ul>{''.join('<li>'+html.escape(s)+'</li>' for s in report['limitations'])}</ul><p>No database records, production models or final-test predictions were changed. Full metrics, interval coverage, probabilities and forecasts are in the accompanying JSON/CSV files.</p></body></html>'''
    (folder/'results.html').write_text(content)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    render(parser.parse_args().folder)
