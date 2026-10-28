Full-stack machine learning project predicting NBA player props.

source /Users/james/Projects/nba-betting/.venv/bin/activate

Stack
- Data: nba_api, basketball reference, nbainjuries(PyPi)
- Datbase: Supabase
- Backend: Node.js / Express
- Frontend: Next.js / TypeScript / Tailwind CSS
- Data Storage / Handling: Python

Install dependencies
- cd api && npm install && cd ../frontend && npm install
- cd ../python && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

Run on localhost
- npm run dev

Train the selected next-game stat model

From the repository root:

```bash
# Default: train/calibrate the selected blend on the first three seasons.
.venv/bin/python -B python/models/train.py

# Explicitly evaluate the fixed 2025-26 test season.
.venv/bin/python -B python/models/train.py --evaluate

# Offline reproduction without writing files.
.venv/bin/python -B python/models/train.py --snapshot python/models/artifacts/full_training_review_v1/source_snapshot.json --evaluate --no-save

# Retained older nine-feature linear/XGBoost comparison.
.venv/bin/python -B python/models/train.py --model baseline --evaluate --stats points --no-save
```

From `python`, the equivalent module command is `python -B -m models.train`.
The default model is 75% CatBoost-MAE (600 trees, depth 4, learning rate .04,
L2 penalty 10, seed 42) plus 25% linear regression. Fixed feature lists contain
42 points, 68 rebound and 48 assist features, defined in
`python/models/features/selected.py`; no ignored experiment manifest is needed.
These include rolling production, minutes/rest/opponent context, shot volume,
position-weighted missing-teammate opportunities and fading absence adjustments.
Rebounds add recent opportunity and team rebounding context; assists add recent
opportunity. Clusters, full-season playtypes/shot zones and tracking profiles are
not selected. The reusable estimator is `SelectedBlendRegressor`.

Default live loading reads only 2022-23, 2023-24 and 2024-25 in one consistent,
read-only SQL snapshot using the existing `python/.env` database configuration.
It requires `player_gamelog`, `team_defensive_stats`, `player_game_absences` and
`players`. `--snapshot PATH` uses a saved capture offline. Training-only runs
filter test records from a supplied snapshot before any feature engineering.
`--evaluate` explicitly includes and scores 2025-26. Evaluation never selects
features, changes blend weights or tunes parameters.

Histories use all stored appearances before eligibility selection. Default
eligibility requires twenty earlier season appearances and ten prior average
minutes; current-game minutes never determine inclusion. Adjust with
`--min-prior-games` / `--min-prior-avg-minutes`. Player rolling/season averages
reset by NBA season and exclude the current game. Opponent averages aggregate
complete team-game totals and use strictly earlier dates. Earlier completed
validation games can update later histories, while trained models remain fixed.

Confidence uses chronological predictions on 2023-24 from models fitted on
2022-23, and on 2024-25 from models fitted on 2022-23/2023-24. Fit the logistic
proxy-direction calibrator and 50/80/90% interval quantiles from those predictions,
then refit regressors on all three seasons. No 2025-26 outcomes enter training or
calibration. This is empirical temporal calibration after refitting, not a
coverage guarantee. Probabilities concern a season-average proxy conditional on
no push; sportsbook-line probabilities require a separate calibration exercise.
Absences are supplied historical observations and positions are current metadata;
this training setup is not yet a live pregame ingestion or forecasting service.

Each saved run creates a new `python/models/artifacts/train_blend_TIMESTAMP/`
directory (or a new `--output-dir PATH`; existing directories are refused).
It contains `blend_points.joblib`, `blend_rebounds.joblib`, `blend_assists.joblib`
(or the `--stats` subset), exact feature lists, report, restricted source snapshot
and out-of-time calibration rows. Each joblib bundle contains the fitted blend,
feature order, calibrator, interval quantiles and training metadata. Evaluation
also saves predictions, metric summary and confidence buckets. `--no-save` writes
no files or directories and cannot be combined with `--output-dir`.
When loading a saved bundle from the repository root, set `PYTHONPATH=python`
so Python can import `models.training.blend`; running from `python` also works.
The regressor is `bundle['model']` and its ordered input list is
`bundle['features']`.

Legacy baseline mode requires `--evaluate` and compares last-five, nine-feature
linear regression and fixed XGBoost. It writes into its own new run directory;
it does not overwrite old `xgboost_STAT.joblib` files. Shared metrics/confidence
code now lives under `models/evaluation/`, separate from experiment runners.

The integration was checked offline on the saved full-training snapshot: the
new entry point reproduces the selected blend's 2025-26 forecasts and calibrated
results (MAE 4.6870/1.9139/1.3892). Verification artifacts are in
`python/models/artifacts/train_entrypoint_validation_v1/`.

Experiment with pregame feature groups

```bash
.venv/bin/python -B -u python/models/experiment_features.py --output-dir python/models/artifacts/new_feature_experiment
```

Run this command from the repository root. It captures only 2022-23 through
2024-25 game logs and team defense data in a read-only, consistent database
snapshot. `--snapshot PATH` reuses a saved `source_snapshot.json` offline. The
output directory must be new and is covered by the model-artifact ignore rule.
The experiment never reads or scores 2025-26, writes database records, or
overwrites existing fitted models. The production baseline feature list remains
unchanged; experimental features live in `models/features/matchup.py`.

Five predefined variants test baseline, pace, opponent defense, player
opportunity, and their combination with fixed linear/XGBoost settings. Validation
folds train on 2022-23 and score 2023-24, then train on 2022-23/2023-24 and score
2024-25. All variants use the same eligible appearances per target and fold, with
coverage recorded. The runner saves exact features, predictions, error metrics,
pregame-average proxy direction scores, and paired date-block bootstrap intervals
for MAE changes. These exploratory intervals do not adjust for multiple
comparisons or feature-selection uncertainty.

Pace uses prior team/opponent season and last-ten game pace averages. Defense
uses prior defensive ratings, shooting/FTA ratios calculated from cumulative
counts, and last-ten target totals allowed. Opportunity uses prior minutes
variability/trends, cumulative and recent production per minute, and simple
minutes-times-rate interactions. These are historical proxies, not an injury-aware
minutes forecast. No current-day team results or full-season player profiles are
used as predictors. All team and player histories reset each NBA season.

The first run (`python/models/artifacts/pregame_feature_experiment_v1/`) found
small, consistent opportunity-feature gains, mixed pace/defense results, and no
clear XGBoost advantage over linear regression. The improvements do not establish
a meaningful betting advantage. All eligible appearances had complete experimental
features. No experimental models were promoted.

Test offensive volume and involvement using the same validation design:

```bash
.venv/bin/python -B -u python/models/experiment_features.py --suite offense --output-dir python/models/artifacts/new_offense_experiment
```

This suite additionally captures game-level FGA, FTA, and three-point attempts.
It compares the baseline, previous opportunity features, shot volume, involvement,
opportunity plus shots, and the combined offensive features. Shot features use
earlier 5/10-game and season averages, attempt trends, and attempts per minute.
Involvement measures earlier shares of team FGA, FTA, and assists plus their
trends and turnovers per minute. Shares use complete team-game totals for the
player's team in each historical game, then shift the resulting appearance
history. A team with zero FTA/assists contributes a zero share for that category.
These are descriptive involvement proxies, not official usage percentage.
No current-game shot counts or teammate outcomes are model inputs. Snapshot
replay requires the offense suite's expanded source snapshot. Results and source
data for the first run are in `python/models/artifacts/pregame_offense_experiment_v1/`.
That run reproduced the earlier batch's exact eligible cohorts and baseline
predictions. Shot-volume features reduced points MAE for both models in both
validation seasons, by roughly 0.015–0.026 points; gains remain below 1%.
Combining all offensive features did not consistently beat the simpler additions.
Linear regression still outperformed XGBoost. The features remain experimental;
no model was promoted and 2025-26 was not read or scored by the experiment.

Evaluate confidence and accuracy using chronological calibration

```bash
.venv/bin/python -B -u python/models/evaluate_confidence.py --snapshot python/models/artifacts/pregame_offense_experiment_v1/source_snapshot.json --output-dir python/models/artifacts/new_confidence_review
```

Run from the repository root. This offline development assessment trains fixed
baseline and shot-volume regressors on 2022-23, calibrates uncertainty on 2023-24,
and evaluates on 2024-25. Models are not refitted on the calibration data. It uses
less training data than the two-season feature comparison, and 2024-25 has already
been explored; results are not a fresh final holdout. No 2025-26 data is used.

Prediction intervals use split-conformal absolute residuals normalized by each
player's prior ten-appearance stat standard deviation (floor one), with finite
sample order-statistic thresholds at nominal 50/80/90% coverage. Actual coverage
and interval widths are reported. Temporal dependence and season drift prevent
assuming future coverage guarantees from these descriptive checks alone.

A one-feature logistic direction calibrator learns from 2023-24 outcomes using
the regressor's distance from the pregame season-average proxy divided by the
same prior-only scale. Its probabilities concern above/below that proxy,
conditional on a non-push. Reported accuracy by confidence bucket is measured on
2024-25, with counts, Brier score, and log loss; very small high-confidence bins
are unstable. This is not the earlier arbitrary sigmoid conversion, sportsbook
accuracy, or evidence of profitability. Reports, per-appearance intervals and
probabilities, and new calibration artifacts are saved only to the new directory.
Prior model files and database rows remain unchanged. The first results are in
`python/models/artifacts/confidence_review_v1/`.

Build historical player game absences

```bash
.venv/bin/python -B python/nba/build_player_absences.py --prepare database/repairs/new_player_absences
.venv/bin/python -B python/nba/build_player_absences.py --apply database/repairs/new_player_absences
```

Preparation reads live game logs, team game dates, team identifiers and player
team stints for 2022-23 through 2025-26. It saves a source snapshot, exact rows,
checksums, counts and schema SQL. An absence means a team played strictly between
the player's first/last recorded appearances in that team stint, with no player
game-log row on that date. Reasons are not inferred. Any existing player log,
including zero minutes, prevents an inferred absence. No dates before/after a
stint or between teams are extrapolated; missing rows do not establish that a
player was available. Validation rejects duplicate dates, ambiguous/overlapping
stints, missing calendar games and appearances outside supplied stints.
Team-calendar dates with no player appearances for that entire team are unknown
coverage and generate no absences. The initial builder omitted this coverage
check. The correction, backed up in
`database/repairs/player_absences_unobserved_team_games/`, removed 98 records
(30/34/34 in 2023-24/2024-25/2025-26) directly from Supabase, leaving 43,774.
Remaining rows were checked against the corrected calculation before commit
and again independently afterward. The targeted repair is available as
`--repair-unobserved-games NEW_BACKUP_DIRECTORY`; it refuses unrelated changes.

Application rechecks live source checksums while locking source tables against
writes, creates `public.player_game_absences`, inserts the prepared rows, and
compares all stored fields before commit and independently after commit. It
refuses to overwrite an existing target table. Source tables are read only.
The new table enables row-level security without public client policies;
database/service-role access can read it. Add application access separately if
needed. This is a historical table, with `pregame_confirmed=false`; it does not
provide timestamped pregame availability. Inferred stint boundaries also use
later appearances. Model experiments using these rows must distinguish a known
absence scenario from a forecast of availability.

Retest matchup features with the fixed linear/CatBoost blend

```bash
.venv/bin/python -B python/models/retest_blend_features.py --training-source python/models/artifacts/full_training_review_v1 --blend-source python/models/artifacts/algorithm_comparison_v1 --output-dir python/models/artifacts/new_blend_feature_retest
```

The regression/loss/weight settings remain fixed at the previously selected
25% linear / 75% CatBoost-MAE blend. Profiles are captured directly from live
Supabase in one read-only transaction, querying only 2022-23/2023-24/2024-25.
Reuse its `profile_snapshot.json` with `--profile-snapshot PATH` for offline
reproduction. The saved training snapshot contains four seasons; every outcome
table is filtered to the first three before histories are engineered. No 2025-26
profiles are queried and no 2025-26 observations are engineered or scored.

First compare baseline, prior-game pace, prior-game opponent defense and both
for each target on the full expanding development folds. Keep a pace/defense
addition only if it lowers MAE in both folds and mean relative MAE by >=.1%.
Then test playtype and shot-zone additions for points, with the selected pace
addition held identical in all profile variants. Each has a matched baseline,
availability-flags-only control, player profile, opponent profile, both profiles
and explicit matchup interactions. Model hyperparameters/weights are not retuned.

For each profile source, compare previous-season and completed-current-season
timing on identical rows and folds. Current timing is an explicitly retrospective
diagnostic: it includes scored/future games in both player and opponent profiles.
Columns/reports label it `retrospective_*`/`retrospective_lookahead`; it cannot
drive a prospective feature recommendation. Since profiles start in 2022-23,
both timing modes use early eligible 2023-24 -> February onward, then eligible
2023-24 -> 2024-25. Pace comparisons have larger training splits, so their raw
scores should not be compared directly with profile scores. Every variant retains
players without profiles using training-only median imputation and missing flags.

For confidence, split each outer training set by unique dates: the first 75%
fit an inner blend, the final 25% calibrate probabilities and interval widths.
Refit the same blend on all outer training rows before validation. No validation
outcome fits the calibrator; completed-current profile features nevertheless
contain deliberate future information. This is empirical temporal calibration,
not guaranteed split-conformal coverage. All seasons are previously explored
development data, so the comparisons are exploratory.

Outputs include per-appearance predictions/probabilities/intervals, MAE/RMSE/bias,
proxy accuracy, Brier/log loss, confidence buckets, matched timing comparisons,
paired date-block intervals against baseline and relevant profile controls,
coverage counts and missing-profile counts. The experiment does not replace any
production model or change the database.

The completed `blend_feature_retest_v1` has 72 matched comparisons. No
pace/defense addition met the preset improvement threshold for any target. Using
player and opponent profiles together, points MAE on late 2023-24 / full 2024-25
was baseline 4.7770 / 4.8165; previous playtypes 4.7730 / 4.8072; current
playtypes 4.7635 / 4.8063; previous shot zones 4.7719 / 4.8109; current shot
zones 4.7675 / 4.8115. Current playtype combined-profile gains had paired
date-block intervals below zero in both folds versus baseline, but differences
versus the availability-only control remained uncertain. Previous-profile
content did not consistently beat its availability control. Explicit interaction
features did not consistently beat additive profiles. Most profile changes were
around .1-.3% of points MAE and did not establish a large practical improvement.
No feature was promoted and 2025-26 was not re-engineered or re-scored.

All original full-fold blend baseline MAEs were reproduced, timing-mode baseline
rows/predictions/probabilities/intervals match, and all 72 exported metric and
confidence-bin counts were checked. Render the human report with:

```bash
MPLCONFIGDIR=/tmp/nba-model-comparison-mpl .venv/bin/python -B python/models/render_blend_feature_report.py python/models/artifacts/blend_feature_retest_v1
```

The self-contained `results.html` includes a reviewed chart, all controls, exact
metrics, confidence buckets, coverage and direct timing comparisons. Raw results
are in `results.csv`, `timing_comparison.csv` and `report.json`.

Compare algorithms with the frozen optimized features

```bash
.venv/bin/python -B python/models/compare_algorithms.py --source-dir python/models/artifacts/full_training_review_v1 --output-dir python/models/artifacts/new_algorithm_comparison
```

This offline comparison reuses the prior live-database snapshot and exact frozen
42/68/48 feature lists. It neither reads nor writes Supabase. Install the pinned
CatBoost dependency from `python/requirements.txt` first. The prespecified grid
contains linear regression, the previous XGBoost configuration, four regularized
XGBoost configurations for each of squared-error and absolute-error objectives,
and CatBoost depths 4/6/8 for each of RMSE and MAE (600 iterations, learning rate
.04, L2 penalty 10). All use seed 42 and one compute thread. There is no early
stopping against the final test season.

For each target, fit each candidate on 2022-23 to validate 2023-24, then fit on
2022-23/2023-24 to validate 2024-25. Select configurations within each tree/loss
family by equal mean validation MAE relative to linear in each fold. Test blend
tree weights 0/.25/.5/.75/1 with those family winners; learn both tree choice and
weight from these development predictions. Endpoints explicitly allow a pure
model. Freeze every target's configuration, blend and overall development choice
in `frozen_comparison.json` before any final prediction. Features stay fixed even
when a different algorithm might favor different features.

Refit the selected models on all three training seasons, score the identical
2025-26 appearances, and verify the linear reference reproduces the previous run.
Report linear, previous XGBoost, tuned XGBoost for both losses, CatBoost for both
losses and the selected blend. Confidence calibration reuses expanding development
predictions; no test label fits probabilities or interval widths. Outputs include
development search scores, all frozen choices, paired date-block MAE comparisons,
RMSE/bias/R2, proxy accuracy, Brier/log loss, interval coverage/widths, confidence
bins/thresholds, monthly scores, predictions, calibration rows and saved models.
The tuning/calibration reuse of development predictions may make calibration
optimistic; the test assessment measures how it transfers after refitting.
2025-26 has already been inspected, and known historical absences/current position
metadata remain conditional inputs. Results do not establish betting profitability.

The completed `algorithm_comparison_v1` run evaluated sixteen configurations per
target (96 development fits) and seven final comparisons per target. Development
selected the same 25% linear / 75% CatBoost-MAE blend for all three targets.
Its CatBoost component has 600 trees, depth 4, learning rate .04 and L2 penalty 10.
On 2025-26, blend MAE was 4.6870/1.9139/1.3892 versus linear
4.7509/1.9320/1.4156: reductions of 1.35%/.93%/1.86%. Paired date-block MAE
intervals favored the blend in all three comparisons. Pure CatBoost-MAE had
slightly lower point/assist test MAE, but no weights or model choices were changed
after inspecting these results. MAE objectives produced larger gains than tuning
squared-error objectives. Blend RMSE improved for points and was slightly worse
for rebounds/assists, so the MAE improvement is not universal across metrics.

Blend calibrated proxy direction accuracy was 59.45%/57.37%/58.27%, versus
58.86%/57.17%/57.38% for linear. It had the lowest observed Brier/log loss for
all targets. Nominal 90% intervals covered 90.62%/90.18%/90.30% of outcomes.
High-confidence buckets remain variable and are not sportsbook probabilities.
Original linear/XGBoost predictions were reproduced exactly; all 21 saved models
were reloaded and checked against recorded forecasts. Calibration chronology,
saved metrics and confidence-bin counts were verified for every comparison.

Render the completed report with:

```bash
MPLCONFIGDIR=/tmp/nba-model-comparison-mpl .venv/bin/python -B python/models/render_algorithm_report.py python/models/artifacts/algorithm_comparison_v1
```

The resulting `results.html` is self-contained and includes the checked
`comparison.png`, all metrics, confidence bins/thresholds, development scores and
frozen choices. `report.json` and `summary.csv` preserve complete numeric results.
These models remain separate local candidates; no database or existing production
models were changed.

Train three seasons and evaluate calibrated 2025-26 predictions

```bash
.venv/bin/python -B python/models/full_training_review.py --output-dir python/models/artifacts/new_full_training_review
```

This reads fresh game logs, team histories, the corrected absence table and
player positions in one read-only database snapshot. Reproduce offline with
`--snapshot PATH`. Final regressors fit eligible 2022-23/2023-24/2024-25 rows and
evaluate 2025-26. Eligibility is twenty prior season appearances and at least ten
prior average minutes; only observed appearances are scored. Test-game outcomes
update histories only after their dates; fitted models/calibrators stay fixed.

The primary model family is fixed to linear regression before scoring. Features
start with the established baseline, shot volume and position-weighted absence
loads/adaptation. Recent player opportunity/rate features and (rebounds only)
team rebounding context are considered on expanding development folds only.
Retain additions only when OLS MAE improves in both folds and mean relative MAE
improves by at least 0.1%; choose the largest qualifying average improvement.
`frozen_features.json` is written before any test prediction. Compare the selected
features with shot-volume controls and XGBoost using the same inputs and rows;
last-five and season-average naive scores are included. No clusters, lagged
tracking percentages or completed-season matchup profiles enter these models.

Confidence calibrators use 2023-24 predictions from models fitted on 2022-23 and
2024-25 predictions from models fitted on 2022-23/2023-24. Fit a logistic
calibrator to the scaled prediction margin over the pregame season-average proxy,
excluding actual ties. Interval widths use historical out-of-time absolute errors
divided by prior ten-appearance standard deviation (floor one), with finite-sample
50/80/90% quantiles. Transfer these calibrators to regressors refitted on all three
seasons. This is empirical temporal calibration, not strict split-conformal
coverage after refitting. Feature selection and calibration reuse development
folds, but no test labels tune features, intervals or direction probabilities.

Results include MAE/RMSE/bias/R2, MAE and accuracy date-block intervals, calibrated
and raw proxy direction accuracy, Brier/log loss and constant-direction controls,
confidence bins/threshold counts, observed interval coverage and widths, monthly
scores, per-appearance predictions and training-period calibration predictions.
New regressors and calibrators are saved only in the output folder. Optimized
results assume supplied actual absences and current master-table positions;
they are not timestamped pregame availability forecasts. The 2025-26 season was
already scored in the original baseline review, so it is not a pristine unseen
holdout. Proxy accuracy is not sportsbook accuracy. No database records or
existing production model files are changed.

The completed run is `python/models/artifacts/full_training_review_v1/`, with
46,330 training appearances and 15,572 test appearances from 409 players. Fixed
feature selection retained 42 points features (absence core), 68 rebound
features (core + recent opportunity + team rebounding), and 48 assist features
(core + recent opportunity). Selected linear MAE was 4.7509/1.9320/1.4156;
XGBoost was 4.7541/1.9422/1.4216. Calibrated proxy direction accuracy was
58.86%/57.17%/57.38% for linear and 59.47%/57.01%/57.55% for XGBoost. Linear
90% interval coverage was 90.39%/90.28%/90.14%, with average full widths
19.98/8.00/5.38. Compared with shot-volume controls, paired intervals favored
the selected point/rebound models, while assist MAE gains were uncertain.
High confidence bins above 80% were small and overconfident overall. No model
family or feature package was changed in response to these test scores.

`results.html` is a self-contained human report with the reviewed chart;
`report.json` contains complete numeric results/limitations, `summary.csv` all
twelve comparisons, and `frozen_features.json` the exact feature lists. Every
saved prediction file was checked against reported MAE, all confidence bin counts
were reconciled, and saved regressor dimensions match their feature lists.
Calibration predictions were verified to come from strictly earlier training
dates, with no 2025-26 calibration rows. All twelve regressor/calibrator artifacts
are saved in this new directory; prior saved production models are untouched.

Test position-weighted absence opportunities and adaptation

```bash
.venv/bin/python -B python/models/experiment_absences.py --game-snapshot python/models/artifacts/pregame_offense_experiment_v1/source_snapshot.json --output-dir python/models/artifacts/new_absence_experiment
```

The default reads live `player_game_absences` and `players.position` for the
2022-23 through 2024-25 experiment. Supply `--supplemental-snapshot PATH` to
replay the saved experiment inputs. Game logs reuse the earlier complete
modeling snapshot. No 2025-26 observations are used and no clusters are included.
This is conditional on knowing who actually missed each game, rather than a
forecast of availability or timestamped pregame validation. Positions are current
master-table labels, not historical snapshots; the actual labels are mostly
Guard/Forward/Center and hybrids. Broad G/F labels expand to both detailed slots.
Identical position sets receive weight 1, overlapping sets .6, disjoint/unknown
sets .3. Missing-position minutes are separately flagged.

At each absence onset, freeze the teammate's last ten appearances for the same
team within that stint. Frozen loads include average minutes, estimated used
possessions (FGA + .44*FTA + TOV), estimated usage fraction times minutes, FGA,
assists and rebounds. The box-score usage fraction is pooled prior player used
possessions times (prior team minutes / 5), divided by (prior player minutes times
prior team used possessions). It is an estimate, not official play-by-play usage.
Profiles with fewer than five prior appearances are counted explicitly. A
continuous episode requires consecutive missed team games; a return resets it.
Dates with no team player logs are unknown coverage and must not be inferred
absences. Any existing player log, including zero minutes, prevents an absence.

For each remaining player, sum the missing loads with and without position
weights. The adaptation features multiply each position-weighted missing load by
2**(-completed appearances by the target during this episode / 5). They retain
100% at onset, 50% after five appearances, 25% after ten and 6.25% after twenty.
Ongoing missing loads remain alongside the fading disruption features. Current
and later outcomes never enter a frozen profile; the current target appearance
does not count toward adaptation. Rolling baseline features continue updating.

Compare baseline + shot volume, unweighted absences, added position weights, and
added adaptation features using fixed OLS/XGBoost for all three targets. Splits
are 2022-23 -> 2023-24 and 2022-23/2023-24 -> 2024-25 with identical appearances.
Reports include overall results, missing-20-plus-minutes and recent-disruption
subgroups, paired date-block intervals against baseline and incremental controls,
and uncalibrated season-average proxy direction scores. Existing production
models are not overwritten.

The reviewed run is `python/models/artifacts/absence_experiment_v2/` (v1 stopped
on the unknown-team-game coverage check before fitting). It captures the corrected
live absence table and current player positions. All earlier offense-experiment
appearance keys and baseline predictions match. The complete feature set improved
MAE for all three targets in both folds and both model types. In 2024-25, linear
MAE changed points 4.8834 -> 4.8389, rebounds 2.0054 -> 1.9999, and assists
1.4331 -> 1.4247. XGBoost changed 4.9110 -> 4.8433, 2.0156 -> 1.9999, and
1.4431 -> 1.4272 respectively. Paired date-block intervals favored the complete
set over baseline in all twelve comparisons. Position weights' incremental
benefit was mixed; adaptation helped several linear comparisons and 2024-25
XGBoost points. The five-appearance half life is an initial fixed setting, not a
selected optimum. These are modest conditional-known-absence development gains;
no production model was replaced. `results.csv` and `slice_results.csv` preserve
all controls, metrics, counts and intervals.

Test rolling-baseline player-cluster matchups for all three stats

```bash
.venv/bin/python -B -u python/models/experiment_cluster_matchups.py --game-snapshot python/models/artifacts/pregame_offense_experiment_v1/source_snapshot.json --output-dir python/models/artifacts/new_cluster_matchup_experiment
```

This reads assignments directly from live `public.player_clusters` in a read-only
transaction; it does not use the old repair backups. The game logs reuse the
earlier experiment snapshot. Only 2022-23 and 2023-24 cluster assignments are
queried, then lagged into 2023-24 and 2024-25. Cluster IDs select peers within each
season; they are neither numeric model inputs nor matched between seasons.
The captured assignments are saved for reproducibility; `--cluster-snapshot PATH`
can explicitly replay that experiment snapshot offline.

For each completed appearance, calculate points/rebounds/assists minus the
player's prior five- or ten-appearance average. A second residual subtracts
actual minutes times the ratio of prior-window stat totals to minutes totals.
Actual minutes are used only in completed historical appearances, never as a
prediction input for the upcoming game. Peer source appearances require ten
earlier games and at least ten average minutes over those ten games; eligibility
never depends on minutes or production in the source game itself.

Before a prediction, aggregate strictly earlier dates against the upcoming
opponent in the same season, excluding the target player's own performances.
Weights halve every 30 calendar days. Each effect is the weighted residual sum
divided by (sum of weights + 20), shrinking sparse histories toward zero.
Weight mass is included as a feature, but is not an independent-game sample
size. Entire dates are predicted before adding their outcomes. Histories reset
each NBA season, follow the actual opponent in each game, and do not filter
players according to their current roster. Unassigned players stay in the
evaluation with zero cluster effects/count and an availability flag.

Fixed linear and XGBoost models compare the existing baseline plus shot-volume
features against: an availability-only control, general opponent residuals,
cluster peer residuals, both types together, and a five-game version of both.
The existing baseline features remain unchanged; only the new matchup residuals
use rolling averages instead of season averages. Also report the subgroup with
available clusters using predictions from the same fitted models.

Use the prior profile experiments' chronological splits: eligible 2023-24 games
before February 1 train the later 2023-24 evaluation; then all eligible 2023-24
games train the 2024-25 evaluation. Baselines are retrained on identical rows.
Models stay fixed within a validation fold while earlier completed validation
games update histories. No 2025-26 game data is read or scored. Reports include
MAE/RMSE/bias, season-average proxy direction accuracy, and paired date-block
bootstrap intervals. These are development comparisons, not sportsbook accuracy,
fresh holdout evidence, or a test of dated current-season clusters. Database rows
and saved production models are unchanged.

The reviewed run is `python/models/artifacts/cluster_matchup_experiment_v2/`
(72 fits, fresh live database assignments). Ten-game cluster-only linear MAE
on 2024-25 was points 4.8788 -> 4.8823, rebounds 2.0058 -> 2.0053, and assists
1.4268 -> 1.4270. Five-game and combined variants did not establish a consistent
improvement across folds. About 21.3% of 2024-25 appearances lacked a lagged
assignment and were retained. The assigned-player subgroup showed a small
2024-25 rebound improvement, but it did not repeat in the earlier fold; assists
had a small earlier-fold improvement without clear later-fold confirmation.
Availability-only controls and assigned-player subgroup results are recorded
separately. Prior points/rebounds cohorts and baseline predictions reproduce
the earlier profile experiments. These features remain experimental.

To compare each season's own completed-season clusters, add
`--profile-timing completed_current` and use a new output directory. This queries
live 2023-24/2024-25 assignments and matches them to those same seasons. Reports
set `retrospective_lookahead=true` and label assignment fields `retrospective_*`.
Peer game histories and rolling baselines still exclude the prediction date,
but the completed-season assignment itself includes future information. This is
a retrospective diagnostic, not a simulation of available pregame clusters.
All models, splits, rolling windows, decay and shrinkage settings stay fixed,
allowing direct paired comparison with the previous-season experiment.

The completed-current diagnostic is saved in
`python/models/artifacts/cluster_matchup_current_retrospective_v1/`. Identical
appearance keys and baseline/general-opponent predictions were verified against
the previous-season run; `timing_comparison.csv` records paired comparisons.
Ten-game cluster-only linear MAE in 2024-25 was points 4.8788 -> 4.8772,
rebounds 2.0058 -> 2.0040, and assists 1.4268 -> 1.4251. Assists improved in
both evaluation periods, including versus the assignment-availability control.
The rebound gain was clearer only in 2024-25; point changes were uncertain.
XGBoost assists improved slightly, while points/rebounds did not consistently
improve. Missing assignments fell from 21.3% to 2.3% of 2024-25 appearances, so
the timing comparison changes both coverage and membership. Gains remain small
and retrospective; these are not validated prospective model replacements.

Test player tracking and team rebounding features

```bash
.venv/bin/python -B -u python/models/experiment_rebounding.py --game-snapshot python/models/artifacts/pregame_offense_experiment_v1/source_snapshot.json --output-dir python/models/artifacts/new_rebounding_experiment
```

Run from the repository root. The read-only capture includes team game counts
for 2022-23 through 2024-25 and only 2022-23/2023-24 player tracking profiles.
Replay entirely offline with `--supplemental-snapshot PATH`. The actual table is
`player_tracking_stats`; no `player_tracking_data` table exists. Its four stored
rebounds fields are `reb_chance_pct`, `dreb_chance_pct`, `oreb_chance_pct`, and
`reb_contest_pct`. These are lagged one NBA season, with missing-profile flags
and train-only median imputation. No raw rebound-chance counts are stored there.

Team histories use strictly earlier dates in the current season. In a row for
team X, `opponent_orb` means offensive boards conceded by X. Joining the upcoming
opponent's history therefore describes its allowed offensive rebounds, not its
own offensive rebounds. Defensive boards are TRB minus ORB. Reciprocal team-game
rows recover own shooting misses; opponent misses are opponent FGA minus FGM.
Offensive/defensive rebound rates use ratios of earlier rebound counts. Last-ten
and season averages reset each NBA season, including all stored game types.
Two tracking-by-opponent products are fitted interactions, not individual rebound
opportunity forecasts.

Lagged-profile folds compare baseline, team context, tracking, both, and their
interactions on early/late 2023-24 and 2023-24/2024-25 splits. Additional team-only
controls use the original full expanding-season folds and reproduce the exact
original direct-model cohorts and baseline predictions. Tracking is not included
in these controls. The baseline remains direct rebounds prediction using the
existing pregame/shot-volume features; no validation appearances are removed for
tracking availability.

The reviewed run, `python/models/artifacts/rebounding_experiment_v2/`, found
small team-feature improvements for linear regression in the larger training
folds (MAE 1.9611 -> 1.9585 and 2.0054 -> 2.0030), but no consistent benefit
from the previous-season tracking percentages. These are exploratory development
results with date-block paired intervals, not fresh holdout or sportsbook
performance. No model was promoted, no database rows or existing artifacts
were overwritten, and 2025-26 was not used.

Test previous-season shot-zone matchup profiles

```bash
.venv/bin/python -B -u python/models/experiment_shot_zones.py --game-snapshot python/models/artifacts/pregame_offense_experiment_v1/source_snapshot.json --output-dir python/models/artifacts/new_shot_zone_experiment
```

Run from the repository root. This captures only completed 2022-23/2023-24 player
and opponent shot-zone profiles with a consistent read-only transaction. The
opponent's NBA team ID is joined to its abbreviation via `teams`. Replay offline
with `--profile-snapshot PATH`. It uses the same adjusted chronological splits,
baseline, parameters, and appearances as the previous-season playtype experiment.
No 2025-26 data or current-season completed profiles enter the model.

The five variants compare baseline, player six-zone frequencies, opponent zone
FG percentages and frequencies, both profiles, and interactions. Opponent rates
are centered on the average team for each zone in the prior season, not an exact
shot-weighted league FG percentage. Scoring interactions multiply player zone
share by the opponent FG-percentage gap and the zone's two/three-point value.
Frequency interactions measure alignment between player shooting locations and
the opponent's allowed locations. Backcourt shots and free throws are outside
these six-zone interactions. Missing player profiles remain in the cohort with
indicators and train-only median imputation. Missing zones within an available
player profile become zero. Entirely unavailable or incomplete opponent profiles
remain flagged/missing rather than silently receiving average defense.

The first run, `python/models/artifacts/shot_zone_experiment_v1/`, reproduced the
exact playtype-experiment cohorts and baseline predictions. Interactions did not
consistently improve points MAE: linear errors were slightly worse in both folds;
XGBoost's tiny improvement in the first fold did not persist in the second.
Confidence intervals are exploratory, date-block paired, and not adjusted for
the sequence of feature experiments. No experimental model was promoted, and no
database rows or existing model artifacts were changed. Older profiles can miss
current-season roles; the result concerns this feature setup, not every possible
use of shot-location data.

Test previous-season playtype matchup profiles

```bash
.venv/bin/python -B -u python/models/experiment_playtypes.py --game-snapshot python/models/artifacts/pregame_offense_experiment_v1/source_snapshot.json --output-dir python/models/artifacts/new_playtype_experiment
```

Run from the repository root. This reads only completed 2022-23/2023-24 player
and opponent playtype profiles in a consistent read-only transaction. Use
`--profile-snapshot PATH` to replay offline. Player proportions multiply opponent
PPP differences from the possession-weighted league mean for each playtype. Five
variants compare baseline, player profiles, opponent profiles, both profiles
without interactions, and both plus interactions. Missing player profiles remain
in the cohort with availability flags; imputation uses training medians only.
Unreported types within an available player profile are zero-filled.

Because 2021-22 profiles are unavailable, a full 2023-24 validation fold would
have no earlier training data with lagged profiles. All variants instead train
on eligible 2023-24 games before 2024-02-01 and validate later that season, then
train on eligible 2023-24 and validate 2024-25. Baselines are retrained on those
same splits. These scores must not be compared directly with earlier experiments
using different training periods. The previous-season run is
`python/models/artifacts/playtype_experiment_v1/`. Its interactions did not
consistently improve points error; no new model was promoted.

An explicitly retrospective current-season comparison is available with
`--profile-timing completed_current`. It reads completed 2023-24/2024-25 profiles
and uses the identical game folds, with all new profile columns named
`retrospective_*`. Those profiles contain current/future-game information and
cannot establish attainable pregame performance. This is not a season-to-date
profile or a tested switch from prior-year to current-year data. Dated in-season
snapshots would be required for that comparison. The completed diagnostic is
`python/models/artifacts/playtype_current_retrospective_v2/`, with a saved
`timing_comparison.csv` verifying identical cohorts and baseline predictions.
Neither timing experiment overwrites model artifacts or database rows, and
2025-26 profiles or games are not used in modeling.

Test separate minutes and production-rate models

```bash
.venv/bin/python -B -u python/models/experiment_minutes.py --experiment-dir python/models/artifacts/pregame_offense_experiment_v1 --output-dir python/models/artifacts/new_minutes_experiment
```

Run from the repository root. This offline experiment reuses the offense snapshot
and exact saved direct-model forecasts for the same expanding 2023-24/2024-25
validation folds. It never reads Supabase or 2025-26 and never overwrites existing
production models. All previous eligible appearances remain in evaluation,
including short-minute games. DNP/availability forecasting is outside this
appearance-only dataset.

Minutes forecasts compare prior five/ten-game and season averages with fixed
linear/XGBoost models using prior minutes variability/trends, rest, home/away, and
earlier offensive involvement. Rate models use the existing direct features plus
prior per-minute production, train on observed stats divided by observed minutes,
and weight training appearances by minutes (normalized to mean one). Undefined
zero-minute rate labels are excluded only from rate training. Actual validation
minutes never enter any forecast. Predicted minutes and rates are clipped at zero
before multiplying; minutes are not capped at 48 because of overtime.

Separate ablations compare recent minutes/recent rates, learned minutes/recent
rates, recent minutes/learned rates, and both learned components. Direct models
with the union of both stages' input features control for additional information.
The report includes MAE/RMSE/bias, average-proxy direction accuracy, paired
date-block intervals, component forecasts, and cohort checks. Existing confidence
calibrators do not automatically apply to these new forecasts. Development
comparisons and their intervals remain exploratory rather than final holdout
evidence. The reviewed run is `python/models/artifacts/minutes_experiment_v2/`.
In that run, linear minutes MAE improved from 5.013 to 4.949 in 2023-24 and
5.010 to 4.886 in 2024-25. The two-stage linear model essentially tied direct
points forecasts, slightly reduced rebound error (not clearly in the later
fold), and reduced assist error in both folds. The assist advantage remained
small after comparing against direct models with matched inputs. No variant
was promoted; these results do not justify replacing all direct models.

Diagnose saved predictions by time, role, and confidence

```bash
.venv/bin/python -B -u python/models/diagnose_predictions.py --experiment-dir python/models/artifacts/pregame_offense_experiment_v1 --confidence-dir python/models/artifacts/confidence_review_v1 --output-dir python/models/artifacts/new_prediction_diagnostics
```

Run from the repository root. This uses only saved pre-2025-26 snapshots and
predictions, without refitting or database access. It compares linear baseline
and shot-volume errors against last-five and season-average predictions across
both development folds. Confidence diagnostics separately compare the calibrated
2024-25 model with its identically trained/calibrated baseline, a fixed direction
learned from the calibration season, and last-five-average directions. Last-five
direction comparisons exclude its exact prediction ties on both sides.

Slices use month, pregame season scoring average (<10/10–20/20+), prior five-game
minutes (<20/20–30/30+), minutes change from the prior season average (at least five
minutes up/down), and relative FGA change (at least 25% up/down). No slice uses
current-game minutes or outcomes. Confidence is also checked within months and
minutes-role groups. Reports include sample/player/date counts, prediction
interval coverage, paired date-block bootstrap intervals, and small-sample flags.
The many exploratory slices are not multiple-testing-adjusted; repeated players
can remain dependent across dates. Eligibility excludes the first twenty player
appearances, so these are mostly December–April diagnostics, not early-season
validation. The reviewed run is `python/models/artifacts/prediction_diagnostics_v2/`.

Audit collected game logs (read-only)

With your Python environment activated, run from the `python` directory:

```bash
python -m nba.audit_gamelogs --season 2024-25
python -m nba.audit_gamelogs --season 2022-23 2023-24 2025-26 --json /tmp/nba-gamelog-audit.json
```

The command reads `SUPABASE_URL` and `SUPABASE_KEY` from `python/.env`
or environment variables. It compares NBA regular-season appearances with
all stored game logs, reporting missing and unexpected appearances, duplicate
records, and NBA players absent from the `players` table. Matching uses player,
date, team, and opponent because stored game IDs currently use a custom format.
It checks appearance coverage, not individual box-score values or stint records.
No database records or ingestion scripts are changed. `--json` optionally writes
all discrepancies to a new local file; existing files are never overwritten.
Console output includes samples; use `--sample-size 25` to show more details.
Exit codes: `0` = clean, `1` = discrepancies, `2` = incomplete audit or error.
Results reflect the NBA response at audit time; an ongoing season can change.

Prepare a game-log repair (read-only until you execute the exported SQL)

```bash
python -m nba.repair_gamelogs --season 2025-26 --audit /tmp/nba-gamelog-audit.json
python -m nba.repair_gamelogs --season 2025-26 --audit /tmp/nba-gamelog-audit.json --sql /tmp/nba-2025-26-repair.sql
```

This separate command checks the audit against fresh NBA and database reads,
fetches missing player profiles and box scores, and previews stint changes.
It does not modify the database or existing ingestion scripts. `--sql` exports
a new SQL file; it never executes SQL or overwrites an existing file.
Review that file, then run its entire contents in Supabase SQL Editor.
The transaction adds missing players/game logs, preserves existing box scores,
and reconciles only affected players' stints for the selected season, including
removing obsolete stint numbers if missing games change the inferred sequence.
It validates coverage before committing and refuses an unexpected database
snapshot. No other seasons are changed. An audit checks appearances, not the
accuracy of every stored statistic. After applying the SQL, rerun the audit.

Prepare season-long player play-type totals (read-only preview)

```bash
python -m nba.aggregate_playtypes --season 2022-23 2023-24 2024-25 2025-26 --sql /tmp/player-playtype-season-totals.sql --json /tmp/player-playtype-season-totals-review.json
```

This keeps the existing player/season/play-type table. It combines distinct
team contributions, prefers an existing NBA whole-season total when supplied,
and normalizes possession frequencies after aggregation. Merged GP sums source
team GPs for that category; it is not necessarily every appearance that season.
FG% uses combined makes/attempts. eFG% weights published source eFG% by attempts,
so it retains source rounding limitations. PPP retains the existing two-decimal
storage precision.

Percentiles are calculated from unrounded combined PPP within each season and
play type, across all available source players with at least 10 reported
possessions. They use average tied ranks and `(rank - 1) / (N - 1)` on a 0–1
scale, with 0.5 for a singleton cohort. These are calculated efficiency ranks,
not official NBA ranks or usage percentiles. Percentiles are recalculated for
every comparison cohort, including players who stayed with one team.

The command never writes to Supabase or changes existing ingestion scripts.
Review the JSON before/after report and execute the exported SQL separately.
The SQL uses a transaction, refuses changed input data, and preserves row keys
and creation timestamps. Existing output files are never overwritten.
`--source-dir` can reuse reviewed NBA Totals JSON files named `SEASON_PLAYTYPE.json`;
omit it to fetch fresh data. If a fresh source no longer matches database keys,
the command stops for review instead of silently inserting or dropping rows.

Player shot-location repair and verification

The shot-location repair uses exact NBA season zone totals and verified
`player_gamelog` appearance counts. `frequency_pct` is zone attempts divided by
attempts across the existing six zones, excluding Backcourt. `fgm` and `fga`
remain per-game values: season counts divided by season appearances, rounded
to two decimals. `fg_pct` uses exact makes/attempts, rounded to four decimals.
Positive season-attempt zones are retained even when the NBA PerGame response
rounds them to zero. Null source counts are recovered only when the game-log
totals determine them uniquely; otherwise preparation stops.

From the repository root:

```bash
.venv/bin/python -B python/nba/repair_shot_locations.py --prepare /tmp/shot-location-review.json
.venv/bin/python -B python/nba/repair_shot_locations.py --apply-plan /tmp/shot-location-review.json
.venv/bin/python -B python/nba/repair_shot_locations.py --verify-plan database/repairs/player_shot_locations_repair_review.json
```

Preparation is read-only and saves source totals and complete before/after
snapshots. Application uses the PostgreSQL connection settings in `python/.env`,
locks the input tables, refuses changed snapshots, and validates before committing.
It preserves existing creation timestamps and performs no deletes or schema changes.
Verification makes a separate read-only connection. The earlier
`audit_shot_locations` command compares against the original NBA PerGame import
policy, so it will flag intentional differences after this repair; use
`--verify-plan` to validate the repaired policy.

Audit player tracking statistics (read-only)

From the repository root:

```bash
.venv/bin/python -B python/nba/audit_tracking.py --json /tmp/player-tracking-audit.json --source-dir /tmp/player-tracking-sources
```

This sends an explicit `TeamID=0` all-team filter and compares all six imported tracking categories across 2022-23 through
2025-26 with NBA PerGame and Totals responses, checking database precision,
duplicates, player coverage, null metrics, value ranges, and tracking game counts
against verified game logs. Counts may differ because tracking availability is
not the same as box-score availability. Blank team filters were observed to
return different tracking values from explicit all-team requests. The optional
new `--source-dir` saves raw public NBA responses, exact request parameters, and
checksums so findings can be reproduced. NBA responses can disagree or change
between reads; reported differences require source review before a repair.
No database writes are performed. Credentials are loaded without being printed.

Repair tracking statistics from frozen responses

```bash
.venv/bin/python -B python/nba/repair_tracking.py --prepare /tmp/tracking-review.json --source-dir database/repairs/tracking_explicit_all_teams_sources
.venv/bin/python -B python/nba/repair_tracking.py --apply-plan /tmp/tracking-review.json
.venv/bin/python -B python/nba/repair_tracking.py --verify-plan database/repairs/player_tracking_stats_repair_review.json
```

Preparation is read-only. It verifies source checksums, explicit all-team filters,
all six categories, per-game/totals consistency, and tracking GP against game-log
coverage, then saves a complete before/after backup. Application reconstructs the
reviewed plan from the frozen sources, locks the affected tables, refuses changed
database inputs, and updates differing metrics in one transaction. No rows are
inserted or deleted, and no schema changes occur. Existing creation timestamps
and nonmetric fields are preserved; unchanged rows retain their update timestamps.
Source nulls remain null rather than being filled with invented zeros. A null
shooting eFG% is accepted only when the source reports zero attempts. Published
per-game units and existing numeric precision are retained. Verification uses a
separate read-only connection and compares every scoped row with the reviewed plan.

Audit team defensive play types and shot locations (read-only)

```bash
.venv/bin/python -B python/nba/audit_team_season_stats.py --json /tmp/team-season-defense-audit.json --source-dir /tmp/team-season-defense-sources
```

The audit checks all 30 teams, 11 play types, six shot zones, and four seasons
against NBA PerGame and Totals responses. It saves raw responses, request filters,
and checksums. It separately checks the original import transformation and the
precision of PPP/frequency calculations against season totals. Opponent shot
totals are cross-checked against the audited regular-season team game logs,
excluding Cup championship extras through the verified player schedule. Synergy
GP can cover fewer games than the box-score schedule; this source limitation
is reported without inventing missing play-type events. No database writes occur.

For a read-only GP investigation using fresh offensive/defensive responses and
the verified schedule:

```bash
.venv/bin/python -B python/nba/investigate_synergy_gp.py --output-dir /tmp/synergy-gp-investigation
```

The output checks whether one schedule cutoff can explain all team GP counts,
compares offense and defense, and contrasts source shot/point totals with box
scores. It preserves source snapshots and documents unresolved limitations.
The public season-summary endpoint does not expose game IDs or team/date filters,
so it cannot definitively identify excluded games. GP gaps must not be repaired by
replacing source GP with 82 or inventing possessions.

Repair the two confirmed team calculations from the saved source snapshots:

```bash
.venv/bin/python -B python/nba/repair_team_calculations.py --prepare /tmp/team-calculations-review.json --source-dir database/repairs/team_season_defense_sources
.venv/bin/python -B python/nba/repair_team_calculations.py --apply-plan /tmp/team-calculations-review.json
.venv/bin/python -B python/nba/repair_team_calculations.py --verify-plan database/repairs/team_defensive_calculations_repair_review.json
```

Preparation is read-only and saves complete before/after records. The repair
changes only `team_defensive_playtype_stats.points_allowed_per_possession`
(exact source season points / possessions) and
`team_defensive_shot_locations.frequency_pct` (exact zone attempts / attempts
across the existing six zones). Existing numeric precision is retained. All GP,
per-game counts, other statistics, keys, and creation timestamps are preserved.
No permanent rows are inserted or deleted and no schema changes occur. Both
tables are checked and updated together in one transaction, followed by a new
read-only connection that verifies every result. Changed snapshots are refused.
The original audit also compares with the legacy per-game calculation, so use
`--verify-plan` to check the repaired policy without flagging intentional changes.

Audit identity tables and unresolved pace fields (read-only)

```bash
.venv/bin/python -B python/nba/audit_master_and_pace.py --output-dir /tmp/master-and-pace-audit --reference-season 2026-27
```

This audits `players` and `teams` IDs, names, duplicates, internal team links,
NBA-ID references across the statistical tables, and team conference/division
mappings. Fresh NBA directory/standings responses are saved with checksums.
Current roster differences are reported separately from historical data integrity;
roster status and an active-career flag are not identical definitions. Optional
profile fields receive missing-value/plausibility checks, and Basketball Reference
IDs receive uniqueness/format checks without an external crosswalk guarantee.

The pace review distinguishes the importer's existing calculation from the
opponent's own pace average. Full-season averages and current-game deviations
are descriptive statistics, not pregame-safe features. The review evaluates prior
history using strictly earlier dates, batching same-day games to avoid leakage.
No database records or original ingestion scripts are changed.

Repair opponent pace baselines using prior games only

```bash
.venv/bin/python -B python/nba/repair_pace.py --prepare /tmp/pace-review.json
.venv/bin/python -B python/nba/repair_pace.py --apply-plan /tmp/pace-review.json
.venv/bin/python -B python/nba/repair_pace.py --verify-plan database/repairs/team_defensive_pace_repair_review.json
```

`opponent_avg_team_pace` averages the opponent's own `team_pace` from strictly
earlier dates in that season. `opponent_avg_game_pace` similarly averages the
opponent's earlier game `pace`. All existing games in the table contribute,
including Cup finals. Same-day games never see one another; histories reset each
season. With no earlier opponent game, both averages and deviations are NULL.
Existing one-decimal precision is retained.

The deviation columns are kept: `opponent_pace_deviation` is current
`opponent_pace` minus the stored rounded opponent team-pace baseline;
`opponent_game_pace_deviation` is current `pace` minus the stored rounded
opponent game-pace baseline. These are completed-game outcomes and must not be
inputs when predicting that same game. The two averages are pregame-safe.

Preparation saves every before/after record and performs no database writes.
Application refuses changed inputs, updates only these four fields in one
transaction, and preserves raw pace, box scores, ratings, keys, creation
timestamps, and unchanged-row update timestamps. No rows or columns are removed.
Both in-transaction and separate read-only checks compare every result, including
an independent SQL window calculation excluding current/future games. The default
scope is 2022-23 through 2025-26; `--season` can select other stored seasons during
preparation. Original ingestion scripts remain unchanged.

Audit season-specific clusters for comparable-player selection

```bash
.venv/bin/python -B python/models/audit_clusters.py --snapshot database/repairs/player_clusters_audit_snapshot.json --output-dir /tmp/player-cluster-audit
```

This uses the saved read-only cluster/input snapshot, original feature matrices
and scalers, and source-repair backups. It checks weight validity, sums, primary
assignments, and coverage at the existing 500-minute threshold. The existing
feature-building functions run against a read-only snapshot adapter, so changes
in pre-standardization inputs can be distinguished from rescaling. Original
inputs are reconstructed from the before-repair data to verify the comparison.
Current style matrices are written only to the new audit directory; original
training artifacts and database assignments are preserved. No GMM is refitted.

Different cluster counts across seasons are allowed for this intended use.
Within-season style-neighbor agreement and silhouette are descriptive diagnostics,
not evidence of improved prop predictions. Cluster weights are model posteriors,
not measured prediction accuracy. Opponent-comparison features require only
earlier games and profiles available before the predicted matchup; full-season
profiles must be lagged or replaced by dated in-season snapshots.

Select cluster counts and generate every season's feature CSV automatically

```bash
.venv/bin/python -B python/models/select_clusters.py --output-dir python/models/clustering_runs/new_run
```

This captures a fresh read-only Supabase snapshot and rebuilds the existing 17
style features for qualifying players (at least 500 season minutes). Each season
gets its own CSV, scaler, independent diagonal-covariance GMM search, score chart,
selected model, player-assignment preview, and cluster profiles. No manual CSV
filename changes are needed. Existing training artifacts and database assignments
are preserved; the output directory must be new. Use `--snapshot PATH` to repeat
the search offline from a saved `source_snapshot.json`.

The search tests 1–20 clusters with five seeds and three initializations per seed,
extending to 30 if a winner is at the initial upper boundary. Lowest BIC among
consistently converged models determines the unrestricted winner. A separate
comparable-player recommendation requires at least five hard-assigned players
per component. Both winners are reported; previous cluster counts do not affect
selection. Silhouette, seed agreement, and ten 80% subsample refits are diagnostics,
not proof of predictive value. Sampling diagnostics retain the full-cohort scaler.

The corrected-data run in
`python/models/clustering_runs/corrected_data_selection/selection_report.json`
selected 13, 8, 7, and 7 clusters for 2022-23 through 2025-26, respectively; each
also won unrestricted BIC. These are the best tested counts for the existing
features/model, not a universal optimum. Membership stability is moderate.
The current database has only eight archetype-weight columns, so storing the
13-component model requires a storage change before applying its preview.
Full-season models remain retrospective and require lagged or dated profiles
for historical predictions.

Review all candidate counts using separation and refit stability

```bash
.venv/bin/python -B python/models/evaluate_clusters.py --run-dir python/models/clustering_runs/corrected_data_selection --output-dir python/models/clustering_runs/new_evaluation
```

This offline review tests each count with 50 initializations (five seeds),
then ten independent 80% subsamples with five initializations per fit. The
standardization is refitted inside each subsample. It saves BIC/AIC, silhouette,
Davies–Bouldin, seed and sample agreement, group sizes, negative silhouettes,
nearby-player agreement, all candidate models, and representative players.
Distances use the same standardized features, so nearby-player agreement is
descriptive rather than independent validation. No metrics are averaged into an
arbitrary score and no database records are changed.

The completed `multicriterion_evaluation/assessment.json` records reviewed working
choices of 7, 6, 9, and 7 for the four seasons. The last choice retains the original
seven-group fit: a higher-likelihood refit at the same count isolated one player.
These choices balance practical peer groups with fit and stability; they are not
proven predictive optima. The original 13/8/7/7 BIC-only choices should not be
treated as final recommendations. Use style distance to rank comparables and
allow nearby players across cluster boundaries. Predictive usefulness still
requires a separate prior-only, out-of-time matchup evaluation.

Back up and apply the reviewed cluster assignments

```bash
.venv/bin/python -B python/models/apply_clusters.py --prepare database/repairs/new_cluster_refresh
.venv/bin/python -B python/models/apply_clusters.py --apply database/repairs/new_cluster_refresh
.venv/bin/python -B python/models/apply_clusters.py --verify database/repairs/new_cluster_refresh
```

Preparation checks current inputs against the model-selection snapshot, checks
previews against saved models and scalers, and writes a full-table CSV plus an
exact-value JSON backup and desired-record plan. It requires the existing
`archetype_8_weight` column; it makes no schema changes. Application locks the
table against concurrent writes, refuses any changed assignments since backup,
and upserts the reviewed 7/6/9/7 models in one transaction. Existing `created_at`
values remain unchanged; unused weight columns become NULL; numeric storage
precision is respected. Every result is checked before commit and again using
a separate read-only connection. No rows are deleted. Backups and results for
this refresh are in `database/repairs/player_clusters_reviewed_refresh/`.

Run the audit and repair tests from `python` with:

```bash
python -B -m unittest discover -s tests -p 'test_*.py'
```

Data Collection and Datbase:

- Database tables:
  - players : collect basic player identification data as well as injury status.
    - insert_player_teams.py get all players and teams for a given season. All categories filled for players other than bball_ref_id. Use       update_bball_ids.py and Player_Shooting.csv to collect those ids.
  - teams : collect basic team identification data
    - insert_player_teams.py fills all this data.
  - player_gamelog : collects all simple box score stats of every player from every game in a given season.
    - Use player_season_gamelog.py to collect this data.


- Data I still need to add:
  - Location based shooting data for players from bball reference, although this is season based and not game by game.
  - Location based shooting data for defensive teams to see where they give up their points.
  - Team defensive stats
  - Use gamelog data to make distinctions such as days of rest, and home or away
  - Injury data from nbainjuries, look for how much usage increases when a certain player gets injured
  - Try and make comparisons based on positional and similar production in a given stat to see if for example if devin booker and donovan mitchell are both high scoring guards, so if devin booker does score a lot against okc maybe donovan mitchell wont either and look at how big of a trend it is. Iffy, game plan matters more probably, but there are some defenders like stephon castle that I think can lock up any guard.
  - If possible try and find stats for what play type a player scores from, and then see how well a team defends against those play types.
  - These are mostly stats for points so maybe look more for other data to predict other stats.
  
  - Pace and possessions : More possessions = more opportunity to get stats
  - Vegas lines : Need those so know what to compare predictions against
  - Minutes projection : More minutes = more opportunity
  - Opponent defensive matchup at the position level : positional defense can be more accurate than full team ranks
  - Recency weighting : maybe weigh recent performances more than season data
  - Game total (over/under) : more or less scoring opportunities based on the total


Practical Workflow for testing features:
1. Train baseline model with core features
2. Look at SHAP values → find which features the model leans on most
3. Form a basketball-logic hypothesis about an interaction
4. Test that specific combination
5. Did it improve calibration? Keep it. If not, discard it.
6. Repeat
