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
