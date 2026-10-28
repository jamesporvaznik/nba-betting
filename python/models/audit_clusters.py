"""Audit season-specific player clusters for comparable-player selection.

Uses a saved read-only database snapshot and the existing feature functions.
Does not fit clustering models, change the database, or overwrite training files.
"""

import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.features.build_player_features import (
    fetch_playtype_data, fetch_tracking_data, normalize_and_merge,
)


class SnapshotQuery:
    """Read-only subset of the query interface used by the feature builders."""
    def __init__(self, rows):
        self.rows, self.columns, self.bounds = rows, None, None

    def select(self, columns):
        self.columns = None if columns == "*" else [c.strip() for c in columns.split(",")]
        return self

    def eq(self, column, value):
        self.rows = [r for r in self.rows if r[column] == value]
        return self

    def range(self, start, end):
        self.bounds = start, end
        return self

    def execute(self):
        rows = self.rows if self.bounds is None else self.rows[self.bounds[0]:self.bounds[1]+1]
        selected = []
        for row in rows:
            new = {c: row[c] for c in (self.columns or row.keys())}
            for field, value in new.items():
                if isinstance(value, str) and field not in ("season", "play_type", "created_at", "updated_at"):
                    try:
                        new[field] = float(value)
                    except ValueError:
                        pass
            selected.append(new)
        return SimpleNamespace(data=selected)


class SnapshotClient:
    def __init__(self, playtypes, tracking):
        self.tables = {"player_playtype_stats": playtypes, "player_tracking_stats": tracking}

    def table(self, name):
        return SnapshotQuery(self.tables[name])


def weight_checks(rows):
    columns = [f"archetype_{i}_weight" for i in range(8)]
    invalid, sums, primary = [], [], []
    peaks, margins = [], []
    for row in rows:
        weights = {i: float(row[c]) for i, c in enumerate(columns) if row.get(c) is not None}
        if not weights or any(not np.isfinite(v) or not 0 <= v <= 1 for v in weights.values()):
            invalid.append(row["player_id"])
            continue
        if abs(sum(weights.values()) - 1) > .00041:
            sums.append(row["player_id"])
        label = row["primary_archetype"]
        if label not in weights or max(weights.values()) - weights[label] > .00011:
            primary.append(row["player_id"])
        ordered = sorted(weights.values(), reverse=True)
        peaks.append(ordered[0])
        margins.append(ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0])
    return {"rows": len(rows), "duplicate_player_ids": len(rows)-len({r["player_id"] for r in rows}),
            "invalid_weight_players": invalid, "weight_sums_outside_rounding": sums,
            "primary_label_weight_mismatches": primary,
            "nonnull_component_rows": {c: sum(r.get(c) is not None for r in rows) for c in columns},
            "cluster_sizes": dict(Counter(r["primary_archetype"] for r in rows)),
            "median_largest_weight": float(np.median(peaks)) if peaks else None,
            "players_with_top_two_weights_within_point_one": sum(m < .1 for m in margins)}


def rebuild_inputs(client, season, ids):
    log = io.StringIO()
    with redirect_stdout(log):
        plays = fetch_playtype_data(client, season)
        tracking = fetch_tracking_data(client, season)
        plays = plays[plays["player_id"].isin(ids)]
        tracking = tracking[tracking["player_id"].isin(ids)]
        scaled, scaler = normalize_and_merge(plays, tracking)
    columns = [c for c in scaled if c != "player_id"]
    raw = pd.DataFrame(scaler.inverse_transform(scaled[columns].to_numpy()),
                       columns=columns, index=scaled["player_id"].astype(int))
    return scaled.set_index("player_id"), raw


def compare_inputs(archived_raw, current_raw):
    common = sorted(set(archived_raw.index) & set(current_raw.index))
    if list(archived_raw.columns) != list(current_raw.columns):
        raise ValueError("Feature columns/order differ from archived training matrix")
    a, b = archived_raw.loc[common].to_numpy(), current_raw.loc[common].to_numpy()
    differences = ~np.isclose(a, b, rtol=1e-7, atol=1e-8, equal_nan=False)
    return {"common_players": len(common),
            "changed_players": [int(pid) for pid, changed in zip(common, differences.any(axis=1)) if changed],
            "changed_players_by_feature": {column: int(differences[:, i].sum()) for i, column in enumerate(archived_raw.columns)},
            "new_current_players": sorted(map(int, set(current_raw.index)-set(archived_raw.index))),
            "archived_players_no_longer_in_current_cohort": sorted(map(int, set(archived_raw.index)-set(current_raw.index)))}


def neighbor_checks(features, rows, names):
    labels = {r["player_id"]: r["primary_archetype"] for r in rows}
    ids = sorted(set(features.index) & set(labels))
    matrix = features.loc[ids].to_numpy()
    finite = np.isfinite(matrix).all(axis=1)
    invalid = [int(pid) for pid, valid in zip(ids, finite) if not valid]
    ids = [pid for pid, valid in zip(ids, finite) if valid]
    matrix = matrix[finite]
    if len(ids) < 7:
        return {"invalid_feature_players": invalid, "insufficient_players": True}
    neighbors = NearestNeighbors(n_neighbors=6).fit(matrix).kneighbors(matrix, return_distance=False)
    same = []
    examples = []
    for index, pid in enumerate(ids):
        near = [i for i in neighbors[index] if i != index][:5]
        same.append(sum(labels[ids[i]] == labels[pid] for i in near) / len(near))
        if pid in (201939, 1629029, 203999, 203497):
            examples.append({"player_id": int(pid), "name": names.get(pid), "primary_cluster": labels[pid],
                             "five_nearest_styles": [{"player_id": int(ids[i]), "name": names.get(ids[i]),
                                                      "primary_cluster": labels[ids[i]]} for i in near]})
    counts = Counter(labels[pid] for pid in ids)
    random_baseline = sum(n*(n-1) for n in counts.values()) / (len(ids)*(len(ids)-1))
    return {"invalid_feature_players": invalid, "players_compared": len(ids),
            "mean_same_primary_cluster_among_five_nearest_styles": float(np.mean(same)),
            "random_same_cluster_baseline": random_baseline,
            "silhouette_of_existing_labels_on_current_features": float(silhouette_score(matrix, [labels[pid] for pid in ids])) if 1 < len(counts) < len(ids) else None,
            "examples": examples,
            "interpretation": "Style cohesion diagnostics only; neither nearest-neighbor agreement nor silhouette measures prediction improvement against an opponent."}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=root / "database/repairs/player_clusters_audit_snapshot.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    snapshot = json.loads(args.snapshot.read_text())
    names = {r["nba_api_id"]: r["display_name"] for r in snapshot["players"]}
    current_client = SnapshotClient(snapshot["playtypes"], snapshot["tracking"])
    play_backup = root / "database/repairs/player_playtypes_season_totals_review.json"
    tracking_backup = root / "database/repairs/player_tracking_stats_repair_review.json"
    original_client = SnapshotClient(json.loads(play_backup.read_text())["before"], json.loads(tracking_backup.read_text())["before"])
    report = {"database_writes": False, "scope": "Within-season comparable-player selection; varying cluster counts are allowed. No cross-season label alignment is required by this audit. No models are retrained.",
              "snapshot_captured_at": snapshot["captured_at"],
              "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(), "seasons": []}
    for season in sorted({r["season"] for r in snapshot["clusters"]}):
        print(f"Auditing clusters and corrected style inputs: {season}", flush=True)
        rows = [r for r in snapshot["clusters"] if r["season"] == season]
        qualified = {r["playerid"] for r in snapshot["minutes"] if r["season"] == season and float(r["minutes"]) >= 500}
        play_ids = {r["player_id"] for r in snapshot["playtypes"] if r["season"] == season}
        tracking_ids = {r["player_id"] for r in snapshot["tracking"] if r["season"] == season}
        expected = qualified & play_ids & tracking_ids
        stored_ids = {r["player_id"] for r in rows}
        folder = root / "python/models/clustering_data" / season
        archived = pd.read_csv(folder / f"player_features_{season}.csv").set_index("player_id")
        archived = archived.loc[:, ~archived.columns.str.startswith("Unnamed:")]
        # These are existing, locally owned training artifacts from this repo.
        scaler = joblib.load(folder / f"feature_scaler_{season}.pkl")
        archived_raw = pd.DataFrame(scaler.inverse_transform(archived.to_numpy()),
                                    index=archived.index.astype(int), columns=archived.columns)
        current_scaled, current_raw = rebuild_inputs(current_client, season, expected)
        _, original_raw = rebuild_inputs(original_client, season, set(archived.index))
        current_scaled.to_csv(args.output_dir / f"current_style_features_{season}.csv")
        result = {"season": season, "weight_checks": weight_checks(rows),
                  "coverage": {"current_expected_players_at_500_minutes": len(expected),
                               "missing_cluster_players": sorted(expected-stored_ids),
                               "stored_players_outside_current_expected_cohort": sorted(stored_ids-expected),
                               "archived_feature_players_missing_clusters": sorted(set(archived.index)-stored_ids)},
                  "cached_inputs_vs_corrected_inputs": compare_inputs(archived_raw, current_raw),
                  "cached_inputs_vs_rebuilt_original_inputs": compare_inputs(archived_raw, original_raw),
                  "current_style_neighbor_diagnostics": neighbor_checks(current_scaled, rows, names)}
        report["seasons"].append(result)
        print(json.dumps({"season": season, "cluster_rows": len(rows), "current_expected_players": len(expected),
                          "missing_clusters": len(expected-stored_ids),
                          "players_with_changed_inputs": len(result["cached_inputs_vs_corrected_inputs"]["changed_players"]),
                          "original_input_reproduction_differences": len(result["cached_inputs_vs_rebuilt_original_inputs"]["changed_players"]),
                          "same_cluster_among_five_nearest": result["current_style_neighbor_diagnostics"].get("mean_same_primary_cluster_among_five_nearest_styles")}), flush=True)
    report["recommendation"] = "Refresh per-season clustering inputs after source repairs, inspect nearest style matches rather than treating an entire primary cluster as interchangeable, and validate opponent-matchup features on chronological holdouts. All comparison games and profiles must be available before the predicted matchup; full-season profiles are retrospective and should be lagged or replaced with dated snapshots. Predictive effectiveness has not been established by this audit."
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Audit saved: {args.output_dir / 'report.json'}. No database changes or clustering model fits.")


if __name__ == "__main__":
    main()
