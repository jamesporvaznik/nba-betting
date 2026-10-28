"""Back up and apply reviewed season cluster previews with transactional checks.

--prepare DIR only reads the database, validates saved models against current
inputs, and saves a full-table CSV/JSON backup and exact desired records.
--apply DIR refuses changed database rows, applies one transaction, and verifies
the result before committing. --verify DIR independently checks committed rows.
"""

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from psycopg2.extras import RealDictCursor, execute_values
from models.select_clusters import collect_snapshot, build_features
from nba.repair_shot_locations import connection

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "python/models/clustering_runs"
WEIGHTS = [f"archetype_{i}_weight" for i in range(9)]
VALUES = ["primary_archetype", *WEIGHTS]
KEYS = ["player_id", "season"]
CHOICES = {"2022-23": (7, "multicriterion_evaluation/2022-23/k_07"),
           "2023-24": (6, "multicriterion_evaluation/2023-24/k_06"),
           "2024-25": (9, "multicriterion_evaluation/2024-25/k_09"),
           "2025-26": (7, "corrected_data_selection/2025-26")}


def serial(value):
    return json.loads(json.dumps(value, default=str))


def key(row):
    return row["player_id"], row["season"]


def numeric(row):
    return tuple(None if row.get(c) is None else Decimal(str(row[c])) for c in VALUES)


def expected_row(row, season, count, scales):
    weights = [Decimal(row[f"archetype_{i}_weight"]) for i in range(count)]
    if any(not w.is_finite() or not 0 <= w <= 1 for w in weights) or abs(sum(weights)-1) > Decimal("1e-8"):
        raise ValueError("Invalid probability vector")
    primary = int(row["primary_archetype"])
    if primary != max(range(count), key=lambda i: weights[i]) or row["season"] != season:
        raise ValueError("Invalid primary cluster or season")
    result = {"player_id": int(row["player_id"]), "season": season, "primary_archetype": primary}
    for i, column in enumerate(WEIGHTS):
        value = weights[i] if i < count else None
        if value is not None and scales[column] is not None:
            value = value.quantize(Decimal(1).scaleb(-scales[column]), rounding=ROUND_HALF_UP)
        result[column] = None if value is None else str(value)
    return result


def fetch_rows(cursor):
    cursor.execute("SELECT * FROM public.player_clusters ORDER BY season,player_id")
    return serial([dict(r) for r in cursor.fetchall()])


def prepare(folder):
    snapshot = collect_snapshot(list(CHOICES))
    saved = json.loads((RUNS / "corrected_data_selection/source_snapshot.json").read_text())
    for source in ("playtypes", "tracking", "minutes"):
        if snapshot[source] != saved[source]:
            raise ValueError(f"Current {source} inputs changed since model selection; rebuild before applying")
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("SELECT column_name,numeric_scale FROM information_schema.columns WHERE table_schema='public' AND table_name='player_clusters' ORDER BY ordinal_position")
            columns = [dict(r) for r in c.fetchall()]
            scales = {r["column_name"]: r["numeric_scale"] for r in columns}
            if any(w not in scales for w in WEIGHTS):
                raise ValueError("Table must contain archetype_0_weight through archetype_8_weight")
            before = fetch_rows(c)
        conn.rollback()
    after, sources, summaries = [], [], []
    for season, (count, relative) in CHOICES.items():
        source = RUNS / relative
        preview_path = source / f"player_clusters_preview_{season}.csv"
        model_path = source / f"selected_gmm_{season}.joblib"
        scaler_path = source / f"feature_scaler_{season}.joblib"
        features, scaler, _ = build_features(snapshot, season, 500)
        matrix = features.drop(columns="player_id").to_numpy()
        model = joblib.load(model_path)
        stored_scaler = joblib.load(scaler_path)
        if model.n_components != count or not np.allclose(scaler.mean_, stored_scaler.mean_) or not np.allclose(scaler.scale_, stored_scaler.scale_):
            raise ValueError("Saved model/scaler does not match current season features")
        preview = pd.read_csv(preview_path).sort_values("player_id").reset_index(drop=True)
        if not np.array_equal(features.player_id, preview.player_id) or not np.array_equal(model.predict(matrix), preview.primary_archetype):
            raise ValueError("Preview membership differs from saved model or current player cohort")
        if not np.allclose(model.predict_proba(matrix), preview[WEIGHTS[:count]].to_numpy(), atol=1e-12, rtol=1e-10):
            raise ValueError("Preview probabilities differ from saved model")
        with preview_path.open(newline="") as handle:
            desired = [expected_row(r, season, count, scales) for r in csv.DictReader(handle)]
        if len({key(r) for r in desired}) != len(desired):
            raise ValueError("Duplicate preview keys")
        existing = {key(r) for r in before if r["season"] == season}
        if existing - {key(r) for r in desired}:
            raise ValueError("Applying preview would leave stale season assignments; review required")
        summaries.append({"season": season, "clusters": count, "players": len(desired),
                          "existing": len(existing), "inserted": len(desired)-len(existing)})
        after.extend(desired)
        for path in (preview_path, model_path, scaler_path):
            sources.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "player_clusters_before.json").write_text(json.dumps(before, indent=2))
    with (folder / "player_clusters_before.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[r["column_name"] for r in columns])
        writer.writeheader()
        writer.writerows(before)
    plan = {"prepared_at": datetime.now(timezone.utc).isoformat(), "columns": columns,
            "policy": "Upsert reviewed 7/6/9/7 models. Preserve existing created_at; clear unused weights to NULL; keep database numeric precision. No deletes or schema changes.",
            "source_artifacts": sources, "summaries": summaries,
            "backup_sha256": hashlib.sha256((folder / "player_clusters_before.json").read_bytes()).hexdigest(),
            "before": before, "after": after}
    (folder / "plan.json").write_text(json.dumps(plan, indent=2))
    print(json.dumps({"backup_rows": len(before), "summaries": summaries, "database_writes": False}, indent=2))


def verify_rows(actual, plan):
    actual_by_key = {key(r): r for r in actual}
    desired = {key(r): r for r in plan["after"]}
    before = {key(r): r for r in plan["before"]}
    if len(actual_by_key) != len(actual) or set(actual_by_key) != set(before) | set(desired):
        raise ValueError("Unexpected row coverage after application")
    for k, row in actual_by_key.items():
        if k in desired:
            if numeric(row) != numeric(desired[k]):
                raise ValueError(f"Stored cluster values differ from reviewed plan: {k}")
            weights = [Decimal(str(row[c])) for c in WEIGHTS if row[c] is not None]
            if any(not 0 <= w <= 1 for w in weights) or abs(sum(weights)-1) > Decimal(".0005"):
                raise ValueError("Stored weights invalid after rounding")
            if k in before and row["created_at"] != before[k]["created_at"]:
                raise ValueError("Existing creation timestamp changed")
        elif row != before[k]:
            raise ValueError("Unrelated season row changed")


def apply_or_verify(folder, apply):
    plan = json.loads((folder / "plan.json").read_text())
    if hashlib.sha256((folder / "player_clusters_before.json").read_bytes()).hexdigest() != plan["backup_sha256"]:
        raise ValueError("Backup checksum mismatch")
    if json.loads((folder / "player_clusters_before.json").read_text()) != plan["before"]:
        raise ValueError("Plan and backup disagree")
    for artifact in plan["source_artifacts"]:
        if hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("Reviewed model artifact changed")
    with connection() as conn:
        conn.set_session(readonly=not apply)
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            if apply:
                c.execute("SET LOCAL lock_timeout='10s'")
                c.execute("LOCK TABLE public.player_clusters IN SHARE ROW EXCLUSIVE MODE")
                if fetch_rows(c) != plan["before"]:
                    raise ValueError("Database assignments changed since backup; no changes applied")
                columns = KEYS + VALUES
                rows = [tuple(r[col] for col in columns) for r in plan["after"]]
                assignments = ",".join(f"{col}=EXCLUDED.{col}" for col in VALUES)
                sql = f"INSERT INTO public.player_clusters ({','.join(columns)}) VALUES %s ON CONFLICT (player_id,season) DO UPDATE SET {assignments},updated_at=now()"
                execute_values(c, sql, rows, page_size=500)
            actual = fetch_rows(c)
            verify_rows(actual, plan)
        if apply:
            conn.commit()
        else:
            conn.rollback()
    result = {"verified_at": datetime.now(timezone.utc).isoformat(), "database_writes": apply,
              "table_rows": len(actual), "reviewed_rows_verified": len(plan["after"]), "summaries": plan["summaries"]}
    (folder / ("apply_result.json" if apply else "verification.json")).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if apply:
        apply_or_verify(folder, False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path)
    group.add_argument("--apply", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    else:
        apply_or_verify(args.apply or args.verify, bool(args.apply))


if __name__ == "__main__":
    main()
