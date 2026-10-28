"""Prepare, apply, or verify a tracking repair using reviewed frozen NBA responses."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import sys

from psycopg2.extras import RealDictCursor, execute_values

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba.audit_tracking import SEASONS, load_measures
from nba.repair_shot_locations import connection, serial


def key(row):
    return row["player_id"], row["season"]


def numeric(value):
    return None if value is None else Decimal(str(value))


def fields():
    return tuple(field for config in load_measures().values() for field in config["rename"].values())


def read_sources(directory):
    sources, manifest = {}, []
    for season in SEASONS:
        sources[season] = {}
        for measure, config in load_measures().items():
            modes = {}
            for mode in ("PerGame", "Totals"):
                name = f"{season}_{measure}_{mode}"
                raw = (directory / f"{name}.json").read_text()
                metadata = json.loads((directory / f"{name}_request.json").read_text())
                digest = hashlib.sha256(raw.encode()).hexdigest()
                if digest != metadata["raw_sha256"]:
                    raise ValueError("Source checksum changed")
                data = json.loads(raw)
                expected = {"Season": season, "SeasonType": "Regular Season", "PerMode": mode,
                            "PtMeasureType": measure, "TeamID": 0, "PlayerOrTeam": "Player", "LeagueID": "00"}
                for params in (metadata["parameters"], data["parameters"]):
                    if any(params.get(k) != v for k, v in expected.items()):
                        raise ValueError("Source filters do not match reviewed all-team regular season")
                params = metadata["parameters"]
                if any(params.get(k) not in ("", None) for k in ("DateFrom", "DateTo", "SeasonSegment", "Location", "Outcome")) or any(str(params.get(k)) != "0" for k in ("LastNGames", "Month", "OpponentTeamID")):
                    raise ValueError("Unexpected source restriction")
                result = data["resultSets"][0]
                headers = result["headers"]
                if not {"PLAYER_ID", "GP", *config["columns"]}.issubset(headers):
                    raise ValueError("Source is missing required columns")
                records = [dict(zip(headers, r)) for r in result["rowSet"]]
                modes[mode] = records
                manifest.append({"file": f"{name}.json", "sha256": digest})
            sources[season][measure] = modes
    return sources, manifest


def build_plan(sources, before, logs, scales, measures=None):
    measures = measures or load_measures()
    metric_fields = tuple(field for config in measures.values() for field in config["rename"].values())
    stored = {key(r): r for r in before}
    if len(stored) != len(before):
        raise ValueError("Duplicate database player-season keys")
    game_counts = {(r["playerid"], r["season"]): r["gp"] for r in logs}
    after = {k: dict(r) for k, r in stored.items()}
    changes, summaries = [], []
    for season, categories in sources.items():
        if set(categories) != set(measures):
            raise ValueError("Incomplete tracking categories")
        union = set()
        for measure, config in measures.items():
            modes = {}
            for mode in ("PerGame", "Totals"):
                rows = categories[measure][mode]
                indexed = {int(r["PLAYER_ID"]): r for r in rows}
                if not rows or len(indexed) != len(rows):
                    raise ValueError("Empty source or duplicate source player IDs")
                modes[mode] = indexed
            if modes["PerGame"].keys() != modes["Totals"].keys():
                raise ValueError("Source modes have different player coverage")
            union.update(modes["PerGame"])
            for pid, row in modes["PerGame"].items():
                k = pid, season
                total = modes["Totals"][pid]
                if k not in stored or k not in game_counts:
                    raise ValueError("Source player missing from stored tracking or verified game logs")
                if row["GP"] != total["GP"] or not 0 < row["GP"] <= game_counts[k]:
                    raise ValueError("Invalid or inconsistent source tracking game counts")
                for src, field in config["rename"].items():
                    a, b = numeric(row[src]), numeric(total[src])
                    if a is None or b is None:
                        if a != b:
                            raise ValueError("Null source fields disagree across modes")
                        if src.endswith("_EFG_PCT") and numeric(total[src.replace("EFG_PCT", "FGA")]) != 0:
                            raise ValueError("Unknown shooting efficiency despite positive attempts")
                    else:
                        if not a.is_finite() or not b.is_finite() or min(a, b) < 0:
                            raise ValueError("Invalid numeric source value")
                        ratio = src.startswith("AVG_") or src.endswith("_PCT")
                        if (ratio and a != b) or (not ratio and abs(a - b / total["GP"]) > Decimal(".051")):
                            raise ValueError("NBA source modes are inconsistent")
                        if src.endswith("_PCT") and a > (Decimal("1.5") if "EFG" in src else 1):
                            raise ValueError("Source percentage outside supported fraction scale")
                    after[k][field] = None if a is None else str(a.quantize(Decimal(1).scaleb(-scales[field]), rounding=ROUND_HALF_UP))
            for k, old in stored.items():
                if k[1] == season and k[0] not in modes["PerGame"]:
                    if any(old[field] is not None for field in config["rename"].values()):
                        raise ValueError("Source category absent for a player with stored values")
        if union != {pid for pid, s in stored if s == season}:
            raise ValueError("Source/database player coverage differs")
    if set(sources) != {r["season"] for r in before}:
        raise ValueError("Season coverage differs")
    for k in sorted(stored):
        changed = {field: {"before": stored[k][field], "after": after[k][field]}
                   for field in metric_fields if numeric(stored[k][field]) != numeric(after[k][field])}
        if changed:
            changes.append({"player_id": k[0], "season": k[1], "changes": changed})
    for season in sources:
        selected = [r for r in changes if r["season"] == season]
        summaries.append({"season": season, "rows": sum(k[1] == season for k in stored),
                          "changed_rows": len(selected), "changed_values": sum(len(r["changes"]) for r in selected),
                          "changed_fields": dict(Counter(f for r in selected for f in r["changes"]))})
    return {"prepared_at": datetime.now(timezone.utc).isoformat(), "seasons": list(sources),
            "metric_fields": list(metric_fields), "scales": scales, "summaries": summaries,
            "policy": "Copy frozen, checksum-verified NBA PerGame tracking values using explicit TeamID=0; verify corresponding Totals. Keep published per-game units and existing database precision. Preserve source nulls, existing keys, creation timestamps, and all nonmetric fields. No inserts, deletes, schema changes, or source refetches during application.",
            "before": before, "after": [after[k] for k in sorted(after)],
            "gamelog_snapshot": logs, "changes": changes}


def snapshot(cursor, seasons):
    cursor.execute("SELECT * FROM public.player_tracking_stats WHERE season = ANY(%s) ORDER BY player_id,season", (seasons,))
    rows = serial([dict(r) for r in cursor.fetchall()])
    cursor.execute("SELECT season,playerid,count(*) AS gp FROM public.player_gamelog WHERE season = ANY(%s) GROUP BY season,playerid ORDER BY playerid,season", (seasons,))
    return rows, serial([dict(r) for r in cursor.fetchall()])


def check_after(actual, plan):
    observed = {key(r): r for r in actual}
    desired = {key(r): r for r in plan["after"]}
    if len(observed) != len(actual) or observed.keys() != desired.keys():
        raise ValueError("Post-repair row keys changed")
    changed = {(r["player_id"], r["season"]) for r in plan["changes"]}
    for k, expected in desired.items():
        for field, value in expected.items():
            if field in plan["metric_fields"]:
                matches = numeric(observed[k][field]) == numeric(value)
            elif field == "updated_at" and k in changed:
                continue
            else:
                matches = observed[k][field] == value
            if not matches:
                raise ValueError(f"Post-repair mismatch in {field}")


def verify(plan):
    with connection() as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            rows, _ = snapshot(cursor, plan["seasons"])
            check_after(rows, plan)
        conn.rollback()
    print(f"Independent read-only verification passed: {len(rows)} rows, values and preserved metadata match.")


def apply(plan):
    sources, manifest = read_sources(Path(plan["source_dir"]))
    rebuilt = build_plan(sources, plan["before"], plan["gamelog_snapshot"], plan["scales"])
    if manifest != plan["source_manifest"] or any(rebuilt[f] != plan[f] for f in ("after", "changes", "summaries", "metric_fields")):
        raise ValueError("Reviewed plan no longer matches saved source responses")
    with connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SET LOCAL lock_timeout = '5s'")
            cursor.execute("SET LOCAL statement_timeout = '120s'")
            cursor.execute("LOCK TABLE public.player_tracking_stats IN SHARE ROW EXCLUSIVE MODE")
            cursor.execute("LOCK TABLE public.player_gamelog IN SHARE MODE")
            before, logs = snapshot(cursor, plan["seasons"])
            if before != plan["before"] or logs != plan["gamelog_snapshot"]:
                raise ValueError("Input snapshot changed; repair refused")
            metrics = plan["metric_fields"]
            columns = ["player_id", "season", *metrics]
            cursor.execute(f"CREATE TEMP TABLE _tracking_repair ON COMMIT DROP AS SELECT {','.join(columns)} FROM public.player_tracking_stats WHERE false")
            execute_values(cursor, f"INSERT INTO _tracking_repair ({','.join(columns)}) VALUES %s",
                           [tuple(r[c] for c in columns) for r in plan["after"]], page_size=1000)
            cursor.execute(f"UPDATE public.player_tracking_stats AS p SET {','.join(f'{c}=d.{c}' for c in metrics)},updated_at=now() FROM _tracking_repair AS d WHERE p.player_id=d.player_id AND p.season=d.season AND ROW({','.join(f'p.{c}' for c in metrics)}) IS DISTINCT FROM ROW({','.join(f'd.{c}' for c in metrics)})")
            updated = cursor.rowcount
            if updated != len(plan["changes"]):
                raise ValueError("Unexpected number of changed rows")
            result, _ = snapshot(cursor, plan["seasons"])
            check_after(result, plan)
    print(f"Committed tracking repair: {updated} rows updated; all transaction checks passed.", flush=True)
    verify(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path)
    group.add_argument("--apply-plan", type=Path)
    group.add_argument("--verify-plan", type=Path)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    if args.apply_plan or args.verify_plan:
        plan = json.loads((args.apply_plan or args.verify_plan).read_text())
        (apply if args.apply_plan else verify)(plan)
        return
    if args.prepare.exists() or not args.prepare.parent.is_dir() or args.source_dir is None:
        parser.error("Preparation needs a new report filename and the reviewed --source-dir")
    sources, manifest = read_sources(args.source_dir)
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            before, logs = snapshot(cursor, list(SEASONS))
            cursor.execute("SELECT column_name,numeric_scale FROM information_schema.columns WHERE table_schema='public' AND table_name='player_tracking_stats' AND data_type='numeric'")
            scales = {r["column_name"]: r["numeric_scale"] for r in cursor.fetchall()}
        conn.rollback()
    plan = build_plan(sources, before, logs, scales)
    plan.update(source_dir=str(args.source_dir.resolve()), source_manifest=manifest)
    with args.prepare.open("x") as handle:
        json.dump(plan, handle, indent=2)
        handle.write("\n")
    print(json.dumps(plan["summaries"], indent=2))
    print(f"Before/after backup and review saved: {args.prepare}. No database writes.")


if __name__ == "__main__":
    main()
