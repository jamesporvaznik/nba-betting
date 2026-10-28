"""Read-only audit of stored tracking metrics against NBA season responses."""

import argparse
import ast
import hashlib
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import sys
import time

from nba_api.stats.endpoints import leaguedashptstats
from psycopg2.extras import RealDictCursor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba.repair_shot_locations import connection, serial


SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")


def load_measures():
    # Read configuration without importing or executing an ingestion script.
    path = Path(__file__).resolve().parent / "player_tracking_insertion" / "player_tracking.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PT_MEASURE_TYPES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Tracking measure configuration not found")


def fetch(season, measure, mode, source_dir=None):
    for attempt in range(2):
        try:
            endpoint = leaguedashptstats.LeagueDashPtStats(
                season=season, season_type_all_star="Regular Season", per_mode_simple=mode,
                player_or_team="Player", pt_measure_type=measure, league_id_nullable="00",
                last_n_games=0, month=0, opponent_team_id=0, timeout=35,
                team_id_nullable=0,
            )
            frame = endpoint.get_data_frames()[0]
            rows = json.loads(frame.to_json(orient="records"))
            if not rows:
                raise ValueError("Empty NBA tracking response")
            if source_dir:
                name = f"{season}_{measure}_{mode}"
                raw = endpoint.nba_response.get_response()
                (source_dir / f"{name}.json").write_text(raw)
                (source_dir / f"{name}_request.json").write_text(json.dumps({
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "request_url": endpoint.nba_response.get_url(), "parameters": endpoint.parameters,
                    "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                }, indent=2))
            return rows
        except Exception:
            if attempt:
                raise
            time.sleep(2)


def compare_measure(season, measure, config, per_game, totals, stored, logs, scales):
    source_counts = Counter(int(r["PLAYER_ID"]) for r in per_game)
    total_counts = Counter(int(r["PLAYER_ID"]) for r in totals)
    duplicates = sorted(pid for pid, n in source_counts.items() if n > 1)
    duplicate_totals = sorted(pid for pid, n in total_counts.items() if n > 1)
    if duplicates or duplicate_totals:
        return {"measure": measure, "complete": False, "duplicate_per_game_players": duplicates,
                "duplicate_totals_players": duplicate_totals}, set(source_counts)
    source = {int(r["PLAYER_ID"]): r for r in per_game}
    total_source = {int(r["PLAYER_ID"]): r for r in totals}
    missing_fields = sorted({f for r in per_game + totals for f in config["columns"] + ["GP"] if f not in r})
    if missing_fields:
        return {"measure": measure, "complete": False, "missing_source_columns": missing_fields}, set(source)
    mismatches, invalid, gp_differences, rate_differences, traded = [], [], [], [], []
    source_nulls, database_nulls, mismatch_counts = Counter(), Counter(), Counter()
    totals_mismatches, totals_mismatch_counts = [], Counter()
    ranges = {}
    for pid, row in source.items():
        player = stored.get(pid)
        if player is None:
            continue
        for src, field in config["rename"].items():
            value, actual = row[src], player.get(field)
            source_nulls[field] += int(value is None)
            database_nulls[field] += int(actual is None)
            if value is None or actual is None:
                matches = value is None and actual is None
            else:
                expected = Decimal(str(value)).quantize(Decimal(1).scaleb(-scales[field]), rounding=ROUND_HALF_UP)
                actual_number = Decimal(str(actual))
                matches = actual_number.is_finite() and actual_number == expected
                if not actual_number.is_finite() or actual_number < 0 or (field.endswith("_pct") and actual_number > (Decimal("1.5") if "efg" in field else 1)):
                    invalid.append({"player_id": pid, "field": field, "value": actual})
                if actual_number.is_finite():
                    low, high = ranges.get(field, (actual_number, actual_number))
                    ranges[field] = (min(low, actual_number), max(high, actual_number))
            if not matches:
                mismatch_counts[field] += 1
                mismatches.append({"player_id": pid, "player_name": row["PLAYER_NAME"],
                                   "field": field, "stored": actual, "nba_source": value})
            total_row = total_source.get(pid)
            if total_row is not None:
                total_value = total_row[src]
                is_rate = not (src.startswith("AVG_") or src.endswith("_PCT"))
                if total_value is None:
                    total_expected = None
                elif is_rate and total_row["GP"]:
                    # The importer copied NBA rates published to one decimal.
                    total_expected = (Decimal(str(total_value)) / Decimal(str(total_row["GP"]))).quantize(Decimal(".1"), rounding=ROUND_HALF_UP)
                elif is_rate:
                    total_expected = None
                else:
                    total_expected = Decimal(str(total_value)).quantize(Decimal(1).scaleb(-scales[field]), rounding=ROUND_HALF_UP)
                total_matches = actual is None if total_expected is None else actual is not None and Decimal(str(actual)) == total_expected
                if not total_matches:
                    totals_mismatch_counts[field] += 1
                    totals_mismatches.append({"player_id": pid, "player_name": row["PLAYER_NAME"],
                                              "field": field, "stored": actual,
                                              "totals_based_expected": total_expected})
        log = logs.get(pid)
        if log is not None:
            if row["GP"] != log["gp"]:
                gp_differences.append({"player_id": pid, "player_name": row["PLAYER_NAME"],
                                       "source_gp": row["GP"], "verified_gamelog_gp": log["gp"],
                                       "teams": log["teams"]})
            if len(log["teams"]) > 1:
                traded.append({"player_id": pid, "player_name": row["PLAYER_NAME"], "teams": log["teams"],
                               "source_gp": row["GP"], "verified_gamelog_gp": log["gp"],
                               "source_team_label": row.get("TEAM_ABBREVIATION")})
        total = total_source.get(pid)
        if total is not None and total["GP"] and total["GP"] == row["GP"]:
            for src in config["columns"]:
                # Averages per touch and percentage fields do not sum by game.
                if src.startswith("AVG_") or src.endswith("_PCT") or row[src] is None or total[src] is None:
                    continue
                exact_rate = Decimal(str(total[src])) / Decimal(str(total["GP"]))
                if abs(exact_rate - Decimal(str(row[src]))) > Decimal(".051"):
                    rate_differences.append({"player_id": pid, "field": src,
                                             "per_game": row[src], "totals_divided_by_gp": exact_rate})
    return {"measure": measure, "complete": True, "source_players": len(source),
            "missing_database_players": sorted(source.keys() - stored.keys()),
            "totals_vs_per_game_player_differences": sorted(source.keys() ^ total_source.keys()),
            "mismatch_counts": dict(mismatch_counts), "mismatches": mismatches,
            "invalid_values": invalid, "source_null_counts": dict(source_nulls),
            "database_null_counts_for_source_players": dict(database_nulls), "stored_ranges": ranges,
            "source_gp_vs_verified_gamelog_differences": gp_differences,
            "traded_player_checks": traded, "totals_vs_per_game_rate_differences": rate_differences,
            "totals_based_stored_mismatch_counts": dict(totals_mismatch_counts),
            "totals_based_stored_mismatches": totals_mismatches}, set(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, help="Save raw public NBA responses in a new directory")
    args = parser.parse_args()
    if args.json.exists() or not args.json.parent.is_dir():
        parser.error("Choose a new report filename in an existing directory")
    if args.source_dir:
        args.source_dir.mkdir(exist_ok=False)
    measures = load_measures()
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM public.player_tracking_stats ORDER BY season,player_id")
            rows = serial([dict(r) for r in cursor.fetchall()])
            cursor.execute("SELECT season,playerid,count(*) AS gp,sum(minutes) AS minutes,array_agg(DISTINCT btrim(team) ORDER BY btrim(team)) AS teams FROM public.player_gamelog WHERE season = ANY(%s) GROUP BY season,playerid", (list(SEASONS),))
            logs = serial([dict(r) for r in cursor.fetchall()])
            cursor.execute("SELECT column_name,numeric_scale FROM information_schema.columns WHERE table_schema='public' AND table_name='player_tracking_stats' AND data_type='numeric'")
            scales = {r["column_name"]: r["numeric_scale"] for r in cursor.fetchall()}
            cursor.execute("SELECT nba_api_id FROM public.players")
            player_ids = {r["nba_api_id"] for r in cursor.fetchall()}
        conn.rollback()
    report = {"audited_at": datetime.now(timezone.utc).isoformat(), "read_only": True,
              "scope": "Six tracking measures, four regular seasons, fresh NBA PerGame and Totals responses. Stored values compared at actual database precision. GP checked against verified full-season game logs; tracking availability can differ from box-score coverage.",
              "unit_policy": "Count metrics are per game; AVG_* are per touch; percentages are source fractions (0-1), with eFG permitting values up to 1.5.",
              "team_filter": "Explicit TeamID=0. Blank TeamID responses were observed to disagree with explicit all-team responses; source request parameters and raw captures should be retained.",
              "seasons": []}
    for season in SEASONS:
        season_rows = [r for r in rows if r["season"] == season]
        counts = Counter(r["player_id"] for r in season_rows)
        stored = {r["player_id"]: r for r in season_rows}
        season_logs = {r["playerid"]: r for r in logs if r["season"] == season}
        result = {"season": season, "database_rows": len(season_rows),
                  "duplicate_database_players": sorted(pid for pid, n in counts.items() if n > 1),
                  "missing_player_master_ids": sorted(stored.keys() - player_ids), "measures": []}
        union = set()
        for measure, config in measures.items():
            print(f"{season}/{measure}: fetching NBA PerGame and Totals...", flush=True)
            try:
                per_game, totals = fetch(season, measure, "PerGame", args.source_dir), fetch(season, measure, "Totals", args.source_dir)
                comparison, available = compare_measure(season, measure, config, per_game, totals,
                                                        stored, season_logs, scales)
                union.update(available)
                result["measures"].append(comparison)
                print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in comparison.items()
                                  if k not in ("stored_ranges", "database_null_counts_for_source_players", "source_null_counts")}, default=str), flush=True)
            except Exception as exc:
                result["measures"].append({"measure": measure, "complete": False, "error_type": type(exc).__name__})
                print(f"Incomplete {measure}: {type(exc).__name__}", flush=True)
            time.sleep(.7)
        result["missing_database_players_from_any_source"] = sorted(union - stored.keys())
        result["database_players_absent_from_all_sources"] = sorted(stored.keys() - union)
        result["gamelog_players_without_tracking"] = [{"player_id": pid, "gp": r["gp"], "minutes": r["minutes"]}
                                                     for pid, r in season_logs.items() if pid not in union]
        result["database_null_counts_all_rows"] = {field: sum(r.get(field) is None for r in season_rows) for field in scales}
        report["seasons"].append(result)
    with args.json.open("x") as handle:
        json.dump(report, handle, indent=2, default=str)
        handle.write("\n")
    print(f"Report saved: {args.json}. No database writes.")


if __name__ == "__main__":
    main()
