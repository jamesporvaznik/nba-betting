"""Prepare or apply a snapshot-checked repair of season shot-zone frequencies.

Preparation only reads Supabase/NBA. --apply-plan executes the reviewed plan
in one transaction and independently verifies the committed records.
--verify-plan only reads and checks the committed records.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import sys

from dotenv import dotenv_values
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba.audit_shot_locations import ZONES, fetch


KEYS = ("player_id", "season", "shot_zone")
VALUES = ("fgm", "fga", "fg_pct", "frequency_pct")
COLUMNS = KEYS + VALUES
SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")


def serial(value):
    return json.loads(json.dumps(value, default=str))


def rounded(value, places):
    return str(Decimal(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def key(row):
    return tuple(row[c] for c in KEYS)


def values(row):
    return tuple(None if row[c] is None else Decimal(str(row[c])) for c in VALUES)


def recover_counts(row, logs):
    """Recover nulls only when nonnegative integer totals determine them uniquely."""
    row = dict(row)
    recovered = []
    for stat in ("FGA", "FGM"):
        fields = [f"{zone}_{stat}" for zone in ZONES + ("Backcourt",)]
        known = [Decimal(str(row[f])) for f in fields if row[f] is not None]
        if any(v < 0 or v != v.to_integral_value() for v in known):
            raise ValueError("NBA Totals contains invalid counts")
        missing = [f for f in fields if row[f] is None]
        residual = Decimal(logs[stat.lower()]) - sum(known)
        if residual < 0 or (not missing and residual != 0):
            raise ValueError("NBA shot totals disagree with verified game logs")
        if missing:
            if len(missing) != 1 and residual != 0:
                raise ValueError("Unknown source zones cannot be resolved uniquely; no repair prepared")
            for field in missing:
                row[field] = int(residual) if len(missing) == 1 else 0
                recovered.append({"field": field, "value": row[field]})
    for zone in ZONES + ("Backcourt",):
        if Decimal(str(row[f"{zone}_FGM"])) > Decimal(str(row[f"{zone}_FGA"])):
            raise ValueError("Zone makes exceed attempts")
    return row, recovered


def plan_repair(source_by_season, stored, gamelogs):
    before = {key(r): r for r in stored}
    if len(before) != len(stored):
        raise ValueError("Duplicate stored zone keys")
    logs = {(r["playerid"], r["season"]): r for r in gamelogs}
    desired, recoveries, summaries = {}, [], []
    for season, source in source_by_season.items():
        seen = set()
        for source_row in source:
            pid = int(source_row["PLAYER_ID"])
            if pid in seen:
                raise ValueError("Multiple NBA rows per player require review")
            seen.add(pid)
            log = logs.get((pid, season))
            if log is None or log["gp"] <= 0:
                raise ValueError("Source player has no verified season game logs")
            row, recovered = recover_counts(source_row, log)
            if recovered:
                recoveries.append({"player_id": pid, "season": season, "fields": recovered})
            attempts = sum(Decimal(str(row[f"{zone}_FGA"])) for zone in ZONES)
            for zone in ZONES:
                fga = Decimal(str(row[f"{zone}_FGA"]))
                fgm = Decimal(str(row[f"{zone}_FGM"]))
                if not fga:
                    continue
                result = {"player_id": pid, "season": season, "shot_zone": zone,
                          "fgm": rounded(fgm / log["gp"], 2),
                          "fga": rounded(fga / log["gp"], 2),
                          "fg_pct": rounded(fgm / fga, 4),
                          "frequency_pct": rounded(fga / attempts, 4)}
                if Decimal(result["fga"]) <= 0:
                    raise ValueError("Storage precision would erase positive attempts")
                desired[key(result)] = result
        if any(r["season"] == season and r["fga"] > 0 and r["playerid"] not in seen for r in gamelogs):
            raise ValueError("NBA source is missing a player with recorded attempts")
        added = [k for k in desired if k[1] == season and k not in before]
        changed = [k for k in desired if k[1] == season and k in before and values(desired[k]) != values(before[k])]
        summaries.append({"season": season, "source_players": len(seen), "inserted_zone_rows": len(added),
                          "updated_zone_rows": len(changed),
                          "frequency_rows_changed": sum(desired[k]["frequency_pct"] is not None and Decimal(str(before[k]["frequency_pct"])) != Decimal(desired[k]["frequency_pct"]) for k in changed),
                          "new_player_seasons": len({k[0] for k in added} - {k[0] for k in before if k[1] == season})})
    if before.keys() - desired.keys():
        raise ValueError("Repair would remove existing rows; review source before proceeding")
    frequencies = defaultdict(Decimal)
    for r in desired.values():
        frequencies[(r["player_id"], r["season"])] += Decimal(r["frequency_pct"])
    if any(abs(total - 1) > Decimal(".00031") for total in frequencies.values()):
        raise ValueError("Season zone shares do not sum to one within rounding precision")
    return {"prepared_at": datetime.now(timezone.utc).isoformat(), "seasons": list(source_by_season),
            "policy": "Frequency = exact zone FGA / exact FGA across six zones (Backcourt excluded). FGM/FGA remain per-game, calculated as full-season counts / verified season appearances and rounded to 2 decimals. FG% = exact zone makes / attempts, rounded to 4 decimals. No schema changes or deletes. Existing created_at values preserved. Null source counts resolved only when verified game totals determine them uniquely.",
            "summaries": summaries, "recovered_source_counts": recoveries,
            "gamelog_snapshot": gamelogs, "before": stored,
            "after": [desired[k] for k in sorted(desired)]}


def connection():
    config = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    return psycopg2.connect(host=config["DB_HOST"], port=config.get("DB_PORT") or "5432",
                           dbname=config.get("DB_NAME") or config.get("DB") or "postgres",
                           user=config["DB_USER"], password=config["DB_PASSWORD"],
                           sslmode="require", connect_timeout=15)


def snapshot(cursor, seasons):
    cursor.execute("SELECT * FROM public.player_shot_locations WHERE season = ANY(%s) ORDER BY player_id,season,shot_zone", (seasons,))
    stored = serial([dict(r) for r in cursor.fetchall()])
    cursor.execute("SELECT season,playerid,count(*) AS gp,sum(fgm) AS fgm,sum(fga) AS fga FROM public.player_gamelog WHERE season = ANY(%s) GROUP BY season,playerid ORDER BY playerid,season", (seasons,))
    return stored, serial([dict(r) for r in cursor.fetchall()])


def check_after(actual, plan):
    observed = {key(r): r for r in actual}
    expected = {key(r): r for r in plan["after"]}
    if len(actual) != len(observed) or observed.keys() != expected.keys():
        raise ValueError("Post-repair row coverage differs from reviewed plan")
    if any(values(observed[k]) != values(expected[k]) for k in expected):
        raise ValueError("Post-repair values differ from reviewed plan")
    for r in plan["before"]:
        if observed[key(r)]["created_at"] != r["created_at"]:
            raise ValueError("An existing creation timestamp changed")


def apply(plan):
    with connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SET LOCAL lock_timeout = '5s'")
            cursor.execute("SET LOCAL statement_timeout = '120s'")
            cursor.execute("LOCK TABLE public.player_shot_locations IN SHARE ROW EXCLUSIVE MODE")
            cursor.execute("LOCK TABLE public.player_gamelog IN SHARE MODE")
            actual, logs = snapshot(cursor, plan["seasons"])
            if actual != plan["before"] or logs != plan["gamelog_snapshot"]:
                raise ValueError("Input records changed since preparation; repair refused")
            columns = ",".join(COLUMNS)
            updates = ",".join(f"{c}=EXCLUDED.{c}" for c in VALUES)
            old_values = ",".join(f"p.{c}" for c in VALUES)
            new_values = ",".join(f"EXCLUDED.{c}" for c in VALUES)
            execute_values(cursor, f"INSERT INTO public.player_shot_locations AS p ({columns}) VALUES %s ON CONFLICT (player_id,season,shot_zone) DO UPDATE SET {updates},updated_at=now() WHERE ROW({old_values}) IS DISTINCT FROM ROW({new_values})",
                           [tuple(r[c] for c in COLUMNS) for r in plan["after"]], page_size=1000)
            result, _ = snapshot(cursor, plan["seasons"])
            check_after(result, plan)
        # The context commits only after every validation has passed.
    print("Repair committed; all in-transaction validations passed.", flush=True)
    verify(plan)


def verify(plan):
    with connection() as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            result, _ = snapshot(cursor, plan["seasons"])
            check_after(result, plan)
        conn.rollback()
    print(f"Independent read-only verification passed: {len(result)} rows; existing creation timestamps preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path, help="Write a new before/after report; no database writes")
    group.add_argument("--apply-plan", type=Path, help="Apply the reviewed report atomically")
    group.add_argument("--verify-plan", type=Path, help="Read-only verification against the reviewed result")
    args = parser.parse_args()
    if args.verify_plan:
        verify(json.loads(args.verify_plan.read_text()))
        return
    if args.apply_plan:
        plan = json.loads(args.apply_plan.read_text())
        print(json.dumps(plan["summaries"], indent=2), flush=True)
        apply(plan)
        return
    if args.prepare.exists() or not args.prepare.parent.is_dir():
        parser.error("Choose a new report filename in an existing directory")
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            before, logs = snapshot(cursor, list(SEASONS))
        conn.rollback()
    source = {}
    for season in SEASONS:
        print(f"Fetching exact NBA zone totals: {season}", flush=True)
        source[season] = fetch(season, "Totals")
    plan = plan_repair(source, before, logs)
    plan["source_totals"] = source
    with args.prepare.open("x") as handle:
        json.dump(plan, handle, indent=2, default=str)
        handle.write("\n")
    print(json.dumps(plan["summaries"], indent=2))
    print(f"Review saved: {args.prepare}. No database changes.")


if __name__ == "__main__":
    main()
