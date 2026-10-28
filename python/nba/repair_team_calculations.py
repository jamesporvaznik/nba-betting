"""Repair only defensive PPP and shot-zone frequency using frozen NBA totals."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import sys

from psycopg2.extras import RealDictCursor, execute_values

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba.aggregate_playtypes import PLAY_TYPES
from nba.audit_shot_locations import ZONES
from nba.audit_tracking import SEASONS
from nba.repair_shot_locations import connection, serial


PLAY_TABLE = "team_defensive_playtype_stats"
SHOT_TABLE = "team_defensive_shot_locations"
TABLES = {
    PLAY_TABLE: {"keys": ("team_id", "season", "play_type"), "field": "points_allowed_per_possession"},
    SHOT_TABLE: {"keys": ("team_id", "season", "shot_zone"), "field": "frequency_pct"},
}


def dec(value):
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("Non-finite numeric value")
    return number


def round_value(value, scale):
    return str(dec(value).quantize(Decimal(1).scaleb(-scale), rounding=ROUND_HALF_UP))


def row_key(row, table):
    return tuple(row[c] for c in TABLES[table]["keys"])


def source_rows(data, shots):
    if not shots:
        result = data["resultSets"][0]
        return [dict(zip(result["headers"], r)) for r in result["rowSet"]]
    result = data["resultSets"]
    grouping, headers = result["headers"]
    if grouping["columnsToSkip"] != 2 or grouping["columnSpan"] != 3:
        raise ValueError("Unexpected source shot header layout")
    names = headers["columnNames"][:2]
    for i, zone in enumerate(grouping["columnNames"]):
        stats = headers["columnNames"][2 + 3*i:5 + 3*i]
        if stats != ["OPP_FGM", "OPP_FGA", "OPP_FG_PCT"]:
            raise ValueError("Unexpected source shot metrics")
        names += [f"{zone}_{stat}" for stat in stats]
    return [dict(zip(names, r)) for r in result["rowSet"]]


def read_sources(directory):
    sources = {table: {mode: [] for mode in ("PerGame", "Totals")} for table in TABLES}
    manifest = []
    for season in SEASONS:
        for table in TABLES:
            for category in PLAY_TYPES if table == PLAY_TABLE else (None,):
                for mode in ("PerGame", "Totals"):
                    name = f"{season}_defense_{category}_{mode}" if category else f"{season}_opponent_shots_{mode}"
                    raw = (directory / f"{name}.json").read_text()
                    metadata = json.loads((directory / f"{name}_request.json").read_text())
                    digest = hashlib.sha256(raw.encode()).hexdigest()
                    if digest != metadata["raw_sha256"]:
                        raise ValueError("Source checksum changed")
                    data = json.loads(raw)
                    expected = {"LeagueID": "00", "SeasonType": "Regular Season", "PerMode": mode}
                    expected.update({"SeasonYear": season, "PlayerOrTeam": "T", "PlayType": category, "TypeGrouping": "defensive"}
                                    if category else {"Season": season, "MeasureType": "Opponent", "TeamID": 0, "DistanceRange": "By Zone", "PaceAdjust": "N"})
                    for params in (data["parameters"], metadata["parameters"]):
                        if any(params.get(k) != v for k, v in expected.items()):
                            raise ValueError("Saved source filters differ from reviewed request")
                    records = source_rows(data, table == SHOT_TABLE)
                    if not records:
                        raise ValueError("Empty source response")
                    sources[table][mode] += [dict(r, season=season) for r in records]
                    manifest.append({"file": f"{name}.json", "sha256": digest})
    return sources, manifest


def build_plan(sources, before, scales):
    after, changes, summaries = {}, {}, []
    for table, config in TABLES.items():
        old = {row_key(r, table): r for r in before[table]}
        if len(old) != len(before[table]):
            raise ValueError("Duplicate stored keys")
        indexed = {}
        for mode in ("PerGame", "Totals"):
            indexed[mode] = {}
            for row in sources[table][mode]:
                k = (int(row["TEAM_ID"]), row["season"], row["PLAY_TYPE"] if table == PLAY_TABLE else None)
                if k in indexed[mode]:
                    raise ValueError("Duplicate source keys")
                indexed[mode][k] = row
            if not indexed[mode]:
                raise ValueError("Empty source")
        if indexed["PerGame"].keys() != indexed["Totals"].keys():
            raise ValueError("Source mode coverage differs")
        desired = {k: dict(r) for k, r in old.items()}
        field = config["field"]
        denominators = defaultdict(Decimal)
        if table == SHOT_TABLE:
            for (team, season, _), row in indexed["Totals"].items():
                for zone in ZONES:
                    attempts, made = dec(row[f"{zone}_OPP_FGA"]), dec(row[f"{zone}_OPP_FGM"])
                    if not 0 <= made <= attempts or attempts != attempts.to_integral_value():
                        raise ValueError("Invalid source zone totals")
                    denominators[(team, season)] += attempts
        expected_keys = set()
        for source_key, total in indexed["Totals"].items():
            team, season, play = source_key
            per = indexed["PerGame"][source_key]
            categories = (play,) if table == PLAY_TABLE else ZONES
            for category in categories:
                k = team, season, category
                expected_keys.add(k)
                if k not in old:
                    raise ValueError("Source key is absent from stored rows; inserts are not authorized")
                row = old[k]
                if table == PLAY_TABLE:
                    if str(total["SEASON_ID"]) != "2" + season[:4] or str(total["TYPE_GROUPING"]).lower() != "defensive":
                        raise ValueError("Unexpected source season or grouping")
                    if total["GP"] != per["GP"] or row["games_played"] != total["GP"] or total["GP"] <= 0:
                        raise ValueError("Stored/source GP differs; GP will not be altered")
                    for column, src in (("possessions_defended", "POSS"), ("points_allowed", "PTS")):
                        if dec(row[column]) != dec(round_value(per[src], scales[table][column])):
                            raise ValueError("Stored/source per-game count differs")
                        if dec(total[src]) < 0 or dec(total[src]) != dec(total[src]).to_integral_value():
                            raise ValueError("Invalid source integer total")
                    if dec(total["POSS"]) <= 0:
                        raise ValueError("Zero source possessions")
                    result = dec(total["PTS"]) / dec(total["POSS"])
                else:
                    for column, src in (("fgm_allowed", "FGM"), ("fga_allowed", "FGA")):
                        if dec(row[column]) != dec(round_value(per[f"{category}_OPP_{src}"], scales[table][column])):
                            raise ValueError("Stored/source per-game shooting count differs")
                    denominator = denominators[(team, season)]
                    if denominator <= 0:
                        raise ValueError("Zero source zone denominator")
                    result = dec(total[f"{category}_OPP_FGA"]) / denominator
                desired[k][field] = round_value(result, scales[table][field])
        if expected_keys != old.keys():
            raise ValueError("Source/stored key coverage differs; no rows may be omitted")
        changes[table] = [{**{c: r[c] for c in config["keys"]}, "field": field,
                           "before": r[field], "after": desired[k][field]}
                          for k, r in sorted(old.items()) if dec(r[field]) != dec(desired[k][field])]
        after[table] = [desired[k] for k in sorted(desired)]
        for season in sorted({k[1] for k in old}):
            selected = [r for r in changes[table] if r["season"] == season]
            summaries.append({"table": table, "season": season, "changed_rows": len(selected),
                              "maximum_absolute_change": str(max((abs(dec(r["after"]) - dec(r["before"])) for r in selected), default=Decimal(0)))})
    return {"prepared_at": datetime.now(timezone.utc).isoformat(), "seasons": list(SEASONS),
            "policy": "Change only defensive PPP to exact source PTS/POSS and defensive shot frequency to exact zone FGA/sum FGA across the existing six zones. Retain numeric precision. Preserve all source GP, per-game counts, other metrics, row keys, creation timestamps, and unchanged-row update timestamps. No inserts, deletes, schema changes, or source refetches.",
            "scales": scales, "summaries": summaries, "before": before, "after": after, "changes": changes}


def snapshot(cursor, seasons):
    rows = {}
    for table, config in TABLES.items():
        cursor.execute(f"SELECT * FROM public.{table} WHERE season = ANY(%s) ORDER BY {','.join(config['keys'])}", (seasons,))
        rows[table] = serial([dict(r) for r in cursor.fetchall()])
    return rows


def check_after(actual, plan):
    for table, config in TABLES.items():
        desired = {row_key(r, table): r for r in plan["after"][table]}
        observed = {row_key(r, table): r for r in actual[table]}
        changed = {row_key(r, table) for r in plan["changes"][table]}
        if len(observed) != len(actual[table]) or observed.keys() != desired.keys():
            raise ValueError("Post-repair keys differ")
        for k, row in desired.items():
            for column, value in row.items():
                if column == config["field"]:
                    matches = dec(value) == dec(observed[k][column])
                elif column == "updated_at" and k in changed:
                    continue
                else:
                    matches = value == observed[k][column]
                if not matches:
                    raise ValueError(f"Post-repair mismatch: {table}.{column}")


def verify(plan):
    with connection() as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            actual = snapshot(cursor, plan["seasons"])
            check_after(actual, plan)
        conn.rollback()
    print(f"Independent read-only verification passed: {sum(map(len, actual.values()))} rows; GP, other metrics, keys, and creation timestamps preserved.")


def apply(plan):
    sources, manifest = read_sources(Path(plan["source_dir"]))
    rebuilt = build_plan(sources, plan["before"], plan["scales"])
    if manifest != plan["source_manifest"] or any(rebuilt[f] != plan[f] for f in ("after", "changes", "summaries")):
        raise ValueError("Plan no longer matches frozen source data")
    with connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SET LOCAL lock_timeout = '5s'")
            cursor.execute("SET LOCAL statement_timeout = '120s'")
            cursor.execute("LOCK TABLE public.team_defensive_playtype_stats, public.team_defensive_shot_locations IN SHARE ROW EXCLUSIVE MODE")
            if snapshot(cursor, plan["seasons"]) != plan["before"]:
                raise ValueError("Database input changed; regenerate the repair")
            for i, (table, config) in enumerate(TABLES.items()):
                columns = [*config["keys"], config["field"]]
                temp = f"_team_calc_{i}"
                cursor.execute(f"CREATE TEMP TABLE {temp} ON COMMIT DROP AS SELECT {','.join(columns)} FROM public.{table} WHERE false")
                execute_values(cursor, f"INSERT INTO {temp} ({','.join(columns)}) VALUES %s",
                               [tuple(r[c] for c in columns) for r in plan["after"][table]], page_size=1000)
                field = config["field"]
                cursor.execute(f"UPDATE public.{table} AS p SET {field}=d.{field},updated_at=now() FROM {temp} AS d WHERE {' AND '.join(f'p.{c}=d.{c}' for c in config['keys'])} AND p.{field} IS DISTINCT FROM d.{field}")
                if cursor.rowcount != len(plan["changes"][table]):
                    raise ValueError("Unexpected update count")
            check_after(snapshot(cursor, plan["seasons"]), plan)
    print("Both calculation repairs committed in one transaction; validation passed.", flush=True)
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
        parser.error("Choose a new review filename and specify the frozen --source-dir")
    sources, manifest = read_sources(args.source_dir)
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            before = snapshot(cursor, list(SEASONS))
            cursor.execute("SELECT table_name,column_name,numeric_scale FROM information_schema.columns WHERE table_schema='public' AND table_name = ANY(%s) AND data_type='numeric'", (list(TABLES),))
            scales = defaultdict(dict)
            for r in cursor.fetchall():
                scales[r["table_name"]][r["column_name"]] = r["numeric_scale"]
        conn.rollback()
    plan = build_plan(sources, before, dict(scales))
    plan.update(source_dir=str(args.source_dir.resolve()), source_manifest=manifest)
    with args.prepare.open("x") as handle:
        json.dump(plan, handle, indent=2)
        handle.write("\n")
    print(json.dumps(plan["summaries"], indent=2))
    print(f"Full before/after backup: {args.prepare}. No database writes.")


if __name__ == "__main__":
    main()
