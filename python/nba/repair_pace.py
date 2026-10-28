"""Repair opponent pace baselines using only earlier games within each season.

Keeps both deviations as descriptive completed-game values. Preparation is
read-only; application is snapshot-checked and atomic, with a separate SQL-window
verification. Existing game pace, box scores, rows, and schema are preserved.
"""

import argparse
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import sys

from psycopg2.extras import RealDictCursor, execute_values

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba.audit_tracking import SEASONS
from nba.repair_shot_locations import connection, serial


FIELDS = ("opponent_avg_team_pace", "opponent_avg_game_pace",
          "opponent_pace_deviation", "opponent_game_pace_deviation")


def dec(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Non-finite pace")
    return result


def rounded(value, scale):
    return str(value.quantize(Decimal(1).scaleb(-scale), rounding=ROUND_HALF_UP))


def equivalent(a, b):
    return a is None and b is None if a is None or b is None else dec(a) == dec(b)


def compute_pregame(rows, scales):
    if set(scales) != set(FIELDS):
        raise ValueError("Unexpected pace baseline columns")
    grouped, games, desired = defaultdict(list), {}, {}
    for row in rows:
        season, day = row["season"], str(row["date"])
        date.fromisoformat(day)
        team, opponent = row["team"], row["opponent_abbrev"]
        if not team or not opponent or team == opponent:
            raise ValueError("Invalid team or opponent")
        game = season, day, team
        if game in games or row["id"] in desired:
            raise ValueError("Duplicate game or row ID")
        for field in ("team_pace", "opponent_pace", "pace"):
            if row[field] is None or dec(row[field]) <= 0:
                raise ValueError("Missing or invalid game pace; cannot establish history")
        games[game] = row
        desired[row["id"]] = dict(row)
        grouped[(season, day)].append(row)
    if not rows:
        raise ValueError("Empty game snapshot")
    for (season, day, team), row in games.items():
        other = games.get((season, day, row["opponent_abbrev"]))
        if other is None or other["opponent_abbrev"] != team:
            raise ValueError("Missing reciprocal team game")
        if dec(other["team_pace"]) != dec(row["opponent_pace"]) or dec(other["pace"]) != dec(row["pace"]):
            raise ValueError("Paired game pace values disagree")
    history = defaultdict(lambda: {"games": 0, "team_pace": Decimal(0), "pace": Decimal(0)})
    for (season, day), day_rows in sorted(grouped.items()):
        for row in day_rows:
            prior = history[(season, row["opponent_abbrev"])]
            new = desired[row["id"]]
            if prior["games"] == 0:
                new.update({field: None for field in FIELDS})
                continue
            new["opponent_avg_team_pace"] = rounded(prior["team_pace"] / prior["games"], scales["opponent_avg_team_pace"])
            new["opponent_avg_game_pace"] = rounded(prior["pace"] / prior["games"], scales["opponent_avg_game_pace"])
            # Use the stored rounded baseline so subtraction can be reproduced
            # directly from the displayed columns.
            new["opponent_pace_deviation"] = rounded(dec(row["opponent_pace"]) - dec(new["opponent_avg_team_pace"]), scales["opponent_pace_deviation"])
            new["opponent_game_pace_deviation"] = rounded(dec(row["pace"]) - dec(new["opponent_avg_game_pace"]), scales["opponent_game_pace_deviation"])
        # The entire date is evaluated before any of its games enter history.
        for row in day_rows:
            prior = history[(season, row["team"])]
            prior["games"] += 1
            prior["team_pace"] += dec(row["team_pace"])
            prior["pace"] += dec(row["pace"])
    return [desired[row_id] for row_id in sorted(desired)]


def build_plan(before, scales):
    after = compute_pregame(before, scales)
    old = {r["id"]: r for r in before}
    changes = []
    for row in after:
        fields = {f: {"before": old[row["id"]][f], "after": row[f]}
                  for f in FIELDS if not equivalent(old[row["id"]][f], row[f])}
        if fields:
            changes.append({"id": row["id"], "season": row["season"], "date": str(row["date"]),
                            "team": row["team"], "opponent": row["opponent_abbrev"], "changes": fields})
    seasons = sorted({r["season"] for r in before})
    return {"prepared_at": datetime.now(timezone.utc).isoformat(), "seasons": seasons, "scales": scales,
            "policy": "For each opponent, average its own team_pace and game pace from strictly earlier dates in the same season. Include every existing table game, including Cup finals. Reset histories each season. If no prior game exists, both averages and deviations are NULL. Deviations are current opponent/game pace minus the corresponding stored rounded pregame baseline. Deviations remain postgame outcomes, not pregame inputs. Change only these four derived columns; preserve raw pace, box scores, ratings, keys, creation timestamps, and unchanged update timestamps. No schema changes, inserts, or deletes.",
            "summaries": [{"season": season, "rows": sum(r["season"] == season for r in before),
                           "changed_rows": sum(r["season"] == season for r in changes),
                           "rows_without_prior_opponent_history": sum(r["season"] == season and r["opponent_avg_team_pace"] is None for r in after)} for season in seasons],
            "before": before, "after": after, "changes": changes}


def snapshot(cursor, seasons):
    cursor.execute("SELECT * FROM public.team_defensive_stats WHERE season = ANY(%s) ORDER BY id", (seasons,))
    return serial([dict(r) for r in cursor.fetchall()])


def read_scales(cursor):
    cursor.execute("SELECT column_name,numeric_scale,is_nullable FROM information_schema.columns WHERE table_schema='public' AND table_name='team_defensive_stats' AND column_name = ANY(%s)", (list(FIELDS),))
    records = cursor.fetchall()
    if len(records) != len(FIELDS) or any(r["is_nullable"] != "YES" or r["numeric_scale"] is None for r in records):
        raise ValueError("Pace baseline columns must be numeric and nullable")
    return {r["column_name"]: r["numeric_scale"] for r in records}


def check_after(rows, plan):
    actual = {r["id"]: r for r in rows}
    desired = {r["id"]: r for r in plan["after"]}
    changed = {r["id"] for r in plan["changes"]}
    if len(actual) != len(rows) or actual.keys() != desired.keys():
        raise ValueError("Post-repair row keys differ")
    for row_id, expected in desired.items():
        for field, value in expected.items():
            if field in FIELDS:
                matches = equivalent(value, actual[row_id][field])
            elif field == "updated_at" and row_id in changed:
                continue
            else:
                matches = value == actual[row_id][field]
            if not matches:
                raise ValueError(f"Unexpected post-repair change in {field}")


def verify_sql(cursor, plan):
    s = plan["scales"]
    # Independent SQL windows check that no current/future game contributes.
    cursor.execute(f"""WITH history AS (
        SELECT season,team,date,
               round(avg(team_pace) OVER prior_games, {s['opponent_avg_team_pace']}) AS prior_team_pace,
               round(avg(pace) OVER prior_games, {s['opponent_avg_game_pace']}) AS prior_game_pace
        FROM public.team_defensive_stats WHERE season = ANY(%s)
        WINDOW prior_games AS (PARTITION BY season,team ORDER BY date
                               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) SELECT count(*) AS checked,
        count(*) FILTER (WHERE
            p.opponent_avg_team_pace IS DISTINCT FROM h.prior_team_pace OR
            p.opponent_avg_game_pace IS DISTINCT FROM h.prior_game_pace OR
            p.opponent_pace_deviation IS DISTINCT FROM round(p.opponent_pace-h.prior_team_pace, {s['opponent_pace_deviation']}) OR
            p.opponent_game_pace_deviation IS DISTINCT FROM round(p.pace-h.prior_game_pace, {s['opponent_game_pace_deviation']})
        ) AS mismatches
        FROM public.team_defensive_stats p JOIN history h
          ON h.season=p.season AND h.team=p.opponent_abbrev AND h.date=p.date
        WHERE p.season = ANY(%s)""", (plan["seasons"], plan["seasons"]))
    result = cursor.fetchone()
    if result["checked"] != len(plan["after"]) or result["mismatches"] != 0:
        raise ValueError("Independent SQL prior-game verification failed")


def verify(plan):
    with connection() as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            check_after(snapshot(cursor, plan["seasons"]), plan)
            verify_sql(cursor, plan)
        conn.rollback()
    print(f"Independent read-only verification passed: {len(plan['after'])} rows match; SQL windows confirm strictly prior-game averages and correct deviations.")


def apply(plan):
    rebuilt = build_plan(plan["before"], plan["scales"])
    if any(rebuilt[f] != plan[f] for f in ("after", "changes", "summaries", "seasons")):
        raise ValueError("Reviewed plan no longer reproduces from its saved snapshot")
    with connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SET LOCAL lock_timeout = '5s'")
            cursor.execute("SET LOCAL statement_timeout = '120s'")
            cursor.execute("LOCK TABLE public.team_defensive_stats IN SHARE ROW EXCLUSIVE MODE")
            if snapshot(cursor, plan["seasons"]) != plan["before"] or read_scales(cursor) != plan["scales"]:
                raise ValueError("Database input or precision changed; repair refused")
            columns = ("id",) + FIELDS
            cursor.execute(f"CREATE TEMP TABLE _pace_repair ON COMMIT DROP AS SELECT {','.join(columns)} FROM public.team_defensive_stats WHERE false")
            execute_values(cursor, f"INSERT INTO _pace_repair ({','.join(columns)}) VALUES %s",
                           [tuple(r[c] for c in columns) for r in plan["after"]], page_size=1000)
            cursor.execute(f"UPDATE public.team_defensive_stats p SET {','.join(f'{f}=d.{f}' for f in FIELDS)},updated_at=now() FROM _pace_repair d WHERE p.id=d.id AND ROW({','.join(f'p.{f}' for f in FIELDS)}) IS DISTINCT FROM ROW({','.join(f'd.{f}' for f in FIELDS)})")
            if cursor.rowcount != len(plan["changes"]):
                raise ValueError("Unexpected updated row count")
            check_after(snapshot(cursor, plan["seasons"]), plan)
            verify_sql(cursor, plan)
    print(f"Pace repair committed: {len(plan['changes'])} rows; all transaction checks passed.", flush=True)
    verify(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path)
    group.add_argument("--apply-plan", type=Path)
    group.add_argument("--verify-plan", type=Path)
    parser.add_argument("--season", nargs="+", default=list(SEASONS))
    args = parser.parse_args()
    if args.apply_plan or args.verify_plan:
        plan = json.loads((args.apply_plan or args.verify_plan).read_text())
        (apply if args.apply_plan else verify)(plan)
        return
    if args.prepare.exists() or not args.prepare.parent.is_dir():
        parser.error("Choose a new report filename in an existing directory")
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            before = snapshot(cursor, args.season)
            scales = read_scales(cursor)
        conn.rollback()
    plan = build_plan(before, scales)
    with args.prepare.open("x") as handle:
        json.dump(plan, handle, indent=2)
        handle.write("\n")
    print(json.dumps(plan["summaries"], indent=2))
    print(f"Before/after backup: {args.prepare}. No database changes.")


if __name__ == "__main__":
    main()
