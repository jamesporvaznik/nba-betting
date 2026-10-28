"""Read-only audit of player shot zones against NBA per-game and totals responses."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

from dotenv import dotenv_values
from nba_api.stats.endpoints import leaguedashplayershotlocations
import psycopg2
from psycopg2.extras import RealDictCursor


ZONES = ("Restricted Area", "In The Paint (Non-RA)", "Mid-Range", "Left Corner 3",
         "Right Corner 3", "Above the Break 3")


def dec(value):
    return Decimal(str(value))


def fetch(season, mode):
    frame = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
        season=season, season_type_all_star="Regular Season", league_id_nullable="00",
        per_mode_detailed=mode, pace_adjust="N", distance_range="By Zone", timeout=40,
    ).get_data_frames()[0]
    frame.columns = ["_".join(map(str, column)).strip("_") for column in frame.columns]
    rows = json.loads(frame.to_json(orient="records"))
    if not rows or len({r["PLAYER_ID"] for r in rows}) != len(rows):
        raise ValueError("Empty source or multiple source rows per player; review before comparison")
    for row in rows:
        for zone in ZONES + ("Backcourt",):
            for stat in ("FGM", "FGA", "FG_PCT"):
                if f"{zone}_{stat}" not in row:
                    raise ValueError("Incomplete NBA zone columns")
    return rows


def compare(season, totals, per_game, stored, gamelog_totals):
    source = {int(r["PLAYER_ID"]): r for r in totals}
    averages = {int(r["PLAYER_ID"]): r for r in per_game}
    if source.keys() != averages.keys():
        raise ValueError("Totals and PerGame responses disagree on player coverage")
    keys = Counter((int(r["player_id"]), r["shot_zone"]) for r in stored)
    actual = {(int(r["player_id"]), r["shot_zone"]): r for r in stored}
    expected, total_keys, rounded_away, rounding_effects, source_nulls = {}, set(), [], [], []
    for pid, row in averages.items():
        denominator = sum(dec(row[f"{zone}_FGA"]) for zone in ZONES if row[f"{zone}_FGA"] is not None and dec(row[f"{zone}_FGA"]) > 0)
        total_denominator = sum(dec(source[pid][f"{zone}_FGA"]) for zone in ZONES if source[pid][f"{zone}_FGA"] is not None)
        for zone in ZONES:
            key = (pid, zone)
            if any(r[f"{zone}_{stat}"] is None for r in (row, source[pid]) for stat in ("FGM", "FGA", "FG_PCT")):
                source_nulls.append({"player_id": pid, "shot_zone": zone})
                continue
            total_attempts = dec(source[pid][f"{zone}_FGA"])
            if total_attempts > 0:
                total_keys.add(key)
            attempts = dec(row[f"{zone}_FGA"])
            if attempts <= 0:
                if total_attempts > 0:
                    rounded_away.append({"player_id": pid, "player_name": row["PLAYER_NAME"],
                                         "shot_zone": zone, "season_fga": total_attempts})
                continue
            expected[key] = {"fgm": row[f"{zone}_FGM"], "fga": row[f"{zone}_FGA"],
                             "fg_pct": row[f"{zone}_FG_PCT"],
                             "frequency_pct": attempts / denominator}
            exact_frequency = total_attempts / total_denominator if total_denominator else Decimal(0)
            if abs(exact_frequency - expected[key]["frequency_pct"]) > Decimal(".005"):
                rounding_effects.append({"player_id": pid, "player_name": row["PLAYER_NAME"],
                                         "shot_zone": zone,
                                         "per_game_frequency": expected[key]["frequency_pct"],
                                         "totals_frequency": exact_frequency})
    mismatch_counts, mismatches, invalid, frequency_sums = Counter(), [], [], defaultdict(Decimal)
    for key, row in actual.items():
        if row["frequency_pct"] is not None:
            frequency_sums[key[0]] += dec(row["frequency_pct"])
        if (any(row.get(c) is None for c in ("fga", "fgm", "fg_pct", "frequency_pct"))
                or not 0 <= dec(row["fgm"]) <= dec(row["fga"])
                or not 0 <= dec(row["fg_pct"]) <= 1
                or not 0 <= dec(row["frequency_pct"]) <= 1):
            invalid.append({"player_id": key[0], "shot_zone": key[1]})
        if key not in expected:
            continue
        differences = {}
        for field, value in expected[key].items():
            tolerance = Decimal(".000051")
            current = row.get(field)
            if current is None or abs(dec(current) - dec(value)) > tolerance:
                mismatch_counts[field] += 1
                differences[field] = {"stored": current, "source_expected": value}
        if differences:
            mismatches.append({"player_id": key[0], "shot_zone": key[1], "changes": differences})

    totals_mismatches, traded_checked, omitted_backcourt, incomplete_totals = [], 0, [], []
    for pid, logs in gamelog_totals.items():
        if pid not in source:
            if logs["fga"]:
                totals_mismatches.append({"player_id": pid, "reason": "Absent from NBA shot source despite gamelog FGA"})
            continue
        row = source[pid]
        traded_checked += int(logs["teams"] > 1)
        for stat in ("FGM", "FGA"):
            if any(row[f"{zone}_{stat}"] is None for zone in ZONES + ("Backcourt",)):
                incomplete_totals.append({"player_id": pid, "stat": stat})
                continue
            total = sum(dec(row[f"{zone}_{stat}"]) for zone in ZONES + ("Backcourt",))
            if total != dec(logs[stat.lower()]):
                totals_mismatches.append({"player_id": pid, "player_name": row["PLAYER_NAME"],
                                          "teams": logs["teams"], "stat": stat,
                                          "zone_total": total, "gamelog_total": logs[stat.lower()]})
        if row["Backcourt_FGA"] is not None and dec(row["Backcourt_FGA"]) > 0:
            omitted_backcourt.append({"player_id": pid, "player_name": row["PLAYER_NAME"],
                                      "fgm": row["Backcourt_FGM"], "fga": row["Backcourt_FGA"]})

    describe = lambda keys: [{"player_id": p, "shot_zone": z} for p, z in sorted(keys)]
    return {"season": season, "database_rows": len(stored), "database_players": len(frequency_sums),
            "source_players": len(source), "source_per_game_positive_zone_rows": len(expected),
            "missing_per_game_rows": describe(expected.keys() - actual.keys()),
            "unexpected_per_game_rows": describe(actual.keys() - expected.keys()),
            "duplicate_keys": describe(k for k, count in keys.items() if count > 1),
            "invalid_rows": invalid, "mismatch_counts": dict(mismatch_counts), "mismatches": mismatches,
            "frequency_sums_outside_rounding": [{"player_id": pid, "sum": total} for pid, total in frequency_sums.items() if abs(total - 1) > Decimal(".00031")],
            "missing_positive_totals_zones": describe(total_keys - actual.keys()),
            "positive_zones_rounded_to_zero_in_per_game_source": rounded_away,
            "per_game_vs_totals_frequency_difference_over_half_percentage_point": rounding_effects,
            "multi_team_player_seasons_present_in_source_and_gamelogs": traded_checked,
            "source_null_zone_records_not_assumed_zero": source_nulls,
            "incomplete_source_player_totals_not_compared": incomplete_totals,
            "zone_totals_vs_gamelog_mismatches": totals_mismatches,
            "players_with_backcourt_attempts_excluded_by_six_zone_design": omitted_backcourt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    if args.json.exists() or not args.json.parent.is_dir():
        parser.error("Choose a new report filename in an existing directory")
    config = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    with psycopg2.connect(host=config["DB_HOST"], port=config.get("DB_PORT") or "5432",
                         dbname=config.get("DB_NAME") or config.get("DB") or "postgres",
                         user=config["DB_USER"], password=config["DB_PASSWORD"],
                         sslmode="require", connect_timeout=15) as conn:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM public.player_shot_locations ORDER BY season,player_id,shot_zone")
            stored = [dict(r) for r in cursor.fetchall()]
            cursor.execute("SELECT season,playerid,count(DISTINCT btrim(team)) AS teams,sum(fgm) AS fgm,sum(fga) AS fga FROM public.player_gamelog GROUP BY season,playerid")
            logs = [dict(r) for r in cursor.fetchall()]
        conn.rollback()
    report = {"audited_at": datetime.now(timezone.utc).isoformat(),
              "units": "Stored FGM/FGA are published NBA PerGame values, not season totals. Their zone shares can differ from season-count shares; this is not assumed to be rounding alone. Frequency is a 0-1 share of six selected zones, excluding Backcourt; Corner 3 is a duplicate combined category and is not added again.",
              "seasons": []}
    seasons = sorted({r["season"] for r in stored} | {r["season"] for r in logs})
    for season in seasons:
        print(f"Fetching {season} NBA shot-location Totals and PerGame...", flush=True)
        result = compare(season, fetch(season, "Totals"), fetch(season, "PerGame"),
                         [r for r in stored if r["season"] == season],
                         {r["playerid"]: r for r in logs if r["season"] == season})
        report["seasons"].append(result)
        print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in result.items()}, default=str), flush=True)
    with args.json.open("x") as handle:
        json.dump(report, handle, indent=2, default=str)
        handle.write("\n")
    print(f"Report saved: {args.json}. No database writes.")


if __name__ == "__main__":
    main()
