"""Read-only comparison of team defensive game records with NBA regular-season logs."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys

from dotenv import dotenv_values
from nba_api.stats.endpoints import teamgamelogs
import psycopg2
from psycopg2.extras import RealDictCursor


OPPONENT_FIELDS = {
    "opponent_fgm": "FGM", "opponent_fga": "FGA", "opponent_fg_pct": "FG_PCT",
    "opponent_3pm": "FG3M", "opponent_3pa": "FG3A", "opponent_3p_pct": "FG3_PCT",
    "opponent_ftm": "FTM", "opponent_fta": "FTA", "opponent_ft_pct": "FT_PCT",
    "opponent_orb": "OREB", "opponent_trb": "REB", "opponent_ast": "AST",
    "opponent_stl": "STL", "opponent_blk": "BLK", "opponent_tov": "TOV",
    "opponent_pf": "PF", "opponent_pts": "PTS",
}
TEAM_FIELDS = {"team_pts": "PTS", "team_stl": "STL", "team_blk": "BLK",
               "team_tov": "TOV", "team_pf": "PF", "team_orb": "OREB", "team_trb": "REB"}
PACE_FIELDS = ("pace", "team_pace", "opponent_pace", "off_rating", "def_rating")
AVERAGE_FIELDS = ("opponent_avg_team_pace", "opponent_pace_deviation",
                  "opponent_avg_game_pace", "opponent_game_pace_deviation")


def number(value):
    return Decimal(str(value))


def possessions(row):
    return number(row["FGA"]) + Decimal(".44") * number(row["FTA"]) - number(row["OREB"]) + number(row["TOV"])


def compare(season, source, stored, player_schedule):
    nba = {}
    for row in source:
        key = (str(row["GAME_DATE"])[:10], row["TEAM_ABBREVIATION"].strip())
        if key in nba:
            raise ValueError("Duplicate NBA team/date source key")
        nba[key] = row
    if not nba:
        raise ValueError("Empty NBA source")
    duplicates = Counter((str(r["date"]), r["team"].strip()) for r in stored)
    actual = {(str(r["date"]), r["team"].strip()): r for r in stored}
    mismatches, counts, expected_metrics = [], Counter(), {}
    for key, row in nba.items():
        matchup = row["MATCHUP"].split()
        if len(matchup) != 3 or matchup[0] != key[1] or matchup[1] not in ("@", "vs."):
            raise ValueError("Unexpected NBA matchup")
        opp = nba.get((key[0], matchup[2]))
        if opp is None or opp["GAME_ID"] != row["GAME_ID"]:
            raise ValueError("Incomplete NBA game pairing")
        own_poss, opp_poss = possessions(row), possessions(opp)
        minutes = number(row["MIN"])
        if minutes <= 0 or own_poss <= 0 or opp_poss <= 0:
            raise ValueError("Invalid NBA minutes or possessions")
        expected = {field: opp[src] for field, src in OPPONENT_FIELDS.items()}
        expected.update({field: row[src] for field, src in TEAM_FIELDS.items()})
        expected.update({
            "home": int(matchup[1] == "vs."), "opponent_abbrev": matchup[2],
            "game_id": f"{key[0]}_{key[1]}_{matchup[2]}",
            "pace": (own_poss + opp_poss) / 2 * 48 / minutes,
            "team_pace": own_poss * 48 / minutes,
            "opponent_pace": opp_poss * 48 / minutes,
            "off_rating": number(row["PTS"]) * 100 / own_poss,
            "def_rating": number(opp["PTS"]) * 100 / opp_poss,
        })
        expected_metrics[key] = expected
        if key not in actual:
            continue
        changes = {}
        for field, value in expected.items():
            current = actual[key].get(field)
            if field in ("opponent_abbrev", "game_id"):
                matches = current is not None and str(current).strip() == value
            else:
                tolerance = Decimal(".051") if field in PACE_FIELDS else Decimal(".00051") if field.endswith("_pct") else Decimal(0)
                matches = current is not None and abs(number(current) - number(value)) <= tolerance
            if not matches:
                counts[field] += 1
                changes[field] = {"stored": current, "nba_expected": value}
        if changes:
            mismatches.append({"date": key[0], "team": key[1], "changes": changes})

    # Reproduce the importer's averages to test internal consistency separately
    # from whether that definition is appropriate for pregame modeling.
    grouped = defaultdict(list)
    for row in stored:
        grouped[row["opponent_abbrev"].strip()].append(row)
    average_mismatches = Counter()
    for row in stored:
        group = grouped[row["opponent_abbrev"].strip()]
        if any(r["team_pace"] is None or r["pace"] is None for r in group):
            average_mismatches["null_pace_inputs"] += 1
            continue
        team_avg = sum(number(r["team_pace"]) for r in group) / len(group)
        game_avg = sum(number(r["pace"]) for r in group) / len(group)
        expected = (team_avg, number(row["opponent_pace"]) - team_avg,
                    game_avg, number(row["pace"]) - game_avg)
        for field, value in zip(AVERAGE_FIELDS, expected):
            if row[field] is None or abs(number(row[field]) - value) > Decimal(".051"):
                average_mismatches[field] += 1

    return {
        "season": season, "nba_rows": len(source), "database_rows": len(stored),
        "missing_games": [{"date": d, "team": t} for d, t in sorted(nba.keys() - actual.keys())],
        "extra_games": [{"date": d, "team": t, "opponent": actual[(d, t)]["opponent_abbrev"],
                         "game_id": actual[(d, t)]["game_id"]} for d, t in sorted(actual.keys() - nba.keys())],
        "duplicate_groups": [{"date": d, "team": t, "count": n} for (d, t), n in duplicates.items() if n > 1],
        "mismatched_rows": len(mismatches), "mismatch_counts": dict(counts), "mismatches": mismatches,
        "stored_average_formula_mismatch_counts": dict(average_mismatches),
        "missing_vs_verified_player_schedule": [list(k) for k in sorted(player_schedule - actual.keys())],
        "extra_vs_verified_player_schedule": [list(k) for k in sorted(actual.keys() - player_schedule)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", nargs="+", default=["2022-23", "2023-24", "2024-25", "2025-26"])
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists() or not args.json.parent.is_dir():
        parser.error("Choose a new JSON filename in an existing directory")
    config = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    with psycopg2.connect(host=config["DB_HOST"], port=config.get("DB_PORT") or "5432",
                         dbname=config.get("DB_NAME") or config.get("DB") or "postgres",
                         user=config["DB_USER"], password=config["DB_PASSWORD"],
                         sslmode="require", connect_timeout=15) as conn:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM public.team_defensive_stats WHERE season = ANY(%s) ORDER BY season,date,team", (args.season,))
            stored = [dict(r) for r in cursor.fetchall()]
            cursor.execute("SELECT DISTINCT season,date,btrim(team) AS team FROM public.player_gamelog WHERE season = ANY(%s)", (args.season,))
            schedule = [dict(r) for r in cursor.fetchall()]
        conn.rollback()
    report = {"audited_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Read-only NBA Regular Season TeamGameLogs versus stored rows; box scores, matchup, pace, ratings, and importer-average consistency. No date filters. Counts do not imply every player statistic was reaudited.",
              "modeling_note": "Stored full-season pace averages/deviations include future games. The opponent_avg_team_pace importer groups team_pace by opponent, which averages that opponent's adversaries, rather than the opponent's own team pace.",
              "seasons": []}
    for season in args.season:
        print(f"Fetching NBA regular-season team logs: {season}", flush=True)
        frame = teamgamelogs.TeamGameLogs(season_nullable=season,
                    season_type_nullable="Regular Season", timeout=40).get_data_frames()[0]
        source = json.loads(frame.to_json(orient="records", date_format="iso"))
        result = compare(season, source, [r for r in stored if r["season"] == season],
                         {(str(r["date"]), r["team"]) for r in schedule if r["season"] == season})
        report["seasons"].append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("mismatches",)}, default=str), flush=True)
    with args.json.open("x") as handle:
        json.dump(report, handle, indent=2, default=str)
        handle.write("\n")
    print(f"Full audit: {args.json}; no database writes.")


if __name__ == "__main__":
    main()
