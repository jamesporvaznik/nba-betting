"""Read-only identity/reference audit and review of derived defensive pace fields."""

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
import sys
import unicodedata

from nba_api.stats.endpoints import commonallplayers, leaguestandingsv3
from nba_api.stats.static import teams as static_teams
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba.audit_team_season_stats import capture
from nba.repair_shot_locations import connection, serial


PLAYER_REFERENCES = {"player_gamelog": "playerid", "player_playtype_stats": "player_id",
                     "player_shot_locations": "player_id", "player_tracking_stats": "player_id",
                     "player_team_stints": "player_id", "player_clusters": "player_id", "predictions": "playerid"}
TEAM_REFERENCES = {"team_defensive_playtype_stats": "team_id", "team_defensive_shot_locations": "team_id",
                   "player_team_stints": "team_id"}


def normalize(value):
    value = unicodedata.normalize("NFKD", str(value or ""))
    return re.sub(r"[^a-z0-9]", "", value.encode("ascii", "ignore").decode().lower())


def duplicates(rows, field):
    counts = Counter(r[field] for r in rows if r.get(field) not in (None, ""))
    return [{"value": value, "count": n} for value, n in counts.items() if n > 1]


def audit_identities(players, teams, nba_players, standings, references):
    official = {int(r["PERSON_ID"]): r for r in nba_players}
    nba_teams = {int(r["TeamID"]): r for r in standings}
    static = {r["id"]: r for r in static_teams.get_teams()}
    if len(nba_teams) != 30 or len(official) != len(nba_players):
        raise ValueError("Incomplete or duplicate official directory")
    internal_teams = {r["id"]: r for r in teams}
    master_ids = {r["nba_api_id"] for r in players}
    team_ids = {r["nba_api_id"] for r in teams}
    player_checks = {"rows": len(players), "duplicate_nba_ids": duplicates(players, "nba_api_id"),
                     "duplicate_internal_ids": duplicates(players, "id"),
                     "duplicate_bball_ref_ids": duplicates(players, "bball_ref_id"),
                     "ambiguous_display_names_not_necessarily_errors": duplicates(players, "display_name"),
                     "ids_absent_from_nba_directory": [], "blank_identity_fields": [],
                     "normalized_name_differences": [], "raw_name_differences": [],
                     "invalid_internal_team_references": [], "current_roster_team_refresh_candidates": [],
                     "active_flag_vs_current_roster_differences_not_equivalent_definitions": [],
                     "profile_plausibility_flags": [], "bball_ref_format_flags": [],
                     "missing_optional_fields": {field: sum(r.get(field) in (None, "") for r in players)
                                                 for field in ("birth_date", "height_inches", "weight_lbs", "position", "draft_year", "bball_ref_id")}}
    for row in players:
        pid = row["nba_api_id"]
        for field in ("nba_api_id", "display_name", "first_name"):
            if row.get(field) in (None, ""):
                player_checks["blank_identity_fields"].append({"player_id": pid, "field": field})
        source = official.get(pid)
        if source is None:
            player_checks["ids_absent_from_nba_directory"].append(pid)
        else:
            if row["display_name"] != source["DISPLAY_FIRST_LAST"]:
                entry = {"player_id": pid, "stored": row["display_name"], "nba": source["DISPLAY_FIRST_LAST"]}
                player_checks["raw_name_differences"].append(entry)
                if normalize(row["display_name"]) != normalize(source["DISPLAY_FIRST_LAST"]):
                    player_checks["normalized_name_differences"].append(entry)
            on_roster = bool(int(source["ROSTERSTATUS"]))
            if row["is_active"] is not None and bool(row["is_active"]) != on_roster:
                player_checks["active_flag_vs_current_roster_differences_not_equivalent_definitions"].append({"player_id": pid, "name": row["display_name"], "stored_is_active": row["is_active"], "nba_current_roster": on_roster})
            team = internal_teams.get(row["team_id"])
            stored_team = team["nba_api_id"] if team else None
            if on_roster and source["TEAM_ID"] and stored_team != source["TEAM_ID"]:
                player_checks["current_roster_team_refresh_candidates"].append({"player_id": pid, "name": row["display_name"], "stored_team_nba_id": stored_team, "nba_current_team_id": source["TEAM_ID"]})
        if row["team_id"] is not None and row["team_id"] not in internal_teams:
            player_checks["invalid_internal_team_references"].append({"player_id": pid, "team_id": row["team_id"]})
        for field, low, high in (("height_inches", 55, 95), ("weight_lbs", 100, 450)):
            if row.get(field) is not None and not low <= row[field] <= high:
                player_checks["profile_plausibility_flags"].append({"player_id": pid, "field": field, "value": row[field]})
        if row.get("birth_date") and not date(1900, 1, 1) <= date.fromisoformat(str(row["birth_date"])) <= date.today():
            player_checks["profile_plausibility_flags"].append({"player_id": pid, "field": "birth_date", "value": row["birth_date"]})
        if row.get("bball_ref_id") and not re.fullmatch(r"[a-z]+[0-9]{2}", row["bball_ref_id"]):
            player_checks["bball_ref_format_flags"].append({"player_id": pid, "value": row["bball_ref_id"]})
    player_checks["current_nba_roster_ids_not_in_historical_master"] = sorted(pid for pid, r in official.items() if int(r["ROSTERSTATUS"]) and pid not in master_ids)
    team_checks = {"rows": len(teams), "duplicate_nba_ids": duplicates(teams, "nba_api_id"),
                   "duplicate_internal_ids": duplicates(teams, "id"), "duplicate_abbreviations": duplicates(teams, "abbreviation"),
                   "missing_nba_team_ids": sorted(nba_teams.keys() - team_ids),
                   "unexpected_nba_team_ids": sorted(team_ids - nba_teams.keys()), "mapping_differences": []}
    for row in teams:
        source = nba_teams.get(row["nba_api_id"])
        if source is None:
            continue
        expected = {"conference": source["Conference"], "division": source["Division"]}
        if row["nba_api_id"] in static:
            expected["abbreviation"] = static[row["nba_api_id"]]["abbreviation"]
            expected["full_name"] = static[row["nba_api_id"]]["full_name"]
        for field, value in expected.items():
            a, b = normalize(row.get(field)), normalize(value)
            if field == "conference":
                a, b = a.removesuffix("ern"), b.removesuffix("ern")
            if field == "full_name":
                a, b = a.replace("losangeles", "la"), b.replace("losangeles", "la")
            if a != b:
                team_checks["mapping_differences"].append({"team_id": row["nba_api_id"], "field": field, "stored": row.get(field), "reference": value})
    orphaned = []
    for namespace, table, column, records in references:
        allowed = master_ids if namespace == "player_nba_id" else team_ids
        missing = [r for r in records if r["value"] not in allowed]
        orphaned.append({"table": table, "column": column, "namespace": namespace,
                         "rows_checked": sum(r["records"] for r in records), "orphaned_ids": missing})
    return {"players": player_checks, "teams": team_checks, "reference_checks": orphaned,
            "limitations": "Identity IDs/names and current roster mappings use the fresh NBA directory. Profile heights/weights/birth dates receive plausibility checks, not individual live profile re-fetches. Basketball Reference IDs receive uniqueness/format checks only, not an external crosswalk verification. Current roster changes and absent 2026-27 rookies are not proof of broken historical-season data. NBA roster status is not identical to the importer's active-career flag."}


def audit_pace(rows):
    results = []
    for season in sorted({r["season"] for r in rows}):
        season_rows = [r for r in rows if r["season"] == season]
        own, facing = defaultdict(list), defaultdict(list)
        for r in season_rows:
            own[r["team"]].append(Decimal(str(r["team_pace"])))
            facing[r["opponent_abbrev"]].append(Decimal(str(r["team_pace"])))
        own_means = {t: sum(v) / len(v) for t, v in own.items()}
        legacy_means = {t: sum(v) / len(v) for t, v in facing.items()}
        definition_differences, legacy_differences = [], 0
        prior, dates = defaultdict(list), defaultdict(list)
        for r in season_rows:
            dates[str(r["date"])].append(r)
        no_prior, prior_mismatches, nonnull_without_history = 0, 0, 0
        for day, games in sorted(dates.items()):
            for r in games:
                opp = r["opponent_abbrev"]
                stored = None if r["opponent_avg_team_pace"] is None else Decimal(str(r["opponent_avg_team_pace"]))
                if stored is None or abs(stored - legacy_means[opp]) > Decimal(".051"):
                    legacy_differences += 1
                correct = own_means[opp]
                if stored is None or abs(stored - correct) > Decimal(".051"):
                    definition_differences.append({"date": day, "team": r["team"], "opponent": opp,
                                                    "stored": stored, "opponent_own_full_season_team_pace": correct})
                if not prior[opp]:
                    no_prior += 1
                    nonnull_without_history += int(stored is not None)
                elif stored is None or abs(stored - sum(prior[opp]) / len(prior[opp])) > Decimal(".051"):
                    prior_mismatches += 1
            # Commit only after every game on the date is evaluated.
            for r in games:
                prior[r["team"]].append(Decimal(str(r["team_pace"])))
        results.append({"season": season, "rows": len(season_rows),
                        "stored_average_disagrees_with_importer_formula": legacy_differences,
                        "stored_average_disagrees_with_opponent_own_pace": len(definition_differences),
                        "definition_difference_examples": definition_differences[:8],
                        "max_absolute_average_definition_difference": max((abs(r["stored"] - r["opponent_own_full_season_team_pace"]) for r in definition_differences if r["stored"] is not None), default=Decimal(0)),
                        "rows_without_any_prior_opponent_games": no_prior,
                        "nonnull_baselines_without_any_prior_opponent_games": nonnull_without_history,
                        "rows_with_prior_history_where_full_season_average_differs": prior_mismatches})
    return {"seasons": results,
            "strict_prior_team_average_checks_pass": all(r["nonnull_baselines_without_any_prior_opponent_games"] == 0 and r["rows_with_prior_history_where_full_season_average_differs"] == 0 for r in results),
            "findings": ["Importer/full-season comparisons are retrospective diagnostics; differences are expected after the prior-game pace repair, not proof of incorrect pregame baselines.",
                         "The strict-prior checks compare opponent_avg_team_pace with the opponent's own team pace from earlier dates and require NULL when no history exists.",
                         "Both pace deviations use the current game's pace; they remain descriptive outcomes even when the averages use strictly earlier games.",
                         "The current training feature list does not consume these four derived pace fields; leave historical box-score pace/ratings intact and build explicitly prior-only features for prediction.",
                         "This check includes the existing Cup-final rows; regular-season-only modeling must explicitly choose its game scope."],
            "recommendation": "Use repair_pace --verify-plan for an exact four-column check with independent SQL windows. Keep current-game pace and deviations out of inputs for that game; opponent averages must use only earlier dates."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-season", default="2026-27")
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM public.players ORDER BY nba_api_id")
            players = serial([dict(r) for r in cursor.fetchall()])
            cursor.execute("SELECT * FROM public.teams ORDER BY nba_api_id")
            teams = serial([dict(r) for r in cursor.fetchall()])
            cursor.execute("SELECT season,team,opponent_abbrev,date,team_pace,opponent_pace,pace,opponent_avg_team_pace,opponent_avg_game_pace,opponent_pace_deviation,opponent_game_pace_deviation FROM public.team_defensive_stats ORDER BY season,date,team")
            pace = serial([dict(r) for r in cursor.fetchall()])
            references = []
            for namespace, mapping in (("player_nba_id", PLAYER_REFERENCES), ("team_nba_id", TEAM_REFERENCES)):
                for table, column in mapping.items():
                    cursor.execute(sql.SQL("SELECT {} AS value,count(*) AS records FROM public.{} GROUP BY {}").format(sql.Identifier(column), sql.Identifier(table), sql.Identifier(column)))
                    references.append((namespace, table, column, serial([dict(r) for r in cursor.fetchall()])))
            cursor.execute("SELECT DISTINCT btrim(team) AS team,btrim(opponent) AS opponent FROM public.player_gamelog")
            abbreviations = serial([dict(r) for r in cursor.fetchall()])
        conn.rollback()
    nba_players = capture(lambda: commonallplayers.CommonAllPlayers(is_only_current_season=0,season=args.reference_season,league_id="00",timeout=35), "nba_player_directory", args.output_dir)
    nba_standings = capture(lambda: leaguestandingsv3.LeagueStandingsV3(league_id="00",season=args.reference_season,season_type="Regular Season",timeout=35), "nba_team_standings", args.output_dir)
    identities = audit_identities(players, teams, nba_players, nba_standings, references)
    known = {t["abbreviation"] for t in teams}
    identities["unknown_game_log_team_abbreviations"] = sorted({r[f] for r in abbreviations for f in ("team", "opponent")} - known)
    report = {"audited_at": datetime.now(timezone.utc).isoformat(), "database_writes": False,
              "current_reference_season": args.reference_season, "identities": identities,
              "pace": audit_pace(pace)}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print("Players:", len(players), "Teams:", len(teams), flush=True)
    for group in ("players", "teams"):
        print(group, json.dumps({k: len(v) if isinstance(v, list) else v for k, v in identities[group].items()}, default=str), flush=True)
    print("Orphaned reference groups:", sum(bool(r["orphaned_ids"]) for r in identities["reference_checks"]), flush=True)
    print("Pace:", json.dumps(report["pace"]["seasons"], default=str), flush=True)
    print(f"Saved report: {args.output_dir / 'report.json'}. No database changes.")


if __name__ == "__main__":
    main()
