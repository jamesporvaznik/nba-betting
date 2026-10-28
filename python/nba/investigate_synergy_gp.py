"""Read-only investigation of Synergy team GP versus verified regular-season games."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from nba_api.stats.endpoints import synergyplaytypes
from psycopg2.extras import RealDictCursor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba.aggregate_playtypes import PLAY_TYPES
from nba.audit_team_season_stats import capture
from nba.audit_tracking import SEASONS
from nba.repair_shot_locations import connection, serial


def decode(path):
    data = json.loads(path.read_text())
    result = data["resultSets"][0]
    return [dict(zip(result["headers"], r)) for r in result["rowSet"]]


def cutoff_check(games, source_gp):
    grouped = {team: sorted([g for g in games if g["team"] == team], key=lambda g: g["date"]) for team in source_gp}
    bounds = []
    for team, gp in source_gp.items():
        if not 0 < gp <= len(grouped[team]):
            raise ValueError("Source GP outside verified schedule")
        bounds.append({"team": team, "source_gp": gp,
                       "last_included_date_if_prefix": grouped[team][gp - 1]["date"],
                       "next_game_date_if_prefix": grouped[team][gp]["date"] if gp < len(grouped[team]) else None})
    lower = max(r["last_included_date_if_prefix"] for r in bounds)
    upper = min((r["next_game_date_if_prefix"] for r in bounds if r["next_game_date_if_prefix"]), default=None)
    candidates = []
    for date in sorted({g["date"] for g in games}):
        counts = {t: sum(g["date"] <= date for g in gs) for t, gs in grouped.items()}
        candidates.append({"date": date, "absolute_team_game_count_difference": sum(abs(counts[t] - source_gp[t]) for t in source_gp),
                           "matching_teams": sum(counts[t] == source_gp[t] for t in source_gp)})
    return {"common_cutoff_possible": upper is None or lower < upper,
            "latest_required_included_date": lower, "earliest_required_excluded_date": upper,
            "closest_cutoff": min(candidates, key=lambda r: (r["absolute_team_game_count_difference"], -r["matching_teams"])),
            "team_boundaries_if_prefix": bounds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    baseline_dir = Path(__file__).resolve().parents[2] / "database" / "repairs" / "team_season_defense_sources"
    with connection() as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT d.season,d.team,d.date,d.team_pts,d.opponent_pts,d.opponent_fgm,d.opponent_fga FROM public.team_defensive_stats d WHERE EXISTS (SELECT 1 FROM public.player_gamelog p WHERE p.season=d.season AND p.date=d.date AND btrim(p.team)=d.team) ORDER BY d.season,d.team,d.date")
            games = serial([dict(r) for r in cursor.fetchall()])
        conn.rollback()
    (args.output_dir / "verified_schedule_snapshot.json").write_text(json.dumps(games, indent=2))
    report = {"captured_at": datetime.now(timezone.utc).isoformat(), "database_writes": False,
              "endpoint_limitations": "The public synergyplaytypes endpoint and installed SDK expose season, play type, offensive/defensive grouping, team/player selection, and mode, but no team-ID or date filters. Team-level comparisons select rows from full-league responses; they are not independently filtered team requests. No game IDs are included in these season responses.",
              "seasons": []}
    for season in SEASONS:
        season_games = [g for g in games if g["season"] == season]
        print(f"Checking {season} offense versus defense", flush=True)
        offensive, defensive = {}, {}
        for grouping, target in (("offensive", offensive), ("defensive", defensive)):
            plays = PLAY_TYPES if season == "2025-26" and grouping == "offensive" else ("Transition",)
            for play in plays:
                rows = capture(lambda: synergyplaytypes.SynergyPlayTypes(
                    league_id="00", per_mode_simple="Totals", player_or_team_abbreviation="T",
                    season_type_all_star="Regular Season", season=season, play_type_nullable=play,
                    type_grouping_nullable=grouping, timeout=35),
                    f"{season}_{grouping}_{play}_Totals", args.output_dir)
                target[play] = {r["TEAM_ABBREVIATION"]: r for r in rows}
                time.sleep(.3)
        source_gp = {team: r["GP"] for team, r in defensive["Transition"].items()}
        if len(source_gp) != 30 or set(source_gp) != set(offensive["Transition"]):
            raise ValueError("Incomplete team coverage in source")
        baseline_gp = {r["TEAM_ABBREVIATION"]: r["GP"] for r in decode(baseline_dir / f"{season}_defense_Transition_Totals.json")}
        side_differences = [{"team": team, "offensive_gp": offensive["Transition"][team]["GP"],
                             "defensive_gp": gp} for team, gp in source_gp.items() if offensive["Transition"][team]["GP"] != gp]
        gp_categories = defaultdict(set)
        sums = defaultdict(Counter)
        for play in PLAY_TYPES:
            for r in decode(baseline_dir / f"{season}_defense_{play}_Totals.json"):
                team = r["TEAM_ABBREVIATION"]
                gp_categories[team].add(r["GP"])
                for field in ("PTS", "FGM", "FGA"):
                    sums[team][field] += r[field]
        box_comparisons = []
        for team, gp in sorted(source_gp.items()):
            team_games = [g for g in season_games if g["team"] == team]
            box = {"PTS": sum(g["opponent_pts"] for g in team_games),
                   "FGM": sum(g["opponent_fgm"] for g in team_games),
                   "FGA": sum(g["opponent_fga"] for g in team_games)}
            box_comparisons.append({"team": team, "source_gp": gp, "box_score_gp": len(team_games),
                                    "defensive_type_gp_values": sorted(gp_categories[team]),
                                    "synergy_totals_across_eleven_types": dict(sums[team]),
                                    "box_score_season_totals": box,
                                    "source_minus_box_score": {f: sums[team][f] - box[f] for f in box}})
        result = {"season": season, "gp_matches_prior_defensive_snapshot": source_gp == baseline_gp,
                  "offensive_defensive_gp_differences": side_differences,
                  "offensive_gp_values_across_types": {team: sorted({by_team[team]["GP"] for by_team in offensive.values()}) for team in source_gp},
                  "cutoff_analysis": cutoff_check(season_games, source_gp), "team_comparisons": box_comparisons}
        if season == "2025-26":
            result["league_offensive_points_across_types"] = sum(row["PTS"] for by_team in offensive.values() for row in by_team.values())
            result["league_defensive_points_across_types"] = sum(s["PTS"] for s in sums.values())
            result["league_offensive_gp_across_types_consistent"] = all(len({by_team[team]["GP"] for by_team in offensive.values()}) == 1 for team in source_gp)
        report["seasons"].append(result)
        print(json.dumps({"season": season, "same_as_previous_source": source_gp == baseline_gp,
                          "offense_defense_gp_differences": len(side_differences),
                          "common_cutoff_possible": result["cutoff_analysis"]["common_cutoff_possible"],
                          "closest_cutoff": result["cutoff_analysis"]["closest_cutoff"]}), flush=True)
    (args.output_dir / "findings.json").write_text(json.dumps(report, indent=2))
    print(f"Investigation saved: {args.output_dir}. No database changes.")


if __name__ == "__main__":
    main()
