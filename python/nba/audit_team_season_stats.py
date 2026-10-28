"""Read-only audit of team defensive play types and shot zones against NBA sources."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import sys
import time

from nba_api.stats.endpoints import synergyplaytypes, leaguedashteamshotlocations
from psycopg2.extras import RealDictCursor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba.aggregate_playtypes import PLAY_TYPES
from nba.audit_shot_locations import ZONES
from nba.audit_tracking import SEASONS
from nba.repair_shot_locations import connection, serial


def dec(value):
    return Decimal(str(value))


def rounded(value, scale):
    return None if value is None else dec(value).quantize(Decimal(1).scaleb(-scale), rounding=ROUND_HALF_UP)


def capture(factory, name, directory, flatten=False):
    for attempt in range(2):
        try:
            endpoint = factory()
            raw = endpoint.nba_response.get_response()
            frame = endpoint.get_data_frames()[0]
            if flatten:
                frame.columns = ["_".join(map(str, column)).strip("_") for column in frame.columns]
            rows = json.loads(frame.to_json(orient="records"))
            if not rows:
                raise ValueError("Empty NBA response")
            (directory / f"{name}.json").write_text(raw)
            (directory / f"{name}_request.json").write_text(json.dumps({
                "captured_at": datetime.now(timezone.utc).isoformat(), "parameters": endpoint.parameters,
                "request_url": endpoint.nba_response.get_url(),
                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest()}, indent=2))
            return rows
        except Exception:
            if attempt:
                raise
            time.sleep(2)


def indexed(rows, columns):
    counts = Counter(tuple(r[c] for c in columns) for r in rows)
    return {tuple(r[c] for c in columns): r for r in rows}, [list(k) for k, n in counts.items() if n > 1]


def compare_fields(actual, expected, scales):
    differences = {}
    for field, value in expected.items():
        stored = actual.get(field)
        if field in scales:
            match = stored is None if value is None else stored is not None and dec(stored) == rounded(value, scales[field])
        else:
            match = stored == value
        if not match:
            differences[field] = {"stored": stored, "source_expected": value}
    return differences


def compare_plays(season, per_game, totals, stored, scales, teams, box_totals):
    per, dup_per = indexed(per_game, ("TEAM_ID", "PLAY_TYPE"))
    total, dup_total = indexed(totals, ("TEAM_ID", "PLAY_TYPE"))
    actual, duplicates = indexed(stored, ("team_id", "play_type"))
    required = {(team, play) for team in teams for play in PLAY_TYPES}
    if dup_per or dup_total or set(per) != required or set(total) != required:
        raise ValueError("NBA play-type coverage is incomplete or duplicated")
    mismatches, counts, rate_differences, ppp_differences, gp_differences = [], Counter(), [], [], []
    total_possessions, frequencies = defaultdict(Decimal), defaultdict(Decimal)
    for (team, play), row in total.items():
        if str(row["SEASON_ID"]) != "2" + season[:4] or str(row["TYPE_GROUPING"]).lower() != "defensive":
            raise ValueError("Wrong source season or grouping")
        for field in ("POSS", "PTS", "FGM", "FGA", "GP"):
            if row[field] is None or dec(row[field]) < 0 or dec(row[field]) != dec(row[field]).to_integral_value():
                raise ValueError("Invalid NBA total counts")
        if row["FGM"] > row["FGA"] or row["POSS"] <= 0 or row["GP"] <= 0:
            raise ValueError("Invalid NBA makes, possessions, or game count")
        total_possessions[team] += dec(row["POSS"])
    source_frequency_differences = []
    for k, source in per.items():
        team, play = k
        season_total = total[k]
        if str(source["SEASON_ID"]) != "2" + season[:4] or str(source["TYPE_GROUPING"]).lower() != "defensive" or source["GP"] != season_total["GP"]:
            raise ValueError("Per-game source filters or GP disagree")
        if season_total["GP"] != box_totals[team]["gp"]:
            gp_differences.append({"team_id": team, "play_type": play, "source_gp": season_total["GP"], "regular_season_gp": box_totals[team]["gp"]})
        for field in ("POSS", "PTS"):
            exact_rate = dec(season_total[field]) / season_total["GP"]
            if abs(dec(source[field]) - exact_rate) > Decimal(".051"):
                rate_differences.append({"team_id": team, "play_type": play, "field": field,
                                         "per_game": source[field], "totals_divided_by_source_gp": exact_rate})
        for field in ("PPP", "POSS_PCT", "FG_PCT", "EFG_PCT", "PERCENTILE"):
            if source[field] != season_total[field]:
                rate_differences.append({"team_id": team, "play_type": play, "field": field,
                                         "per_game": source[field], "totals": season_total[field]})
        exact_frequency = dec(season_total["POSS"]) / total_possessions[team]
        if abs(dec(season_total["POSS_PCT"]) - exact_frequency) > Decimal(".00051"):
            source_frequency_differences.append({"team_id": team, "play_type": play,
                                                 "source": season_total["POSS_PCT"], "share_of_eleven_types": exact_frequency})
        if k not in actual:
            continue
        old = actual[k]
        expectations = {
            "team_abbreviation": source["TEAM_ABBREVIATION"],
            "possessions_defended": source["POSS"], "points_allowed": source["PTS"],
            "poss_pct": source["POSS_PCT"], "percentile": source["PERCENTILE"],
            "games_played": source["GP"], "fg_pct_allowed": source["FG_PCT"],
            "efg_pct_allowed": source["EFG_PCT"],
            "points_allowed_per_possession": round(float(source["PTS"]) / float(source["POSS"]), 3),
        }
        changes = compare_fields(old, expectations, scales)
        if changes:
            counts.update(changes)
            mismatches.append({"team_id": team, "play_type": play, "changes": changes})
        exact_ppp = dec(season_total["PTS"]) / dec(season_total["POSS"])
        if old["points_allowed_per_possession"] is None or dec(old["points_allowed_per_possession"]) != rounded(exact_ppp, scales["points_allowed_per_possession"]):
            ppp_differences.append({"team_id": team, "play_type": play,
                                    "stored": old["points_allowed_per_possession"],
                                    "exact_totals_ppp": exact_ppp,
                                    "expected_at_storage_precision": rounded(exact_ppp, scales["points_allowed_per_possession"])})
        if old["poss_pct"] is not None:
            frequencies[team] += dec(old["poss_pct"])
    return {"season": season, "database_rows": len(stored), "expected_rows": len(required),
            "missing_keys": [list(k) for k in sorted(required - actual.keys())],
            "unexpected_keys": [list(k) for k in sorted(actual.keys() - required)],
            "duplicate_keys": duplicates, "source_replay_mismatch_counts": dict(counts),
            "source_replay_mismatches": mismatches,
            "source_per_game_totals_inconsistencies": rate_differences,
            "stored_ppp_different_from_rounded_exact_totals": ppp_differences,
            "source_gp_different_from_regular_season_appearances": gp_differences,
            "source_frequency_different_from_share_of_eleven_types": source_frequency_differences,
            "stored_frequency_sums": frequencies}


def compare_shots(season, per_game, totals, stored, scales, teams, box_totals):
    per, dup_per = indexed(per_game, ("TEAM_ID",))
    total, dup_total = indexed(totals, ("TEAM_ID",))
    actual, duplicates = indexed(stored, ("team_id", "shot_zone"))
    if dup_per or dup_total or {k[0] for k in per} != set(teams) or per.keys() != total.keys():
        raise ValueError("NBA shot-zone team coverage is incomplete or duplicated")
    required = {(team, zone) for team in teams for zone in ZONES}
    mismatches, counts, frequencies, rate_differences, total_differences = [], Counter(), [], [], []
    stored_sums = defaultdict(Decimal)
    for (team,), row in per.items():
        t = total[(team,)]
        per_denominator = sum(dec(row[f"{z}_OPP_FGA"]) for z in ZONES)
        denominator = sum(dec(t[f"{z}_OPP_FGA"]) for z in ZONES)
        if denominator <= 0 or per_denominator <= 0:
            raise ValueError("No opponent attempts for a source team")
        for stat in ("FGM", "FGA"):
            field = f"opponent_{stat.lower()}"
            total_count = sum(dec(t[f"{z}_OPP_{stat}"]) for z in ZONES + ("Backcourt",))
            if total_count != dec(box_totals[team][field]):
                total_differences.append({"team_id": team, "field": field, "zone_sum": total_count,
                                          "regular_season_team_gamelogs": box_totals[team][field]})
        for zone in ZONES:
            k = team, zone
            fga, fgm = dec(t[f"{zone}_OPP_FGA"]), dec(t[f"{zone}_OPP_FGM"])
            if not 0 <= fgm <= fga or fga <= 0:
                raise ValueError("Invalid source zone attempts or makes")
            exact_frequency = fga / denominator
            for stat in ("FGM", "FGA"):
                exact_rate = dec(t[f"{zone}_OPP_{stat}"]) / box_totals[team]["gp"]
                if abs(dec(row[f"{zone}_OPP_{stat}"]) - exact_rate) > Decimal(".051"):
                    rate_differences.append({"team_id": team, "shot_zone": zone, "field": stat,
                                             "per_game": row[f"{zone}_OPP_{stat}"], "total_divided_by_gp": exact_rate})
            if k not in actual:
                continue
            old = actual[k]
            expected = {"fgm_allowed": row[f"{zone}_OPP_FGM"], "fga_allowed": row[f"{zone}_OPP_FGA"],
                        "fg_pct_allowed": row[f"{zone}_OPP_FG_PCT"],
                        "frequency_pct": dec(row[f"{zone}_OPP_FGA"]) / per_denominator}
            changes = compare_fields(old, expected, scales)
            if changes:
                counts.update(changes)
                mismatches.append({"team_id": team, "shot_zone": zone, "changes": changes})
            if old["frequency_pct"] is not None:
                stored_sums[team] += dec(old["frequency_pct"])
                if dec(old["frequency_pct"]) != rounded(exact_frequency, scales["frequency_pct"]):
                    frequencies.append({"team_id": team, "shot_zone": zone,
                                        "stored": old["frequency_pct"], "totals_frequency": exact_frequency,
                                        "difference_percentage_points": abs(dec(old["frequency_pct"]) - exact_frequency) * 100})
    return {"season": season, "database_rows": len(stored), "expected_rows": len(required),
            "missing_keys": [list(k) for k in sorted(required - actual.keys())],
            "unexpected_keys": [list(k) for k in sorted(actual.keys() - required)],
            "duplicate_keys": duplicates, "source_replay_mismatch_counts": dict(counts),
            "source_replay_mismatches": mismatches, "source_per_game_totals_inconsistencies": rate_differences,
            "season_zone_totals_vs_regular_team_games": total_differences,
            "frequency_differences_from_exact_totals": frequencies,
            "frequency_differences_over_half_percentage_point": sum(r["difference_percentage_points"] > Decimal(".5") for r in frequencies),
            "stored_frequency_sums": stored_sums}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists() or not args.json.parent.is_dir():
        parser.error("Choose a new report filename in an existing directory")
    args.source_dir.mkdir(exist_ok=False)
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT nba_api_id,abbreviation FROM public.teams")
            teams = {r["nba_api_id"]: r["abbreviation"] for r in cursor.fetchall()}
            cursor.execute("SELECT * FROM public.team_defensive_playtype_stats ORDER BY season,team_id,play_type")
            stored_plays = serial([dict(r) for r in cursor.fetchall()])
            cursor.execute("SELECT * FROM public.team_defensive_shot_locations ORDER BY season,team_id,shot_zone")
            stored_shots = serial([dict(r) for r in cursor.fetchall()])
            cursor.execute("SELECT table_name,column_name,numeric_scale FROM information_schema.columns WHERE table_schema='public' AND table_name IN ('team_defensive_playtype_stats','team_defensive_shot_locations') AND data_type='numeric'")
            scales = defaultdict(dict)
            for r in cursor.fetchall():
                scales[r["table_name"]][r["column_name"]] = r["numeric_scale"]
            # The verified player schedule excludes the Cup championship extras.
            cursor.execute("SELECT d.season,d.team,count(*) AS gp,sum(d.opponent_fgm) AS opponent_fgm,sum(d.opponent_fga) AS opponent_fga FROM public.team_defensive_stats d WHERE EXISTS (SELECT 1 FROM public.player_gamelog p WHERE p.season=d.season AND p.date=d.date AND btrim(p.team)=d.team) GROUP BY d.season,d.team")
            game_totals = serial([dict(r) for r in cursor.fetchall()])
        conn.rollback()
    report = {"audited_at": datetime.now(timezone.utc).isoformat(), "database_writes": False,
              "policies": "Stored count metrics are per game. Compare stored values with original transformation and separately with exact season totals. Shots use the same six zones as player profiles, excluding Backcourt and the duplicate combined Corner 3 category. Tracking/play-type source GP can differ from all regular-season appearances. No source nulls are silently replaced with zero.",
              "team_defensive_playtype_stats": [], "team_defensive_shot_locations": []}
    inverse = {name: team for team, name in teams.items()}
    for season in SEASONS:
        box = {inverse[r["team"]]: r for r in game_totals if r["season"] == season}
        if set(box) != set(teams):
            raise ValueError("Incomplete verified team schedule")
        plays = {mode: [] for mode in ("PerGame", "Totals")}
        try:
            for play in PLAY_TYPES:
                print(f"{season}/{play}: defensive team PerGame and Totals", flush=True)
                for mode in plays:
                    rows = capture(lambda: synergyplaytypes.SynergyPlayTypes(
                        league_id="00", per_mode_simple=mode, player_or_team_abbreviation="T",
                        season_type_all_star="Regular Season", season=season,
                        play_type_nullable=play, type_grouping_nullable="defensive", timeout=35),
                        f"{season}_defense_{play}_{mode}", args.source_dir)
                    plays[mode].extend(rows)
                    time.sleep(.3)
            result = compare_plays(season, plays["PerGame"], plays["Totals"], [r for r in stored_plays if r["season"] == season], scales["team_defensive_playtype_stats"], teams, box)
            report["team_defensive_playtype_stats"].append(result)
            print("Play-type results:", json.dumps({k: len(v) if isinstance(v, list) else v for k, v in result.items() if k != "stored_frequency_sums"}, default=str), flush=True)
        except Exception as exc:
            report["team_defensive_playtype_stats"].append({"season": season, "incomplete": True, "error_type": type(exc).__name__})
            print(f"Incomplete play-type audit: {type(exc).__name__}", flush=True)
        try:
            shots = {}
            for mode in ("PerGame", "Totals"):
                print(f"{season}: defensive shot locations {mode}", flush=True)
                shots[mode] = capture(lambda: leaguedashteamshotlocations.LeagueDashTeamShotLocations(
                    season=season, season_type_all_star="Regular Season", league_id_nullable="00",
                    per_mode_detailed=mode, measure_type_simple="Opponent", pace_adjust="N",
                    distance_range="By Zone", team_id_nullable=0, timeout=35),
                    f"{season}_opponent_shots_{mode}", args.source_dir, flatten=True)
            result = compare_shots(season, shots["PerGame"], shots["Totals"], [r for r in stored_shots if r["season"] == season], scales["team_defensive_shot_locations"], teams, box)
            report["team_defensive_shot_locations"].append(result)
            print("Shot results:", json.dumps({k: len(v) if isinstance(v, list) else v for k, v in result.items() if k != "stored_frequency_sums"}, default=str), flush=True)
        except Exception as exc:
            report["team_defensive_shot_locations"].append({"season": season, "incomplete": True, "error_type": type(exc).__name__})
            print(f"Incomplete shot audit: {type(exc).__name__}", flush=True)
    with args.json.open("x") as handle:
        json.dump(report, handle, indent=2, default=str)
        handle.write("\n")
    print(f"Saved audit: {args.json}. No database writes.")


if __name__ == "__main__":
    main()
