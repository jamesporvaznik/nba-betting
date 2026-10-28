"""Capture public NBA tracking responses and cross-check season/game endpoints.

Only reads database records. Saves raw public source responses, request parameters,
and checksums locally; never saves connection credentials.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import time

from nba_api.stats.endpoints import leaguedashptstats, playergamelogs, boxscoreplayertrackv3
from psycopg2.extras import RealDictCursor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba.audit_tracking import load_measures
from nba.repair_shot_locations import connection, serial


SAMPLE_IDS = (1630639, 1629029, 1628964, 203471)


def capture(endpoint, name, directory):
    raw = endpoint.nba_response.get_response()
    (directory / f"{name}.json").write_text(raw)
    frame = endpoint.get_data_frames()[0]
    rows = json.loads(frame.to_json(orient="records"))
    metadata = {"captured_at": datetime.now(timezone.utc).isoformat(),
                "request_url": endpoint.nba_response.get_url(), "parameters": endpoint.parameters,
                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "records_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                "rows": len(rows), "raw_file": f"{name}.json"}
    (directory / f"{name}_request.json").write_text(json.dumps(metadata, indent=2))
    return rows, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM public.player_tracking_stats WHERE player_id = ANY(%s) ORDER BY player_id,season", (list(SAMPLE_IDS),))
            stored = serial([dict(r) for r in cursor.fetchall()])
        conn.rollback()
    summary = {"started_at": datetime.now(timezone.utc).isoformat(), "database_writes": False,
               "stored_sample_rows": stored, "season_captures": [], "game_checks": []}
    sources = {}
    for season in ("2024-25", "2025-26"):
        for measure in load_measures():
            for mode in ("PerGame", "Totals"):
                print(f"Capturing {season}/{measure}/{mode}", flush=True)
                endpoint = leaguedashptstats.LeagueDashPtStats(
                    season=season, season_type_all_star="Regular Season", per_mode_simple=mode,
                    player_or_team="Player", pt_measure_type=measure, league_id_nullable="00",
                    last_n_games=0, month=0, opponent_team_id=0, timeout=35)
                rows, meta = capture(endpoint, f"{season}_{measure}_{mode}", args.output_dir)
                meta["sample_players"] = [r for r in rows if r["PLAYER_ID"] in SAMPLE_IDS]
                summary["season_captures"].append(meta)
                sources[(season, measure, mode)] = {int(r["PLAYER_ID"]): r for r in rows}
                time.sleep(.3)
    for repeat in (1, 2):
        endpoint = leaguedashptstats.LeagueDashPtStats(
            season="2024-25", season_type_all_star="Regular Season", per_mode_simple="Totals",
            player_or_team="Player", pt_measure_type="Possessions", league_id_nullable="00",
            last_n_games=0, month=0, opponent_team_id=0, timeout=35)
        rows, meta = capture(endpoint, f"2024-25_Possessions_Totals_repeat{repeat}", args.output_dir)
        meta["sample_players"] = [r for r in rows if r["PLAYER_ID"] in SAMPLE_IDS]
        summary["season_captures"].append(meta)

    for season, pid in (("2024-25", 1630639), ("2024-25", 1629029)):
        endpoint = playergamelogs.PlayerGameLogs(player_id_nullable=pid, season_nullable=season,
                    season_type_nullable="Regular Season", timeout=35)
        games, meta = capture(endpoint, f"{season}_{pid}_gamelogs", args.output_dir)
        game_ids = [str(r["GAME_ID"]) for r in games]
        if not games or len(game_ids) != len(set(game_ids)):
            raise ValueError("Empty or duplicate player game IDs")
        counts, unavailable, game_rows = Counter(), [], []
        for index, gid in enumerate(game_ids):
            print(f"Checking game tracking {season}/{pid}: {index+1}/{len(game_ids)}", flush=True)
            endpoint = boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=gid, timeout=35)
            records, _ = capture(endpoint, f"game_{gid}", args.output_dir)
            matches = [r for r in records if r["personId"] == pid]
            if len(matches) != 1 or not matches[0].get("minutes"):
                unavailable.append(gid)
                continue
            row = matches[0]
            game_rows.append(row)
            for field in ("touches", "passes", "assists", "reboundChancesOffensive", "reboundChancesDefensive", "reboundChancesTotal"):
                if row.get(field) is None:
                    raise ValueError("Null game-level tracking counts")
                counts[field] += row[field]
            time.sleep(.2)
        old = next(r for r in stored if r["player_id"] == pid and r["season"] == season)
        comparison = {"season": season, "player_id": pid, "games": len(games), "games_checked": len(game_rows),
                      "unavailable_game_ids": unavailable, "game_sum": dict(counts),
                      "teams": sorted({r["teamTricode"] for r in game_rows}), "comparisons": []}
        for measure, api_field, game_field, db_field in (
                ("Possessions", "TOUCHES", "touches", "touches"),
                ("Passing", "PASSES_MADE", "passes", "passes_made")):
            total = sources[(season, measure, "Totals")][pid]
            pergame = sources[(season, measure, "PerGame")][pid]
            comparison["comparisons"].append({"field": db_field, "game_sum": counts[game_field],
                         "league_totals": total[api_field], "league_per_game": pergame[api_field],
                         "league_gp": total["GP"], "stored_per_game": old[db_field],
                         "game_based_rate": str(Decimal(counts[game_field]) / len(game_rows)) if game_rows else None})
        summary["game_checks"].append(comparison)
        print(json.dumps(comparison), flush=True)
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Read-only investigation saved: {args.output_dir}")


if __name__ == "__main__":
    main()
