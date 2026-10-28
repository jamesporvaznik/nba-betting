"""Read-only NBA/Supabase appearance audit. Run from python/ with -m nba.audit_gamelogs."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time

import pandas as pd
from dotenv import dotenv_values
from nba_api.stats.endpoints import playergamelogs
from supabase import create_client


class AuditError(Exception):
    """The audit could not establish a reliable comparison."""


def season_argument(value):
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise argparse.ArgumentTypeError("Use a season such as 2024-25.")
    if (int(value[:4]) + 1) % 100 != int(value[-2:]):
        raise argparse.ArgumentTypeError("The season's years must be consecutive.")
    return value


def read_all(query_factory, identity_column):
    """Read every page, including when the API caps pages below 1,000 rows."""
    rows = []
    expected = None
    seen = set()
    while True:
        response = query_factory().range(len(rows), len(rows) + 999).execute()
        if response.count is None:
            raise AuditError("Database did not return an exact row count.")
        if expected is None:
            expected = response.count
        if response.count != expected:
            raise AuditError("Database row count changed during pagination; rerun the audit.")
        page = response.data or []
        if not page:
            if len(rows) != expected:
                raise AuditError("Database pagination ended before all rows were retrieved.")
            return rows
        for row in page:
            identity = row[identity_column]
            if identity in seen:
                raise AuditError("Database pagination repeated a row; rerun the audit.")
            seen.add(identity)
        rows.extend(page)
        if len(rows) > expected:
            raise AuditError("Database returned more rows than its exact count.")
        if len(rows) == expected:
            return rows


def appearance_key(player_id, date, team, opponent):
    if any(pd.isna(value) for value in (player_id, date, team, opponent)):
        raise AuditError("An appearance contains a missing player, date, team, or opponent.")
    team, opponent = str(team).strip().upper(), str(opponent).strip().upper()
    if not team or not opponent or not str(date).strip():
        raise AuditError("An appearance contains an empty date, team, or opponent.")
    parsed_date = pd.Timestamp(date)
    if pd.isna(parsed_date):
        raise AuditError("An appearance contains an invalid date.")
    return int(player_id), parsed_date.strftime("%Y-%m-%d"), team, opponent


def describe_key(key):
    return dict(zip(("player_id", "date", "team", "opponent"), key))


def compare_appearances(season, nba_rows, database_rows, players):
    if not nba_rows:
        raise AuditError(f"NBA returned no appearances for {season}; not treating this as a pass.")

    source = Counter()
    source_details = {}
    names = {}
    for row in nba_rows:
        matchup = str(row["MATCHUP"]).split()
        if len(matchup) != 3 or matchup[1] not in ("@", "vs."):
            raise AuditError("NBA returned an unrecognized matchup format.")
        key = appearance_key(row["PLAYER_ID"], row["GAME_DATE"],
                             row["TEAM_ABBREVIATION"], matchup[-1])
        if matchup[0].upper() != key[2]:
            raise AuditError("NBA matchup team disagrees with its team abbreviation.")
        source[key] += 1
        names[key[0]] = str(row["PLAYER_NAME"])
        source_details[key] = {"player_name": names[key[0]],
                               "nba_game_id": str(row["GAME_ID"])}

    stored = Counter()
    database_details = {}
    for row in database_rows:
        key = appearance_key(row["playerid"], row["date"], row["team"], row["opponent"])
        stored[key] += 1
        database_details.setdefault(key, []).append({
            "gamelogid": row["gamelogid"], "gameid": row["gameid"],
        })

    master_names = {int(p["nba_api_id"]): p["display_name"]
                    for p in players if p["nba_api_id"] is not None}
    missing_keys = sorted(source.keys() - stored.keys())
    unexpected_keys = sorted(stored.keys() - source.keys())
    missing_by_player = Counter(key[0] for key in missing_keys)
    source_player_ids = {key[0] for key in source}
    missing_master_ids = sorted(source_player_ids - master_names.keys())

    missing = [dict(describe_key(key), **source_details[key]) for key in missing_keys]
    unexpected = [dict(describe_key(key), player_name=master_names.get(key[0]),
                       database_records=database_details[key]) for key in unexpected_keys]
    duplicates = [dict(describe_key(key), row_count=count,
                       database_records=database_details[key])
                  for key, count in sorted(stored.items()) if count > 1]
    source_duplicates = [dict(describe_key(key), row_count=count)
                         for key, count in sorted(source.items()) if count > 1]
    summary = {
        "nba_rows": len(nba_rows),
        "nba_unique_appearances": len(source),
        "database_rows": len(database_rows),
        "database_unique_appearances": len(stored),
        "missing_appearances": len(missing),
        "unexpected_appearances": len(unexpected),
        "duplicate_database_groups": len(duplicates),
        "extra_duplicate_database_rows": sum(n - 1 for n in stored.values()),
        "duplicate_nba_groups": len(source_duplicates),
        "missing_player_records": len(missing_master_ids),
    }
    return {
        "season": season,
        "clean": not (missing or unexpected or duplicates or source_duplicates or missing_master_ids),
        "summary": summary,
        "missing_by_player": [
            {"player_id": pid, "player_name": names[pid], "missing_games": count,
             "exists_in_players": pid in master_names}
            for pid, count in sorted(missing_by_player.items())
        ],
        "missing_player_records": [
            {"player_id": pid, "player_name": names[pid]} for pid in missing_master_ids
        ],
        "missing_appearances": missing,
        "unexpected_appearances": unexpected,
        "duplicate_database_appearances": duplicates,
        "duplicate_nba_appearances": source_duplicates,
    }


def fetch_nba_rows(season, timeout):
    for attempt in range(3):
        try:
            frame = playergamelogs.PlayerGameLogs(
                season_nullable=season, season_type_nullable="Regular Season",
                timeout=timeout,
            ).get_data_frames()[0]
            if frame.empty:
                raise AuditError(f"NBA returned no appearances for {season}.")
            return frame.to_dict("records")
        except AuditError:
            raise
        except Exception as exc:
            if attempt == 2:
                raise AuditError(f"NBA request failed after 3 attempts ({type(exc).__name__}).") from None
            print(f"NBA request failed; retrying ({attempt + 1}/3)...", file=sys.stderr, flush=True)
            time.sleep(2 ** (attempt + 1))


def print_report(report, sample_size):
    s = report["summary"]
    print(f"\n{report['season']}: {'PASS' if report['clean'] else 'DIFFERENCES FOUND'}")
    print(f"  NBA: {s['nba_rows']:,} rows | Supabase: {s['database_rows']:,} rows")
    print(f"  Missing: {s['missing_appearances']:,} | Unexpected: {s['unexpected_appearances']:,}")
    print(f"  Extra duplicate database rows: {s['extra_duplicate_database_rows']:,}")
    print(f"  Duplicate NBA groups: {s['duplicate_nba_groups']:,}")
    print(f"  NBA players absent from players table: {s['missing_player_records']:,}")
    for player in report["missing_by_player"]:
        master = "exists" if player["exists_in_players"] else "MISSING"
        print(f"    {player['player_name']} ({player['player_id']}): "
              f"{player['missing_games']} missing games; player record {master}")
    for label, key in (("Missing player records", "missing_player_records"),
                       ("Missing appearance sample", "missing_appearances"),
                       ("Unexpected appearance sample", "unexpected_appearances"),
                       ("Database duplicate sample", "duplicate_database_appearances"),
                       ("NBA duplicate sample", "duplicate_nba_appearances")):
        if report[key] and sample_size:
            print(f"  {label} (up to {sample_size}):")
            for entry in report[key][:sample_size]:
                print("    " + json.dumps(entry, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", nargs="+", required=True, type=season_argument,
                        help="One or more seasons, e.g. --season 2022-23 2023-24")
    parser.add_argument("--json", type=Path, dest="json_path",
                        help="Save every discrepancy to a new local JSON file (never overwrites).")
    parser.add_argument("--sample-size", type=int, default=10,
                        help="Maximum detailed entries printed per category (default: 10).")
    parser.add_argument("--timeout", type=int, default=30, help="NBA request timeout in seconds.")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.sample_size < 0:
        parser.error("Timeout must be positive and sample size cannot be negative.")
    if args.json_path and (args.json_path.exists() or not args.json_path.parent.is_dir()):
        parser.error("Choose a new JSON filename inside an existing directory.")

    config = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    url = os.environ.get("SUPABASE_URL") or config.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or config.get("SUPABASE_KEY")
    if not url or not key:
        parser.error("Set SUPABASE_URL and SUPABASE_KEY in python/.env or the environment.")

    print("Read-only audit: no database records or ingestion scripts will be changed.", flush=True)
    reports = []
    errors = []
    try:
        client = create_client(url, key)
        players = read_all(lambda: client.table("players")
                           .select("id,nba_api_id,display_name", count="exact").order("id"), "id")
    except Exception as exc:
        message = str(exc) if isinstance(exc, AuditError) else f"Database read failed ({type(exc).__name__})."
        print(message, file=sys.stderr)
        return 2

    for season in dict.fromkeys(args.season):
        print(f"\nAuditing {season}: fetching NBA and Supabase appearances...", flush=True)
        try:
            nba_rows = fetch_nba_rows(season, args.timeout)
            database_rows = read_all(lambda: client.table("player_gamelog")
                                     .select("gamelogid,playerid,gameid,date,team,opponent", count="exact")
                                     .eq("season", season).order("gamelogid"), "gamelogid")
            report = compare_appearances(season, nba_rows, database_rows, players)
            reports.append(report)
            print_report(report, args.sample_size)
        except Exception as exc:
            message = str(exc) if isinstance(exc, AuditError) else f"Audit failed ({type(exc).__name__})."
            errors.append({"season": season, "error": message})
            print(f"{season}: ERROR: {message}", file=sys.stderr, flush=True)

    if args.json_path:
        payload = {"generated_at": datetime.now(timezone.utc).isoformat(),
                   "scope": "Regular-season appearance coverage and player master presence; not box-score accuracy.",
                   "reports": reports, "errors": errors}
        try:
            with args.json_path.open("x") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
            print(f"\nFull report saved to {args.json_path}")
        except OSError as exc:
            print(f"Could not save report ({type(exc).__name__}).", file=sys.stderr)
            return 2
    return 2 if errors else (1 if any(not r["clean"] for r in reports) else 0)


if __name__ == "__main__":
    sys.exit(main())
