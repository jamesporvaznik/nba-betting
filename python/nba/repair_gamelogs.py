"""Preview missing game-log repairs and optionally export transactional SQL.

This command only reads Supabase. Review and run the exported SQL separately.
Existing ingestion scripts are not imported or modified.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import pandas as pd
from dotenv import dotenv_values
from nba_api.stats.endpoints import commonplayerinfo
from supabase import create_client

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba.audit_gamelogs import (
    AuditError, appearance_key, compare_appearances, fetch_nba_rows,
    read_all, season_argument,
)


GAME_COLUMNS = (
    "playerid", "gameid", "season", "date", "team", "opponent", "homeaway", "result",
    "minutes", "points", "rebounds", "assists", "steals", "blocks", "turnovers",
    "personalfouls", "fgm", "fga", "fg_pct", "3PM", "3PA", "3P_PCT", "ftm",
    "fta", "ft_pct", "oreb", "dreb", "plusminus", "gamescore",
)
PLAYER_COLUMNS = (
    "nba_api_id", "first_name", "last_name", "display_name", "birth_date",
    "height_inches", "weight_lbs", "draft_year", "position", "is_active",
)
STINT_COLUMNS = (
    "player_id", "team_id", "season", "stint_number", "first_game_date", "last_game_date",
)


def number(value, integer=False, required=False):
    if value is None or pd.isna(value) or not str(value).strip():
        if required:
            raise AuditError("NBA returned a missing required box-score value.")
        return None
    result = float(value)
    if not math.isfinite(result) or (integer and not result.is_integer()):
        raise AuditError("NBA returned an invalid numeric value.")
    return int(result) if integer else result


def game_record(row, season):
    matchup = str(row["MATCHUP"]).split()
    if len(matchup) != 3 or matchup[1] not in ("@", "vs."):
        raise AuditError("NBA returned an unrecognized matchup.")
    pid, date, team, opponent = appearance_key(
        row["PLAYER_ID"], row["GAME_DATE"], row["TEAM_ABBREVIATION"], matchup[-1],
    )
    record = {
        "playerid": pid, "gameid": f"{date}_{team}_{opponent}", "season": season,
        "date": date, "team": team, "opponent": opponent,
        "homeaway": "A" if matchup[1] == "@" else "H", "result": row["WL"],
    }
    minutes = row["MIN"]
    if isinstance(minutes, str) and ":" in minutes:
        minute_part, seconds = minutes.split(":")
        minutes = float(minute_part) + float(seconds) / 60
    record["minutes"] = number(minutes, required=True)
    for field, source in (
        ("points", "PTS"), ("rebounds", "REB"), ("assists", "AST"),
        ("steals", "STL"), ("blocks", "BLK"), ("turnovers", "TOV"),
        ("personalfouls", "PF"), ("fgm", "FGM"), ("fga", "FGA"),
        ("3PM", "FG3M"), ("3PA", "FG3A"), ("ftm", "FTM"), ("fta", "FTA"),
        ("oreb", "OREB"), ("dreb", "DREB"),
    ):
        record[field] = number(row[source], integer=True, required=True)
        if record[field] < 0:
            raise AuditError("NBA returned a negative box-score count.")
    for field, source, made, attempted in (
        ("fg_pct", "FG_PCT", "fgm", "fga"),
        ("3P_PCT", "FG3_PCT", "3PM", "3PA"),
        ("ft_pct", "FT_PCT", "ftm", "fta"),
    ):
        if record[made] > record[attempted]:
            raise AuditError("NBA shot makes exceed attempts.")
        percentage = number(row.get(source))
        if percentage is None and record[attempted]:
            percentage = record[made] / record[attempted]
        if percentage is not None and not 0 <= percentage <= 1:
            raise AuditError("NBA shooting percentage is outside 0–1.")
        record[field] = round(percentage, 4) if percentage is not None else None
    if record["minutes"] < 0 or record["result"] not in ("W", "L"):
        raise AuditError("NBA minutes or game result are invalid.")
    if record["rebounds"] != record["oreb"] + record["dreb"]:
        raise AuditError("NBA rebound totals do not agree.")
    record["plusminus"] = number(row.get("PLUS_MINUS"), integer=True)
    r = record
    record["gamescore"] = round(
        r["points"] + .4 * r["fgm"] - .7 * r["fga"] - .4 * (r["fta"] - r["ftm"])
        + .7 * r["oreb"] + .3 * r["dreb"] + r["steals"] + .7 * r["assists"]
        + .7 * r["blocks"] - .4 * r["personalfouls"] - r["turnovers"], 2,
    )
    return record


def player_record(info, player_id):
    if number(info["PERSON_ID"], integer=True, required=True) != player_id:
        raise AuditError("NBA player profile does not match the requested player.")
    height = None
    raw_height = str(info.get("HEIGHT") or "").strip()
    if raw_height and raw_height.lower() != "nan":
        feet, inches = raw_height.split("-")
        height = int(feet) * 12 + int(inches)
    birth = info.get("BIRTHDATE")
    birth = pd.Timestamp(birth).strftime("%Y-%m-%d") if birth and not pd.isna(birth) else None
    status = str(info.get("ROSTERSTATUS") or "").strip().lower()
    if not info.get("DISPLAY_FIRST_LAST"):
        raise AuditError("NBA profile is missing its display name.")
    return {
        "nba_api_id": player_id, "first_name": info.get("FIRST_NAME") or None,
        "last_name": info.get("LAST_NAME") or None,
        "display_name": info["DISPLAY_FIRST_LAST"], "birth_date": birth,
        "height_inches": height, "weight_lbs": number(info.get("WEIGHT"), integer=True),
        "draft_year": info.get("DRAFT_YEAR") or None, "position": info.get("POSITION") or None,
        "is_active": {"active": True, "inactive": False}.get(status),
    }


def fetch_profile(player_id, timeout):
    for attempt in range(3):
        try:
            frame = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=timeout).get_data_frames()[0]
            if len(frame) != 1:
                raise AuditError(f"Expected one NBA profile for player {player_id}.")
            return player_record(frame.iloc[0].to_dict(), player_id)
        except AuditError:
            raise
        except Exception as exc:
            if attempt == 2:
                raise AuditError(f"Profile fetch failed for {player_id} ({type(exc).__name__}).") from None
            time.sleep(2 ** (attempt + 1))


def database_key(row):
    return appearance_key(row["playerid"], row["date"], row["team"], row["opponent"])


def fingerprint(keys):
    text = "\n".join("|".join(map(str, key)) for key in sorted(keys))
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_stints(games, team_ids):
    stints = []
    last_group = None
    previous_team = None
    stint_number = 0
    for game in sorted(games, key=lambda g: (g["playerid"], g["season"], g["date"], g["gameid"], g["team"])):
        group = game["playerid"], game["season"]
        if group != last_group:
            previous_team = None
            stint_number = 0
        if game["team"] not in team_ids:
            raise AuditError(f"Team {game['team']} has no NBA ID in teams.")
        if previous_team != game["team"]:
            stint_number += 1
            stints.append({"player_id": group[0], "team_id": team_ids[game["team"]],
                           "season": group[1], "stint_number": stint_number,
                           "first_game_date": game["date"], "last_game_date": game["date"]})
        else:
            stints[-1]["last_game_date"] = game["date"]
        last_group, previous_team = group, game["team"]
    return stints


def sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def sql_records(records):
    # One public-data record per line; no credentials are included.
    payload = "[\n" + ",\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in records) + "\n]"
    return sql_literal(payload) + "::jsonb"


def quoted_columns(columns):
    return ", ".join('"' + c + '"' for c in columns)


def render_sql(season, players, games, stints, before_fingerprint, after_fingerprint, source_count):
    season_sql = sql_literal(season)
    fingerprint_query = f"""SELECT md5(COALESCE(string_agg(
        g.playerid::text || '|' || g.date::text || '|' || UPPER(TRIM(g.team)) || '|' || UPPER(TRIM(g.opponent)),
        E'\\n' ORDER BY g.playerid, g.date, UPPER(TRIM(g.team)) COLLATE "C", UPPER(TRIM(g.opponent)) COLLATE "C"
    ), '')) FROM public.player_gamelog AS g WHERE g.season = {season_sql}"""
    parts = [f"""-- Missing NBA regular-season appearances: {season}.
-- Generated {datetime.now(timezone.utc).isoformat()} from fresh NBA and Supabase reads.
-- Expected missing records: {len(players)} players, {len(games)} appearances.
-- Review before running. Execute this entire file in Supabase SQL Editor.
-- Existing player and box-score records are preserved. Only affected players' stints are reconciled.
-- Player profiles are current at fetch time, not historical roster assignments.
-- Source game IDs retain the existing date_team_opponent convention.
-- Stint team_id references teams.nba_api_id. First/last dates are observed appearances.
-- Preflight and final checks roll back the transaction if records differ unexpectedly.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL standard_conforming_strings = on;
LOCK TABLE public.players, public.player_gamelog, public.player_team_stints
    IN SHARE ROW EXCLUSIVE MODE;

DO $preflight$
DECLARE actual text;
BEGIN
    actual := ({fingerprint_query});
    IF actual NOT IN ({sql_literal(before_fingerprint)}, {sql_literal(after_fingerprint)}) THEN
        RAISE EXCEPTION 'Game logs changed since this repair was generated. Regenerate the repair; no changes applied.';
    END IF;
END;
$preflight$;
"""]
    for name, table, records in (("players", "players", players), ("games", "player_gamelog", games),
                                 ("stints", "player_team_stints", stints)):
        parts.append(f"CREATE TEMP TABLE _nba_repair_{name} ON COMMIT DROP AS\n"
                     f"SELECT * FROM jsonb_populate_recordset(NULL::public.{table},\n{sql_records(records)});\n")
    pc, gc, sc = map(quoted_columns, (PLAYER_COLUMNS, GAME_COLUMNS, STINT_COLUMNS))
    parts.extend([f"""INSERT INTO public.players ({pc})
SELECT {pc} FROM _nba_repair_players
ON CONFLICT (nba_api_id) DO NOTHING;

INSERT INTO public.player_gamelog ({gc})
SELECT {', '.join('r.' + '"' + c + '"' for c in GAME_COLUMNS)}
FROM _nba_repair_games AS r
WHERE NOT EXISTS (
    SELECT 1 FROM public.player_gamelog AS existing
    WHERE existing.playerid = r.playerid AND existing.season = r.season
      AND existing.date = r.date
      AND UPPER(TRIM(existing.team)) = UPPER(TRIM(r.team))
      AND UPPER(TRIM(existing.opponent)) = UPPER(TRIM(r.opponent))
)
ON CONFLICT DO NOTHING;

INSERT INTO public.player_team_stints ({sc})
SELECT {sc} FROM _nba_repair_stints
ON CONFLICT (player_id, season, stint_number) DO UPDATE SET
    team_id = EXCLUDED.team_id,
    first_game_date = EXCLUDED.first_game_date,
    last_game_date = EXCLUDED.last_game_date;

-- Remove only obsolete stint numbers for affected player-seasons.
-- Missing historical appearances can split or merge previously inferred stints.
DELETE FROM public.player_team_stints AS existing
WHERE existing.season = {season_sql}
  AND existing.player_id IN (SELECT player_id FROM _nba_repair_stints)
  AND NOT EXISTS (
      SELECT 1 FROM _nba_repair_stints AS desired
      WHERE desired.player_id = existing.player_id
        AND desired.season = existing.season
        AND desired.stint_number = existing.stint_number
  );

DO $validate$
BEGIN
    IF ({fingerprint_query}) IS DISTINCT FROM {sql_literal(after_fingerprint)} THEN
        RAISE EXCEPTION 'Final appearance coverage differs from NBA reference; repair rolled back.';
    END IF;
    IF EXISTS (
        SELECT 1 FROM _nba_repair_stints AS s
        LEFT JOIN public.players AS p ON p.nba_api_id = s.player_id
        WHERE p.nba_api_id IS NULL
    ) THEN
        RAISE EXCEPTION 'A repaired player is absent from players; repair rolled back.';
    END IF;
    IF EXISTS (
        (SELECT {sc} FROM public.player_team_stints
         WHERE season = {season_sql} AND player_id IN (SELECT player_id FROM _nba_repair_stints))
        EXCEPT SELECT {sc} FROM _nba_repair_stints
    ) OR EXISTS (
        (SELECT {sc} FROM _nba_repair_stints)
        EXCEPT SELECT {sc} FROM public.player_team_stints
    ) THEN
        RAISE EXCEPTION 'Repaired stints do not match expected appearances; repair rolled back.';
    END IF;
END;
$validate$;

COMMIT;

SELECT season, COUNT(*) AS stored_appearances, {source_count} AS nba_reference_appearances
FROM public.player_gamelog WHERE season = {season_sql} GROUP BY season;
"""])
    return "\n".join(parts)


def check_audit(path, season, fresh_report):
    if not path:
        return
    payload = json.loads(path.read_text())
    previous = next((r for r in payload.get("reports", []) if r["season"] == season), None)
    if previous is None:
        raise AuditError(f"Audit report has no successful comparison for {season}.")
    def missing_keys(report):
        return {(r["player_id"], r["date"], r["team"], r["opponent"]) for r in report["missing_appearances"]}
    if (previous["summary"] != fresh_report["summary"] or missing_keys(previous) != missing_keys(fresh_report)
            or previous["missing_player_records"] != fresh_report["missing_player_records"]):
        raise AuditError("The supplied audit is stale. Run a fresh audit and review its findings first.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, type=season_argument)
    parser.add_argument("--audit", type=Path, help="Require fresh differences to match this audit JSON.")
    parser.add_argument("--sql", type=Path, help="Export SQL to a new file; does not execute it.")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("Timeout must be positive.")
    if args.sql and (args.sql.exists() or not args.sql.parent.is_dir()):
        parser.error("Choose a new SQL filename inside an existing directory.")
    if args.audit and not args.audit.is_file():
        parser.error("Audit JSON file does not exist.")
    print("Read-only repair preview: no database writes or ingestion-script changes.", flush=True)
    try:
        config = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
        url = os.environ.get("SUPABASE_URL") or config.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY") or config.get("SUPABASE_KEY")
        if not url or not key:
            raise AuditError("Set SUPABASE_URL and SUPABASE_KEY in python/.env or the environment.")
        client = create_client(url, key)
        season = args.season
        players = read_all(lambda: client.table("players")
                           .select("id,nba_api_id,display_name", count="exact").order("id"), "id")
        stored = read_all(lambda: client.table("player_gamelog")
                          .select("gamelogid,playerid,gameid,date,team,opponent", count="exact")
                          .eq("season", season).order("gamelogid"), "gamelogid")
        nba_rows = fetch_nba_rows(season, args.timeout)
        report = compare_appearances(season, nba_rows, stored, players)
        check_audit(args.audit, season, report)
        s = report["summary"]
        if s["unexpected_appearances"] or s["extra_duplicate_database_rows"] or s["duplicate_nba_groups"]:
            raise AuditError("Unexpected or duplicate appearances need review before generating a missing-row repair.")
        if report["clean"]:
            print(f"{season}: nothing to repair. No SQL generated.")
            return 0
        print(f"{season}: {s['missing_appearances']} missing appearances; "
              f"{s['missing_player_records']} missing player records.", flush=True)
        master_ids = {r["player_id"] for r in report["missing_player_records"]}
        target_ids = master_ids | {r["player_id"] for r in report["missing_appearances"]}
        missing = {(r["player_id"], r["date"], r["team"], r["opponent"]) for r in report["missing_appearances"]}
        teams = read_all(lambda: client.table("teams")
                         .select("id,nba_api_id,abbreviation", count="exact").order("id"), "id")
        team_ids = {str(t["abbreviation"]).strip().upper(): int(t["nba_api_id"])
                    for t in teams if t["abbreviation"] and t["nba_api_id"] is not None}
        games, projected, source_keys = [], [], []
        for row in nba_rows:
            matchup = str(row["MATCHUP"]).split()
            key = appearance_key(row["PLAYER_ID"], row["GAME_DATE"], row["TEAM_ABBREVIATION"], matchup[-1])
            source_keys.append(key)
            if key[0] not in target_ids:
                continue
            game = game_record(row, season)
            if team_ids.get(game["team"]) != int(row["TEAM_ID"]):
                raise AuditError(f"Team mapping disagrees with NBA team ID for {game['team']}.")
            projected.append(game)
            if key in missing:
                games.append(game)
        profiles = []
        for pid in sorted(master_ids):
            print(f"Fetching missing player profile {pid}...", flush=True)
            profiles.append(fetch_profile(pid, args.timeout))
            time.sleep(.7)
        desired = build_stints(projected, team_ids)
        existing_stints = read_all(lambda: client.table("player_team_stints")
                                   .select("id," + ",".join(STINT_COLUMNS), count="exact")
                                   .eq("season", season).order("id"), "id")
        existing = {(r["player_id"], r["stint_number"]): r for r in existing_stints if r["player_id"] in target_ids}
        if len(existing) != sum(r["player_id"] in target_ids for r in existing_stints):
            raise AuditError("Duplicate stint keys need review first.")
        expected = {(r["player_id"], r["stint_number"]): r for r in desired}
        added = len(expected.keys() - existing.keys())
        removed = len(existing.keys() - expected.keys())
        updated = sum(any(existing[k][c] != expected[k][c] for c in STINT_COLUMNS)
                      for k in expected.keys() & existing.keys())
        print(f"Affected players: {len(target_ids)}")
        for p in profiles:
            print(f"  Add player: {p['display_name']} ({p['nba_api_id']})")
        print(f"Stints: {added} new, {updated} changed, {removed} obsolete; "
              "other players/seasons untouched.")
        # Recheck appearance coverage after the slower profile requests.
        refreshed = read_all(lambda: client.table("player_gamelog")
                             .select("gamelogid,playerid,gameid,date,team,opponent", count="exact")
                             .eq("season", season).order("gamelogid"), "gamelogid")
        if fingerprint(map(database_key, refreshed)) != fingerprint(map(database_key, stored)):
            raise AuditError("Game logs changed during preparation. Rerun the preview.")
        sql = render_sql(season, profiles, sorted(games, key=database_key), desired,
                         fingerprint(map(database_key, stored)), fingerprint(source_keys), len(source_keys))
        if args.sql:
            with args.sql.open("x") as handle:
                handle.write(sql)
            print(f"SQL saved to {args.sql}. Review and run the entire file in Supabase SQL Editor.")
        else:
            print("Preview only. Add --sql /path/to/new-repair.sql to export the transaction.")
        print("No database records were changed.")
        return 0
    except Exception as exc:
        message = str(exc) if isinstance(exc, AuditError) else f"Repair preparation failed ({type(exc).__name__})."
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
