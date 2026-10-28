"""Preview season-long player play-type aggregation and export SQL for review.

Reads NBA/Supabase data only. No SQL is executed and no ingestion scripts are changed.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
import os
from pathlib import Path
import re
import sys
import time

from dotenv import dotenv_values
from nba_api.stats.endpoints import synergyplaytypes
from supabase import create_client

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba.audit_gamelogs import AuditError, season_argument


PLAY_TYPES = (
    "Transition", "Isolation", "Postup", "PRBallHandler", "PRRollMan", "Spotup",
    "Handoff", "Cut", "OffScreen", "OffRebound", "Misc",
)
KEY_COLUMNS = ("player_id", "season", "play_type")
VALUE_COLUMNS = (
    "possessions", "points", "poss_pct", "percentile", "games_played",
    "points_per_possession", "fg_pct", "efg_pct",
)
COLUMNS = KEY_COLUMNS + VALUE_COLUMNS


def decimal(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise AuditError("Source contains a non-finite number.")
    return result


def rounded(value, places):
    return float(decimal(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def select_team_rows(rows):
    """Prefer a source season total, otherwise combine distinct team records."""
    aggregates = []
    teams = {}
    for row in rows:
        team = int(row["TEAM_ID"])
        abbreviation = str(row.get("TEAM_ABBREVIATION", "")).strip().upper()
        if team == 0 or abbreviation in ("TOT", "TOTAL") or re.fullmatch(r"\d+TM", abbreviation):
            if row not in aggregates:
                aggregates.append(row)
        elif team not in teams:
            teams[team] = row
        elif teams[team] != row:
            raise AuditError("Conflicting source records for one player/team/play type.")
    if len(aggregates) > 1:
        raise AuditError("Multiple conflicting whole-season aggregate rows in NBA source.")
    return aggregates if aggregates else list(teams.values())


def aggregate_group(rows):
    selected = select_team_rows(rows)
    if not selected:
        raise AuditError("No usable NBA source rows.")
    for row in selected:
        for field in ("POSS", "PTS", "FGM", "FGA", "GP"):
            value = decimal(row[field])
            if value < 0 or value != value.to_integral_value():
                raise AuditError("Expected nonnegative integer totals; use NBA PerMode=Totals.")
        if decimal(row["FGM"]) > decimal(row["FGA"]):
            raise AuditError("Source field goals made exceed attempts.")
        if not 0 <= decimal(row["EFG_PCT"]) <= Decimal("1.5"):
            raise AuditError("Source eFG% is outside 0–1.5.")
    merged = len(selected) > 1
    possessions = sum(decimal(r["POSS"]) for r in selected)
    points = sum(decimal(r["PTS"]) for r in selected)
    attempts = sum(decimal(r["FGA"]) for r in selected)
    made = sum(decimal(r["FGM"]) for r in selected)
    if merged:
        # The endpoint publishes rounded eFG%, but not three-point makes.
        # Weight those published values by attempts; do not invent exact 3PM.
        weighted_efg = sum(decimal(r["EFG_PCT"]) * decimal(r["FGA"]) for r in selected)
        values = {
            "possessions": int(possessions), "points": int(points),
            "games_played": sum(int(r["GP"]) for r in selected),
            "points_per_possession": rounded(points / possessions, 2) if possessions else None,
            "fg_pct": rounded(made / attempts, 4) if attempts else None,
            "efg_pct": rounded(weighted_efg / attempts, 4) if attempts else None,
            # A team-specific percentile is not a combined-season percentile.
            "percentile": None,
        }
    else:
        r = selected[0]
        values = {
            "possessions": int(possessions), "points": int(points), "games_played": int(r["GP"]),
            "points_per_possession": rounded(r["PPP"], 2),
            "fg_pct": rounded(r["FG_PCT"], 4), "efg_pct": rounded(r["EFG_PCT"], 4),
            "percentile": float(r["PERCENTILE"]),
        }
    return values, merged, selected


def season_percentiles(grouped):
    """Calculate PPP midrank percentiles within each season/play-type cohort.

    Uses exact combined PTS/POSS, not the two-decimal stored PPP. Cohorts
    contain available NBA source players with at least 10 reported possessions.
    Rank ties receive their average position, scaled to 0–1. A singleton or
    all-tied cohort receives .5. This is calculated, not an official NBA rank.
    """
    cohorts = defaultdict(list)
    percentiles = {key: None for key in grouped}
    for key, rows in grouped.items():
        selected = select_team_rows(rows)
        possessions = sum(decimal(r["POSS"]) for r in selected)
        points = sum(decimal(r["PTS"]) for r in selected)
        if possessions >= 10:
            cohorts[key[1:]].append((points / possessions, key))
    for cohort in cohorts.values():
        ordered = sorted(cohort)
        size = len(ordered)
        start = 0
        while start < size:
            end = start + 1
            while end < size and ordered[end][0] == ordered[start][0]:
                end += 1
            average_rank = (Decimal(start + 1) + Decimal(end)) / 2
            rank = (average_rank - 1) / (size - 1) if size > 1 else Decimal(".5")
            for _, key in ordered[start:end]:
                percentiles[key] = rounded(rank, 4)
            start = end
    return percentiles


def plan_repair(source_by_season, stored_rows):
    stored = {tuple(r[k] for k in KEY_COLUMNS): r for r in stored_rows}
    if len(stored) != len(stored_rows):
        raise AuditError("Duplicate stored player/season/play-type keys.")
    grouped = defaultdict(list)
    names = {}
    for season, rows in source_by_season.items():
        available_types = set()
        for row in rows:
            if str(row["SEASON_ID"]) != "2" + season[:4]:
                raise AuditError("Source row belongs to a different season.")
            play = row["PLAY_TYPE"]
            if play not in PLAY_TYPES or str(row["TYPE_GROUPING"]).lower() != "offensive":
                raise AuditError("Unexpected source play type or grouping.")
            available_types.add(play)
            pid = int(row["PLAYER_ID"])
            grouped[(pid, season, play)].append(row)
            names[(pid, season)] = row["PLAYER_NAME"]
        if available_types != set(PLAY_TYPES):
            raise AuditError(f"Incomplete source play-type coverage for {season}.")
    if grouped.keys() != stored.keys():
        raise AuditError("Source and database keys differ. Review missing/unexpected records before aggregation.")

    computed, merged_keys, affected = {}, set(), set()
    for key, rows in grouped.items():
        values, merged, _ = aggregate_group(rows)
        computed[key] = values
        # Include a supplied whole-season aggregate if multiple team rows were present.
        if merged or len({int(r["TEAM_ID"]) for r in rows}) > 1:
            affected.add(key[:2])
        if merged:
            merged_keys.add(key)
    rankings = season_percentiles(grouped)
    totals = defaultdict(Decimal)
    for key, row in computed.items():
        totals[key[:2]] += decimal(row["possessions"])
    proposed, changes = {}, []
    for key in sorted(stored):
        old = {c: stored[key][c] for c in COLUMNS}
        new = dict(old)
        if key[:2] in affected:
            new.update(computed[key])
            new["poss_pct"] = rounded(decimal(new["possessions"]) / totals[key[:2]], 4) if totals[key[:2]] else None
        new["percentile"] = rankings[key]
        proposed[key] = new
        changed_fields = {c: {"before": old[c], "after": new[c]} for c in VALUE_COLUMNS if old[c] != new[c]}
        if changed_fields:
            changes.append({"player_id": key[0], "player_name": names[key[:2]],
                            "season": key[1], "play_type": key[2], "changes": changed_fields})
    # Every member of each selected season is part of the ranking snapshot,
    # even if its rounded percentile happens to remain unchanged.
    before = [{c: stored[key][c] for c in COLUMNS} for key in sorted(stored)]
    after = [proposed[key] for key in sorted(stored)]
    summaries = []
    for season in source_by_season:
        merged = [key for key in merged_keys if key[1] == season]
        summaries.append({"season": season,
                          "affected_player_seasons": sum(s == season for _, s in affected),
                          "merged_play_type_rows": len(merged),
                          "changed_rows": sum(r["season"] == season for r in changes),
                          "percentile_rows_changed": sum(r["season"] == season and "percentile" in r["changes"] for r in changes),
                          "additional_possessions": sum(computed[k]["possessions"] - stored[k]["possessions"] for k in merged),
                          "additional_points": sum(computed[k]["points"] - stored[k]["points"] for k in merged)})
    return {"summaries": summaries, "distinct_players": len({pid for pid, _ in affected}),
            "percentile_policy": "Calculated across all available player season totals within each play type, using exact PPP, at least 10 reported possessions, average tied ranks, and (rank-1)/(N-1). Singletons receive 0.5. Not an official NBA percentile.",
            "games_played_policy": "Sum source GP across distinct contributing teams for each play type; not total season appearances when source coverage is partial.",
            "frequency_policy": "Normalize combined possessions across available qualifying play types for each player-season.",
            "efg_policy": "FGA-weighted published team eFG%; limited by source rounding because exact 3PM is not supplied.",
            "ppp_policy": "Combined points/possessions rounded to the existing two-decimal storage precision.",
            "before": before, "after": after, "changes": changes}


def sql_json(rows):
    text = "[\n" + ",\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in rows) + "\n]"
    return "'" + text.replace("'", "''") + "'::jsonb"


def render_sql(plan):
    cols = ", ".join('"' + c + '"' for c in COLUMNS)
    def actual():
        return f"""SELECT {', '.join('p.' + '"' + c + '"' for c in COLUMNS)}
            FROM public.player_playtype_stats AS p
            WHERE p.season IN (SELECT DISTINCT season FROM _pt_before)"""
    def differs(table):
        return f"""EXISTS (({actual()}) EXCEPT SELECT {cols} FROM {table})
            OR EXISTS ((SELECT {cols} FROM {table}) EXCEPT {actual()})"""
    return f"""-- Season-long player play-type repair; generated {datetime.now(timezone.utc).isoformat()}.
-- Updates existing rows only. No permanent tables or schema changes.
-- Team-specific rows are combined; a supplied whole-season total is never double-counted.
-- Frequency = share of available qualifying play-type possessions.
-- Percentiles are calculated PPP ranks across combined season totals, not official NBA ranks.
-- Cohort: available source players with >=10 reported possessions in that season/play type.
-- Ties use average rank; percentile=(rank-1)/(N-1), or .5 for a singleton cohort.
-- Recalculates all cohort members so combined rows use the same comparison scale.
-- Merged GP sums the source team GPs for that play type, not all season appearances.
-- eFG% weights rounded source eFG% by attempts; exact 3PM is unavailable.
-- PPP retains existing two-decimal precision. Existing created_at values are preserved.
-- Run the entire file; validation failures roll back the transaction.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL standard_conforming_strings = on;
LOCK TABLE public.player_playtype_stats IN SHARE ROW EXCLUSIVE MODE;
CREATE TEMP TABLE _pt_before ON COMMIT DROP AS
SELECT {cols} FROM jsonb_populate_recordset(NULL::public.player_playtype_stats,
{sql_json(plan['before'])});
CREATE TEMP TABLE _pt_after ON COMMIT DROP AS
SELECT {cols} FROM jsonb_populate_recordset(NULL::public.player_playtype_stats,
{sql_json(plan['after'])});

DO $check_snapshot$
BEGIN
    IF ({differs('_pt_before')}) AND ({differs('_pt_after')}) THEN
        RAISE EXCEPTION 'Affected play-type records changed since preparation; regenerate this repair.';
    END IF;
END;
$check_snapshot$;

UPDATE public.player_playtype_stats AS p
SET {', '.join('"'+c+'" = desired."'+c+'"' for c in VALUE_COLUMNS)},
    updated_at = now()
FROM _pt_after AS desired
WHERE p.player_id = desired.player_id AND p.season = desired.season
  AND p.play_type = desired.play_type
  AND ROW({', '.join('p."'+c+'"' for c in VALUE_COLUMNS)})
      IS DISTINCT FROM ROW({', '.join('desired."'+c+'"' for c in VALUE_COLUMNS)});

DO $check_result$
BEGIN
    IF {differs('_pt_after')} THEN
        RAISE EXCEPTION 'Season aggregation validation failed; repair rolled back.';
    END IF;
END;
$check_result$;
COMMIT;
"""


def fetch_source(season, play, timeout):
    for attempt in range(3):
        try:
            frame = synergyplaytypes.SynergyPlayTypes(
                league_id="00", per_mode_simple="Totals", player_or_team_abbreviation="P",
                season_type_all_star="Regular Season", season=season,
                play_type_nullable=play, type_grouping_nullable="offensive", timeout=timeout,
            ).get_data_frames()[0]
            if frame.empty:
                raise AuditError(f"Empty NBA response for {season}/{play}; cannot aggregate safely.")
            return json.loads(frame.to_json(orient="records"))
        except AuditError:
            raise
        except Exception as exc:
            if attempt == 2:
                raise AuditError(f"NBA source failed for {season}/{play} ({type(exc).__name__}).") from None
            time.sleep(2 ** (attempt + 1))


def read_stored(client, seasons):
    rows, seen, expected = [], set(), None
    while True:
        response = (client.table("player_playtype_stats").select("*", count="exact")
                    .in_("season", seasons).order("player_id").order("season").order("play_type")
                    .range(len(rows), len(rows) + 999).execute())
        if expected is None:
            expected = response.count
        if expected is None or response.count != expected:
            raise AuditError("Database count changed or was unavailable during pagination.")
        if not response.data:
            if len(rows) != expected:
                raise AuditError("Database pagination ended early.")
            return rows
        for row in response.data:
            key = tuple(row[c] for c in KEY_COLUMNS)
            if key in seen:
                raise AuditError("Duplicate database keys or repeated pages.")
            seen.add(key)
        rows.extend(response.data)
        if len(rows) == expected:
            return rows
        if len(rows) > expected:
            raise AuditError("Database row count changed during pagination.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", nargs="+", required=True, type=season_argument)
    parser.add_argument("--source-dir", type=Path,
                        help="Use previously reviewed Totals JSON files named SEASON_PLAYTYPE.json.")
    parser.add_argument("--sql", type=Path, help="Export a new transactional SQL file; never executes it.")
    parser.add_argument("--json", type=Path, help="Export the full before/after review to a new JSON file.")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("Timeout must be positive.")
    for path in (args.sql, args.json):
        if path and (path.exists() or not path.parent.is_dir()):
            parser.error("Choose new output filenames inside existing directories.")
    if args.sql and args.json and args.sql.resolve() == args.json.resolve():
        parser.error("SQL and JSON outputs must use different filenames.")
    print("Read-only season aggregation preview; no database writes or ingestion-script changes.", flush=True)
    try:
        config = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
        url = os.environ.get("SUPABASE_URL") or config.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY") or config.get("SUPABASE_KEY")
        if not url or not key:
            raise AuditError("Set SUPABASE_URL and SUPABASE_KEY in python/.env or the environment.")
        client = create_client(url, key)
        seasons = list(dict.fromkeys(args.season))
        source = {}
        for season in seasons:
            source[season] = []
            for play in PLAY_TYPES:
                if args.source_dir:
                    rows = json.loads((args.source_dir / f"{season}_{play}.json").read_text())
                    if not rows or any(r["PLAY_TYPE"] != play for r in rows):
                        raise AuditError(f"Invalid cached source for {season}/{play}.")
                else:
                    print(f"Fetching {season}/{play}...", flush=True)
                    rows = fetch_source(season, play, args.timeout)
                    time.sleep(.7)
                source[season].extend(rows)
        stored = read_stored(client, seasons)
        plan = plan_repair(source, stored)
        print(f"Distinct affected players: {plan['distinct_players']}")
        for s in plan["summaries"]:
            print(f"{s['season']}: {s['affected_player_seasons']} player-seasons; "
                  f"{s['merged_play_type_rows']} play types to combine; {s['changed_rows']} changed rows; "
                  f"+{s['additional_possessions']:g} possessions, +{s['additional_points']:g} points.")
        print("Season PPP percentiles recalculated for all comparison cohorts; >=10 reported possessions, average tied ranks.")
        print("Source team GP counts are summed per play type; no usage-percentile calculation.")
        print("eFG% uses attempt-weighted rounded source percentages; PPP retains two decimal places.")
        if args.json:
            with args.json.open("x") as handle:
                json.dump(plan, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
            print(f"Before/after review: {args.json}")
        if args.sql:
            with args.sql.open("x") as handle:
                handle.write(render_sql(plan))
            print(f"SQL prepared for review, not executed: {args.sql}")
        print("No database records were changed.")
        return 0
    except Exception as exc:
        message = str(exc) if isinstance(exc, AuditError) else f"Preparation failed ({type(exc).__name__})."
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
