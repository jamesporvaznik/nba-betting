"""
fetch_all_gamelogs.py
---------------------
Fetches 2025-26 regular season game logs for every player in the local 'players' table
and upserts them into the 'player_gamelog' table in Supabase.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
from nba_api.stats.endpoints import playergamelogs

# ------------------------------------------------------------------ #
# Load environment variables
# ------------------------------------------------------------------ #
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEASON = "2025-26"
REQUEST_DELAY = 0.7          # seconds between NBA API calls
RETRY_DELAY = 10
MAX_RETRIES = 3
BATCH_SIZE = 500             # rows per upsert

# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #
def parse_minutes(min_str):
    """Convert 'MM:SS' to decimal minutes."""
    if min_str is None:
        return 0.0
    try:
        if ":" in str(min_str):
            parts = str(min_str).split(":")
            return round(int(parts[0]) + int(parts[1]) / 60, 1)
        return float(min_str)
    except (ValueError, TypeError):
        return 0.0

def extract_team(matchup: str) -> str:
    return matchup.split()[0] if matchup else None

def extract_opponent(matchup: str) -> str:
    return matchup.split()[-1] if matchup else None

def extract_home_away(matchup: str) -> str:
    if not matchup:
        return None
    if "@" in matchup:
        return "A"
    if "vs." in matchup:
        return "H"
    return None

def compute_game_score(row):
    """John Hollinger's Game Score."""
    pts = row.get('PTS', 0) or 0
    fgm = row.get('FGM', 0) or 0
    fga = row.get('FGA', 0) or 0
    ftm = row.get('FTM', 0) or 0
    fta = row.get('FTA', 0) or 0
    oreb = row.get('OREB', 0) or 0
    dreb = row.get('DREB', 0) or 0
    stl = row.get('STL', 0) or 0
    ast = row.get('AST', 0) or 0
    blk = row.get('BLK', 0) or 0
    pf = row.get('PF', 0) or 0
    tov = row.get('TOV', 0) or 0
    return round(pts + 0.4*fgm - 0.7*fga - 0.4*(fta-ftm) + 0.7*oreb + 0.3*dreb + stl + 0.7*ast + 0.7*blk - 0.4*pf - tov, 2)

def fetch_player_game_logs(player_id, player_name):
    """Fetch game logs for one player from NBA API."""
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY)
            result = playergamelogs.PlayerGameLogs(
                player_id_nullable=player_id,
                season_nullable=SEASON,
                season_type_nullable="Regular Season"
            )
            df = result.get_data_frames()[0]
            if not df.empty:
                log.info(f"Fetched {len(df)} games for {player_name}")
            return df
        except Exception as e:
            log.warning(f"{player_name} attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    log.error(f"Failed to fetch data for {player_name}")
    return None

def transform_row_to_dict(row, playerid):
    """
    Convert a row from the NBA API DataFrame into a dictionary
    that matches the columns of public.player_gamelog.
    """
    # Date
    game_date = row['GAME_DATE']
    if hasattr(game_date, 'strftime'):
        date_str = game_date.strftime('%Y-%m-%d')
    else:
        date_str = str(game_date)[:10]

    # Matchup parsing
    matchup = row.get('MATCHUP', '')
    team = extract_team(matchup)
    opponent = extract_opponent(matchup)
    homeaway = extract_home_away(matchup)

    # GameID: combination of date + team + opponent
    gameid = f"{date_str}_{team}_{opponent}" if team and opponent else None

    # Basic stats
    minutes = parse_minutes(row.get('MIN'))
    points = row.get('PTS', 0) or 0
    rebounds = row.get('REB', 0) or 0
    assists = row.get('AST', 0) or 0
    steals = row.get('STL', 0) or 0
    blocks = row.get('BLK', 0) or 0
    turnovers = row.get('TOV', 0) or 0
    personalfouls = row.get('PF', 0) or 0
    fgm = row.get('FGM', 0) or 0
    fga = row.get('FGA', 0) or 0
    three_pm = row.get('FG3M', 0) or 0
    three_pa = row.get('FG3A', 0) or 0
    ftm = row.get('FTM', 0) or 0
    fta = row.get('FTA', 0) or 0
    oreb = row.get('OREB', 0) or 0
    dreb = row.get('DREB', 0) or 0
    plusminus = row.get('PLUS_MINUS') if 'PLUS_MINUS' in row else None

    # Percentages (use API provided or compute)
    fg_pct = row.get('FG_PCT')
    if fg_pct is None and fga > 0:
        fg_pct = round(fgm / fga, 4)
    elif fg_pct is not None:
        fg_pct = round(fg_pct, 4)

    three_pct = row.get('FG3_PCT')
    if three_pct is None and three_pa > 0:
        three_pct = round(three_pm / three_pa, 4)
    elif three_pct is not None:
        three_pct = round(three_pct, 4)

    ft_pct = row.get('FT_PCT')
    if ft_pct is None and fta > 0:
        ft_pct = round(ftm / fta, 4)
    elif ft_pct is not None:
        ft_pct = round(ft_pct, 4)

    # GameScore
    gamescore = compute_game_score(row)

    # Result (W/L)
    result = row.get('WL')

    # Build dictionary – column names must match player_gamelog exactly
    return {
        "playerid": playerid,
        "gameid": gameid,
        "season": SEASON,
        "date": date_str,
        "team": team,
        "opponent": opponent,
        "homeaway": homeaway,
        "result": result,
        "minutes": minutes,
        "points": points,
        "rebounds": rebounds,
        "assists": assists,
        "steals": steals,
        "blocks": blocks,
        "turnovers": turnovers,
        "personalfouls": personalfouls,
        "fgm": fgm,
        "fga": fga,
        "fg_pct": fg_pct,
        "3PM": three_pm,
        "3PA": three_pa,
        "3P_PCT": three_pct,
        "ftm": ftm,
        "fta": fta,
        "ft_pct": ft_pct,
        "oreb": oreb,
        "dreb": dreb,
        "plusminus": plusminus,
        "gamescore": gamescore,
    }

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1. Get all players from our 'players' table
    log.info("Fetching players from public.players table...")
    response = supabase.table("players").select("nba_api_id, display_name").execute()
    if hasattr(response, 'error') and response.error:
        log.error(f"Error fetching players: {response.error}")
        return

    players_list = response.data
    log.info(f"Found {len(players_list)} players")

    # 2. Process each player
    batch = []
    total_upserted = 0

    for player in players_list:
        nba_api_id = player.get("nba_api_id")
        display_name = player.get("display_name", "Unknown")
        if not nba_api_id:
            log.warning(f"Skipping {display_name}: no nba_api_id")
            continue

        df = fetch_player_game_logs(nba_api_id, display_name)
        if df is None or df.empty:
            log.info(f"No game logs for {display_name}")
            continue

        for _, row in df.iterrows():
            try:
                record = transform_row_to_dict(row, nba_api_id)   # playerid = nba_api_id
                batch.append(record)
                total_upserted += 1
            except Exception as e:
                log.error(f"Error transforming row for {display_name}: {e}")
                continue

            # Upsert when batch is full
            if len(batch) >= BATCH_SIZE:
                result = supabase.table("player_gamelog").upsert(
                    batch, on_conflict="playerid,gameid"
                ).execute()
                if hasattr(result, 'error') and result.error:
                    log.error(f"Upsert error: {result.error}")
                else:
                    log.info(f"Upserted batch of {len(batch)} rows. Total: {total_upserted}")
                batch = []

    # 3. Upsert any remaining rows
    if batch:
        result = supabase.table("player_gamelog").upsert(
            batch, on_conflict="playerid,gameid"
        ).execute()
        if hasattr(result, 'error') and result.error:
            log.error(f"Final upsert error: {result.error}")
        else:
            log.info(f"Upserted final batch of {len(batch)} rows. Total: {total_upserted}")

    log.info(f"✅ Completed. Upserted {total_upserted} game log rows.")

if __name__ == "__main__":
    main()