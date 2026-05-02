"""
retry_missing_players.py
------------------------
Fetches game logs for specific players that were missed in the initial run.
"""

import os
import time
import random
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from supabase import create_client, Client
from nba_api.stats.endpoints import playergamelogs

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEASON = "2022-23"
REQUEST_DELAY = 1.5
RETRY_DELAY = 10
MAX_RETRIES = 5 
BATCH_SIZE = 100

# Players you want to retry (add more as needed)
PROBLEM_PLAYERS = [
    "John Butler Jr.", "Emanuel Miller", "Jordan Schakel"
]

# Helper functions to get and format data
def parse_minutes(min_str: Any) -> float:
    if min_str is None:
        return 0.0
    try:
        if ":" in str(min_str):
            parts = str(min_str).split(":")
            return round(int(parts[0]) + int(parts[1]) / 60, 1)
        return float(min_str)
    except (ValueError, TypeError):
        return 0.0

def extract_team(matchup: str) -> Optional[str]:
    return matchup.split()[0] if matchup else None

def extract_opponent(matchup: str) -> Optional[str]:
    return matchup.split()[-1] if matchup else None

def extract_home_away(matchup: str) -> Optional[str]:
    if not matchup:
        return None
    if "@" in matchup:
        return "A"
    if "vs." in matchup:
        return "H"
    return None

def format_date(game_date) -> str:
    return game_date.strftime('%Y-%m-%d') if hasattr(game_date, 'strftime') else str(game_date)[:10]

def get_safe_int(row: Dict[str, Any], key: str) -> int:
    val = row.get(key, 0)
    return int(val) if val is not None else 0

def calculate_percentage(api_value: Optional[float], made: int, attempted: int) -> Optional[float]:
    if api_value is not None:
        return round(float(api_value), 4)
    if attempted > 0:
        return round(made / attempted, 4)
    return None

def compute_game_score(row: Dict[str, Any]) -> Optional[float]:
    try:
        pts = get_safe_int(row, 'PTS')
        fgm = get_safe_int(row, 'FGM')
        fga = get_safe_int(row, 'FGA')
        ftm = get_safe_int(row, 'FTM')
        fta = get_safe_int(row, 'FTA')
        oreb = get_safe_int(row, 'OREB')
        dreb = get_safe_int(row, 'DREB')
        stl = get_safe_int(row, 'STL')
        ast = get_safe_int(row, 'AST')
        blk = get_safe_int(row, 'BLK')
        pf = get_safe_int(row, 'PF')
        tov = get_safe_int(row, 'TOV')
        
        return round(
            pts + 0.4 * fgm - 0.7 * fga - 0.4 * (fta - ftm) +
            0.7 * oreb + 0.3 * dreb + stl + 0.7 * ast +
            0.7 * blk - 0.4 * pf - tov, 2
        )
    except Exception:
        return None
# get player and fetch their gamelog
def fetch_player_game_logs(player_id: int, player_name: str):
    """Fetch game logs with aggressive retry for problematic players."""
    for attempt in range(MAX_RETRIES):
        try:
            delay = REQUEST_DELAY + random.uniform(0.2, 1.0)
            time.sleep(delay)
            
            result = playergamelogs.PlayerGameLogs(
                player_id_nullable=player_id,
                season_nullable=SEASON,
                season_type_nullable="Regular Season"
            )
            df = result.get_data_frames()[0]
            if not df.empty:
                log.info(f"✓ Fetched {len(df)} games for {player_name}")
            else:
                log.info(f"○ No game logs for {player_name} in {SEASON}")
            return df
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate" in error_msg.lower():
                wait_time = RETRY_DELAY * (attempt + 1)
                log.warning(f"⏳ Rate limited for {player_name}. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                log.warning(f"⚠️ {player_name} attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
    
    log.error(f"❌ Failed to fetch data for {player_name} after {MAX_RETRIES} attempts")
    return None

# put data into a dict
def transform_row_to_dict(row, playerid: int) -> Dict[str, Any]:
    date_str = format_date(row['GAME_DATE'])
    matchup = row.get('MATCHUP', '')
    team = extract_team(matchup)
    opponent = extract_opponent(matchup)
    homeaway = extract_home_away(matchup)
    gameid = f"{date_str}_{team}_{opponent}" if team and opponent else None
    
    minutes = parse_minutes(row.get('MIN'))
    points = get_safe_int(row, 'PTS')
    rebounds = get_safe_int(row, 'REB')
    assists = get_safe_int(row, 'AST')
    steals = get_safe_int(row, 'STL')
    blocks = get_safe_int(row, 'BLK')
    turnovers = get_safe_int(row, 'TOV')
    personalfouls = get_safe_int(row, 'PF')
    fgm = get_safe_int(row, 'FGM')
    fga = get_safe_int(row, 'FGA')
    three_pm = get_safe_int(row, 'FG3M')
    three_pa = get_safe_int(row, 'FG3A')
    ftm = get_safe_int(row, 'FTM')
    fta = get_safe_int(row, 'FTA')
    oreb = get_safe_int(row, 'OREB')
    dreb = get_safe_int(row, 'DREB')
    plusminus = row.get('PLUS_MINUS') if 'PLUS_MINUS' in row else None
    
    fg_pct = calculate_percentage(row.get('FG_PCT'), fgm, fga)
    three_pct = calculate_percentage(row.get('FG3_PCT'), three_pm, three_pa)
    ft_pct = calculate_percentage(row.get('FT_PCT'), ftm, fta)
    
    return {
        "playerid": playerid,
        "gameid": gameid,
        "season": SEASON,
        "date": date_str,
        "team": team,
        "opponent": opponent,
        "homeaway": homeaway,
        "result": row.get('WL'),
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
        "gamescore": compute_game_score(row),
    }

# upsert rows into the database
def upsert_batch(supabase_client: Client, batch: List[Dict], total_upserted: int) -> int:
    if not batch:
        return total_upserted
    
    try:
        result = supabase_client.table("player_gamelog").upsert(
            batch, on_conflict="playerid,gameid"
        ).execute()
        
        if hasattr(result, 'error') and result.error:
            log.error(f"Upsert error: {result.error}")
            return total_upserted
        
        new_total = total_upserted + len(batch)
        log.info(f"📦 Upserted batch of {len(batch)} rows. Total: {new_total}")
        return new_total
    except Exception as e:
        log.error(f"Batch upsert failed: {e}")
        return total_upserted


# Main
def main():
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    log.info(f"Retrying {len(PROBLEM_PLAYERS)} players for season {SEASON}...")
    
    batch = []
    total_upserted = 0
    success_count = 0
    
    for player_name in PROBLEM_PLAYERS:
        # Get player info from database
        response = supabase.table("players") \
            .select("nba_api_id, display_name") \
            .eq("display_name", player_name) \
            .execute()
        
        if not response.data:
            log.warning(f"⚠️ {player_name}: Not found in players table")
            continue
        
        nba_api_id = response.data[0].get("nba_api_id")
        if not nba_api_id:
            log.warning(f"⚠️ {player_name}: No nba_api_id in database")
            continue
        
        log.info(f"\n--- Processing {player_name} (ID: {nba_api_id}) ---")
        
        df = fetch_player_game_logs(nba_api_id, player_name)
        if df is None or df.empty:
            log.info(f"❌ No data for {player_name}")
            continue
        
        success_count += 1
        for _, row in df.iterrows():
            try:
                record = transform_row_to_dict(row, nba_api_id)
                batch.append(record)
            except Exception as e:
                log.error(f"Error transforming row for {player_name}: {e}")
                continue
            
            if len(batch) >= BATCH_SIZE:
                total_upserted = upsert_batch(supabase, batch, total_upserted)
                batch = []
    
    # Final batch
    total_upserted = upsert_batch(supabase, batch, total_upserted)
    
    # Summary
    log.info(f"\n{'='*50}")
    log.info(f"✅ COMPLETED for season {SEASON}")
    log.info(f"   Players successfully fetched: {success_count}/{len(PROBLEM_PLAYERS)}")
    log.info(f"   Total rows upserted: {total_upserted}")
    log.info(f"{'='*50}")

if __name__ == "__main__":
    main()