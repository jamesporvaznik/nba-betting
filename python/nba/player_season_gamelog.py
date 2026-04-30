"""
fetch_all_gamelogs.py
---------------------
Fetches a chosen (line 29) regular season game logs for every player in the local 'players' table
and upserts them into the 'player_gamelog' table in Supabase.
"""

# Import packages
import os
import time
import logging
from typing import Optional, Dict, Any
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

# Constant variables
SEASON = "2022-23"
REQUEST_DELAY = 0.7
RETRY_DELAY = 10
MAX_RETRIES = 3
BATCH_SIZE = 500

# Helper functions

# Convert minutes to fit table format in database
def parse_minutes(min_str: Any) -> float:
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

def extract_team(matchup: str) -> Optional[str]:
    return matchup.split()[0] if matchup else None

def extract_opponent(matchup: str) -> Optional[str]:
    return matchup.split()[-1] if matchup else None

def extract_home_away(matchup: str) -> Optional[str]:
    """Return 'H' for home, 'A' for away, None if unknown."""
    if not matchup:
        return None
    if "@" in matchup:
        return "A"
    if "vs." in matchup:
        return "H"
    return None

def format_date(game_date) -> str:
    """Format game date to string."""
    return game_date.strftime('%Y-%m-%d') if hasattr(game_date, 'strftime') else str(game_date)[:10]

def get_safe_int(row: Dict[str, Any], key: str) -> int:
    """Safely get integer value from row."""
    return row.get(key, 0) or 0

def get_safe_float(row: Dict[str, Any], key: str) -> float:
    """Safely get float value from row."""
    return float(row.get(key, 0) or 0)

def calculate_percentage(api_value: Optional[float], made: int, attempted: int) -> Optional[float]:
    """Calculate percentage from made/attempted if API value not provided."""
    if api_value is not None:
        return round(api_value, 4)
    if attempted > 0:
        return round(made / attempted, 4)
    return None

def compute_game_score(row: Dict[str, Any]) -> float:
    """John Hollinger's Game Score."""
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

def create_gameid(date_str: str, team: Optional[str], opponent: Optional[str]) -> Optional[str]:
    """Create game ID from date, team, and opponent."""
    return f"{date_str}_{team}_{opponent}" if team and opponent else None

def extract_matchup_info(row: Dict[str, Any]) -> tuple:
    """Extract matchup information from row."""
    matchup = row.get('MATCHUP', '')
    team = extract_team(matchup)
    opponent = extract_opponent(matchup)
    homeaway = extract_home_away(matchup)
    return team, opponent, homeaway

def extract_basic_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract basic stats from row."""
    return {
        'minutes': parse_minutes(row.get('MIN')),
        'points': get_safe_int(row, 'PTS'),
        'rebounds': get_safe_int(row, 'REB'),
        'assists': get_safe_int(row, 'AST'),
        'steals': get_safe_int(row, 'STL'),
        'blocks': get_safe_int(row, 'BLK'),
        'turnovers': get_safe_int(row, 'TOV'),
        'personalfouls': get_safe_int(row, 'PF'),
        'fgm': get_safe_int(row, 'FGM'),
        'fga': get_safe_int(row, 'FGA'),
        'three_pm': get_safe_int(row, 'FG3M'),
        'three_pa': get_safe_int(row, 'FG3A'),
        'ftm': get_safe_int(row, 'FTM'),
        'fta': get_safe_int(row, 'FTA'),
        'oreb': get_safe_int(row, 'OREB'),
        'dreb': get_safe_int(row, 'DREB'),
        'plusminus': row.get('PLUS_MINUS') if 'PLUS_MINUS' in row else None,
    }

def extract_percentages(row: Dict[str, Any], stats: Dict[str, Any]) -> tuple:
    """Extract shooting percentages from row."""
    fg_pct = calculate_percentage(row.get('FG_PCT'), stats['fgm'], stats['fga'])
    three_pct = calculate_percentage(row.get('FG3_PCT'), stats['three_pm'], stats['three_pa'])
    ft_pct = calculate_percentage(row.get('FT_PCT'), stats['ftm'], stats['fta'])
    return fg_pct, three_pct, ft_pct

def fetch_player_game_logs(player_id: int, player_name: str):
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

def transform_row_to_dict(row, playerid: int) -> Dict[str, Any]:
    """Convert NBA API row to player_gamelog dictionary."""
    # Date and matchup
    date_str = format_date(row['GAME_DATE'])
    team, opponent, homeaway = extract_matchup_info(row)
    gameid = create_gameid(date_str, team, opponent)
    
    # Stats
    stats = extract_basic_stats(row)
    fg_pct, three_pct, ft_pct = extract_percentages(row, stats)
    
    return {
        "playerid": playerid,
        "gameid": gameid,
        "season": SEASON,
        "date": date_str,
        "team": team,
        "opponent": opponent,
        "homeaway": homeaway,
        "result": row.get('WL'),
        "minutes": stats['minutes'],
        "points": stats['points'],
        "rebounds": stats['rebounds'],
        "assists": stats['assists'],
        "steals": stats['steals'],
        "blocks": stats['blocks'],
        "turnovers": stats['turnovers'],
        "personalfouls": stats['personalfouls'],
        "fgm": stats['fgm'],
        "fga": stats['fga'],
        "fg_pct": fg_pct,
        "3PM": stats['three_pm'],
        "3PA": stats['three_pa'],
        "3P_PCT": three_pct,
        "ftm": stats['ftm'],
        "fta": stats['fta'],
        "ft_pct": ft_pct,
        "oreb": stats['oreb'],
        "dreb": stats['dreb'],
        "plusminus": stats['plusminus'],
        "gamescore": compute_game_score(row),
    }

def upsert_batch(supabase_client: Client, batch: list, total_upserted: int) -> int:
    """Upsert a batch of records and return updated count."""
    if not batch:
        return total_upserted
    
    result = supabase_client.table("player_gamelog").upsert(
        batch, on_conflict="playerid,gameid"
    ).execute()
    
    if hasattr(result, 'error') and result.error:
        log.error(f"Upsert error: {result.error}")
        return total_upserted
    
    new_total = total_upserted + len(batch)
    log.info(f"Upserted batch of {len(batch)} rows. Total: {new_total}")
    return new_total

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    log.info("Fetching players from public.players table...")
    response = supabase.table("players").select("nba_api_id, display_name").execute()
    if hasattr(response, 'error') and response.error:
        log.error(f"Error fetching players: {response.error}")
        return

    players_list = response.data
    log.info(f"Found {len(players_list)} players")

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
                record = transform_row_to_dict(row, nba_api_id)
                batch.append(record)
            except Exception as e:
                log.error(f"Error transforming row for {display_name}: {e}")
                continue

            if len(batch) >= BATCH_SIZE:
                total_upserted = upsert_batch(supabase, batch, total_upserted)
                batch = []

    total_upserted = upsert_batch(supabase, batch, total_upserted)
    log.info(f"Completed. Upserted {total_upserted} game log rows.")

if __name__ == "__main__":
    main()


    # 2024-25 John Collins, Darius Garland, Simone Fontecchio, Adem Flagler, Aaron Nesmith, 
    #         Jusuf Nurkic, Craig Porter Jr., Jason Preston, Joshua Primo, Jahmi'us Ramsey, 
    #         Rodney McGruder, Jaden Springer, Cade Cunningham


    # 2023-24 Simone Fontecchio, Wesley Matthews, Skylar Mays, JaVale McGee, Nathan Mensah
    #          Chimezie Metu, Xavier Moon, Mike Muscala, Boban Marjanovic, Kevin McCullar Jr.
    #          Bruce Brown, Kobe Brown, Kevon Harris, Thomas Bryant, Talen Horton-Tucker,
    #          Reggie Jackson, Ty Jerome, AJ Johnson, Drake Powell, Maxime Raynaud,


    # 2022-23 John Butler Jr., Emanuel Miller, Jordan Schakel