"""
fetch_team_defensive_stats.py
------------------------------
Fetches team game logs for multiple seasons from NBA API,
properly maps opponent stats, calculates advanced metrics,
and upserts data directly into Supabase team_defensive_stats table.
"""

import os
import time
import random
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from dotenv import load_dotenv
from supabase import create_client, Client

from nba_api.stats.endpoints import teamgamelogs

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Test with one season at a time, because seasons start and end at different dates
SEASONS = ["2022-23"]  
# SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

TEAMS = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", 
         "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", 
         "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]

REQUEST_DELAY = 0.7
RETRY_DELAY = 10
MAX_RETRIES = 3
BATCH_SIZE = 500

# Create a session with retry logic
def create_retry_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

import nba_api
nba_api.session = create_retry_session()

# Helper Functions
def parse_minutes(minutes_str):
    """Convert minutes string to float."""
    if minutes_str is None or minutes_str == '':
        return 48.0
    try:
        if ':' in str(minutes_str):
            parts = str(minutes_str).split(':')
            return float(parts[0]) + float(parts[1]) / 60
        return float(minutes_str)
    except (ValueError, TypeError):
        return 48.0

def calculate_possessions(fga, fta, orb, tov):
    """Calculate possessions using standard NBA formula."""
    return fga + (0.44 * fta) - orb + tov

def get_safe_float(value, default=0.0):
    """Safely convert value to float."""
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def get_safe_int(value, default=0):
    """Safely convert value to int."""
    try:
        if value is None or value == '':
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default

def fetch_season_gamelogs(season):
    """Fetch game logs for all teams for a given season."""
    log.info(f"Fetching {season}...")
    
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY)
            
            gamelog = teamgamelogs.TeamGameLogs(
                season_nullable=season,
                timeout=60
            )
            df = gamelog.get_data_frames()[0]
            
            log.info(f"  Raw data: {len(df)} rows")
            
            # Convert date
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            
            # Filter to regular season only
            year = int(season[:4])
            season_start = pd.to_datetime(f'{year}-10-17')
            season_end = pd.to_datetime(f'{year + 1}-04-10')
            
            df = df[(df['GAME_DATE'] >= season_start) & (df['GAME_DATE'] < season_end)]
            log.info(f"  After date filter: {len(df)} rows")
            
            # Filter out rows with missing MATCHUP
            df = df[df['MATCHUP'].notna()]
            log.info(f"  After removing null matchups: {len(df)} rows")
            
            return df
            
        except Exception as e:
            log.warning(f"  Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    log.error(f"  Failed to fetch {season}")
    return pd.DataFrame()

def build_defensive_stats(df, season):
    """Build defensive stats by mapping each game to its opponent's stats."""
    
    log.info(f"  Building defensive stats...")
    
    # Create a dictionary keyed by (date, team) for quick lookup
    team_stats = {}
    for _, row in df.iterrows():
        date_str = row['GAME_DATE'].strftime('%Y-%m-%d')
        team = row['TEAM_ABBREVIATION']
        key = f"{date_str}_{team}"
        team_stats[key] = row
    
    all_games = []
    
    for _, row in df.iterrows():
        date_str = row['GAME_DATE'].strftime('%Y-%m-%d')
        team = row['TEAM_ABBREVIATION']
        matchup = row['MATCHUP']
        
        if matchup is None:
            continue
        
        # Determine opponent from matchup
        opponent = None
        parts = None
        if 'vs.' in matchup:
            parts = matchup.split('vs.')
            if parts[0].strip() == team:
                opponent = parts[1].strip()
            else:
                opponent = parts[0].strip()
        elif '@' in matchup:
            parts = matchup.split('@')
            if parts[0].strip() == team:
                opponent = parts[1].strip()
            else:
                opponent = parts[0].strip()
        
        if opponent is None or opponent not in TEAMS:
            continue
        
        # Look up opponent's stats
        opp_key = f"{date_str}_{opponent}"
        opp_stats = team_stats.get(opp_key)
        
        if opp_stats is None:
            continue
        
        # Determine home/away
        home = 1 if 'vs.' in matchup and parts[0].strip() == team else 0
        
        # Get numeric values
        team_fga = get_safe_float(row['FGA'])
        team_fta = get_safe_float(row['FTA'])
        team_orb = get_safe_float(row['OREB'])
        team_tov = get_safe_float(row['TOV'])
        team_pts = get_safe_float(row['PTS'])
        
        opp_fga = get_safe_float(opp_stats['FGA'])
        opp_fta = get_safe_float(opp_stats['FTA'])
        opp_orb = get_safe_float(opp_stats['OREB'])
        opp_tov = get_safe_float(opp_stats['TOV'])
        opp_pts = get_safe_float(opp_stats['PTS'])
        
        # Calculate possessions
        team_poss = calculate_possessions(team_fga, team_fta, team_orb, team_tov)
        opp_poss = calculate_possessions(opp_fga, opp_fta, opp_orb, opp_tov)
        
        # Average possessions for game pace
        avg_poss = (team_poss + opp_poss) / 2
        if avg_poss <= 0:
            avg_poss = 1
        
        minutes = parse_minutes(row['MIN'])
        
        # Calculate pace metrics
        game_pace = 48 * (avg_poss / minutes)
        team_pace = 48 * (team_poss / minutes)
        opponent_pace = 48 * (opp_poss / minutes)
        
        # Calculate ratings
        off_rating = (team_pts / team_poss) * 100 if team_poss > 0 else 0
        def_rating = (opp_pts / opp_poss) * 100 if opp_poss > 0 else 0
        
        all_games.append({
            'season': season,
            'team': team,
            'opponent': opponent,
            'date': date_str,
            'home': home,
            'team_pts': get_safe_int(row['PTS']),
            'opponent_fgm': get_safe_int(opp_stats['FGM']),
            'opponent_fga': get_safe_int(opp_stats['FGA']),
            'opponent_fg_pct': round(get_safe_float(opp_stats['FG_PCT']), 4),
            'opponent_3pm': get_safe_int(opp_stats['FG3M']),
            'opponent_3pa': get_safe_int(opp_stats['FG3A']),
            'opponent_3p_pct': round(get_safe_float(opp_stats['FG3_PCT']), 4),
            'opponent_ftm': get_safe_int(opp_stats['FTM']),
            'opponent_fta': get_safe_int(opp_stats['FTA']),
            'opponent_ft_pct': round(get_safe_float(opp_stats['FT_PCT']), 4),
            'opponent_orb': get_safe_int(opp_stats['OREB']),
            'opponent_trb': get_safe_int(opp_stats['REB']),
            'opponent_ast': get_safe_int(opp_stats['AST']),
            'opponent_stl': get_safe_int(opp_stats['STL']),
            'opponent_blk': get_safe_int(opp_stats['BLK']),
            'opponent_tov': get_safe_float(opp_stats['TOV']),
            'opponent_pf': get_safe_int(opp_stats['PF']),
            'team_stl': get_safe_int(row['STL']),
            'team_blk': get_safe_int(row['BLK']),
            'team_tov': get_safe_float(row['TOV']),
            'team_pf': get_safe_int(row['PF']),
            'team_orb': get_safe_int(row['OREB']),
            'team_trb': get_safe_int(row['REB']),
            'opponent_pts': get_safe_int(opp_stats['PTS']),
            'pace': round(game_pace, 1),
            'team_pace': round(team_pace, 1),
            'opponent_pace': round(opponent_pace, 1),
            'off_rating': round(off_rating, 1),
            'def_rating': round(def_rating, 1),
        })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(all_games)
    
    if result_df.empty:
        log.info(f"    No games found")
        return result_df

    # Calculate average team pace for each opponent (for opponent_pace_deviation)
    opponent_avg_team_pace = result_df.groupby('opponent')['team_pace'].mean().to_dict()
    
    # Add average opponent team pace and deviation columns
    result_df['opponent_avg_team_pace'] = result_df['opponent'].map(opponent_avg_team_pace)
    result_df['opponent_pace_deviation'] = result_df['opponent_pace'] - result_df['opponent_avg_team_pace']

    # Calculate average game pace for each opponent (for opponent_game_pace_deviation)
    opponent_avg_game_pace = result_df.groupby('opponent')['pace'].mean().to_dict()
    
    # Add average opponent game pace and deviation columns
    result_df['opponent_avg_game_pace'] = result_df['opponent'].map(opponent_avg_game_pace)
    result_df['opponent_game_pace_deviation'] = result_df['pace'] - result_df['opponent_avg_game_pace']

    # Round new columns
    result_df['opponent_avg_team_pace'] = result_df['opponent_avg_team_pace'].round(1)
    result_df['opponent_pace_deviation'] = result_df['opponent_pace_deviation'].round(1)
    result_df['opponent_avg_game_pace'] = result_df['opponent_avg_game_pace'].round(1)
    result_df['opponent_game_pace_deviation'] = result_df['opponent_game_pace_deviation'].round(1)
    
    # Add game_id
    result_df['game_id'] = result_df['date'] + '_' + result_df['team'] + '_' + result_df['opponent']
    
    # Rename opponent column to match DB schema
    result_df = result_df.rename(columns={'opponent': 'opponent_abbrev'})
    
    log.info(f"    Built {len(result_df)} defensive game records")
    log.info(f"    Opponent avg team pace computed for {len(opponent_avg_team_pace)} teams")
    log.info(f"    Opponent avg game pace computed for {len(opponent_avg_game_pace)} teams")
    
    return result_df

def upsert_batch(supabase_client: Client, batch: List[Dict], total_upserted: int) -> int:
    """Upsert a batch of records and return updated count."""
    if not batch:
        return total_upserted
    
    result = supabase_client.table("team_defensive_stats").upsert(
        batch, on_conflict="team,date"
    ).execute()
    
    if hasattr(result, 'error') and result.error:
        log.error(f"Upsert error: {result.error}")
        return total_upserted
    
    new_total = total_upserted + len(batch)
    log.info(f"Upserted batch of {len(batch)} rows. Total: {new_total}")
    return new_total

# Main Execution
def main():
    log.info("="*80)
    log.info("🏀 FETCHING TEAM DEFENSIVE STATS")
    log.info("="*80)
    log.info(f"Seasons: {SEASONS}")
    log.info(f"Teams: {len(TEAMS)}")
    
    # Initialize Supabase client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    all_data = []
    
    for season in SEASONS:
        log.info(f"\n{'='*50}")
        log.info(f"Processing {season}...")
        log.info('='*50)
        
        # Fetch raw game logs
        df_raw = fetch_season_gamelogs(season)
        
        if df_raw.empty:
            log.error(f"  ✗ No data for {season}")
            continue
        
        # Build defensive stats
        df_defensive = build_defensive_stats(df_raw, season)
        
        if df_defensive.empty:
            log.error(f"  ✗ Failed to build defensive stats for {season}")
            continue
        
        log.info(f"\n  ✓ Processed {len(df_defensive)} rows")
        log.info(f"    Teams: {df_defensive['team'].nunique()}")
        log.info(f"    Date range: {df_defensive['date'].min()} to {df_defensive['date'].max()}")
        
        # Show sample verification
        if not df_defensive.empty:
            sample = df_defensive.iloc[0]
            log.info(f"\n    Sample verification:")
            log.info(f"      {sample['team']} vs {sample['opponent_abbrev']}")
            log.info(f"      Team points: {sample['team_pts']} | Opponent points: {sample['opponent_pts']}")
            log.info(f"      Pace: {sample['pace']} | Team Pace: {sample['team_pace']} | Opp Pace: {sample['opponent_pace']}")
            log.info(f"      Off Rating: {sample['off_rating']} | Def Rating: {sample['def_rating']}")
        
        all_data.append(df_defensive)
        
        if season != SEASONS[-1]:
            time.sleep(2)
    
    # Combine all seasons and upsert to Supabase
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        log.info(f"\n{'='*80}")
        log.info(f"✅ DATA COLLECTION COMPLETE!")
        log.info(f"{'='*80}")
        log.info(f"Total rows: {len(final_df):,}")
        log.info(f"Seasons: {final_df['season'].nunique()}")
        log.info(f"Teams: {final_df['team'].nunique()}")
        
        # Convert DataFrame to list of dicts for upsert
        records = final_df.to_dict('records')
        
        log.info(f"\n📤 Uploading to Supabase...")
        
        # Upsert in batches
        batch = []
        total_upserted = 0
        
        for record in records:
            batch.append(record)
            if len(batch) >= BATCH_SIZE:
                total_upserted = upsert_batch(supabase, batch, total_upserted)
                batch = []
        
        # Upsert remaining records
        total_upserted = upsert_batch(supabase, batch, total_upserted)
        
        log.info(f"\n✅ Upload complete! {total_upserted} rows upserted to team_defensive_stats table.")
    else:
        log.error("\n❌ No data was collected")

if __name__ == "__main__":
    main()