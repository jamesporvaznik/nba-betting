"""
fetch_player_playtype_stats.py
------------------------------
Fetches player play type data using nba_api with better error handling,
and upserts directly to Supabase.
"""

import os
import time
import logging
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from dotenv import load_dotenv
from supabase import create_client, Client

from nba_api.stats.endpoints import synergyplaytypes

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
# SEASONS = ["2025-26"]  # Start with one

PLAY_TYPES = [
    "Transition",
    "Isolation",
    "Postup",
    "PRBallHandler",
    "PRRollMan",
    "Spotup",
    "Handoff",
    "Cut",
    "OffScreen",
    "OffRebound",
    "Misc"
]

REQUEST_DELAY = 1.0
RETRY_DELAY = 5
MAX_RETRIES = 2
BATCH_SIZE = 500

def create_retry_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

import nba_api
nba_api.session = create_retry_session()

def fetch_playtype(season: str, play_type: str) -> pd.DataFrame:
    """Fetch data for a specific play type."""
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY)
            
            result = synergyplaytypes.SynergyPlayTypes(
                league_id='00',
                per_mode_simple='PerGame',
                player_or_team_abbreviation='P',
                season_type_all_star='Regular Season',
                season=season,
                play_type_nullable=play_type,
                type_grouping_nullable='offensive'
            )
            df = result.get_data_frames()[0]
            
            if df is not None and not df.empty:
                log.info(f"    ✓ {play_type}: {len(df)} players")
                return df
            else:
                log.info(f"    - {play_type}: No data")
                return pd.DataFrame()
            
        except Exception as e:
            error_msg = str(e)
            if "Expecting value" in error_msg or "resultSet" in error_msg:
                log.info(f"    - {play_type}: No data available")
                return pd.DataFrame()
            else:
                log.warning(f"    Attempt {attempt+1} for {play_type}: {error_msg[:80]}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    log.warning(f"    ✗ {play_type}: Failed")
    return pd.DataFrame()

def fetch_season_data(season: str) -> pd.DataFrame:
    """Fetch all play types for a season."""
    log.info(f"\n{'='*50}")
    log.info(f"Processing {season}")
    log.info('='*50)
    
    all_dfs = []
    
    for play_type in PLAY_TYPES:
        log.info(f"  {play_type}...")
        df = fetch_playtype(season, play_type)
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        log.warning(f"  No data collected for {season}")
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    log.info(f"\n  📊 {season}: {len(combined)} total rows")
    
    return combined

def transform_data(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Clean and transform the data."""
    transformed = pd.DataFrame()
    
    transformed['player_id'] = df.get('PLAYER_ID', 0)
    transformed['season'] = season
    transformed['play_type'] = df.get('PLAY_TYPE', 'Unknown')
    
    # Core stats - possessions as float to preserve decimals
    transformed['possessions'] = pd.to_numeric(df.get('POSS', 0), errors='coerce').fillna(0).round(2)
    transformed['points'] = pd.to_numeric(df.get('PTS', 0), errors='coerce').fillna(0).round(2)
    transformed['poss_pct'] = pd.to_numeric(df.get('POSS_PCT', 0), errors='coerce').fillna(0).round(4)
    transformed['percentile'] = pd.to_numeric(df.get('PERCENTILE', 0), errors='coerce').fillna(0).round(4)
    transformed['games_played'] = pd.to_numeric(df.get('GP', 0), errors='coerce').fillna(0).astype(int)
    
    # Efficiency
    transformed['points_per_possession'] = transformed.apply(
        lambda row: round(row['points'] / row['possessions'], 3) if row['possessions'] > 0 else 0,
        axis=1
    )
    transformed['fg_pct'] = pd.to_numeric(df.get('FG_PCT', 0), errors='coerce').fillna(0).round(4)
    transformed['efg_pct'] = pd.to_numeric(df.get('EFG_PCT', 0), errors='coerce').fillna(0).round(4)
    
    return transformed

def upsert_batch(supabase_client: Client, batch: list, table_name: str, total_upserted: int) -> int:
    """Upsert a batch of records and return updated count."""
    if not batch:
        return total_upserted
    
    try:
        result = supabase_client.table(table_name).upsert(
            batch, on_conflict="player_id,season,play_type"
        ).execute()
        
        if hasattr(result, 'error') and result.error:
            log.error(f"Upsert error: {result.error}")
            return total_upserted
        
        new_total = total_upserted + len(batch)
        log.info(f"  Upserted batch of {len(batch)} rows. Total: {new_total}")
        return new_total
        
    except Exception as e:
        log.error(f"Batch upsert failed: {e}")
        return total_upserted

def main():
    log.info("="*80)
    log.info("🏀 FETCHING PLAYER PLAY TYPE STATS & UPSERTING TO SUPABASE")
    log.info("="*80)
    
    # Initialize Supabase client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    all_data = []
    
    for season in SEASONS:
        raw_df = fetch_season_data(season)
        if not raw_df.empty:
            transformed_df = transform_data(raw_df, season)
            all_data.append(transformed_df)
        
        if season != SEASONS[-1]:
            log.info("\n  ⏳ Waiting 5 seconds before next season...")
            time.sleep(5)
    
    if not all_data:
        log.error("\n❌ No data collected")
        return
    
    final_df = pd.concat(all_data, ignore_index=True)
    
    log.info(f"\n{'='*80}")
    log.info("📊 DATA COLLECTION COMPLETE")
    log.info(f"{'='*80}")
    log.info(f"Total rows: {len(final_df):,}")
    log.info(f"Unique players: {final_df['player_id'].nunique():,}")

    # Remove duplicates before upsert
    final_df = final_df.drop_duplicates(subset=['player_id', 'season', 'play_type'])
    
    # Convert to records for upsert
    records = final_df.to_dict('records')
    
    log.info(f"\n📤 Uploading {len(records)} records to Supabase...")
    
    # Upsert in batches
    batch = []
    total_upserted = 0
    table_name = "player_playtype_stats"
    
    for record in records:
        batch.append(record)
        if len(batch) >= BATCH_SIZE:
            total_upserted = upsert_batch(supabase, batch, table_name, total_upserted)
            batch = []
    
    # Upsert remaining records
    total_upserted = upsert_batch(supabase, batch, table_name, total_upserted)
    
    log.info(f"\n{'='*80}")
    log.info(f"✅ UPSERT COMPLETE!")
    log.info(f"{'='*80}")
    log.info(f"Total rows upserted: {total_upserted}")
    log.info(f"Table: {table_name}")

if __name__ == "__main__":
    main()