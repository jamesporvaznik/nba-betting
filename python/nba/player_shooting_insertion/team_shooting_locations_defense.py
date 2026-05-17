"""
fetch_team_defensive_shot_locations.py
------------------------------
Fetches team defensive shot location data (where teams allow shots)
from NBA API and upserts directly to Supabase.
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

from nba_api.stats.endpoints import leaguedashteamshotlocations

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEASONS = ["2025-26"]
# SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

REQUEST_DELAY = 1.0
RETRY_DELAY = 5
MAX_RETRIES = 2
BATCH_SIZE = 500

# Shot zones we want to extract (same as player version)
SHOT_ZONES = [
    'Restricted Area',
    'In The Paint (Non-RA)',
    'Mid-Range',
    'Left Corner 3',
    'Right Corner 3',
    'Above the Break 3'
]

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

def fetch_team_defensive_shot_locations(season: str) -> pd.DataFrame:
    """Fetch team defensive shot location data for a specific season."""
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY)
            
            result = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
                season=season,
                season_type_all_star='Regular Season',
                league_id_nullable='00',
                per_mode_detailed='PerGame',
                measure_type_simple='Opponent',  # Shows what teams allow
                pace_adjust='N',
                distance_range='By Zone'
            )
            df = result.get_data_frames()[0]
            
            if df is not None and not df.empty:
                log.info(f"    ✓ Got {len(df)} rows for {season}")
                return df
            else:
                log.warning(f"    No data for {season}")
                return pd.DataFrame()
            
        except Exception as e:
            log.warning(f"    Attempt {attempt+1} for {season}: {str(e)[:100]}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    log.error(f"    ✗ Failed to fetch {season}")
    return pd.DataFrame()

def flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns to single level."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

def fetch_season_data(season: str) -> pd.DataFrame:
    """Fetch team defensive shot location data for a season."""
    log.info(f"\n{'='*50}")
    log.info(f"Processing {season}")
    log.info('='*50)
    log.info(f"  Fetching team defensive shot locations...")
    
    df = fetch_team_defensive_shot_locations(season)
    
    if df.empty:
        log.warning(f"  No data collected for {season}")
        return pd.DataFrame()
    
    # Flatten the MultiIndex columns
    df = flatten_multiindex(df)
    
    log.info(f"\n  📊 {season}: {len(df)} total rows")
    log.info(f"    Columns found: {list(df.columns)[:15]}...")
    
    return df

def transform_data(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Clean and transform the team defensive shot location data."""
    
    # Extract team info columns
    team_id_col = next((col for col in df.columns if 'TEAM_ID' in col), None)
    
    if not team_id_col:
        log.error("    Could not find team ID column")
        return pd.DataFrame()
    
    # Create a list to hold all zone data
    all_zones = []
    
    # For each team, extract data for each shot zone
    for idx, row in df.iterrows():
        team_id = row[team_id_col]
        
        for zone in SHOT_ZONES:
            # Look for columns that match this zone (defensive stats allowed)
            fgm_col = next((col for col in df.columns if col.startswith(zone) and col.endswith('_FGM')), None)
            fga_col = next((col for col in df.columns if col.startswith(zone) and col.endswith('_FGA')), None)
            fg_pct_col = next((col for col in df.columns if col.startswith(zone) and col.endswith('_FG_PCT')), None)
            
            if fgm_col and fga_col:
                fgm = row[fgm_col]
                fga = row[fga_col]
                fg_pct = row[fg_pct_col] if fg_pct_col else (fgm / fga if fga > 0 else 0)
                
                all_zones.append({
                    'team_id': team_id,
                    'season': season,
                    'shot_zone': zone,
                    'fgm_allowed': round(float(fgm), 2),
                    'fga_allowed': round(float(fga), 2),
                    'fg_pct_allowed': round(float(fg_pct), 4)
                })
    
    if not all_zones:
        log.warning("    No shot zone data extracted")
        return pd.DataFrame()
    
    transformed = pd.DataFrame(all_zones)
    
    # Calculate frequency percentage per team (what % of opponent shots come from each zone)
    team_totals = transformed.groupby('team_id')['fga_allowed'].sum().reset_index()
    team_totals.columns = ['team_id', 'total_fga_allowed']
    
    transformed = transformed.merge(team_totals, on='team_id')
    transformed['frequency_pct'] = round(transformed['fga_allowed'] / transformed['total_fga_allowed'], 4)
    transformed = transformed.drop('total_fga_allowed', axis=1)
    
    return transformed

def upsert_batch(supabase_client: Client, batch: list, table_name: str, total_upserted: int) -> int:
    """Upsert a batch of records and return updated count."""
    if not batch:
        return total_upserted
    
    try:
        result = supabase_client.table(table_name).upsert(
            batch, on_conflict="team_id,season,shot_zone"
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
    log.info("🏀 FETCHING TEAM DEFENSIVE SHOT LOCATION STATS & UPSERTING TO SUPABASE")
    log.info("="*80)
    log.info(f"Seasons: {SEASONS}")
    
    # Initialize Supabase client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    all_data = []
    
    for season in SEASONS:
        raw_df = fetch_season_data(season)
        if not raw_df.empty:
            transformed_df = transform_data(raw_df, season)
            if not transformed_df.empty:
                all_data.append(transformed_df)
                log.info(f"    ✓ Extracted {len(transformed_df)} shot zone rows")
                log.info(f"    Unique teams: {transformed_df['team_id'].nunique()}")
        
        if season != SEASONS[-1]:
            log.info("\n  ⏳ Waiting 5 seconds before next season...")
            time.sleep(5)
    
    if not all_data:
        log.error("\n❌ No data collected")
        return
    
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates if any
    final_df = final_df.drop_duplicates(subset=['team_id', 'season', 'shot_zone'])
    
    log.info(f"\n{'='*80}")
    log.info("📊 DATA COLLECTION COMPLETE")
    log.info(f"{'='*80}")
    log.info(f"Total rows: {len(final_df):,}")
    log.info(f"Unique teams: {final_df['team_id'].nunique():,}")
    log.info(f"Seasons: {final_df['season'].nunique()}")
    
    # Show what shot zones we got
    print("\n📊 Shot zones collected (team defense):")
    for zone in final_df['shot_zone'].unique():
        zone_data = final_df[final_df['shot_zone'] == zone]
        print(f"  {zone}: {len(zone_data)} entries, {zone_data['team_id'].nunique()} teams")
    
    # Convert to records for upsert
    records = final_df.to_dict('records')
    
    log.info(f"\n📤 Uploading {len(records)} records to Supabase...")
    
    # Upsert in batches
    batch = []
    total_upserted = 0
    table_name = "team_defensive_shot_locations"
    
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