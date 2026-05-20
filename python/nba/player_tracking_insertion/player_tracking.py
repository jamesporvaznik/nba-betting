"""
fetch_player_tracking_stats.py
Using nba_api package to avoid stats.nba.com blocking issues.
Upserts directly to Supabase.
Install: pip install nba_api supabase
"""

from nba_api.stats.endpoints import leaguedashptstats
import pandas as pd
import time
import logging
import os
import math
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEASONS = ["2025-26"]

PT_MEASURE_TYPES = {
    "Possessions": {
        "columns": [
            "AVG_DRIB_PER_TOUCH",
            "AVG_SEC_PER_TOUCH",
            "TOUCHES",
        ],
        "rename": {
            "AVG_DRIB_PER_TOUCH": "avg_drib_per_touch",
            "AVG_SEC_PER_TOUCH": "avg_sec_per_touch",
            "TOUCHES": "touches",
        }
    },
    "Passing": {
        "columns": [
            "AST_TO_PASS_PCT",
            "PASSES_MADE",
            "POTENTIAL_AST",
        ],
        "rename": {
            "AST_TO_PASS_PCT": "ast_to_pass_pct",
            "PASSES_MADE": "passes_made",
            "POTENTIAL_AST": "potential_ast",
        }
    },
    "CatchShoot": {
        "columns": [
            "CATCH_SHOOT_FGA",
            "CATCH_SHOOT_FG3A",
            "CATCH_SHOOT_EFG_PCT",
        ],
        "rename": {
            "CATCH_SHOOT_FGA": "catch_shoot_fga",
            "CATCH_SHOOT_FG3A": "catch_shoot_fg3a",
            "CATCH_SHOOT_EFG_PCT": "catch_shoot_efg_pct",
        }
    },
    "PullUpShot": {
        "columns": [
            "PULL_UP_FGA",
            "PULL_UP_FG3A",
            "PULL_UP_EFG_PCT",
        ],
        "rename": {
            "PULL_UP_FGA": "pull_up_fga",
            "PULL_UP_FG3A": "pull_up_fg3a",
            "PULL_UP_EFG_PCT": "pull_up_efg_pct",
        }
    },
    "Drives": {
        "columns": [
            "DRIVES",
            "DRIVE_FGA",
            "DRIVE_FG_PCT",
            "DRIVE_PTS",
            "DRIVE_PASSES",
            "DRIVE_AST",
        ],
        "rename": {
            "DRIVES": "drives",
            "DRIVE_FGA": "drive_fga",
            "DRIVE_FG_PCT": "drive_fg_pct",
            "DRIVE_PTS": "drive_pts",
            "DRIVE_PASSES": "drive_passes",
            "DRIVE_AST": "drive_ast",
        }
    },
    "Rebounding": {
        "columns": [
            "REB_CHANCE_PCT",
            "DREB_CHANCE_PCT",
            "OREB_CHANCE_PCT",
            "REB_CONTEST_PCT",
        ],
        "rename": {
            "REB_CHANCE_PCT": "reb_chance_pct",
            "DREB_CHANCE_PCT": "dreb_chance_pct",
            "OREB_CHANCE_PCT": "oreb_chance_pct",
            "REB_CONTEST_PCT": "reb_contest_pct",
        }
    },
}

def clean_value(val):
    """Clean values for JSON serialization."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
    return val

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame for Supabase upsert."""
    # Replace NaN, None, and infinite values with None
    df = df.replace([math.inf, -math.inf], None)
    df = df.where(pd.notnull(df), None)
    return df

def fetch_pt_stats(season: str, pt_measure_type: str) -> pd.DataFrame | None:
    """Fetch player tracking stats for a specific measure type."""
    try:
        endpoint = leaguedashptstats.LeagueDashPtStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
            player_or_team="Player",
            pt_measure_type=pt_measure_type,
            league_id_nullable="00",
            last_n_games=0,
            month=0,
            opponent_team_id=0,
        )

        df = endpoint.league_dash_pt_stats.get_data_frame()

        if df.empty:
            log.warning(f"  No data for {pt_measure_type}")
            return None

        log.info(f"    ✓ {pt_measure_type}: {len(df)} players")
        return df

    except Exception as e:
        log.error(f"  ✗ {pt_measure_type} failed: {e}")
        return None

def fetch_all_tracking_stats(season: str) -> dict[str, pd.DataFrame]:
    """Fetch all tracking stats for a season."""
    log.info(f"\n{'='*50}\nProcessing {season}\n{'='*50}")
    results = {}

    for measure_type, config in PT_MEASURE_TYPES.items():
        log.info(f"  Fetching {measure_type}...")
        df = fetch_pt_stats(season, measure_type)

        if df is not None and not df.empty:
            # Keep only player ID and desired columns
            keep_cols = ["PLAYER_ID"] + [col for col in config["columns"] if col in df.columns]
            df = df[keep_cols]
            
            # Rename columns to match database schema
            df = df.rename(columns=config["rename"])
            
            # Clean the DataFrame
            df = clean_dataframe(df)
            
            results[measure_type] = df
            log.info(f"      Kept {len(keep_cols) - 1} metrics")

        time.sleep(2)  # nba_api recommends ~1-2s between calls

    return results

def merge_tracking_data(all_data: dict[str, pd.DataFrame], season: str) -> pd.DataFrame:
    """Merge all tracking data into a single DataFrame."""
    if not all_data:
        return pd.DataFrame()

    dfs = list(all_data.values())
    merged = dfs[0].copy()
    merged["season"] = season

    for df in dfs[1:]:
        merged = merged.merge(df, on="PLAYER_ID", how="outer")

    # Rename PLAYER_ID to player_id for database
    merged = merged.rename(columns={"PLAYER_ID": "player_id"})
    
    # Clean the final merged DataFrame
    merged = clean_dataframe(merged)
    
    return merged

def upsert_batch(supabase_client: Client, batch: list, total_upserted: int) -> int:
    """Upsert a batch of records to Supabase."""
    if not batch:
        return total_upserted
    
    try:
        result = supabase_client.table("player_tracking_stats").upsert(
            batch, on_conflict="player_id,season"
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
    log.info("🏀 FETCHING PLAYER TRACKING STATS & UPSERTING TO SUPABASE")
    log.info("="*80)
    
    # Initialize Supabase client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    all_merged = []
    BATCH_SIZE = 500

    for season in SEASONS:
        data = fetch_all_tracking_stats(season)
        if data:
            merged = merge_tracking_data(data, season)
            log.info(f"\n  📊 {season}: {len(merged)} players, {len(merged.columns)} columns")
            all_merged.append(merged)
        else:
            log.warning(f"  No data for {season}")

        if season != SEASONS[-1]:
            time.sleep(3)

    if not all_merged:
        log.error("No data collected.")
        return

    final_df = pd.concat(all_merged, ignore_index=True)
    
    # Final cleaning pass
    final_df = clean_dataframe(final_df)
    
    log.info(f"\n{'='*80}")
    log.info("📊 DATA COLLECTION COMPLETE")
    log.info(f"{'='*80}")
    log.info(f"Total rows: {len(final_df):,}")
    log.info(f"Unique players: {final_df['player_id'].nunique()}")
    
    # Convert to records for upsert
    records = final_df.to_dict('records')
    
    # Validate records have no NaN/inf values
    valid_records = []
    for record in records:
        clean_record = {}
        for key, value in record.items():
            if value is not None and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                clean_record[key] = value
            else:
                clean_record[key] = None
        valid_records.append(clean_record)
    
    log.info(f"\n📤 Uploading {len(valid_records)} records to Supabase...")
    
    # Upsert in batches
    batch = []
    total_upserted = 0
    
    for record in valid_records:
        batch.append(record)
        if len(batch) >= BATCH_SIZE:
            total_upserted = upsert_batch(supabase, batch, total_upserted)
            batch = []
    
    # Upsert remaining records
    total_upserted = upsert_batch(supabase, batch, total_upserted)
    
    log.info(f"\n{'='*80}")
    log.info(f"✅ UPSERT COMPLETE!")
    log.info(f"{'='*80}")
    log.info(f"Total rows upserted: {total_upserted}")
    log.info(f"Table: player_tracking_stats")
    
    # Show sample of what was uploaded
    print("\n📋 Sample of uploaded data (first 5 rows):")
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 200)
    print(final_df.head().to_string())

if __name__ == "__main__":
    main()