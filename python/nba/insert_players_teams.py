# GETS ALL PLAYERS FROM NBA API FOR A GIVEN SEASON

import os
import time
import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonplayerinfo, commonallplayers
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase Client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Helper Functions for Batch Operations
def chunk_list(data, chunk_size=100):
    """Yield successive chunks from a list for batch processing."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def get_existing_bbref_ids():
    """Fetch existing bball_ref_id values so we don't overwrite them."""
    response = supabase.table('players').select('nba_api_id, bball_ref_id').execute()
    return {row['nba_api_id']: row['bball_ref_id'] for row in response.data}

def upsert_data(table_name, records, chunk_size=100):
    """Perform an upsert (insert or update) on a table in batches."""
    if not records:
        return
    all_errors = []
    for batch in chunk_list(records, chunk_size):
        try:
            supabase.table(table_name).upsert(batch, on_conflict='nba_api_id').execute()
        except Exception as e:
            print(f"Error upserting batch to {table_name}: {e}")
            all_errors.append(e)
    if all_errors:
        print(f"Completed with {len(all_errors)} errors for table {table_name}.")
    else:
        print(f"Successfully upserted {len(records)} records to {table_name}.")

# --- Step 1: Insert / Update Teams ---
def insert_teams():
    print("Fetching teams from NBA API...")
    all_teams = teams.get_teams()
    records = []
    for team in all_teams:
        records.append({
            'nba_api_id': team['id'],
            'abbreviation': team['abbreviation'],
            'full_name': team['full_name'],
            'city': team['city'],
        })
    ''' print(f"Upserting {len(records)} teams into Supabase in batches...")
        upsert_data('teams', records)'''

# --- Step 2: Prepare Player Records ---
def prepare_player_records(season='2024-25'):
    """Fetch player data from NBA API for a specific season and prepare records for upsert."""
    team_id_map = get_team_id_map()
    existing_bbref_ids = get_existing_bbref_ids()

    # Get all players who were active in the given season
    all_players_data = commonallplayers.CommonAllPlayers(
        is_only_current_season=0,
        season=season,
        league_id='00'
    ).get_data_frames()[0]

    # Filter to players active in that season
    season_year = int(season[:4])
    all_players_data = all_players_data[
        (all_players_data['FROM_YEAR'].astype(int) <= season_year) &
        (all_players_data['TO_YEAR'].astype(int) >= season_year)
    ]

    # Build set of current active player IDs to determine is_active
    active_player_ids = {p['id'] for p in players.get_active_players()}

    player_records = []
    total = len(all_players_data)
    print(f"Preparing {total} player records for {season}...")

    for i, (_, row) in enumerate(all_players_data.iterrows()):
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=row['PERSON_ID'])
            d = info.get_data_frames()[0].iloc[0]

            # Height conversion "6-8" -> inches
            height_inches = None
            if pd.notna(d['HEIGHT']) and '-' in str(d['HEIGHT']):
                feet, inches = str(d['HEIGHT']).split('-')
                height_inches = int(feet) * 12 + int(inches)

            # Birth date
            birth_date = None
            if pd.notna(d['BIRTHDATE']):
                birth_date = str(d['BIRTHDATE'])[:10]

            # Team mapping — only set team_id if player is currently active
            nba_team_id = d['TEAM_ID']
            team_id_val = None
            if pd.notna(nba_team_id) and int(nba_team_id) != 0:
                team_id_val = team_id_map.get(int(nba_team_id))

            is_active = int(row['PERSON_ID']) in active_player_ids

            record = {
                'nba_api_id':    int(row['PERSON_ID']),
                'first_name':    row['DISPLAY_FIRST_LAST'].split(' ')[0],
                'last_name':     ' '.join(row['DISPLAY_FIRST_LAST'].split(' ')[1:]),
                'display_name':  row['DISPLAY_FIRST_LAST'],
                'birth_date':    birth_date,
                'height_inches': height_inches,
                'weight_lbs':    int(d['WEIGHT']) if pd.notna(d['WEIGHT']) else None,
                'draft_year':    str(d['DRAFT_YEAR']) if pd.notna(d['DRAFT_YEAR']) else None,
                'bball_ref_id': existing_bbref_ids.get(int(row['PERSON_ID'])),
                'position':      d['POSITION'] if pd.notna(d['POSITION']) else None,
                'team_id':       team_id_val if is_active else None,
                'status':        'active',   # reserved for injury status
                'is_active':     is_active,
            }
            player_records.append(record)
            print(f"  Processed [{i+1}/{total}] {row['DISPLAY_FIRST_LAST']} "
                  f"({'active' if is_active else 'inactive'})")

            time.sleep(0.6)  # Rate limiting

        except Exception as e:
            print(f"  ERROR processing {row['DISPLAY_FIRST_LAST']}: {e}")
            continue

    return player_records

def get_team_id_map():
    """Fetch the mapping of NBA API team IDs to our internal 'teams' table ID."""
    response = supabase.table('teams').select('id, nba_api_id').execute()
    return {row['nba_api_id']: row['id'] for row in response.data}

def insert_players(season='2025-26'):
    player_records = prepare_player_records(season)
    if not player_records:
        print("No player records to upsert.")
        return
    print(f"Upserting {len(player_records)} players into Supabase in batches...")
    upsert_data('players', player_records)

# --- Step 3: Update bbref_id from CSV (if available) ---
# def update_bbref_ids():
#     merged_path = "data/processed/players_merged.csv"
#     if not os.path.exists(merged_path):
#         print("No merged CSV found. Skipping bbref_id update.")
#         return
#     df = pd.read_csv(merged_path)
#     df = df[df['bbref_id'].notna()]
#     print(f"Updating bbref_id for {len(df)} players...")
#     for _, row in df.iterrows():
#         try:
#             supabase.table('players').update({
#                 'bball_ref_id': row['bbref_id']
#             }).eq('nba_api_id', int(row['nba_id'])).execute()
#         except Exception as e:
#             print(f"Error updating bbref_id for NBA ID {row['nba_id']}: {e}")
#     print("bbref ID update complete.")

if __name__ == "__main__":
    seasons = ['2022-23', '2023-24', '2024-25', '2025-26']
    print("🚀 Starting NBA data import (Supabase API version)...\n")
    insert_teams()
    for season in seasons:
        print(f"\n📅 Processing season: {season}")
        insert_players(season)
    # update_bbref_ids()
    print("\n✨ All done!")