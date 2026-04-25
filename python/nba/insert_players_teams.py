# python/nba/insert_player_teams_supabase.py

#GETS ALL PLAYERS FROM NPA API, BUT DOESN'T GET BASKETBALL REFERENCE IDS

import os
import time
import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonplayerinfo
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# --- Initialize Supabase Client ---
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# --- Helper Functions for Batch Operations ---
def chunk_list(data, chunk_size=100):
    """Yield successive chunks from a list for batch processing."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def upsert_data(table_name, records, chunk_size=100):
    """Perform an upsert (insert or update) on a table in batches."""
    if not records:
        return
    all_errors = []
    for batch in chunk_list(records, chunk_size):
        try:
            supabase.table(table_name).upsert(batch).execute()
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
    print(f"Upserting {len(records)} teams into Supabase in batches...")
    upsert_data('teams', records)

# --- Step 2: Prepare Player Records ---
def prepare_player_records():
    """Fetch player data from NBA API and prepare a list of dictionaries for an upsert."""
    team_id_map = get_team_id_map()
    all_players = players.get_active_players()
    player_records = []
    print(f"Preparing {len(all_players)} player records...")

    for i, player in enumerate(all_players):
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=player['id'])
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

            # Team mapping
            nba_team_id = d['TEAM_ID']
            team_id_val = None
            if pd.notna(nba_team_id):
                team_id_val = team_id_map.get(int(nba_team_id))

            record = {
                'nba_api_id': player['id'],
                'first_name': player['first_name'],
                'last_name': player['last_name'],
                'display_name': player['full_name'],
                'birth_date': birth_date,
                'height_inches': height_inches,
                'weight_lbs': int(d['WEIGHT']) if pd.notna(d['WEIGHT']) else None,
                'draft_year': str(d['DRAFT_YEAR']) if pd.notna(d['DRAFT_YEAR']) else None,
                'position': d['POSITION'] if pd.notna(d['POSITION']) else None,
                'team_id': team_id_val,
                'status': 'active',
            }
            player_records.append(record)
            print(f"  Processed [{i+1}/{len(all_players)}] {player['full_name']}")

            time.sleep(0.6)  # Rate limiting to be nice to the NBA API

        except Exception as e:
            print(f"  ERROR processing {player['full_name']}: {e}")
            # Continue to the next player
            continue

    return player_records

def get_team_id_map():
    """Fetch the mapping of NBA API team IDs to our internal 'teams' table ID."""
    response = supabase.table('teams').select('id, nba_api_id').execute()
    # Return a dictionary like { nba_api_id: our_teams_id }
    return {row['nba_api_id']: row['id'] for row in response.data}

def insert_players():
    player_records = prepare_player_records()
    if not player_records:
        print("No player records to upsert.")
        return
    # Upsert all player records in batches
    print(f"Upserting {len(player_records)} players into Supabase in batches...")
    upsert_data('players', player_records)

# --- Step 3: Update bbref_id from CSV (if available) ---
def update_bbref_ids():
    merged_path = "data/processed/players_merged.csv"
    if not os.path.exists(merged_path):
        print("No merged CSV found. Skipping bbref_id update.")
        return

    df = pd.read_csv(merged_path)
    df = df[df['bbref_id'].notna()]
    print(f"Updating bbref_id for {len(df)} players...")

    for _, row in df.iterrows():
        try:
            supabase.table('players').update({
                'bball_ref_id': row['bbref_id']
            }).eq('nba_api_id', int(row['nba_id'])).execute()
        except Exception as e:
            print(f"Error updating bbref_id for NBA ID {row['nba_id']}: {e}")

    print("bbref ID update complete.")

if __name__ == "__main__":
    print("🚀 Starting NBA data import (Supabase API version)...\n")
    insert_teams()
    insert_players()
    update_bbref_ids()
    print("\n✨ All done!")