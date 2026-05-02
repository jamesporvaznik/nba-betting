import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

# Load only Supabase credentials from .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Constant variables
CSV_PATH = "Player Shooting.csv"
TABLE_NAME = "players"           # Supabase table name
NAME_COLUMN = "display_name"     # column with player full name
BBALL_REF_COLUMN = "bball_ref_id" # column to store the ID

def get_existing_bbref_ids():
    """Fetch existing bball_ref_id values so we don't overwrite them."""
    response = supabase.table('players').select('nba_api_id, bball_ref_id, display_name').execute()
    return {
        row['display_name']: {
            'nba_api_id': row['nba_api_id'],
            'bball_ref_id': row['bball_ref_id']
        }
        for row in response.data
    }

if __name__ == "__main__":

    # Get existing basketball reference IDs
    existing_data = get_existing_bbref_ids()
    
    # Load the shooting CSV
    shot_df = pd.read_csv(CSV_PATH)

    # Keep only 2026 season
    shot_2026 = shot_df[shot_df['season'] == 2018]

    # Get unique player name + player_id pairs
    players_2026 = shot_2026[['player', 'player_id']].drop_duplicates()

    # Fetch all current players from your Supabase table
    response = supabase.table(TABLE_NAME).select(NAME_COLUMN).execute()
    supabase_players = response.data
    supabase_names = {p[NAME_COLUMN] for p in supabase_players}

    # Keep only players that exist in your Supabase table
    players_to_update = players_2026[players_2026['player'].isin(supabase_names)]

    # Split into players who need updates
    players_without_id = []
    players_with_mismatch = []
    
    for _, row in players_to_update.iterrows():
        player_name = row['player']
        csv_player_id = row['player_id']
        
        if player_name in existing_data:
            current_id = existing_data[player_name]['bball_ref_id']
            if current_id is None or current_id == "":
                # Player exists but has no bball_ref_id → needs update
                players_without_id.append((player_name, csv_player_id))
            elif current_id != csv_player_id:
                # Player has mismatched ID → optionally update (commented out by default)
                players_with_mismatch.append((player_name, csv_player_id, current_id))
        else:
            # Player not found in existing data (shouldn't happen due to filter above)
            pass

    # Update players without bball_ref_id
    updated_count = 0
    for player_name, csv_player_id in players_without_id:
        result = supabase.table(TABLE_NAME)\
            .update({BBALL_REF_COLUMN: csv_player_id})\
            .eq(NAME_COLUMN, player_name)\
            .execute()
        
        if hasattr(result, 'error') and result.error:
            print(f"Error updating {player_name}: {result.error}")
        else:
            updated_count += 1
            print(f"✓ Updated {player_name} with ID: {csv_player_id}")

    print(f"\n✅ Updated {updated_count} players with missing bball_ref_id.")
    print(f"❌ Skipped {len(players_with_mismatch)} players with existing IDs (mismatched)")
