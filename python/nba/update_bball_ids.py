import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

# Load only Supabase credentials from .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env file")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Hardcoded values (adjust these as needed) ---
CSV_PATH = "Player Shooting.csv"
TABLE_NAME = "players"           # your Supabase table name
NAME_COLUMN = "display_name"             # column with player full name
BBALL_REF_COLUMN = "bball_ref_id" # column to store the ID

# 1. Load the shooting CSV
shot_df = pd.read_csv(CSV_PATH)

# 2. Keep only 2026 season
shot_2026 = shot_df[shot_df['season'] == 2026]

# 3. Get unique player name + player_id pairs
players_2026 = shot_2026[['player', 'player_id']].drop_duplicates()

# 4. Fetch all current players from your Supabase table
response = supabase.table(TABLE_NAME).select(NAME_COLUMN).execute()
supabase_players = response.data
supabase_names = {p[NAME_COLUMN] for p in supabase_players}

# 5. Keep only players that exist in your Supabase table
players_to_update = players_2026[players_2026['player'].isin(supabase_names)]

# 6. Update the bball_ref_id for each matching player
for _, row in players_to_update.iterrows():
    supabase.table(TABLE_NAME)\
        .update({BBALL_REF_COLUMN: row['player_id']})\
        .eq(NAME_COLUMN, row['player'])\
        .execute()

print(f"Updated {len(players_to_update)} players with bball_ref_id.")