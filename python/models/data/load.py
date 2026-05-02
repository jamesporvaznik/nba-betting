import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

# Create supabase connection
load_dotenv()
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Seasons from which im importing the data and how im using them
TRAINING_SEASONS = ['2022-23', '2023-24', '2024-25']
TEST_SEASON      = '2025-26'

# Load the player gamelogs from players who played more than 20 games in a season and averaged over 10 mins/game
def load_gamelogs(seasons: list[str] = None, min_games_per_season: int = 20, min_minutes: float = 10.0) -> pd.DataFrame:
    """Load player gamelogs from Supabase for given seasons with pagination."""
    if seasons is None:
        seasons = TRAINING_SEASONS

    print(f"Loading gamelogs for seasons: {seasons}...")
    
    all_data = []
    page = 0
    page_size = 1000
    
    while True:
        response = (
            supabase.table('player_gamelog')
            .select('*')
            .in_('season', seasons)
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        
        if not response.data:
            break
            
        all_data.extend(response.data)
        print(f"  Fetched page {page + 1}: {len(response.data)} rows (total: {len(all_data)})")
        
        if len(response.data) < page_size:
            break
            
        page += 1
    
    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df):,} raw game logs")
    
    # Filter by minimum minutes per game
    if min_minutes > 0:
        before = len(df)
        df = df[df['minutes'] >= min_minutes]
        print(f"  Filtered out {before - len(df):,} rows with <{min_minutes} minutes played")
    
    # Filter players with minimum games per season
    if min_games_per_season > 0:
        game_counts = df.groupby(['playerid', 'season']).size().reset_index(name='games')
        valid_players = game_counts[game_counts['games'] >= min_games_per_season]
        df = df.merge(valid_players[['playerid', 'season']], on=['playerid', 'season'])
        print(f"  Retained {len(df):,} rows from players with ≥{min_games_per_season} games/season")
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['playerid', 'date']).reset_index(drop=True)
        print(f"  Players: {df['playerid'].nunique()} | Seasons: {df['season'].unique()}")
    
    return df

def load_players() -> pd.DataFrame:
    """Load player info from Supabase."""
    response = supabase.table('players') \
        .select('nba_api_id, display_name, position, is_active') \
        .execute()
    return pd.DataFrame(response.data)

def load_teams() -> pd.DataFrame:
    """Load team info from Supabase."""
    response = supabase.table('teams') \
        .select('nba_api_id, abbreviation, full_name') \
        .execute()
    return pd.DataFrame(response.data)
