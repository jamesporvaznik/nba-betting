import os
from functools import lru_cache
from pathlib import Path
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

@lru_cache(maxsize=1)
def get_client():
    """Connect only when loading data, without exposing credentials."""
    load_dotenv(Path(__file__).resolve().parents[2] / '.env')
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Seasons from which im importing the data and how im using them
TRAINING_SEASONS = ['2022-23', '2023-24', '2024-25']
TEST_SEASON      = '2025-26'

def load_gamelogs(seasons: list[str] = None) -> pd.DataFrame:
    """Load complete histories; apply pregame eligibility after engineering."""
    if seasons is None:
        seasons = TRAINING_SEASONS

    print(f"Loading gamelogs for seasons: {seasons}...")
    
    all_data = []
    page = 0
    page_size = 1000
    
    while True:
        response = (
            get_client().table('player_gamelog')
            .select('*')
            .in_('season', seasons)
            .order('gamelogid')
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
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['playerid', 'date']).reset_index(drop=True)
        print(f"  Players: {df['playerid'].nunique()} | Seasons: {df['season'].unique()}")
    
    return df

def load_players() -> pd.DataFrame:
    """Load player info from Supabase."""
    response = get_client().table('players') \
        .select('nba_api_id, display_name, position, is_active') \
        .execute()
    return pd.DataFrame(response.data)

def load_teams() -> pd.DataFrame:
    """Load team info from Supabase."""
    response = get_client().table('teams') \
        .select('nba_api_id, abbreviation, full_name') \
        .execute()
    return pd.DataFrame(response.data)

def load_clusters() -> pd.DataFrame:
    all_data = []
    page = 0
    page_size = 1000
    
    while True:
        response = (
            get_client().table('player_clusters')
            .select(','.join(['player_id', 'season', 'primary_archetype'] +
                             [f'archetype_{i}_weight' for i in range(9)]))
            .order('season').order('player_id')
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        
        if not response.data:
            break
            
        all_data.extend(response.data)
        
        if len(response.data) < page_size:
            break
            
        page += 1
    
    return pd.DataFrame(all_data)

def load_player_shots() -> pd.DataFrame:
    all_data = []
    page = 0
    page_size = 1000
    
    while True:
        response = (
            get_client().table('player_shot_locations')
            .select('player_id, season, shot_zone, fg_pct, frequency_pct') \
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        
        if not response.data:
            break
            
        all_data.extend(response.data)
        
        if len(response.data) < page_size:
            break
            
        page += 1
    
    return pd.DataFrame(all_data)

def load_team_shots() -> pd.DataFrame:

    response = get_client().table('team_defensive_shot_locations') \
        .select('team_id, season, shot_zone, fg_pct_allowed, frequency_pct') \
        .execute()
    return pd.DataFrame(response.data)

def load_player_plays() -> pd.DataFrame:
    all_data = []
    page = 0
    page_size = 1000
    
    while True:
        response = (
            get_client().table('player_playtype_stats')
            .select('player_id, season, play_type, fg_pct, points_per_possession, poss_pct') \
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        
        if not response.data:
            break
            
        all_data.extend(response.data)
        
        if len(response.data) < page_size:
            break
            
        page += 1
    
    return pd.DataFrame(all_data)

def load_team_plays() -> pd.DataFrame:
    all_data = []
    page = 0
    page_size = 1000
    
    while True:
        response = (
            get_client().table('team_defensive_playtype_stats')
            .select('team_id, season, play_type, fg_pct_allowed, points_allowed_per_possession, poss_pct') \
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        
        if not response.data:
            break
            
        all_data.extend(response.data)
        
        if len(response.data) < page_size:
            break
            
        page += 1
    
    return pd.DataFrame(all_data)

def load_team_defensive_stats() -> pd.DataFrame:
    all_data = []
    page = 0
    page_size = 1000
    
    while True:
        response = (
            get_client().table('team_defensive_stats')
            .select('*') \
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        
        if not response.data:
            break
            
        all_data.extend(response.data)
        
        if len(response.data) < page_size:
            break
            
        page += 1
    
    return pd.DataFrame(all_data)
