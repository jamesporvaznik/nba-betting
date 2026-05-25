"""
features/build_player_cluster_features.py
------------------------------------------
Builds feature matrix for player clustering using data from Supabase.
Features include: play type, shot location, and tracking stats.
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.preprocessing import StandardScaler
from skbio.stats.composition import multi_replace, ilr

load_dotenv()

# Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def get_supabase_client() -> Client:
    """Create and return Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_playtype_data(supabase: Client, season: str) -> pd.DataFrame:
    """
    Fetch player play type stats from Supabase.
    Returns pivoted DataFrame with grouped play types.
    """
    # Implement pagination to get ALL rows
    all_data = []
    page = 0
    page_size = 1000
    
    print(f"  Fetching playtype data for {season}...")
    
    while True:
        query = supabase.table("player_playtype_stats")\
            .select("player_id, play_type, poss_pct")\
            .eq("season", season)\
            .range(page * page_size, (page + 1) * page_size - 1)\
            .execute()
        
        if not query.data:
            break
            
        all_data.extend(query.data)
        
        # If we got less than page_size, we've reached the end
        if len(query.data) < page_size:
            break
            
        page += 1
    
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print(f"  No playtype data for {season}")
        return pd.DataFrame()
    
    print(f"  Total rows fetched: {len(df)}")
    
    # DEBUG: Show all unique play types in the data
    print(f"\n  All play types in data: {sorted(df['play_type'].unique())}")

    # Define play type groupings (including Misc)
    playtype_groups = {
        'ball_dominant_creation': ['Isolation', 'PRBallHandler'],
        'stationary_spacing': ['Spotup'],
        'movement_shooting': ['OffScreen', 'Handoff'],
        'rim_cuts_and_putbacks': ['Cut', 'OffRebound'],
        'interior_big_actions': ['Postup', 'PRRollMan'],
        'transition': ['Transition'],
        'misc': ['Misc']
    }
    
    # Map each play type to its group
    playtype_to_group = {}
    for group, types in playtype_groups.items():
        for play_type in types:
            playtype_to_group[play_type] = group
    
    # Add group column
    df['group'] = df['play_type'].map(playtype_to_group)
    
    # Check for any unmapped play types (shouldn't happen now that Misc is included)
    unmapped = df[df['group'].isna()]['play_type'].unique()
    if len(unmapped) > 0:
        print(f"    Warning: Unmapped play types found: {list(unmapped)}")
    
    # Aggregate by player and group (sum poss_pct within each group)
    grouped_df = df.groupby(['player_id', 'group'])['poss_pct'].sum().reset_index()
    
    # Pivot
    result = grouped_df.pivot_table(
        index='player_id',
        columns='group',
        values='poss_pct',
        fill_value=0
    )
    
    # Add prefix to column names
    result.columns = [f'poss_pct_{col}' for col in result.columns]
    
    # Move player_id from index to column
    result = result.reset_index()
    
    # print(f"    Pivoted {len(df)} rows into {len(result)} players with {len(result.columns)-1} grouped features")
    # print(f"    Groups: {list(result.columns)[1:]}")
    
    # Optional: Verify that poss_pct sums to approximately 1 for each player
    poss_pct_cols = [col for col in result.columns if col.startswith('poss_pct_')]
    sums = result[poss_pct_cols].sum(axis=1)
    # print(f"    Poss_pct sum range: {sums.min():.3f} - {sums.max():.3f} (should be ~1.0)")
    
    return result

def fetch_shot_location_data(supabase: Client, season: str) -> pd.DataFrame:
    """
    Fetch player shot location stats from Supabase.
    Returns pivoted DataFrame with 3 consolidated zones: Rim, Mid-Range, Three.
    """
    # Implement pagination to get ALL rows
    all_data = []
    page = 0
    page_size = 1000
    
    print(f"  Fetching shot location data for {season}...")
    
    while True:
        query = supabase.table("player_shot_locations")\
            .select("player_id, shot_zone, frequency_pct")\
            .eq("season", season)\
            .range(page * page_size, (page + 1) * page_size - 1)\
            .execute()
        
        if not query.data:
            break
            
        all_data.extend(query.data)
        
        # If we got less than page_size, we've reached the end
        if len(query.data) < page_size:
            break
            
        page += 1
    
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print(f"  No shot location data for {season}")
        return pd.DataFrame()
    
    print(f"  Total rows fetched: {len(df)}")
    
    # DEBUG: Show all unique shot zones in the data
    print(f"\n  All shot zones in data: {sorted(df['shot_zone'].unique())}")
    
    # Define zone groupings
    zone_groups = {
        'rim': ['Restricted Area'],
        'mid_range': ['In The Paint (Non-RA)', 'Mid-Range'],
        'three': ['Left Corner 3', 'Right Corner 3', 'Above the Break 3']
    }
    
    # Map each shot zone to its group
    zone_to_group = {}
    for group, zones in zone_groups.items():
        for zone in zones:
            zone_to_group[zone] = group
    
    # Add group column
    df['zone_group'] = df['shot_zone'].map(zone_to_group)
    
    # Check for any unmapped shot zones
    unmapped = df[df['zone_group'].isna()]['shot_zone'].unique()
    if len(unmapped) > 0:
        print(f"    Warning: Unmapped shot zones found: {list(unmapped)}")
        # Drop unmapped zones
        df = df.dropna(subset=['zone_group'])
    
    # Aggregate by player and zone group (sum frequency_pct within each group)
    grouped_df = df.groupby(['player_id', 'zone_group'])['frequency_pct'].sum().reset_index()
    
    # Pivot
    result = grouped_df.pivot_table(
        index='player_id',
        columns='zone_group',
        values='frequency_pct',
        fill_value=0
    )
    
    # Add prefix to column names
    result.columns = [f'freq_{col}' for col in result.columns]
    
    # Ensure all three zones exist (add missing ones with 0)
    for zone in ['rim', 'mid_range', 'three']:
        if f'freq_{zone}' not in result.columns:
            result[f'freq_{zone}'] = 0
    
    # Reorder columns to have consistent order
    result = result[['freq_rim', 'freq_mid_range', 'freq_three']]
    
    # Move player_id from index to column
    result = result.reset_index()
    
    print(f"    Consolidated {len(df)} rows into {len(result)} players with 3 zone features")
    print(f"    Zones: rim ({result['freq_rim'].mean():.2f}%), mid-range ({result['freq_mid_range'].mean():.2f}%), three ({result['freq_three'].mean():.2f}%)")
    
    # Optional: Verify that frequencies sum to approximately 100 for each player
    freq_cols = ['freq_rim', 'freq_mid_range', 'freq_three']
    sums = result[freq_cols].sum(axis=1)
    print(f"    Frequency sum range: {sums.min():.2f}% - {sums.max():.2f}% (should be ~100%)")
    
    return result
    

def fetch_tracking_data(supabase: Client, season: str) -> pd.DataFrame:
    """
    Fetch player tracking stats from Supabase.
    Already has one row per player.
    """
    # Implement pagination to get ALL rows
    all_data = []
    page = 0
    page_size = 1000
    
    print(f"  Fetching tracking data for {season}...")
    
    while True:
        query = supabase.table("player_tracking_stats")\
            .select("*")\
            .eq("season", season)\
            .range(page * page_size, (page + 1) * page_size - 1)\
            .execute()
        
        if not query.data:
            break
            
        all_data.extend(query.data)
        
        # If we got less than page_size, we've reached the end
        if len(query.data) < page_size:
            break
            
        page += 1
    
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print(f"  No tracking data for {season}")
        return pd.DataFrame()
    
    print(f"  Total rows fetched: {len(df)}")
    
    result = pd.DataFrame()

    # Add player_id
    result['player_id'] = df['player_id']

    # Ball-Handling Metrics
    # result['avg_drib_per_touch'] = df['avg_drib_per_touch']
    result['avg_sec_per_touch'] = df['avg_sec_per_touch']

    # Passing & Vision
    # result['ast_to_pass_pct'] = df['ast_to_pass_pct']
    result['potential_ast_rate'] = df['potential_ast'].divide(df['passes_made']).fillna(0)

    # Perimeter Shot Profile
    result['catch_shoot_fga_per_touch_pct'] = df['catch_shoot_fga'].divide(df['touches']).fillna(0)
    result['catch_shoot_fg3a_per_catch_shoot_fga_pct'] = df['catch_shoot_fg3a'].divide(df['catch_shoot_fga']).fillna(0)
    result['pull_up_fga_per_touch_pct'] = df['pull_up_fga'].divide(df['touches']).fillna(0)
    result['pull_up_fg3a_per_pull_up_fga_pct'] = df['pull_up_fg3a'].divide(df['pull_up_fga']).fillna(0)

    # Slasher Choices
    result['drive_fga_per_touch_pct'] = df['drive_fga'].divide(df['touches']).fillna(0)
    result['drive_fga_per_drive_pct'] = df['drive_fga'].divide(df['drives']).fillna(0)
    result['drive_passes_per_drive_pct'] = df['drive_passes'].divide(df['drives']).fillna(0)
    # result['drive_ast_per_drive_pass_pct'] = df['drive_ast'].divide(df['drive_passes']).fillna(0)

    # Rebounding Domain
    result['reb_chance_pct'] = df['reb_chance_pct']
    # result['dreb_chance_pct'] = df['dreb_chance_pct']
    result['oreb_chance_pct'] = df['oreb_chance_pct']
    # result['reb_contest_pct'] = df['reb_contest_pct']

    # Clean up any infinite values from players with zero denominators
    result.replace([np.inf, -np.inf], 0, inplace=True)
    
    print(f"    Created {len(result.columns)-1} tracking features for {len(result)} players")
    print(f"    Features: {list(result.columns)[1:]}")

    return result

# Put this after your other fetch functions
def fetch_qualified_players(supabase: Client, season: str, min_minutes=500) -> set:
    """
    Fetch player IDs who have played at least min_minutes in the season.
    Aggregates minutes from player_gamelog table.
    """
    # Implement pagination to get ALL game logs
    all_data = []
    page = 0
    page_size = 1000
    
    print(f"  Fetching game logs for {season} to calculate minutes...")
    
    while True:
        query = supabase.table("player_gamelog")\
            .select("playerid, minutes")\
            .eq("season", season)\
            .range(page * page_size, (page + 1) * page_size - 1)\
            .execute()
        
        if not query.data:
            break
            
        all_data.extend(query.data)
        
        if len(query.data) < page_size:
            break
            
        page += 1
    
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print(f"  No game log data for {season}")
        return set()
    
    print(f"  Total game log rows fetched: {len(df)}")
    
    # Convert minutes to numeric (handle potential string values like '12:34')
    if df['minutes'].dtype == 'object':
        # Handle minutes in format "MM:SS"
        def convert_minutes(min_str):
            if pd.isna(min_str) or min_str == '':
                return 0
            if ':' in str(min_str):
                parts = str(min_str).split(':')
                return float(parts[0]) + float(parts[1]) / 60
            return float(min_str)
        
        df['minutes'] = df['minutes'].apply(convert_minutes)
    else:
        df['minutes'] = df['minutes'].fillna(0)
    
    # Aggregate minutes by player
    player_minutes = df.groupby('playerid')['minutes'].sum().reset_index()
    
    # Filter players who meet the minute threshold
    qualified_players = player_minutes[player_minutes['minutes'] >= min_minutes]['playerid'].tolist()
    
    print(f"  Found {len(qualified_players)} players with >= {min_minutes} minutes")
    print(f"  Minutes range: {player_minutes['minutes'].min():.0f} - {player_minutes['minutes'].max():.0f}")
    
    return set(qualified_players)

def normalize_and_merge(df_playtypes: pd.DataFrame, df_tracking: pd.DataFrame) -> tuple:
    """
    Build normalized feature matrix from play type and tracking data.
    """
    # --- 1. Align Dataframes via Inner Join ---
    df_pt = df_playtypes.set_index('player_id')
    df_tr = df_tracking.set_index('player_id')
    df_aligned = df_pt.join(df_tr, how='inner', rsuffix='_track')
    print(f"Aligned players: {len(df_aligned)}")

    # --- 2. Define Feature Lists ---
    playtype_cols = [
        'poss_pct_ball_dominant_creation', 'poss_pct_interior_big_actions', 
        'poss_pct_movement_shooting', 'poss_pct_rim_cuts_and_putbacks', 
        'poss_pct_stationary_spacing', 'poss_pct_transition', 'poss_pct_misc'
    ]
    
    tracking_cols = [
        'avg_sec_per_touch', 'potential_ast_rate', 'catch_shoot_fga_per_touch_pct', 'catch_shoot_fg3a_per_catch_shoot_fga_pct',
        'pull_up_fga_per_touch_pct', 'pull_up_fg3a_per_pull_up_fga_pct', 'drive_fga_per_touch_pct', 'drive_fga_per_drive_pct', 
        'drive_passes_per_drive_pct', 'reb_chance_pct', 'oreb_chance_pct'
    ]

    # Verify columns exist
    missing_pt = [col for col in playtype_cols if col not in df_aligned.columns]
    missing_tr = [col for col in tracking_cols if col not in df_aligned.columns]
    if missing_pt:
        print(f"Warning: Missing playtype columns: {missing_pt}")
    if missing_tr:
        print(f"Warning: Missing tracking columns: {missing_tr}")

    # Use only available columns
    playtype_cols = [col for col in playtype_cols if col in df_aligned.columns]
    tracking_cols = [col for col in tracking_cols if col in df_aligned.columns]

    # --- 3. Run ILR Log-Ratio on Play-Types ---
    X_pt_raw = df_aligned[playtype_cols].to_numpy()
    X_pt_clean = multi_replace(X_pt_raw)
    X_pt_ilr = ilr(X_pt_clean)  # Isometric Log-Ratio transform (k play types -> k-1 features)
    print(f"ILR shape: {X_pt_ilr.shape} (expected: {len(playtype_cols)-1} components)")

    # --- 4. Process Tracking Data ---
    df_tr_clean = df_aligned[tracking_cols].copy()
    
    # Apply natural log to smooth out right-skewed guard touch times 
    if 'avg_sec_per_touch' in df_tr_clean.columns:
        df_tr_clean['avg_sec_per_touch'] = np.log(df_tr_clean['avg_sec_per_touch'] + 1e-5)
        
    X_tracking = df_tr_clean.to_numpy()

    # --- 5. Concatenate and Apply Global StandardScaler ---
    X_combined = np.hstack((X_pt_ilr, X_tracking))
    scaler = StandardScaler()
    X_final_scaled = scaler.fit_transform(X_combined)

    # --- 6. Create Final DataFrame with Player IDs ---
    ilr_names = [f'ilr_pt_{i}' for i in range(X_pt_ilr.shape[1])]
    column_names = ilr_names + tracking_cols
    
    final_features = pd.DataFrame(
        X_final_scaled, 
        columns=column_names, 
        index=df_aligned.index
    )
    final_features.index.name = 'player_id'
    final_features = final_features.reset_index()

    print("\n" + "="*50)
    print("✅ Pipeline matrix execution complete.")
    print(f"Shape of final features: {final_features.shape}")
    print(f"Expected: ({len(df_aligned)}, {1 + (len(playtype_cols)-1) + len(tracking_cols)})")
    print("="*50)
    
    print("\nFeature summary:")
    print(f" - {len(ilr_names)} ILR components from {len(playtype_cols)} play types")
    print(f" - {len(tracking_cols)} tracking features")
    print(f" - Total features: {len(ilr_names) + len(tracking_cols)}")

    return final_features, scaler


# Then update your build_feature_matrix function
def build_feature_matrix(season: str = "2022-23", min_minutes: int = 500):
    """
    Main function: builds complete feature matrix for player clustering.
    Filters players who played at least min_minutes.
    """
    print(f"\n{'='*60}")
    print(f"Building player feature matrix for {season}")
    print(f"{'='*60}")
    
    supabase = get_supabase_client()
    
    # First, get qualified players
    print("\n0. Determining qualified players...")
    qualified_players = fetch_qualified_players(supabase, season, min_minutes)
    
    if not qualified_players:
        print("  No qualified players found!")
        return pd.DataFrame(), None
    
    print("\n1. Fetching play type data...")
    playtype_df = fetch_playtype_data(supabase, season)

    # Filter to qualified players
    if not playtype_df.empty:
        playtype_df = playtype_df[playtype_df['player_id'].isin(qualified_players)]
    print(f"   Shape: {playtype_df.shape}")
    
    # print("\n2. Fetching shot location data...")
    # shot_df = fetch_shot_location_data(supabase, season)
    # # Filter to qualified players
    # if not shot_df.empty:
    #     shot_df = shot_df[shot_df['player_id'].isin(qualified_players)]
    #     print(len(shot_df))
    # print(f"   Shape: {shot_df.shape}")
    
    print("\n3. Fetching tracking data...")
    tracking_df = fetch_tracking_data(supabase, season)
    # Filter to qualified players
    if not tracking_df.empty:
        tracking_df = tracking_df[tracking_df['player_id'].isin(qualified_players)]
        print(len(tracking_df))
    print(f"   Shape: {tracking_df.shape}")

    # Get common players
    common_players = set(playtype_df['player_id'])& set(tracking_df['player_id']) 

    # Filter both datasets
    playtype_df = playtype_df[playtype_df['player_id'].isin(common_players)]
    tracking_df = tracking_df[tracking_df['player_id'].isin(common_players)]
    # shot_df =  shot_df[shot_df['player_id'].isin(common_players)]

    features, scaler = normalize_and_merge(playtype_df, tracking_df)

    # Now you can use them:
    print(features.head())  # View the feature matrix
    print(features.shape)   # Should be (n_players, n_features)

    # Save for later use
    features.to_csv("player_features_2022-23.csv", index=False)
    import joblib
    joblib.dump(scaler, "feature_scaler_2022-23.pkl")

    return features, scaler

if __name__ == "__main__":
    # Test with current season
    features, scaler = build_feature_matrix("2022-23")
    
    print("\n📋 Feature preview:")
    print(features.head(10).to_string())
    
    print(f"\n📊 Feature statistics:")
    print(f"   Total players: {len(features)}")
    print(f"   Total features: {len(features.columns) - 1}")  # Excluding player_id