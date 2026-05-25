"""
view_clusters.py
----------------
View cluster assignments with player names.
"""

import pandas as pd
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load clusters
clusters = pd.read_csv("player_archetypes_assigned.csv")

print(f"Columns in clusters file: {clusters.columns.tolist()}")
print(f"Total players: {len(clusters)}")

# Get actual player names from Supabase
print("\n📡 Fetching player names from Supabase...")

# Check what column names exist in your players table
try:
    # First, check what columns are available
    test_query = supabase.table("players").select("*").limit(1).execute()
    if test_query.data:
        available_cols = list(test_query.data[0].keys())
        print(f"Available columns in players table: {available_cols}")
        
        # Find the right column names
        id_col = None
        name_col = None
        
        if 'player_id' in available_cols:
            id_col = 'player_id'
        elif 'nba_api_id' in available_cols:
            id_col = 'nba_api_id'
        elif 'id' in available_cols:
            id_col = 'id'
            
        if 'player_name' in available_cols:
            name_col = 'player_name'
        elif 'display_name' in available_cols:
            name_col = 'display_name'
        elif 'full_name' in available_cols:
            name_col = 'full_name'
        elif 'name' in available_cols:
            name_col = 'name'
        
        if id_col and name_col:
            # Fetch all player names
            player_ids = clusters['player_id'].tolist()
            response = supabase.table("players").select(f"{id_col}, {name_col}").in_(id_col, player_ids).execute()
            players_df = pd.DataFrame(response.data)
            
            # Rename to standard columns
            players_df = players_df.rename(columns={id_col: 'player_id', name_col: 'player_name'})
            print(f"✅ Found {len(players_df)} player names")
        else:
            print(f"⚠️ Could not find appropriate columns. id_col={id_col}, name_col={name_col}")
            players_df = pd.DataFrame({'player_id': clusters['player_id'], 'player_name': clusters['player_name']})
    else:
        print("⚠️ Players table is empty or inaccessible")
        players_df = pd.DataFrame({'player_id': clusters['player_id'], 'player_name': clusters['player_name']})
        
except Exception as e:
    print(f"⚠️ Error fetching from Supabase: {e}")
    print("Using placeholder names...")
    players_df = pd.DataFrame({'player_id': clusters['player_id'], 'player_name': clusters['player_name']})

# Merge with clusters
result = clusters.merge(players_df, on='player_id', how='left')

# If merge created duplicate columns, fix
if 'player_name_x' in result.columns and 'player_name_y' in result.columns:
    result['player_name'] = result['player_name_y'].fillna(result['player_name_x'])
    result = result.drop(['player_name_x', 'player_name_y'], axis=1)
elif 'player_name_x' in result.columns:
    result['player_name'] = result['player_name_x']
    result = result.drop('player_name_x', axis=1)

# Show sample from each cluster
print("\n" + "="*60)
print("📊 SAMPLE PLAYERS FROM EACH CLUSTER")
print("="*60)

for cluster_id in sorted(result['primary_archetype'].unique()):
    cluster_players = result[result['primary_archetype'] == cluster_id]['player_name'].dropna().head(5).tolist()
    
    # Also get top player with highest weight for this archetype
    weight_col = f'archetype_{cluster_id}_weight'
    if weight_col in result.columns:
        top_player = result.nlargest(1, weight_col)
        top_name = top_player['player_name'].iloc[0] if not top_player.empty else "N/A"
        top_weight = top_player[weight_col].iloc[0] if not top_player.empty else 0
        
        print(f"\n⚡ Cluster {cluster_id} (Best Match: {top_name} @ {top_weight:.1%})")
    else:
        print(f"\n⚡ Cluster {cluster_id}:")
    
    for player in cluster_players[:5]:
        if player and player != "Unknown":
            print(f"   - {player}")

# Save full results with real names
output_cols = ['player_id', 'player_name', 'primary_archetype']

# Add top 3 archetype weights for context
weight_cols = [col for col in result.columns if 'archetype_' in col and 'weight' in col]
if weight_cols:
    output_cols.extend(weight_cols[:7])

# Add purity if it exists
if 'archetype_purity' in result.columns:
    output_cols.append('archetype_purity')
else:
    # Calculate purity from weights
    weight_cols_all = [col for col in result.columns if col.startswith('archetype_') and col.endswith('_weight')]
    if weight_cols_all:
        result['archetype_purity'] = result[weight_cols_all].max(axis=1)
        output_cols.append('archetype_purity')

result[output_cols].to_csv("player_clusters_with_names.csv", index=False)
print("\n" + "="*60)
print(f"✅ Saved to player_clusters_with_names.csv")
print(f"   Columns: {output_cols}")

# Show cluster size distribution
print("\n" + "="*60)
print("📊 CLUSTER SIZE DISTRIBUTION")
print("="*60)
cluster_sizes = result['primary_archetype'].value_counts().sort_index()
for cluster_id, size in cluster_sizes.items():
    pct = size / len(result) * 100
    bar = "█" * int(pct / 2)
    print(f"  Cluster {cluster_id}: {size:3d} players ({pct:5.1f}%) {bar}")

# Show purity distribution
print("\n" + "="*60)
print("📊 ARCHETYPE PURITY DISTRIBUTION")
print("="*60)
low_purity = result[result['archetype_purity'] < 0.85]
print(f"  High purity (>85%): {len(result) - len(low_purity)} players")
print(f"  Hybrid players (<85%): {len(low_purity)} players")

if not low_purity.empty:
    print("\n  Top hybrid players (needs scouting):")
    hybrids = low_purity.nsmallest(5, 'archetype_purity')[['player_name', 'primary_archetype', 'archetype_purity']]
    for _, row in hybrids.iterrows():
        if row['player_name'] and row['player_name'] != "Unknown":
            print(f"    - {row['player_name']} (Cluster {row['primary_archetype']}, Purity: {row['archetype_purity']:.1%})")

# Show the most confident player from each cluster
print("\n" + "="*60)
print("🏆 MOST CONFIDENT PLAYER PER CLUSTER")
print("="*60)
for cluster_id in sorted(result['primary_archetype'].unique()):
    weight_col = f'archetype_{cluster_id}_weight'
    if weight_col in result.columns:
        top_player = result.nlargest(1, weight_col)
        if not top_player.empty:
            player_name = top_player['player_name'].iloc[0]
            weight = top_player[weight_col].iloc[0]
            if player_name and player_name != "Unknown":
                print(f"  Cluster {cluster_id}: {player_name} ({weight:.1%})")

print("\n✅ Done!")