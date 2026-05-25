import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# STEP 1: LOAD AND INITIALIZE DATA MATRIX
final_features = pd.read_csv("player_features_2022-23.csv")
if 'Unnamed: 0' in final_features.columns:
    final_features = final_features.drop(columns=['Unnamed: 0'])

# Isolate numeric array for the GMM
X = final_features.drop(columns=['player_id']).to_numpy()


# STEP 2: TRAIN THE FINALIZED 8-CLUSTER GMM
print("🚀 Training finalized 8-cluster GMM model...")
final_gmm = GaussianMixture(n_components=8, covariance_type='diag', random_state=42)
final_gmm.fit(X)

# Extract hard labels and soft probability distributions
hard_labels = final_gmm.predict(X)
soft_probs = final_gmm.predict_proba(X)

# STEP 3: CREATE MASTER OPERATIONAL DataFrame
season = "2022-23"  # Set the season

df_clusters = pd.DataFrame({'player_id': final_features['player_id']})
df_clusters['season'] = season
df_clusters['primary_archetype'] = hard_labels

# Add dedicated columns for each archetype's soft weight percentage
for i in range(8):
    df_clusters[f'archetype_{i}_weight'] = soft_probs[:, i]

# STEP 4: UPSERT TO SUPABASE
print("\n" + "="*60)
print("📤 UPSERTING CLUSTER ASSIGNMENTS TO SUPABASE")
print("="*60)

# Convert DataFrame to list of dictionaries for upsert
records = df_clusters.to_dict('records')

# Add updated timestamp
for record in records:
    record['updated_at'] = datetime.now().isoformat()

# Upsert in batches
BATCH_SIZE = 100
table_name = "player_clusters"

total_upserted = 0
for i in range(0, len(records), BATCH_SIZE):
    batch = records[i:i+BATCH_SIZE]
    try:
        result = supabase.table(table_name).upsert(
            batch, 
            on_conflict="player_id,season"
        ).execute()
        
        total_upserted += len(batch)
        print(f"  Upserted batch {i//BATCH_SIZE + 1}: {len(batch)} rows (Total: {total_upserted})")
        
    except Exception as e:
        print(f"  ❌ Error upserting batch: {e}")

print(f"\n✅ Successfully upserted {total_upserted} records to {table_name}")

# STEP 5: PROFILED ROSTER PRINTING (THE EYE TEST)
print("\n" + "="*60)
print("📋 ARCHETYPE ROSTER PREVIEWS (TOP 5 MODEL PLAYERS PER CLUSTER)")
print("="*60)

for cluster_num in range(8):
    print(f"\n⚡ ARCHETYPE (CLUSTER) {cluster_num}:")
    
    # Filter for players whose primary cluster assignment is this one
    cluster_roster = df_clusters[df_clusters['primary_archetype'] == cluster_num]
    
    # Sort by who has the highest probability weight for this specific archetype
    top_players = cluster_roster.sort_values(by=f'archetype_{cluster_num}_weight', ascending=False).head(5)
    
    for idx, row in top_players.iterrows():
        weight_pct = row[f'archetype_{cluster_num}_weight'] * 100
        print(f"  - Player ID: {row['player_id']} ({weight_pct:.1f}% Match)")

# STEP 6: VERIFICATION
print("\n" + "="*60)
print("🔍 VERIFYING UPSERT")
print("="*60)

try:
    # Check row count for the season
    verify = supabase.table(table_name).select("*", count="exact").eq("season", season).execute()
    print(f"  ✅ Table '{table_name}' now has {verify.count} rows for season {season}")
    
    # Show cluster distribution
    cluster_dist = supabase.table(table_name).select("primary_archetype", count="exact").eq("season", season).execute()
    df_verify = pd.DataFrame(cluster_dist.data)
    
    print(f"\n  📊 Cluster distribution:")
    for cluster_id in sorted(df_verify['primary_archetype'].unique()):
        count = len(df_verify[df_verify['primary_archetype'] == cluster_id])
        pct = count / len(df_verify) * 100
        print(f"     Cluster {cluster_id}: {count} players ({pct:.1f}%)")
    
except Exception as e:
    print(f"  ⚠️ Could not verify: {e}")

print("\n✅ Done!")