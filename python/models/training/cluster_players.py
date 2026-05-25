import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score, adjusted_rand_score


def bic():
    # Load the features you saved earlier
    final_features = pd.read_csv("player_features_2022-23.csv")

    # FIX: Automatically drop any hidden index columns saved by pandas
    if 'Unnamed: 0' in final_features.columns:
        final_features = final_features.drop(columns=['Unnamed: 0'])

    # Isolate just the numeric features (dropping player_id)
    X = final_features.drop(columns=['player_id']).to_numpy()

    print(f"Loaded features shape: {X.shape}")
    print(f"Features: {final_features.columns.tolist()}")

    bic_scores = []
    cluster_range = range(5, 12)  # Testing 5 to 11 archetypes

    for k in cluster_range:
        gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        print(f"Clusters: {k} | BIC Score: {bic_scores[-1]:.2f}")

    optimal_k = cluster_range[np.argmin(bic_scores)]
    print(f"\n🏆 Mathematically optimal number of archetypes: {optimal_k}")

    # Optional: Plot BIC scores
    plt.figure(figsize=(8, 5))
    plt.plot(list(cluster_range), bic_scores, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('BIC Score')
    plt.title('Gaussian Mixture Model BIC Scores')
    plt.grid(True, alpha=0.3)
    plt.show()

def silhouette(X):

    sil_scores = []
    k_range = range(5, 12)

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
        labels = gmm.fit_predict(X)
        score = silhouette_score(X, labels)
        sil_scores.append(score)
        print(f"Clusters: {k} | Silhouette Score: {score:.4f}")

def run_remaining_diagnostics(X):
    """
    Executes Davies-Bouldin Index sweep and random-seed stability checks.
    """
    print("\n" + "="*40)
    print("🔬 RUNNING DIAGNOSTIC TIE-BREAKERS")
    print("="*40)
    
    k_range = range(5, 12)
    
    # TEST 1: Davies-Bouldin Index Sweep (Lower = Better)
    print("\n1. Davies-Bouldin Index (Seeking the Minimum):")
    dbi_scores = {}
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
        labels = gmm.fit_predict(X)
        score = davies_bouldin_score(X, labels)
        dbi_scores[k] = score
        print(f"  Clusters: {k} | DBI Score: {score:.4f}")
        
    optimal_dbi_k = min(dbi_scores, key=dbi_scores.get)
    print(f"👉 DBI prefers: {optimal_dbi_k} clusters")

    # TEST 2: Cluster Stability Checks (Higher/Closer to 1.0 = Better)
    print("\n2. Random Seed Stability (Adjusted Rand Index):")
    print("   (Testing if changing the seed completely shuffles the players)")
    
    for k in [5, 6, 7, 8, 9, 10]:
        # Fit model with Seed A
        gmm_a = GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
        labels_a = gmm_a.fit_predict(X)
        
        # Fit model with Seed B
        gmm_b = GaussianMixture(n_components=k, covariance_type='diag', random_state=99)
        labels_b = gmm_b.fit_predict(X)
        
        # Compare how well the player groupings match across seeds
        stability_score = adjusted_rand_score(labels_a, labels_b)
        print(f"  Clusters: {k} | Stability Score (ARI): {stability_score:.4f}")
        if stability_score > 0.80:
            print(f"    ✅ Highly stable structure.")
        else:
            print(f"    ⚠️ Moderate fluctuation between seeds.")

if __name__ == "__main__":
   bic()

   final_features = pd.read_csv("player_features_2022-23.csv")
   if 'Unnamed: 0' in final_features.columns:
        final_features = final_features.drop(columns=['Unnamed: 0'])
   X_matrix = final_features.drop(columns=['player_id']).to_numpy()
   silhouette(X_matrix)
   run_remaining_diagnostics(X_matrix)
