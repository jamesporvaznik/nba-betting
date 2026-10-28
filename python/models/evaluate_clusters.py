"""Read-only multi-criterion review of a saved season cluster selection run.

No blended score or automatic declaration of a universally best count. Refits
every candidate with more initializations and checks membership stability with
the scaler refitted inside each subsample. All results are local artifacts.
"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.select_clusters import build_features, evaluate_k, fit_model, write_preview
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_samples
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def geometry(matrix, labels):
    distances = pairwise_distances(matrix)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :10]
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    sizes = same.sum(axis=1)
    sil = silhouette_samples(matrix, labels)
    chance = float(np.mean(sizes / (len(labels) - 1)))
    purity = float(np.mean(labels[neighbors[:, :5]] == labels[:, None]))
    return {
        "negative_silhouette_fraction": float(np.mean(sil < 0)),
        "nearest5_same_cluster_fraction": purity,
        "random_pair_same_cluster_fraction": chance,
        "nearest5_chance_adjusted_agreement": (purity - chance) / (1 - chance),
        "nearest10_excluded_by_cluster_fraction": float(np.mean(labels[neighbors] != labels[:, None])),
        "clusters": [{"cluster": int(k), "players": int(np.sum(labels == k)),
                      "mean_silhouette": float(sil[labels == k].mean()),
                      "negative_silhouette_fraction": float(np.mean(sil[labels == k] < 0))}
                     for k in sorted(set(labels))],
    }


def sample_stability(raw, reference_labels, k, repeats=10):
    trials = []
    neighbor_retention = []
    original_same = reference_labels[:, None] == reference_labels[None, :]
    np.fill_diagonal(original_same, False)
    for seed in range(401, 401 + repeats):
        subset = np.random.default_rng(seed).choice(len(raw), int(.8 * len(raw)), replace=False)
        scaler = StandardScaler().fit(raw[subset])
        model, _ = fit_model(scaler.transform(raw[subset]), k, seed, 5, 500)
        labels = model.predict(scaler.transform(raw))
        same = labels[:, None] == labels[None, :]
        retention = float(same[original_same].mean())
        trials.append({"seed": seed, "converged": bool(model.converged_),
                       "ari": float(adjusted_rand_score(reference_labels, labels)),
                       "original_same_cluster_pair_retention": retention})
        neighbor_retention.append(retention)
    return {"median_ari": float(np.median([t["ari"] for t in trials])),
            "minimum_ari": float(min(t["ari"] for t in trials)),
            "median_original_pair_retention": float(np.median(neighbor_retention)),
            "all_converged": all(t["converged"] for t in trials), "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    original = json.loads((args.run_dir / "selection_report.json").read_text())
    snapshot = json.loads((args.run_dir / "source_snapshot.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"database_writes": False, "source_run": str(args.run_dir),
              "method": "Same 17 features and diagonal GMM; 5 seeds x 10 initializations per candidate; 10 independent 80% subsamples x 5 initializations, refitting scaler within each subsample. Geometric metrics use standardized Euclidean distance. All diagnostics are retrospective, not prediction validation. No weighted aggregate or automatic final count recommendation.",
              "seasons": []}
    with threadpool_limits(limits=1):
        for old in original["seasons"]:
            season = old["season"]
            folder = args.output_dir / season
            folder.mkdir()
            features, scaler, profiles = build_features(snapshot, season, 500)
            matrix = features.drop(columns="player_id").to_numpy()
            raw = scaler.inverse_transform(matrix)
            candidates = []
            for previous in old["all_candidates"]:
                k = previous["clusters"]
                if k < 2:
                    continue
                metrics, model = evaluate_k(matrix, k, [7, 42, 99, 123, 2026], 10, 500, 5)
                # Preserve the better original fit if the expanded search took
                # a different initialization path. This can occur with n_init.
                if k == old["selected_clusters"]:
                    saved = joblib.load(args.run_dir / season / f"selected_gmm_{season}.joblib")
                    if saved.bic(matrix) < model.bic(matrix):
                        model = saved
                        metrics["bic"] = float(saved.bic(matrix))
                        metrics["aic"] = float(saved.aic(matrix))
                        metrics["reference_from_original_run"] = True
                labels = model.predict(matrix)
                # Recalculate all reference geometry even when saved fit wins.
                from sklearn.metrics import silhouette_score, davies_bouldin_score
                metrics["silhouette"] = float(silhouette_score(matrix, labels))
                metrics["davies_bouldin"] = float(davies_bouldin_score(matrix, labels))
                sizes = np.bincount(labels, minlength=k)
                metrics["cluster_sizes"] = sizes.tolist()
                metrics["smallest_cluster"] = int(sizes.min())
                metrics["largest_cluster"] = int(sizes.max())
                metrics["eligible_for_comparable_players"] = bool(sizes.min() >= 5 and model.converged_)
                metrics["geometry"] = geometry(matrix, labels)
                metrics["sampling"] = sample_stability(raw, labels, k)
                metrics["median_max_posterior"] = float(np.median(model.predict_proba(matrix).max(axis=1)))
                candidate_folder = folder / f"k_{k:02d}"
                candidate_folder.mkdir()
                metrics["representative_players"] = write_preview(candidate_folder, season, features, scaler, profiles, model, snapshot)
                (candidate_folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
                candidates.append(metrics)
                print(f"{season} K={k}: BIC={metrics['bic']:.1f}; silhouette={metrics['silhouette']:.3f}; sample ARI={metrics['sampling']['median_ari']:.3f}; negative silhouette={metrics['geometry']['negative_silhouette_fraction']:.1%}", flush=True)
            eligible = [c for c in candidates if c["eligible_for_comparable_players"] and c["all_seed_fits_converged"]]
            best = min(eligible, key=lambda c: c["bic"])
            for c in candidates:
                c["bic_above_eligible_best"] = c["bic"] - best["bic"]
            result = {"season": season, "players": len(features), "original_bic_count": old["selected_clusters"],
                      "expanded_search_bic_count": best["clusters"], "candidates": candidates}
            (folder / "evaluation.json").write_text(json.dumps(result, indent=2))
            pd.DataFrame([{**{k: v for k, v in c.items() if not isinstance(v, (list, dict))},
                           **{k: v for k, v in c["geometry"].items() if k != "clusters"},
                           "sampling_ari_median": c["sampling"]["median_ari"],
                           "sampling_ari_minimum": c["sampling"]["minimum_ari"]} for c in candidates]).to_csv(folder / "candidate_comparison.csv", index=False)
            report["seasons"].append(result)
            print(f"FINISHED {season}: expanded BIC winner {best['clusters']}", flush=True)
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
