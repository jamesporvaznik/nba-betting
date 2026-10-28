"""Generate each season's corrected feature CSV and select GMM cluster counts.

Reads Supabase or a supplied snapshot. Writes local previews and fitted artifacts
to a new directory only; never changes database cluster assignments.
"""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nba-cluster-matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from psycopg2.extras import RealDictCursor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.audit_clusters import SnapshotClient
from models.features.build_player_features import fetch_playtype_data, fetch_tracking_data, normalize_and_merge
from nba.audit_tracking import SEASONS
from nba.repair_shot_locations import connection


def collect_snapshot(seasons):
    queries = {
        "clusters": "SELECT * FROM public.player_clusters WHERE season = ANY(%s) ORDER BY season,player_id",
        "playtypes": "SELECT player_id,season,play_type,poss_pct FROM public.player_playtype_stats WHERE season = ANY(%s) ORDER BY season,player_id,play_type",
        "tracking": "SELECT * FROM public.player_tracking_stats WHERE season = ANY(%s) ORDER BY season,player_id",
        "minutes": "SELECT season,playerid,sum(minutes) AS minutes FROM public.player_gamelog WHERE season = ANY(%s) GROUP BY season,playerid ORDER BY season,playerid",
    }
    result = {"captured_at": datetime.now(timezone.utc).isoformat(), "database_writes": False}
    with connection() as conn:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            for name, query in queries.items():
                cursor.execute(query, (seasons,))
                result[name] = [dict(r) for r in cursor.fetchall()]
            cursor.execute("SELECT nba_api_id,display_name,position FROM public.players ORDER BY nba_api_id")
            result["players"] = [dict(r) for r in cursor.fetchall()]
        conn.rollback()
    return json.loads(json.dumps(result, default=lambda v: float(v) if isinstance(v, Decimal) else str(v)))


def build_features(snapshot, season, min_minutes):
    qualified = {r["playerid"] for r in snapshot["minutes"] if r["season"] == season and float(r["minutes"]) >= min_minutes}
    client = SnapshotClient(snapshot["playtypes"], snapshot["tracking"])
    with redirect_stdout(io.StringIO()):
        plays = fetch_playtype_data(client, season)
        tracking = fetch_tracking_data(client, season)
        ids = qualified & set(plays["player_id"]) & set(tracking["player_id"])
        plays = plays[plays["player_id"].isin(ids)]
        tracking = tracking[tracking["player_id"].isin(ids)]
        features, scaler = normalize_and_merge(plays, tracking)
    features = features.sort_values("player_id").reset_index(drop=True)
    matrix = features.drop(columns="player_id").to_numpy(dtype=float)
    if not ids or len(features) != len(ids) or features["player_id"].duplicated().any() or not np.isfinite(matrix).all():
        raise ValueError("Incomplete, duplicated, or nonfinite clustering inputs")
    profiles = plays.set_index("player_id").join(tracking.set_index("player_id"), how="inner").loc[features["player_id"]]
    return features, scaler, profiles


def fit_model(matrix, k, seed, n_init, max_iter):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed,
                                n_init=n_init, max_iter=max_iter, tol=1e-4, reg_covar=1e-6).fit(matrix)
    return model, sum(issubclass(w.category, ConvergenceWarning) for w in caught)


def evaluate_k(matrix, k, seeds, n_init, max_iter, min_size):
    trials, models, labels = [], [], []
    for seed in seeds:
        model, convergence_warnings = fit_model(matrix, k, seed, n_init, max_iter)
        assignment = model.predict(matrix)
        trials.append({"seed": seed, "bic": float(model.bic(matrix)), "converged": bool(model.converged_),
                       "iterations": int(model.n_iter_), "convergence_warnings": convergence_warnings})
        models.append(model)
        labels.append(assignment)
    best_index = min(range(len(models)), key=lambda i: trials[i]["bic"])
    best, assignment = models[best_index], labels[best_index]
    sizes = np.bincount(assignment, minlength=k)
    ari = [adjusted_rand_score(a, b) for a, b in itertools.combinations(labels, 2)]
    assigned_count = len(set(assignment))
    nontrivial = 1 < assigned_count < len(matrix)
    result = {"clusters": k, "bic": trials[best_index]["bic"], "aic": float(best.aic(matrix)),
              "best_seed": trials[best_index]["seed"], "all_seed_fits_converged": all(t["converged"] for t in trials),
              "bic_seed_min": min(t["bic"] for t in trials), "bic_seed_max": max(t["bic"] for t in trials),
              "silhouette": float(silhouette_score(matrix, assignment)) if nontrivial else None,
              "davies_bouldin": float(davies_bouldin_score(matrix, assignment)) if nontrivial else None,
              "seed_ari_median": float(np.median(ari)) if ari else None,
              "seed_ari_min": float(min(ari)) if ari else None,
              "smallest_cluster": int(sizes.min()), "largest_cluster": int(sizes.max()),
              "cluster_sizes": sizes.tolist(),
              "eligible_for_comparable_players": bool(k >= 2 and best.converged_ and assigned_count == k and sizes.min() >= min_size),
              "trials": trials}
    return result, best


def choose_count(results):
    converged = [r for r in results if r["all_seed_fits_converged"]]
    if not converged:
        raise ValueError("No cluster count converged consistently")
    raw = min(converged, key=lambda r: r["bic"])
    eligible = [r for r in converged if r["eligible_for_comparable_players"]]
    if not eligible:
        raise ValueError("No converged count meets the minimum comparable-player group size")
    return raw, min(eligible, key=lambda r: r["bic"])


def sampling_stability(matrix, reference_model, n_init, max_iter, repeats):
    reference = reference_model.predict(matrix)
    results = []
    for seed in range(301, 301 + repeats):
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(matrix), size=int(.8 * len(matrix)), replace=False)
        model, _ = fit_model(matrix[selected], reference_model.n_components, seed, n_init, max_iter)
        results.append({"seed": seed, "converged": bool(model.converged_),
                        "ari_on_full_cohort": float(adjusted_rand_score(reference, model.predict(matrix)))})
    return {"fraction_sampled_without_replacement": .8, "repetitions": repeats,
            "median_ari": float(np.median([r["ari_on_full_cohort"] for r in results])),
            "minimum_ari": min(r["ari_on_full_cohort"] for r in results), "trials": results}


def write_preview(folder, season, features, scaler, profiles, model, snapshot):
    matrix = features.drop(columns="player_id").to_numpy()
    probabilities = model.predict_proba(matrix)
    labels = model.predict(matrix)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1) or not np.array_equal(labels, probabilities.argmax(axis=1)):
        raise ValueError("Invalid model probability preview")
    names = {r["nba_api_id"]: r["display_name"] for r in snapshot["players"]}
    preview = pd.DataFrame({"player_id": features["player_id"], "season": season,
                            "player_name": [names.get(pid, str(pid)) for pid in features["player_id"]],
                            "primary_archetype": labels})
    for i in range(model.n_components):
        preview[f"archetype_{i}_weight"] = probabilities[:, i]
    preview.to_csv(folder / f"player_clusters_preview_{season}.csv", index=False)
    joblib.dump(model, folder / f"selected_gmm_{season}.joblib")
    joblib.dump(scaler, folder / f"feature_scaler_{season}.joblib")
    profile_rows, representatives = [], []
    for k in range(model.n_components):
        indexes = np.flatnonzero(labels == k)
        means = profiles.iloc[indexes].mean().to_dict()
        profile_rows.append({"cluster": k, "players": len(indexes), **means})
        distance = (((matrix[indexes] - model.means_[k]) ** 2) / model.covariances_[k]).sum(axis=1)
        central = indexes[np.argsort(distance)[:8]]
        representatives.append({"cluster": k, "players": len(indexes),
                                "central_players": [names.get(int(features.iloc[i]["player_id"])) for i in central]})
    pd.DataFrame(profile_rows).to_csv(folder / f"cluster_style_profiles_{season}.csv", index=False)
    (folder / f"representative_players_{season}.json").write_text(json.dumps(representatives, indent=2))
    return representatives


def plot_scores(results, selected_k, folder, season):
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "trials"} for r in results]).sort_values("clusters")
    for col in ("cluster_sizes",):
        frame[col] = frame[col].map(json.dumps)
    frame.to_csv(folder / f"cluster_count_scores_{season}.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = frame["clusters"]
    axes[0].plot(x, frame["bic"] - frame["bic"].min(), marker="o", markersize=3)
    axes[0].set_ylabel("BIC above lowest tested score (lower is better)")
    axes[1].plot(x, frame["silhouette"], marker="o", markersize=3)
    axes[1].set_ylabel("Silhouette (higher is better)")
    axes[2].plot(x, frame["seed_ari_median"], marker="o", markersize=3)
    axes[2].plot(x, frame["seed_ari_min"], alpha=.5, label="Minimum across seeds")
    axes[2].set_ylabel("Seed agreement, adjusted Rand index")
    axes[2].set_ylim(-.05, 1.05)
    for ax in axes:
        ax.axvline(selected_k, color="#157f56", linestyle="--", label=f"Selected: {selected_k}")
        ax.set_xlabel("Number of clusters")
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{season}: independent cluster-count search")
    fig.tight_layout()
    fig.savefig(folder / f"cluster_selection_{season}.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path, help="Reuse a captured snapshot instead of reading Supabase")
    parser.add_argument("--season", nargs="+", default=list(SEASONS))
    parser.add_argument("--min-minutes", type=int, default=500)
    parser.add_argument("--initial-max-clusters", type=int, default=20)
    parser.add_argument("--max-clusters", type=int, default=30)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 99, 123, 2026])
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--sampling-repeats", type=int, default=10)
    args = parser.parse_args()
    if not 2 <= args.initial_max_clusters <= args.max_clusters or min(args.n_init, args.min_cluster_size, args.sampling_repeats) <= 0:
        parser.error("Invalid search bounds or positive-count settings")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = json.loads(args.snapshot.read_text()) if args.snapshot else collect_snapshot(args.season)
    snapshot_path = args.output_dir / "source_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "database_writes": False,
              "source_snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
              "sklearn_version": sklearn.__version__,
              "method": {"features": "Existing 17 style features: six ILR play-type components and eleven tracking features, per-season scaling.",
                         "model": "GaussianMixture, diagonal covariance, tol=1e-4, reg_covar=1e-6",
                         "primary_selection": "Lowest BIC among consistently converged models. Report the unrestricted winner and a comparable-player winner meeting the configured minimum hard-assigned players per component. Silhouette, Davies-Bouldin, and stability are diagnostic, not a blended arbitrary score.",
                         "min_cluster_size": args.min_cluster_size, "seeds": args.seeds, "initializations_per_seed": args.n_init,
                         "initial_search": [1, args.initial_max_clusters], "boundary_extension_limit": args.max_clusters,
                         "temporal_scope": "Retrospective full-season role profiles; lag or use dated snapshots before using them for historical predictions."}, "seasons": []}
    with threadpool_limits(limits=1):
        for season in args.season:
            print(f"Building corrected {season} CSV automatically...", flush=True)
            folder = args.output_dir / season
            folder.mkdir()
            features, scaler, profiles = build_features(snapshot, season, args.min_minutes)
            features.to_csv(folder / f"player_features_{season}.csv", index=False)
            matrix = features.drop(columns="player_id").to_numpy()
            results, models = [], {}
            initial_max = min(args.initial_max_clusters, len(matrix)-1)
            for k in range(1, initial_max+1):
                result, model = evaluate_k(matrix, k, args.seeds, args.n_init, args.max_iter, args.min_cluster_size)
                results.append(result)
                models[k] = model
                print(f"{season} K={k}: BIC={result['bic']:.1f}; min group={result['smallest_cluster']}; seed ARI={result['seed_ari_median']:.3f}", flush=True)
            raw, chosen = choose_count(results)
            if raw["clusters"] == initial_max or chosen["clusters"] == initial_max:
                for k in range(initial_max+1, min(args.max_clusters, len(matrix)-1)+1):
                    result, model = evaluate_k(matrix, k, args.seeds, args.n_init, args.max_iter, args.min_cluster_size)
                    results.append(result)
                    models[k] = model
                    print(f"{season} extended K={k}: BIC={result['bic']:.1f}", flush=True)
                raw, chosen = choose_count(results)
            model = models[chosen["clusters"]]
            stability = sampling_stability(matrix, model, args.n_init, args.max_iter, args.sampling_repeats)
            representatives = write_preview(folder, season, features, scaler, profiles, model, snapshot)
            plot_scores(results, chosen["clusters"], folder, season)
            old_rows = [r for r in snapshot["clusters"] if r["season"] == season]
            old_labels = {r["player_id"]: r["primary_archetype"] for r in old_rows}
            new_labels = dict(zip(features["player_id"], model.predict(matrix)))
            common = sorted(old_labels.keys() & new_labels.keys())
            result = {"season": season, "players": len(features), "previous_assigned_cluster_count": len(set(old_labels.values())),
                      "unrestricted_bic_winner": raw["clusters"], "selected_clusters": chosen["clusters"],
                      "selected_metrics": chosen, "selected_sampling_stability": stability,
                      "best_bic_at_search_boundary": raw["clusters"] == max(models),
                      "membership_ari_vs_previous": float(adjusted_rand_score([old_labels[p] for p in common], [new_labels[p] for p in common])),
                      "representative_players": representatives, "all_candidates": results}
            (folder / "selection_report.json").write_text(json.dumps(result, indent=2))
            report["seasons"].append(result)
            print(f"SELECTED {season}: {chosen['clusters']} clusters for {len(features)} players; sampling ARI={stability['median_ari']:.3f}", flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "selection_report.json").write_text(json.dumps(report, indent=2))
    print(f"All season CSVs, scores, charts, and previews saved to {args.output_dir}. Database assignments unchanged.")


if __name__ == "__main__":
    main()
