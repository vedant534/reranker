import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import evaluate_ranking, ranking_metrics


SCORE_COLUMNS = {
    "word_tfidf": "word_tfidf_score",
    "combined_lexical": "lexical_score",
    "lightgbm_ranker": "ranker_score",
}


def per_query_model_metrics(predictions):
    rows = []
    for query_id, group in predictions.groupby("query_id", sort=False):
        query = group["query"].iloc[0]
        for model_name, score_column in SCORE_COLUMNS.items():
            ranked = group.sort_values(
                [score_column, "product_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "model": model_name,
                    **ranking_metrics(ranked["esci_label"].tolist()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_models(predictions, latency, per_query=None):
    per_query = per_query_model_metrics(predictions) if per_query is None else per_query
    metrics = {}
    rows = []
    latency_names = {
        "word_tfidf": "word_tfidf",
        "combined_lexical": "combined_lexical",
        "lightgbm_ranker": "ranker",
    }
    metric_columns = [
        "ndcg_at_10",
        "exact_mrr_at_10",
        "es_recall_at_5",
        "complement_exposure_at_5",
        "irrelevant_exposure_at_5",
    ]
    for model_name in SCORE_COLUMNS:
        model_rows = per_query[per_query["model"] == model_name]
        model_metrics = {
            name: float(model_rows[name].mean(skipna=True))
            if model_rows[name].notna().any()
            else None
            for name in metric_columns
        }
        model_metrics["n_queries"] = int(model_rows["query_id"].nunique())
        model_metrics["es_recall_queries"] = int(model_rows["es_recall_at_5"].notna().sum())
        model_metrics.update(latency[latency_names[model_name]])
        metrics[model_name] = model_metrics
        rows.append({"model": model_name, **model_metrics})
    return metrics, pd.DataFrame(rows)


def paired_bootstrap_comparison(per_query, seed, replicates):
    rows = []
    pivots = {}
    for metric in ("ndcg_at_10", "exact_mrr_at_10"):
        pivot = per_query.pivot(index="query_id", columns="model", values=metric)
        pivot = pivot[["combined_lexical", "lightgbm_ranker"]].dropna()
        differences = (
            pivot["lightgbm_ranker"] - pivot["combined_lexical"]
        ).to_numpy(dtype=float)
        rng = np.random.default_rng(int(seed))
        samples = np.empty(int(replicates), dtype=float)
        for index in range(int(replicates)):
            sampled_indices = rng.integers(0, len(differences), size=len(differences))
            samples[index] = float(differences[sampled_indices].mean())
        lower, upper = np.percentile(samples, [2.5, 97.5])
        if lower > 0:
            conclusion = "positive interval conditional on fixed model and sample"
        elif upper < 0:
            conclusion = "negative interval conditional on fixed model and sample"
        else:
            conclusion = "interval contains zero"
        rows.append(
            {
                "metric": metric,
                "baseline_model": "combined_lexical",
                "challenger_model": "lightgbm_ranker",
                "observed_difference": float(differences.mean()),
                "ci_lower_95": float(lower),
                "ci_upper_95": float(upper),
                "bootstrap_seed": int(seed),
                "bootstrap_replicates": int(replicates),
                "n_queries": len(differences),
                "conclusion": conclusion,
            }
        )
        pivots[metric] = pivot

    ndcg_differences = (
        pivots["ndcg_at_10"]["lightgbm_ranker"]
        - pivots["ndcg_at_10"]["combined_lexical"]
    )
    tied = np.isclose(ndcg_differences.to_numpy(), 0.0, atol=1e-12, rtol=0.0)
    improved = (ndcg_differences.to_numpy() > 0) & ~tied
    worsened = (ndcg_differences.to_numpy() < 0) & ~tied
    total = len(ndcg_differences)
    win_tie_loss = {
        "improved": int(improved.sum()),
        "tied": int(tied.sum()),
        "worsened": int(worsened.sum()),
        "improved_pct": float(100.0 * improved.sum() / total),
        "tied_pct": float(100.0 * tied.sum() / total),
        "worsened_pct": float(100.0 * worsened.sum() / total),
        "n_queries": total,
        "tie_tolerance": 1e-12,
    }
    return pd.DataFrame(rows), win_tie_loss


def build_error_examples(predictions, per_query):
    pivot = per_query.pivot(index="query_id", columns="model", values="ndcg_at_10")
    differences = (
        pivot["lightgbm_ranker"] - pivot["combined_lexical"]
    ).rename("ndcg_difference")
    difference_frame = differences.reset_index()
    tolerance = 1e-12
    selections = []

    positive = difference_frame[difference_frame["ndcg_difference"] > tolerance].copy()
    if not positive.empty:
        largest = positive.sort_values(
            ["ndcg_difference", "query_id"], ascending=[False, True]
        )
        largest_query_id = largest["query_id"].iloc[0]
        selections.append(("largest_improvement", largest_query_id))
        remaining = positive[positive["query_id"] != largest_query_id].copy()
        if not remaining.empty:
            median_positive = float(positive["ndcg_difference"].median())
            remaining["median_distance"] = (
                remaining["ndcg_difference"] - median_positive
            ).abs()
            representative = remaining.sort_values(
                ["median_distance", "query_id"], ascending=[True, True]
            )
            selections.append(
                ("representative_positive", representative["query_id"].iloc[0])
            )

    negative = difference_frame[difference_frame["ndcg_difference"] < -tolerance].copy()
    if not negative.empty:
        regression = negative.sort_values(
            ["ndcg_difference", "query_id"], ascending=[True, True]
        )
        selections.append(("largest_regression", regression["query_id"].iloc[0]))

    rows = []
    methods = (
        ("combined_lexical", "lexical_score", "lexical_rank"),
        ("lightgbm_ranker", "ranker_score", "model_rank"),
    )
    for case_type, query_id in selections:
        query_rows = predictions[predictions["query_id"] == query_id]
        difference = float(differences.loc[query_id])
        for method, score_column, rank_column in methods:
            ranked = query_rows.sort_values(rank_column)
            for item in ranked.itertuples(index=False):
                rows.append(
                    {
                        "query_id": query_id,
                        "query": item.query,
                        "case_type": case_type,
                        "method": method,
                        "rank": int(getattr(item, rank_column)),
                        "product_id": item.product_id,
                        "title": item.product_title,
                        "brand": item.product_brand,
                        "esci_label": item.esci_label,
                        "score": float(getattr(item, score_column)),
                        "query_ndcg_difference": difference,
                    }
                )
    columns = [
        "query_id",
        "query",
        "case_type",
        "method",
        "rank",
        "product_id",
        "title",
        "brand",
        "esci_label",
        "score",
        "query_ndcg_difference",
    ]
    return pd.DataFrame(rows, columns=columns)


def validation_low_overlap_threshold(validation_frame, validation_features):
    coverage = pd.DataFrame(
        {
            "query_id": validation_frame["query_id"].to_numpy(),
            "coverage": validation_features["query_token_coverage"].to_numpy(),
        }
    ).groupby("query_id")["coverage"].mean()
    return float(coverage.quantile(0.25))


def query_slice_metrics(predictions, final_features, low_overlap_threshold):
    query_metadata = predictions.groupby("query_id", sort=False)["query"].first().to_frame()
    query_metadata["query_length"] = query_metadata["query"].map(
        lambda text: len(re.findall(r"[a-z0-9]+", str(text).lower()))
    )
    query_metadata["has_numeric_model_token"] = query_metadata["query"].map(
        lambda text: any(
            any(char.isdigit() for char in token)
            for token in re.findall(r"[a-z0-9]+", str(text).lower())
        )
    )
    coverage = pd.DataFrame(
        {
            "query_id": predictions["query_id"].to_numpy(),
            "coverage": final_features["query_token_coverage"].to_numpy(),
        }
    ).groupby("query_id")["coverage"].mean()
    query_metadata["mean_query_coverage"] = coverage

    slices = {
        "one_or_two_tokens": query_metadata["query_length"] <= 2,
        "contains_number_or_model_token": query_metadata["has_numeric_model_token"],
        "long_query_5_plus_tokens": query_metadata["query_length"] >= 5,
        "low_lexical_overlap": query_metadata["mean_query_coverage"] <= low_overlap_threshold,
    }
    rows = []
    for slice_name, mask in slices.items():
        query_ids = query_metadata.index[mask]
        sliced = predictions[predictions["query_id"].isin(query_ids)]
        for model_name, score_column in SCORE_COLUMNS.items():
            row = evaluate_ranking(sliced, score_column)
            rows.append({"slice": slice_name, "model": model_name, **row})
    return pd.DataFrame(rows)


def save_model_comparison_plot(comparison, path):
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        comparison["model"],
        comparison["ndcg_at_10"],
        color=["#8097b1", "#4f81bd", "#c0504d"],
    )
    axis.set_ylabel("nDCG@10")
    axis.set_title("ESCI reranking model comparison")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_feature_importance_plot(ranker, feature_names, path):
    importance = ranker.booster_.feature_importance(importance_type="gain")
    order = np.argsort(importance)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(np.asarray(feature_names)[order], importance[order], color="#4f81bd")
    axis.set_xlabel("LightGBM gain importance")
    axis.set_title("Selected ranker feature importance")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
