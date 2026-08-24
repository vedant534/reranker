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

from src.metrics import evaluate_ranking


SCORE_COLUMNS = {
    "word_tfidf": "word_tfidf_score",
    "combined_lexical": "lexical_score",
    "lightgbm_ranker": "ranker_score",
}


def evaluate_models(predictions, latency):
    metrics = {}
    rows = []
    latency_names = {
        "word_tfidf": "word_tfidf",
        "combined_lexical": "combined_lexical",
        "lightgbm_ranker": "ranker",
    }
    for model_name, score_column in SCORE_COLUMNS.items():
        model_metrics = evaluate_ranking(predictions, score_column)
        model_metrics.update(latency[latency_names[model_name]])
        metrics[model_name] = model_metrics
        rows.append({"model": model_name, **model_metrics})
    return metrics, pd.DataFrame(rows)


def validation_low_overlap_threshold(validation_frame, validation_features):
    coverage = pd.DataFrame(
        {
            "query_id": validation_frame["query_id"].to_numpy(),
            "coverage": validation_features["query_token_coverage"].to_numpy(),
        }
    ).groupby("query_id")["coverage"].mean()
    return float(coverage.quantile(0.25))


def query_slice_metrics(predictions, test_features, low_overlap_threshold):
    query_metadata = predictions.groupby("query_id", sort=False)["query"].first().to_frame()
    query_metadata["query_length"] = query_metadata["query"].map(
        lambda text: len(re.findall(r"[a-z0-9]+", str(text).lower()))
    )
    query_metadata["has_numeric_model_token"] = query_metadata["query"].map(
        lambda text: any(any(char.isdigit() for char in token) for token in re.findall(r"[a-z0-9]+", str(text).lower()))
    )
    coverage = pd.DataFrame(
        {
            "query_id": predictions["query_id"].to_numpy(),
            "coverage": test_features["query_token_coverage"].to_numpy(),
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
    axis.bar(comparison["model"], comparison["ndcg_at_10"], color=["#8097b1", "#4f81bd", "#c0504d"])
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
    axis.set_title("Reranker feature importance")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
