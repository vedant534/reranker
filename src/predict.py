import time

import numpy as np
import pandas as pd

from src.features import combined_lexical_score, transform_features


def _prepare_candidates(query, candidates):
    prepared = candidates.copy().reset_index(drop=True)
    prepared["query"] = str(query)
    if "query_id" not in prepared:
        prepared["query_id"] = 0
    for column in ("product_title", "product_brand", "product_color"):
        if column not in prepared:
            prepared[column] = ""
        prepared[column] = prepared[column].fillna("").astype(str)
    if "product_id" not in prepared:
        prepared["product_id"] = prepared.index.astype(str)
    return prepared


def score_candidates(query, candidates, feature_bundle, ranker=None, method="ranker"):
    prepared = _prepare_candidates(query, candidates)
    features = transform_features(prepared, feature_bundle)
    if method == "word_tfidf":
        scores = features["word_tfidf_cosine"].to_numpy()
    elif method == "combined_lexical":
        scores = combined_lexical_score(features, feature_bundle["lexical_word_weight"])
    elif method == "ranker":
        if ranker is None:
            raise ValueError("ranker is required for method='ranker'")
        scores = ranker.predict(features, num_iteration=ranker.best_iteration_)
    else:
        raise ValueError(f"Unknown scoring method: {method}")
    prepared["predicted_score"] = np.asarray(scores, dtype=float)
    return prepared.sort_values(
        ["predicted_score", "product_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def predict_reranked(query, candidates, feature_bundle, ranker):
    prepared = _prepare_candidates(query, candidates)
    features = transform_features(prepared, feature_bundle)
    prepared["lexical_score"] = combined_lexical_score(
        features, feature_bundle["lexical_word_weight"]
    )
    prepared["predicted_score"] = ranker.predict(features, num_iteration=ranker.best_iteration_)
    prepared["lexical_rank"] = (
        prepared["lexical_score"].rank(method="first", ascending=False).astype(int)
    )
    prepared["model_rank"] = (
        prepared["predicted_score"].rank(method="first", ascending=False).astype(int)
    )
    return prepared.sort_values(
        ["predicted_score", "product_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def measure_scoring_latency(test_frame, feature_bundle, ranker):
    groups = [(group["query"].iloc[0], group) for _, group in test_frame.groupby("query_id", sort=False)]
    if not groups:
        raise ValueError("Cannot measure latency on an empty test set")
    methods = ("word_tfidf", "combined_lexical", "ranker")
    for method in methods:
        for query, candidates in groups[:3]:
            score_candidates(query, candidates, feature_bundle, ranker, method)

    result = {}
    for method in methods:
        timings = []
        for query, candidates in groups:
            start = time.perf_counter()
            score_candidates(query, candidates, feature_bundle, ranker, method)
            timings.append((time.perf_counter() - start) * 1000.0)
        result[method] = {
            "p50_ms": float(np.percentile(timings, 50)),
            "p95_ms": float(np.percentile(timings, 95)),
            "n_queries": len(timings),
        }
    return result

