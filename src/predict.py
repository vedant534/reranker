import platform
import sys
import time

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from src.features import TFIDF_FEATURE_NAMES, combined_lexical_score, transform_features


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
    prepared["product_id"] = prepared["product_id"].astype(str)
    return prepared


def assign_deterministic_ranks(frame, score_column, rank_column):
    ranked = frame.copy()
    ranked["_product_id_tiebreaker"] = ranked["product_id"].astype(str)
    ordered_index = ranked.sort_values(
        ["query_id", score_column, "_product_id_tiebreaker"],
        ascending=[True, False, True],
        kind="mergesort",
    ).index
    ordered_ranks = (
        ranked.loc[ordered_index]
        .groupby("query_id", sort=False)
        .cumcount()
        .add(1)
        .to_numpy()
    )
    ranked.loc[ordered_index, rank_column] = ordered_ranks
    ranked[rank_column] = ranked[rank_column].astype(int)
    return ranked.drop(columns="_product_id_tiebreaker")


def _sort_by_score(frame, score_column):
    sortable = frame.assign(_product_id_tiebreaker=frame["product_id"].astype(str))
    return (
        sortable.sort_values(
            [score_column, "_product_id_tiebreaker"],
            ascending=[False, True],
            kind="mergesort",
        )
        .drop(columns="_product_id_tiebreaker")
        .reset_index(drop=True)
    )


def _ranker_feature_names(feature_bundle):
    return list(feature_bundle.get("ranker_feature_names", feature_bundle["feature_names"]))


def _inference_threads(feature_bundle):
    return int(feature_bundle.get("inference_threads", 1))


def score_candidates(query, candidates, feature_bundle, ranker=None, method="ranker"):
    prepared = _prepare_candidates(query, candidates)
    if method == "word_tfidf":
        features = transform_features(prepared, feature_bundle, [TFIDF_FEATURE_NAMES[0]])
        scores = features[TFIDF_FEATURE_NAMES[0]].to_numpy()
    elif method == "combined_lexical":
        features = transform_features(prepared, feature_bundle, TFIDF_FEATURE_NAMES)
        scores = combined_lexical_score(features, feature_bundle["lexical_word_weight"])
    elif method == "ranker":
        if ranker is None:
            raise ValueError("ranker is required for method='ranker'")
        feature_names = _ranker_feature_names(feature_bundle)
        features = transform_features(prepared, feature_bundle, feature_names)
        scores = ranker.predict(
            features[feature_names],
            num_iteration=ranker.best_iteration_,
            num_threads=_inference_threads(feature_bundle),
        )
    else:
        raise ValueError(f"Unknown scoring method: {method}")
    prepared["predicted_score"] = np.asarray(scores, dtype=float)
    return _sort_by_score(prepared, "predicted_score")


def predict_reranked(query, candidates, feature_bundle, ranker):
    prepared = _prepare_candidates(query, candidates)
    ranker_features = _ranker_feature_names(feature_bundle)
    requested = list(dict.fromkeys(TFIDF_FEATURE_NAMES + ranker_features))
    features = transform_features(prepared, feature_bundle, requested)
    prepared["lexical_score"] = combined_lexical_score(
        features, feature_bundle["lexical_word_weight"]
    )
    prepared["predicted_score"] = ranker.predict(
        features[ranker_features],
        num_iteration=ranker.best_iteration_,
        num_threads=_inference_threads(feature_bundle),
    )
    prepared = assign_deterministic_ranks(prepared, "lexical_score", "lexical_rank")
    prepared = assign_deterministic_ranks(prepared, "predicted_score", "model_rank")
    return _sort_by_score(prepared, "predicted_score")


def measure_scoring_latency(test_frame, feature_bundle, ranker, warmup_queries=10):
    groups = [
        (group["query"].iloc[0], group)
        for _, group in test_frame.groupby("query_id", sort=False)
    ]
    if not groups:
        raise ValueError("Cannot measure latency on an empty test set")
    methods = ("word_tfidf", "combined_lexical", "ranker")
    warmup_count = min(warmup_queries, len(groups))
    for method in methods:
        for query, candidates in groups[:warmup_count]:
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
            "n_timed_queries": len(timings),
        }
    result["metadata"] = {
        "mean_candidate_count": float(test_frame.groupby("query_id").size().mean()),
        "warmup_queries_per_method": warmup_count,
        "inference_threads": _inference_threads(feature_bundle),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not reported by platform",
        "python_version": sys.version.split()[0],
        "lightgbm_version": lightgbm.__version__,
        "scikit_learn_version": sklearn.__version__,
        "hardware_dependent": True,
    }
    return result
