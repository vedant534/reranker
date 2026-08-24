import argparse
import json
from pathlib import Path

import joblib

from src.data import load_config, load_esci_data, sample_query_splits
from src.evaluate import (
    evaluate_models,
    query_slice_metrics,
    save_feature_importance_plot,
    save_model_comparison_plot,
    validation_low_overlap_threshold,
)
from src.features import combined_lexical_score, fit_feature_bundle, transform_features
from src.metrics import evaluate_ranking
from src.models import train_ranker
from src.predict import measure_scoring_latency


def _add_scores(frame, features, bundle, ranker):
    predictions = frame.copy()
    predictions["word_tfidf_score"] = features["word_tfidf_cosine"].to_numpy()
    predictions["lexical_score"] = combined_lexical_score(
        features, bundle["lexical_word_weight"]
    )
    predictions["ranker_score"] = ranker.predict(features, num_iteration=ranker.best_iteration_)
    for score_column, rank_column in (
        ("word_tfidf_score", "word_tfidf_rank"),
        ("lexical_score", "lexical_rank"),
        ("ranker_score", "model_rank"),
    ):
        predictions[rank_column] = predictions.groupby("query_id")[score_column].rank(
            method="first", ascending=False
        ).astype(int)
    return predictions


def run_pipeline(config_path):
    config = load_config(config_path)
    artifact_dir = Path(config["paths"]["artifact_dir"])
    report_dir = Path(config["paths"]["report_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and sampling ESCI data...")
    data = load_esci_data(config["paths"]["dataset_dir"])
    splits = sample_query_splits(data, config["sampling"])
    print({name: (len(frame), frame["query_id"].nunique()) for name, frame in splits.items()})

    print("Fitting training-only TF-IDF vectorizers and generating features...")
    bundle = fit_feature_bundle(splits["train"], config["tfidf"])
    features = {name: transform_features(frame, bundle) for name, frame in splits.items()}

    validation_scores = []
    for weight in config["tfidf"]["lexical_word_weights"]:
        candidate = splits["validation"].copy()
        candidate["score"] = combined_lexical_score(features["validation"], weight)
        score = evaluate_ranking(candidate, "score")["ndcg_at_10"]
        validation_scores.append((float(weight), score))
    selected_weight, selected_validation_ndcg = max(
        validation_scores, key=lambda item: (item[1], -abs(item[0] - 0.5))
    )
    bundle["lexical_word_weight"] = selected_weight

    print("Training LightGBM LambdaRank model...")
    ranker = train_ranker(
        splits["train"],
        features["train"],
        splits["validation"],
        features["validation"],
        config["lightgbm"],
    )
    low_overlap_threshold = validation_low_overlap_threshold(
        splits["validation"], features["validation"]
    )
    bundle["low_overlap_threshold"] = low_overlap_threshold

    print("Scoring test queries and measuring warmed-up latency...")
    predictions = _add_scores(splits["test"], features["test"], bundle, ranker)
    latency = measure_scoring_latency(splits["test"], bundle, ranker)
    print("Computing aggregate and query-slice metrics...")
    model_metrics, comparison = evaluate_models(predictions, latency)
    slices = query_slice_metrics(predictions, features["test"], low_overlap_threshold)

    feature_path = artifact_dir / "feature_bundle.joblib"
    ranker_path = artifact_dir / "ranker.joblib"
    predictions_path = artifact_dir / "test_predictions.parquet"
    joblib.dump(bundle, feature_path)
    joblib.dump(ranker, ranker_path)
    predictions.to_parquet(predictions_path, index=False)

    metrics = {
        "dataset": {
            name: {"rows": len(frame), "queries": int(frame["query_id"].nunique())}
            for name, frame in splits.items()
        },
        "label_gains": {"I": 0.0, "C": 0.01, "S": 0.1, "E": 1.0},
        "selected_lexical_word_weight": selected_weight,
        "selected_lexical_validation_ndcg_at_10": selected_validation_ndcg,
        "validation_weight_scores": {
            str(weight): score for weight, score in validation_scores
        },
        "low_overlap_validation_q25_mean_coverage": low_overlap_threshold,
        "best_iteration": int(ranker.best_iteration_),
        "models": model_metrics,
    }
    with (report_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    slices.to_csv(report_dir / "query_slice_metrics.csv", index=False)
    print("Generating report plots...")
    save_model_comparison_plot(comparison, report_dir / "model_comparison.png")
    save_feature_importance_plot(
        ranker, bundle["feature_names"], report_dir / "feature_importance.png"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the ESCI reranker")
    parser.add_argument("--config", default="config.yaml")
    arguments = parser.parse_args()
    run_pipeline(arguments.config)
