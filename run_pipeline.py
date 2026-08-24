import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np

from src.data import dataset_files, load_config, load_esci_data, sample_query_splits
from src.evaluate import (
    build_error_examples,
    evaluate_models,
    paired_bootstrap_comparison,
    per_query_model_metrics,
    query_slice_metrics,
    save_feature_importance_plot,
    save_model_comparison_plot,
    validation_low_overlap_threshold,
)
from src.features import (
    FEATURE_NAMES,
    FEATURE_SETS,
    combined_lexical_score,
    fit_feature_bundle,
    transform_features,
)
from src.metrics import LABEL_GAINS, evaluate_ranking
from src.models import train_ranker
from src.predict import assign_deterministic_ranks, measure_scoring_latency


def _peak_rss_mib():
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024.0 * 1024.0 if platform.system() == "Darwin" else 1024.0
        return float(peak / divisor)
    except (ImportError, OSError, ValueError):
        return None


def _add_scores(frame, features, bundle, ranker):
    predictions = frame.copy()
    predictions["product_id"] = predictions["product_id"].astype(str)
    predictions["word_tfidf_score"] = features["word_tfidf_cosine"].to_numpy()
    predictions["lexical_score"] = combined_lexical_score(
        features, bundle["lexical_word_weight"]
    )
    ranker_features = bundle["ranker_feature_names"]
    predictions["ranker_score"] = ranker.predict(
        features[ranker_features],
        num_iteration=ranker.best_iteration_,
        num_threads=bundle["inference_threads"],
    )
    for score_column, rank_column in (
        ("word_tfidf_score", "word_tfidf_rank"),
        ("lexical_score", "lexical_rank"),
        ("ranker_score", "model_rank"),
    ):
        predictions = assign_deterministic_ranks(
            predictions, score_column, rank_column
        )
    return predictions


def _assert_rank_alignment(predictions):
    for score_column, rank_column in (
        ("word_tfidf_score", "word_tfidf_rank"),
        ("lexical_score", "lexical_rank"),
        ("ranker_score", "model_rank"),
    ):
        ordered = predictions.assign(
            _product_id_tiebreaker=predictions["product_id"].astype(str)
        ).sort_values(
            ["query_id", score_column, "_product_id_tiebreaker"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        expected = ordered.groupby("query_id", sort=False).cumcount().add(1).to_numpy()
        if not np.array_equal(ordered[rank_column].to_numpy(), expected):
            raise AssertionError(f"{rank_column} does not agree with deterministic ordering")


def _select_ranker(splits, features, config, report_dir):
    rows = []
    winner = None
    winner_metadata = None
    winner_key = None
    for candidate in config["ranker_candidates"]:
        feature_names = list(FEATURE_SETS[candidate["feature_set"]])
        truncation_level = int(candidate["truncation_level"])
        print(
            f"Training {candidate['name']} with {len(feature_names)} features "
            f"and truncation {truncation_level}..."
        )
        ranker = train_ranker(
            splits["train"],
            features["train"],
            splits["validation"],
            features["validation"],
            config["lightgbm"],
            feature_names=feature_names,
            truncation_level=truncation_level,
        )
        validation = splits["validation"].copy()
        validation["score"] = ranker.predict(
            features["validation"][feature_names],
            num_iteration=ranker.best_iteration_,
            num_threads=int(config["evaluation"]["inference_threads"]),
        )
        validation_ndcg = evaluate_ranking(validation, "score")["ndcg_at_10"]
        row = {
            "candidate_name": candidate["name"],
            "feature_set": candidate["feature_set"],
            "truncation_level": truncation_level,
            "validation_ndcg_at_10": validation_ndcg,
            "best_iteration": int(ranker.best_iteration_),
        }
        rows.append(row)
        key = (
            validation_ndcg,
            -len(feature_names),
            int(truncation_level == 13),
        )
        if winner_key is None or key > winner_key:
            winner = ranker
            winner_metadata = {**row, "feature_names": feature_names}
            winner_key = key

    import pandas as pd

    ablation = pd.DataFrame(rows)
    expected_columns = [
        "candidate_name",
        "feature_set",
        "truncation_level",
        "validation_ndcg_at_10",
        "best_iteration",
    ]
    if list(ablation.columns) != expected_columns:
        raise AssertionError("Validation ablation schema changed unexpectedly")
    if any("test" in column.lower() for column in ablation.columns):
        raise AssertionError("Validation ablation must not contain test metrics")
    ablation.to_csv(report_dir / "validation_ablation.csv", index=False)
    if winner is None:
        raise AssertionError("Validation failed to select a ranker")
    return winner, winner_metadata, ablation


def _verify_reloaded_predictions(final_frame, saved_predictions, bundle_path, ranker_path):
    reloaded_bundle = joblib.load(bundle_path)
    reloaded_ranker = joblib.load(ranker_path)
    reloaded_features = transform_features(final_frame, reloaded_bundle, FEATURE_NAMES)
    reproduced = _add_scores(
        final_frame, reloaded_features, reloaded_bundle, reloaded_ranker
    )
    if not np.array_equal(
        reproduced["product_id"].to_numpy(), saved_predictions["product_id"].to_numpy()
    ):
        raise AssertionError("Reloaded artifacts changed candidate row alignment")
    for score_column in ("word_tfidf_score", "lexical_score", "ranker_score"):
        if not np.allclose(
            reproduced[score_column],
            saved_predictions[score_column],
            rtol=1e-7,
            atol=1e-9,
        ):
            raise AssertionError(f"Reloaded artifacts changed {score_column}")
    for rank_column in ("word_tfidf_rank", "lexical_rank", "model_rank"):
        if not np.array_equal(
            reproduced[rank_column].to_numpy(),
            saved_predictions[rank_column].to_numpy(),
        ):
            raise AssertionError(f"Reloaded artifacts changed {rank_column}")
    _assert_rank_alignment(reproduced)


def run_pipeline(config_path):
    started = time.perf_counter()
    config = load_config(config_path)
    artifact_dir = Path(config["paths"]["artifact_dir"])
    report_dir = Path(config["paths"]["report_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ESCI data and reconstructing query-disjoint splits...")
    data = load_esci_data(config["paths"]["dataset_dir"])
    splits = sample_query_splits(data, config["sampling"])
    split_counts = {
        name: {"rows": len(frame), "queries": int(frame["query_id"].nunique())}
        for name, frame in splits.items()
    }
    print(split_counts)
    if split_counts["legacy_test"]["queries"] != int(
        config["sampling"]["legacy_test_queries"]
    ):
        raise AssertionError("Legacy test reconstruction did not return exactly 1,000 queries")
    official_test_queries = int(data.loc[data["split"] == "test", "query_id"].nunique())
    if split_counts["final_test"]["queries"] != (
        official_test_queries - split_counts["legacy_test"]["queries"]
    ):
        raise AssertionError("Fresh final split is not the official-test remainder")

    print("Fitting training-only TF-IDF vectorizers and train/validation features...")
    bundle = fit_feature_bundle(splits["train"], config["tfidf"])
    features = {
        name: transform_features(splits[name], bundle, FEATURE_NAMES)
        for name in ("train", "validation")
    }

    validation_scores = []
    for weight in config["tfidf"]["lexical_word_weights"]:
        candidate = splits["validation"].copy()
        candidate["score"] = combined_lexical_score(features["validation"], weight)
        validation_scores.append(
            (float(weight), evaluate_ranking(candidate, "score")["ndcg_at_10"])
        )
    selected_weight, selected_validation_ndcg = max(
        validation_scores, key=lambda item: (item[1], -abs(item[0] - 0.5))
    )
    bundle["lexical_word_weight"] = selected_weight

    ranker, selected_candidate, ablation = _select_ranker(
        splits, features, config, report_dir
    )
    bundle.update(
        {
            "ranker_feature_names": selected_candidate["feature_names"],
            "ranker_candidate_name": selected_candidate["candidate_name"],
            "ranker_truncation_level": selected_candidate["truncation_level"],
            "model_seed": int(config["lightgbm"]["random_state"]),
            "inference_threads": int(config["evaluation"]["inference_threads"]),
        }
    )
    low_overlap_threshold = validation_low_overlap_threshold(
        splits["validation"], features["validation"]
    )
    bundle["low_overlap_threshold"] = low_overlap_threshold
    print(
        f"Selected {selected_candidate['candidate_name']} at validation nDCG@10 "
        f"{selected_candidate['validation_ndcg_at_10']:.6f}."
    )

    print("Selection frozen. Transforming and scoring the fresh final split...")
    final_features = transform_features(splits["final_test"], bundle, FEATURE_NAMES)
    predictions = _add_scores(splits["final_test"], final_features, bundle, ranker)
    _assert_rank_alignment(predictions)

    print("Measuring warmed-up, method-specific scoring latency...")
    latency = measure_scoring_latency(splits["final_test"], bundle, ranker)
    print("Computing aggregate, per-query, bootstrap, slice, and example reports...")
    per_query = per_query_model_metrics(predictions)
    model_metrics, comparison = evaluate_models(predictions, latency, per_query)
    bootstrap, win_tie_loss = paired_bootstrap_comparison(
        per_query,
        seed=int(config["evaluation"]["bootstrap_seed"]),
        replicates=int(config["evaluation"]["bootstrap_replicates"]),
    )
    slices = query_slice_metrics(predictions, final_features, low_overlap_threshold)
    examples = build_error_examples(predictions, per_query)

    for bootstrap_row in bootstrap.itertuples(index=False):
        model_difference = (
            model_metrics["lightgbm_ranker"][bootstrap_row.metric]
            - model_metrics["combined_lexical"][bootstrap_row.metric]
        )
        if not np.isclose(
            model_difference, bootstrap_row.observed_difference, atol=1e-12, rtol=0.0
        ):
            raise AssertionError(
                f"Bootstrap observed difference disagrees for {bootstrap_row.metric}"
            )
    for model_name, values in model_metrics.items():
        separated = (
            values["complement_exposure_at_5"]
            + values["irrelevant_exposure_at_5"]
        )
        score_column = {
            "word_tfidf": "word_tfidf_score",
            "combined_lexical": "lexical_score",
            "lightgbm_ranker": "ranker_score",
        }[model_name]
        combined = []
        for _, group in predictions.groupby("query_id", sort=False):
            ranked = group.sort_values(
                [score_column, "product_id"], ascending=[False, True], kind="mergesort"
            )
            top = ranked["esci_label"].tolist()[:5]
            combined.append(sum(label in {"C", "I"} for label in top) / len(top))
        if not np.isclose(separated, float(np.mean(combined)), atol=1e-12, rtol=0.0):
            raise AssertionError("Separated exposure metrics do not match combined exposure")

    feature_path = artifact_dir / "feature_bundle.joblib"
    ranker_path = artifact_dir / "ranker.joblib"
    predictions_path = artifact_dir / "final_test_predictions.parquet"
    joblib.dump(bundle, feature_path)
    joblib.dump(ranker, ranker_path)
    predictions.to_parquet(predictions_path, index=False)
    stale_prediction_path = artifact_dir / "test_predictions.parquet"
    if stale_prediction_path.exists():
        stale_prediction_path.unlink()

    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    per_query.to_csv(report_dir / "per_query_metrics.csv", index=False)
    bootstrap.to_csv(report_dir / "bootstrap_comparison.csv", index=False)
    slices.to_csv(report_dir / "query_slice_metrics.csv", index=False)
    examples.to_csv(report_dir / "error_examples.csv", index=False)
    save_model_comparison_plot(comparison, report_dir / "model_comparison.png")
    save_feature_importance_plot(
        ranker,
        bundle["ranker_feature_names"],
        report_dir / "feature_importance.png",
    )

    print("Reloading saved artifacts and reproducing final rankings...")
    _verify_reloaded_predictions(
        splits["final_test"], predictions, feature_path, ranker_path
    )

    examples_file, products_file = dataset_files(config["paths"]["dataset_dir"])
    runtime_seconds = float(time.perf_counter() - started)
    metrics = {
        "project_title": "Graded-Relevance E-commerce Search Reranker",
        "seeds": {
            "sampling": int(config["sampling"]["random_seed"]),
            "model": int(config["lightgbm"]["random_state"]),
            "bootstrap": int(config["evaluation"]["bootstrap_seed"]),
        },
        "dataset": split_counts,
        "split_protocol": {
            "official_train_queries": int(
                data.loc[data["split"] == "train", "query_id"].nunique()
            ),
            "official_test_queries": official_test_queries,
            "legacy_test_queries_excluded": split_counts["legacy_test"]["queries"],
            "fresh_final_queries": split_counts["final_test"]["queries"],
        },
        "dataset_files_bytes": {
            examples_file.name: examples_file.stat().st_size,
            products_file.name: products_file.stat().st_size,
        },
        "label_gains": LABEL_GAINS,
        "selected_lexical_word_weight": selected_weight,
        "selected_lexical_validation_ndcg_at_10": selected_validation_ndcg,
        "validation_weight_scores": {
            str(weight): score for weight, score in validation_scores
        },
        "selected_ranker": {
            "candidate_name": selected_candidate["candidate_name"],
            "feature_names": selected_candidate["feature_names"],
            "truncation_level": selected_candidate["truncation_level"],
            "validation_ndcg_at_10": selected_candidate["validation_ndcg_at_10"],
            "best_iteration": selected_candidate["best_iteration"],
        },
        "low_overlap_validation_q25_mean_coverage": low_overlap_threshold,
        "models": model_metrics,
        "bootstrap_comparison": bootstrap.to_dict(orient="records"),
        "ndcg_win_tie_loss": win_tie_loss,
        "latency_metadata": latency["metadata"],
        "run": {
            "pipeline_wall_time_seconds": runtime_seconds,
            "peak_rss_mib": _peak_rss_mib(),
            "saved_artifact_reproduction": "passed",
        },
    }
    with (report_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the ESCI reranker")
    parser.add_argument("--config", default="config.yaml")
    arguments = parser.parse_args()
    run_pipeline(arguments.config)
