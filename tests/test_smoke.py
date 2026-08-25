import joblib
import numpy as np
import pandas as pd
import pytest

from run_pipeline import _add_scores, _verify_reloaded_predictions
from src.data import group_sizes, sample_query_splits
from src.features import FEATURE_NAMES, TFIDF_FEATURE_NAMES, fit_feature_bundle, transform_features
from src.models import train_ranker
from src.predict import predict_reranked


def synthetic_data():
    rows = []
    labels = ["E", "S", "C", "I"]
    for split, query_ids in (("train", range(1, 7)), ("test", range(20, 23))):
        for query_id in query_ids:
            query = f"model {query_id} shoe"
            for index, label in enumerate(labels):
                rows.append(
                    {
                        "example_id": len(rows),
                        "query_id": query_id,
                        "query": query,
                        "product_id": f"p-{query_id}-{index}",
                        "product_locale": "us",
                        "product_title": query if label == "E" else f"different product {index}",
                        "product_brand": "model",
                        "product_color": "red" if index == 0 else "",
                        "esci_label": label,
                        "small_version": 1,
                        "split": split,
                    }
                )
    return pd.DataFrame(rows)


def test_query_splits_groups_and_prediction_smoke():
    data = synthetic_data()
    splits = sample_query_splits(
        data,
        {
            "train_queries": 4,
            "validation_queries": 2,
            "legacy_test_queries": 2,
            "random_seed": 42,
        },
    )
    repeated = sample_query_splits(
        data,
        {
            "train_queries": 4,
            "validation_queries": 2,
            "legacy_test_queries": 2,
            "random_seed": 42,
        },
    )
    train_ids = set(splits["train"]["query_id"])
    validation_ids = set(splits["validation"]["query_id"])
    legacy_ids = set(splits["legacy_test"]["query_id"])
    final_ids = set(splits["final_test"]["query_id"])
    assert legacy_ids == set(repeated["legacy_test"]["query_id"])
    assert final_ids == set(repeated["final_test"]["query_id"])
    split_sets = [train_ids, validation_ids, legacy_ids, final_ids]
    assert all(
        not left & right
        for index, left in enumerate(split_sets)
        for right in split_sets[index + 1 :]
    )
    assert group_sizes(splits["train"]).sum() == len(splits["train"])

    config = {
        "word_ngram_range": [1, 2],
        "word_max_features": 100,
        "char_ngram_range": [3, 4],
        "char_max_features": 200,
        "min_df": 1,
    }
    bundle = fit_feature_bundle(splits["train"], config)
    expected_corpus = len(
        pd.concat(
            [
                splits["train"]["query"].drop_duplicates(),
                splits["train"]["product_title"].drop_duplicates(),
            ],
            ignore_index=True,
        ).drop_duplicates()
    )
    assert bundle["fit_corpus_size"] == expected_corpus
    train_features = transform_features(splits["train"], bundle)
    validation_features = transform_features(splits["validation"], bundle)
    ranker = train_ranker(
        splits["train"],
        train_features,
        splits["validation"],
        validation_features,
        {
            "objective": "lambdarank",
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 1,
            "random_state": 42,
            "n_jobs": 1,
            "verbosity": -1,
            "early_stopping_rounds": 3,
        },
    )
    bundle["lexical_word_weight"] = 0.5
    bundle["ranker_feature_names"] = FEATURE_NAMES
    bundle["inference_threads"] = 1
    query_group = splits["final_test"].groupby("query_id").get_group(next(iter(final_ids)))
    ranked = predict_reranked(query_group["query"].iloc[0], query_group, bundle, ranker)
    assert len(ranked) == len(query_group)
    assert ranked["predicted_score"].is_monotonic_decreasing
    assert {"lexical_rank", "model_rank"}.issubset(ranked.columns)


class TiedRanker:
    best_iteration_ = 1

    def __init__(self):
        self.columns = None

    def predict(self, features, num_iteration=None, num_threads=None):
        self.columns = list(features.columns)
        return np.zeros(len(features))


def test_verification_reloads_saved_prediction_parquet(tmp_path):
    candidates = pd.DataFrame(
        {
            "query_id": [1, 1],
            "query": ["fitbit charge 3", "fitbit charge 3"],
            "product_id": ["tracker", "band"],
            "product_title": ["Fitbit Charge 3 tracker", "Fitbit Charge 3 band"],
            "product_brand": ["Fitbit", "Accessory Co"],
            "product_color": ["black", "black"],
        }
    )
    bundle = fit_feature_bundle(
        candidates,
        {
            "word_ngram_range": [1, 1],
            "word_max_features": 20,
            "char_ngram_range": [3, 3],
            "char_max_features": 40,
            "min_df": 1,
        },
    )
    bundle["lexical_word_weight"] = 0.5
    bundle["ranker_feature_names"] = TFIDF_FEATURE_NAMES
    bundle["inference_threads"] = 1
    ranker = TiedRanker()
    features = transform_features(candidates, bundle)
    predictions = _add_scores(candidates, features, bundle, ranker)

    bundle_path = tmp_path / "feature_bundle.joblib"
    ranker_path = tmp_path / "ranker.joblib"
    predictions_path = tmp_path / "final_test_predictions.parquet"
    joblib.dump(bundle, bundle_path)
    joblib.dump(ranker, ranker_path)
    predictions.to_parquet(predictions_path, index=False)

    _verify_reloaded_predictions(
        candidates, predictions_path, bundle_path, ranker_path
    )

    corrupted = pd.read_parquet(predictions_path)
    corrupted.loc[0, "lexical_score"] = np.float32(
        corrupted.loc[0, "lexical_score"] + 0.1
    )
    corrupted.to_parquet(predictions_path, index=False)
    with pytest.raises(AssertionError, match="lexical_score"):
        _verify_reloaded_predictions(
            candidates, predictions_path, bundle_path, ranker_path
        )


def test_tied_scores_use_product_id_after_shuffled_input():
    training = pd.DataFrame(
        {
            "query": ["anchor"],
            "product_title": ["anchor"],
            "product_brand": [""],
            "product_color": [""],
        }
    )
    bundle = fit_feature_bundle(
        training,
        {
            "word_ngram_range": [1, 1],
            "word_max_features": 10,
            "char_ngram_range": [3, 3],
            "char_max_features": 10,
            "min_df": 1,
        },
    )
    bundle["lexical_word_weight"] = 0.5
    bundle["ranker_feature_names"] = TFIDF_FEATURE_NAMES
    bundle["inference_threads"] = 1
    candidates = pd.DataFrame(
        {
            "product_id": ["b", "a", "c"],
            "product_title": ["", "", ""],
            "product_brand": ["", "", ""],
            "product_color": ["", "", ""],
        }
    )
    ranker = TiedRanker()
    ranked = predict_reranked("unseen", candidates, bundle, ranker)
    assert ranked["product_id"].tolist() == ["a", "b", "c"]
    assert ranked["lexical_rank"].tolist() == [1, 2, 3]
    assert ranked["model_rank"].tolist() == [1, 2, 3]
    assert ranker.columns == TFIDF_FEATURE_NAMES
