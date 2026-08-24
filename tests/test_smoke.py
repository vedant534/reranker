import pandas as pd

from src.data import group_sizes, sample_query_splits
from src.features import fit_feature_bundle, transform_features
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
        {"train_queries": 4, "validation_queries": 2, "test_queries": 2, "random_seed": 42},
    )
    train_ids = set(splits["train"]["query_id"])
    validation_ids = set(splits["validation"]["query_id"])
    test_ids = set(splits["test"]["query_id"])
    assert not (train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids)
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
            "label_gain": [0.0, 0.01, 0.1, 1.0],
            "early_stopping_rounds": 3,
        },
    )
    bundle["lexical_word_weight"] = 0.5
    query_group = splits["test"].groupby("query_id").get_group(next(iter(test_ids)))
    ranked = predict_reranked(query_group["query"].iloc[0], query_group, bundle, ranker)
    assert len(ranked) == len(query_group)
    assert ranked["predicted_score"].is_monotonic_decreasing
    assert {"lexical_rank", "model_rank"}.issubset(ranked.columns)

