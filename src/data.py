from pathlib import Path

import numpy as np
import pandas as pd
import yaml


EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"


def load_config(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    base = config_path.parent
    for key in ("dataset_dir", "artifact_dir", "report_dir"):
        path = Path(config["paths"][key])
        config["paths"][key] = str(path if path.is_absolute() else (base / path).resolve())
    return config


def dataset_files(dataset_dir):
    dataset_dir = Path(dataset_dir)
    return dataset_dir / EXAMPLES_FILE, dataset_dir / PRODUCTS_FILE


def load_esci_data(dataset_dir):
    examples_path, products_path = dataset_files(dataset_dir)
    missing = [str(path) for path in (examples_path, products_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing ESCI parquet file(s): " + ", ".join(missing))

    example_columns = [
        "example_id",
        "query",
        "query_id",
        "product_id",
        "product_locale",
        "esci_label",
        "small_version",
        "split",
    ]
    product_columns = [
        "product_id",
        "product_locale",
        "product_title",
        "product_brand",
        "product_color",
    ]
    examples = pd.read_parquet(
        examples_path,
        columns=example_columns,
        filters=[("small_version", "==", 1), ("product_locale", "==", "us")],
    )
    products = pd.read_parquet(
        products_path,
        columns=product_columns,
        filters=[("product_locale", "==", "us")],
    )
    data = examples.merge(
        products,
        on=["product_id", "product_locale"],
        how="left",
        validate="many_to_one",
    )
    if data.empty:
        raise ValueError("The ESCI US reduced-ranking filter returned no rows")
    if data["product_title"].isna().all():
        raise ValueError("Product merge failed: all product titles are missing")
    for column in ("query", "product_title", "product_brand", "product_color"):
        data[column] = data[column].fillna("").astype(str)
    if not set(data["esci_label"].unique()).issubset({"E", "S", "C", "I"}):
        raise ValueError("Unexpected ESCI labels found")
    return data


def sample_query_splits(data, sampling):
    seed = int(sampling["random_seed"])
    train_count = int(sampling["train_queries"])
    validation_count = int(sampling["validation_queries"])
    test_count = int(sampling["test_queries"])

    official_train = np.sort(data.loc[data["split"] == "train", "query_id"].unique())
    official_test = np.sort(data.loc[data["split"] == "test", "query_id"].unique())
    official_overlap = set(official_train) & set(official_test)
    if official_overlap:
        raise ValueError("Official train/test query IDs are not disjoint")
    if train_count + validation_count > len(official_train):
        raise ValueError("Requested train + validation queries exceed the official training split")
    if test_count > len(official_test):
        raise ValueError("Requested test queries exceed the official test split")

    rng = np.random.default_rng(seed)
    shuffled_train = rng.permutation(official_train)
    validation_ids = set(shuffled_train[:validation_count].tolist())
    train_ids = set(shuffled_train[validation_count : validation_count + train_count].tolist())
    test_ids = set(rng.permutation(official_test)[:test_count].tolist())

    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Sampled query splits overlap")

    split_data = {
        "train": data[data["query_id"].isin(train_ids)].copy(),
        "validation": data[data["query_id"].isin(validation_ids)].copy(),
        "test": data[data["query_id"].isin(test_ids)].copy(),
    }
    expected = {"train": train_count, "validation": validation_count, "test": test_count}
    for name, frame in split_data.items():
        if frame["query_id"].nunique() != expected[name]:
            raise AssertionError(f"{name} query sampling lost query IDs")
        split_data[name] = frame.sort_values(["query_id", "product_id"]).reset_index(drop=True)
    return split_data


def group_sizes(frame):
    sizes = frame.groupby("query_id", sort=False).size().to_numpy(dtype=np.int32)
    if int(sizes.sum()) != len(frame):
        raise AssertionError("LightGBM group sizes do not sum to the number of rows")
    return sizes

