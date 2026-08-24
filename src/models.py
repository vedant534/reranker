import lightgbm as lgb

from src.data import group_sizes
from src.metrics import LABEL_LEVELS, LIGHTGBM_LABEL_GAINS


def train_ranker(
    train_frame,
    train_features,
    validation_frame,
    validation_features,
    config,
    feature_names=None,
    truncation_level=None,
):
    selected_features = list(
        train_features.columns if feature_names is None else feature_names
    )
    train_order = train_frame.sort_values(["query_id", "product_id"]).index
    validation_order = validation_frame.sort_values(["query_id", "product_id"]).index
    train_sorted = train_frame.loc[train_order]
    validation_sorted = validation_frame.loc[validation_order]
    x_train = train_features.loc[train_order, selected_features]
    x_validation = validation_features.loc[validation_order, selected_features]
    y_train = train_sorted["esci_label"].map(LABEL_LEVELS).to_numpy()
    y_validation = validation_sorted["esci_label"].map(LABEL_LEVELS).to_numpy()

    parameters = dict(config)
    early_stopping_rounds = int(parameters.pop("early_stopping_rounds"))
    parameters["label_gain"] = LIGHTGBM_LABEL_GAINS
    if truncation_level is not None:
        parameters["lambdarank_truncation_level"] = int(truncation_level)
    ranker = lgb.LGBMRanker(**parameters)
    ranker.fit(
        x_train,
        y_train,
        group=group_sizes(train_sorted),
        eval_set=[(x_validation, y_validation)],
        eval_group=[group_sizes(validation_sorted)],
        eval_metric="ndcg",
        eval_at=[10],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return ranker
