import lightgbm as lgb

from src.data import group_sizes


LABEL_LEVELS = {"I": 0, "C": 1, "S": 2, "E": 3}


def train_ranker(train_frame, train_features, validation_frame, validation_features, config):
    train_order = train_frame.sort_values(["query_id", "product_id"]).index
    validation_order = validation_frame.sort_values(["query_id", "product_id"]).index
    train_sorted = train_frame.loc[train_order]
    validation_sorted = validation_frame.loc[validation_order]
    x_train = train_features.loc[train_order]
    x_validation = validation_features.loc[validation_order]
    y_train = train_sorted["esci_label"].map(LABEL_LEVELS).to_numpy()
    y_validation = validation_sorted["esci_label"].map(LABEL_LEVELS).to_numpy()

    parameters = dict(config)
    early_stopping_rounds = int(parameters.pop("early_stopping_rounds"))
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

