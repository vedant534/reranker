import numpy as np


LABEL_GAINS = {"I": 0.0, "C": 0.01, "S": 0.1, "E": 1.0}


def ndcg_at_k(labels, k=10):
    gains = np.asarray([LABEL_GAINS[label] for label in labels], dtype=float)[:k]
    if len(gains) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains / discounts))
    ideal = np.sort(np.asarray([LABEL_GAINS[label] for label in labels], dtype=float))[::-1][:k]
    idcg = float(np.sum(ideal / discounts))
    return dcg / idcg if idcg > 0 else 0.0


def exact_mrr_at_k(labels, k=10):
    for rank, label in enumerate(labels[:k], start=1):
        if label == "E":
            return 1.0 / rank
    return 0.0


def exact_substitute_recall_at_k(labels, k=5):
    total = sum(label in {"E", "S"} for label in labels)
    if total == 0:
        return np.nan
    return sum(label in {"E", "S"} for label in labels[:k]) / total


def bad_exposure_at_k(labels, k=5):
    top = labels[:k]
    if not top:
        return 0.0
    return sum(label in {"C", "I"} for label in top) / len(top)


def evaluate_ranking(frame, score_column):
    values = {"ndcg_at_10": [], "exact_mrr_at_10": [], "es_recall_at_5": [], "ci_exposure_at_5": []}
    for _, group in frame.groupby("query_id", sort=False):
        ranked = group.sort_values([score_column, "product_id"], ascending=[False, True], kind="mergesort")
        labels = ranked["esci_label"].tolist()
        values["ndcg_at_10"].append(ndcg_at_k(labels, 10))
        values["exact_mrr_at_10"].append(exact_mrr_at_k(labels, 10))
        values["es_recall_at_5"].append(exact_substitute_recall_at_k(labels, 5))
        values["ci_exposure_at_5"].append(bad_exposure_at_k(labels, 5))
    result = {}
    for name, metric_values in values.items():
        array = np.asarray(metric_values, dtype=float)
        result[name] = float(np.nanmean(array)) if np.any(~np.isnan(array)) else None
    result["n_queries"] = int(frame["query_id"].nunique())
    result["es_recall_queries"] = int(np.sum(~np.isnan(values["es_recall_at_5"])))
    return result

