import math

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    complement_exposure_at_k,
    evaluate_ranking,
    exact_mrr_at_k,
    exact_substitute_recall_at_k,
    irrelevant_exposure_at_k,
    ndcg_at_k,
)


def test_ndcg_uses_esci_gains():
    actual = ndcg_at_k(["S", "E", "I"], k=3)
    dcg = 0.1 + 1.0 / math.log2(3)
    ideal = 1.0 + 0.1 / math.log2(3)
    assert actual == pytest.approx(dcg / ideal)


def test_hand_calculated_ranking_metrics():
    labels = ["S", "I", "E", "C", "S", "E"]
    assert exact_mrr_at_k(labels, 10) == pytest.approx(1.0 / 3.0)
    assert exact_substitute_recall_at_k(labels, 5) == pytest.approx(3.0 / 4.0)
    assert complement_exposure_at_k(labels, 5) == pytest.approx(1.0 / 5.0)
    assert irrelevant_exposure_at_k(labels, 5) == pytest.approx(1.0 / 5.0)


def test_ndcg_truncation_uses_best_candidates_for_ideal_order():
    labels = ["S", "I", "E"]
    expected = 0.1 / (1.0 + 0.1 / math.log2(3))
    assert ndcg_at_k(labels, k=2) == pytest.approx(expected)


def test_query_without_exact_or_substitute_has_undefined_recall():
    assert np.isnan(exact_substitute_recall_at_k(["C", "I"], 5))
    frame = pd.DataFrame(
        {
            "query_id": [1, 1],
            "product_id": ["a", "b"],
            "esci_label": ["C", "I"],
            "score": [1.0, 0.0],
        }
    )
    result = evaluate_ranking(frame, "score")
    assert result["exact_mrr_at_10"] == 0.0
    assert result["es_recall_at_5"] is None
    assert result["es_recall_queries"] == 0
    assert result["ndcg_at_10"] == pytest.approx(1.0)
    assert ndcg_at_k(["I", "I"], 10) == 0.0
