import math

import pytest

from src.metrics import ndcg_at_k


def test_ndcg_uses_esci_gains():
    actual = ndcg_at_k(["S", "E", "I"], k=3)
    dcg = 0.1 + 1.0 / math.log2(3)
    ideal = 1.0 + 0.1 / math.log2(3)
    assert actual == pytest.approx(dcg / ideal)
