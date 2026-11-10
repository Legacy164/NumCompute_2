import pytest

from numcompute_stream.tree import entropy, gini


def test_gini_imbalanced_labels():
    assert gini([0, 0, 0, 1]) == 0.375


def test_entropy_imbalanced_labels():
    assert entropy([0, 0, 0, 1]) == pytest.approx(0.8112781244591328)


def test_gini_balanced_nonconsecutive_labels():
    assert gini([2, 2, 7, 7]) == 0.5


def test_entropy_balanced_nonconsecutive_labels():
    assert entropy([2, 2, 7, 7]) == 1.0