import math

from numcompute_stream.tree import gini, entropy, best_numeric_split


def test_gini_and_entropy():
    assert math.isclose(gini([0, 0, 0, 1]), 0.375)
    assert math.isclose(entropy([0, 0, 0, 1]), 0.8112781244591328)
    assert math.isclose(gini([2, 2, 7, 7]), 0.5)
    assert math.isclose(entropy([2, 2, 7, 7]), 1.0)


def test_best_numeric_split():
    result = best_numeric_split([0, 1, 4, 5], [0, 0, 1, 1])
    assert result == (2.5, 0.0), result
    assert best_numeric_split([0, 0, 2, 2], [0, 0, 1, 1]) == (1.0, 0.0)
    assert best_numeric_split([3, 3, 3], [0, 1, 0]) is None

    threshold, score = best_numeric_split([0, 1, 2], [0, 1, 0])
    assert threshold == 0.5
    assert math.isclose(score, 1 / 3)


if __name__ == "__main__":
    test_gini_and_entropy()
    test_best_numeric_split()
    print("All tests passed")