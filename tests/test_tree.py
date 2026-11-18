import math
import pytest
import numpy as np
from numcompute.utils import NotFittedError
from numcompute_stream.tree import gini, entropy, best_numeric_split, DecisionTreeClassifier

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

def test_decision_tree():
    
    X = [[9, 0], [9, 1], [9, 4], [9, 5]]
    y = [0, 0, 1, 1]
    tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert tree.root_["feature"] == 1
    assert list(tree.predict([[9, 1], [9, 4]])) == [0, 1]

    X = [[0], [1], [2], [3]]
    y = [0, 1, 1, 0]
    shallow = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert list(shallow.predict(X)) != y
    deep = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert list(deep.predict(X)) == y

    X = [[3], [3], [3]]
    y = [0, 1, 0]
    tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert tree.root_["leaf"]
    assert list(tree.predict([[3], [10]])) == [0, 0]

def test_partial_fit_retrains_when_buffer_reaches_limit():
    tree = DecisionTreeClassifier(max_depth=1, buffer_size=4)
    tree.fit([[0], [1], [4], [5]], [0, 0, 1, 1])

    tree.partial_fit([[0], [1]], [1, 1])
    assert tree.predict([[0], [5]]).tolist() == [0, 1]

    tree.partial_fit([[4], [5], [6]], [0, 0, 0])
    assert tree.predict([[0], [5]]).tolist() == [1, 0]

    tree.partial_fit([[0], [1]], [0, 0])
    assert tree.predict([[0], [5]]).tolist() == [1, 0]

def test_max_features_limits_split_search():
    X = [[0, 0], [1, 0], [4, 0], [5, 0]]
    y = [0, 0, 1, 1]

    full_tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
    limited_tree = DecisionTreeClassifier(
        max_depth=1, max_features=1, random_state=0
    ).fit(X, y)

    assert full_tree.root_["feature"] == 0
    assert limited_tree.root_["leaf"]

@pytest.mark.parametrize("parameter,value,error", [
    ("max_depth", -1, ValueError),
    ("max_depth", 1.5, TypeError),
    ("max_depth", True, TypeError),
    ("min_samples_split", 0, ValueError),
    ("min_samples_split", 1, ValueError),
    ("min_samples_split", 2.5, TypeError),
    ("min_samples_split", True, TypeError),
    ("impurity", "gini", TypeError),
])
def test_tree_rejects_invalid_parameters(parameter, value, error):
    with pytest.raises(error, match=parameter):
        DecisionTreeClassifier(**{parameter: value})


def test_predict_before_fit_raises_not_fitted():
    with pytest.raises(NotFittedError):
        DecisionTreeClassifier().predict([[1.0]])

@pytest.mark.parametrize("impurity", [gini, entropy])
def test_split_criteria_and_ties(impurity):
    assert best_numeric_split(
        [0, 1, 4, 5], [0, 0, 1, 1], impurity
    ) == (2.5, 0.0)

    threshold, _ = best_numeric_split(
        [0, 1, 2], [0, 1, 0], impurity
    )
    assert threshold == 0.5

    assert best_numeric_split(
        [3, 3, 3], [0, 1, 0], impurity
    ) is None

    assert best_numeric_split([], [], impurity) is None

def test_prediction_preserves_order_and_labels():
    tree = DecisionTreeClassifier().fit(
        [[0], [1], [4], [5]],
        ["a", "a", "long", "long"],
    )

    assert tree.predict([[5], [0], [4], [1]]).tolist() == [
        "long", "a", "long", "a"
    ]

    assert tree.predict(np.empty((0, 1))).shape == (0,)

def test_custom_impurity_uses_loop_and_matches_gini():
    x = [0, 1, 2, 4, 5, 7]
    y = [0, 0, 1, 1, 1, 0]
    expected = best_numeric_split(x, y, gini)
    result = best_numeric_split(x, y, lambda labels: gini(labels))
    assert result[0] == expected[0]
    assert math.isclose(result[1], expected[1])  

