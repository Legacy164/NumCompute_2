import numpy as np
import pytest

from numcompute_stream.metrics import StreamingAUC, StreamingMetrics
from numcompute_stream.preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


@pytest.mark.parametrize("buffer_size", [0, 1.5])
def test_tree_rejects_invalid_buffer_size(buffer_size):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(buffer_size=buffer_size)


def test_tree_rejects_different_feature_count_while_buffering():
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit([[0]], [0])

    with pytest.raises(ValueError):
        tree.partial_fit([[1, 2]], [1])


def test_tree_copies_buffered_input():
    tree = DecisionTreeClassifier(max_depth=1, buffer_size=4)
    X = np.array([[0.0], [1.0]])
    tree.partial_fit(X, [0, 0])
    X[:] = 100.0

    tree.partial_fit([[4.0], [5.0]], [1, 1])
    assert tree.predict([[0.0], [5.0]]).tolist() == [0, 1]


def test_standard_scaler_fit_replaces_stream_history():
    scaler = StandardScaler().partial_fit([[100.0]])
    scaler.fit([[2.0], [4.0]])
    np.testing.assert_allclose(scaler.transform([[2.0], [4.0]]), [[-1], [1]])


def test_minmax_fit_replaces_stream_history():
    scaler = MinMaxScaler().partial_fit([[100.0]])
    scaler.fit([[2.0], [4.0]])
    np.testing.assert_allclose(scaler.transform([[2.0], [4.0]]), [[0], [1]])


def test_imputer_ignores_custom_missing_value_across_chunks():
    imputer = Imputer(missing_values=-999).partial_fit([[2.0], [-999]])
    imputer.partial_fit([[6.0]])
    np.testing.assert_allclose(imputer.transform([[-999]]), [[4.0]])


def test_encoder_ignores_unseen_category():
    encoder = OneHotEncoder(handle_unknown="ignore").partial_fit(["blue", "red"])
    np.testing.assert_array_equal(encoder.transform(["amber"]), [[0, 0]])


def test_metrics_reject_unknown_label():
    with pytest.raises(ValueError):
        StreamingMetrics().update([0], [2])


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        StreamingMetrics().update([0, 1], [0])


def test_auc_scores_tied_predictions_as_half():
    assert StreamingAUC().update([0, 1], [0.5, 0.5]).result() == pytest.approx(0.5)


def test_auc_rejects_nan_score():
    with pytest.raises(ValueError):
        StreamingAUC().update([1], [np.nan])