import numpy as np

from numcompute_stream.metrics import StreamingMetrics, StreamingAUC

def test_metrics_accumulate_across_chunks():
    metrics = StreamingMetrics()
    metrics.update([0, 1], [0, 0])
    metrics.update([1, 1], [1, 1])

    result = metrics.result()
    assert result["accuracy"] == 0.75
    np.testing.assert_array_equal(result["confusion_matrix"], [[1, 0], [1, 2]])
    np.testing.assert_allclose(result["precision"], [0.5, 1.0])
    np.testing.assert_allclose(result["recall"], [1.0, 2 / 3])
    np.testing.assert_allclose(result["f1"], [2 / 3, 0.8])


def test_rolling_metrics_keep_latest_rows():
    metrics = StreamingMetrics(window_size=3)
    metrics.update([0, 0], [1, 1])
    metrics.update([1, 1], [1, 1])

    np.testing.assert_array_equal(metrics.result()["confusion_matrix"], [[0, 1], [0, 2]])


def test_reset_clears_metrics():
    metrics = StreamingMetrics(window_size=2)
    metrics.update([1], [1])
    metrics.reset()
    metrics.update([0], [0])

    np.testing.assert_array_equal(metrics.result()["confusion_matrix"], [[1, 0], [0, 0]])

def test_auc_compares_scores_across_chunks():
    metric = StreamingAUC().update([0, 0], [0.1, 0.4])
    assert np.isnan(metric.result())

    metric.update([1, 1], [0.35, 0.8])
    assert metric.result() == 0.75


def test_rolling_auc_discards_old_scores():
    metric = StreamingAUC(window_size=2)
    metric.update([1, 0], [0.1, 0.9])
    assert metric.result() == 0.0

    metric.update([0, 1], [0.2, 0.8])
    assert metric.result() == 1.0

    metric.reset()
    assert np.isnan(metric.result())