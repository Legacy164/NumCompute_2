import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


def test_metric_plot_saves_file(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    path = tmp_path / "accuracy.png"

    fig = plot_metric_over_time(
        [0.5, 0.75],
        "Accuracy",
        "Score",
        save_path=path,
        show=True,
    )

    try:
        ax = fig.axes[0]
        np.testing.assert_array_equal(
            ax.lines[0].get_ydata(), [0.5, 0.75]
        )
        np.testing.assert_array_equal(
            ax.lines[0].get_xdata(), [1, 2]
        )
        assert ax.get_title() == "Accuracy"
        assert ax.get_ylabel() == "Score"
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert shown == [True]
    finally:
        plt.close(fig)


def test_compare_models():
    fig = compare_models(
        [0.5, 0.6],
        [0.4, 0.8],
        ["Buffer 30", "Buffer 60"],
    )

    try:
        ax = fig.axes[0]
        np.testing.assert_array_equal(
            ax.lines[0].get_ydata(), [0.5, 0.6]
        )
        np.testing.assert_array_equal(
            ax.lines[1].get_ydata(), [0.4, 0.8]
        )
        assert [
            text.get_text() for text in ax.get_legend().get_texts()
        ] == ["Buffer 30", "Buffer 60"]
    finally:
        plt.close(fig)


def test_predictions_vs_ground_truth():
    fig = plot_predictions_vs_ground_truth(
        [0, 1, 0], [0, 0, 1]
    )

    try:
        actual, predicted = fig.axes[0].collections
        np.testing.assert_array_equal(
            actual.get_offsets(), [[1, 0], [2, 1], [3, 0]]
        )
        np.testing.assert_array_equal(
            predicted.get_offsets(), [[1, 0], [2, 0], [3, 1]]
        )
    finally:
        plt.close(fig)