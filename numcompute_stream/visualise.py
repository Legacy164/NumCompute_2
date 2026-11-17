import matplotlib.pyplot as plt


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig


def plot_metric_over_time(metric_values, title, ylabel, *, save_path=None, show=False):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(metric_values) + 1), metric_values, marker="o")
    ax.set(title=title, xlabel="Chunk", ylabel=ylabel)
    return _finish(fig, save_path, show)


def compare_models(metric1, metric2, labels, *, save_path=None, show=False):
    fig, ax = plt.subplots()
    for values, label in zip((metric1, metric2), labels):
        ax.plot(range(1, len(values) + 1), values, marker="o", label=label)
    ax.set(title="Cumulative accuracy by buffer size", xlabel="Chunk", ylabel="Accuracy")
    ax.legend()
    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, *, save_path=None, show=False):
    fig, ax = plt.subplots()
    rows = range(1, len(y_true) + 1)
    ax.scatter(rows, y_true, marker="o", facecolors="none", edgecolors="C0", label="Actual")
    ax.scatter(rows, y_pred, marker="x", color="C1", label="Predicted")
    ax.set(title="Latest scored chunk", xlabel="Row", ylabel="Class")
    ax.legend()
    return _finish(fig, save_path, show)