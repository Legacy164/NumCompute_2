from pathlib import Path
from time import perf_counter
from timeit import repeat

import numpy as np

from numcompute.io import iter_csv_chunks
from numcompute_stream.metrics import StreamingMetrics
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer
from numcompute_stream.tree import DecisionTreeClassifier, best_numeric_split, gini

def _loop_gini(labels):
    """Same value as gini; passing it makes best_numeric_split use its loop."""
    return gini(labels)

def _best_time(run, repeats=3):
    """Fastest of several single runs, in seconds."""
    return min(repeat(run, number=1, repeat=repeats))

def benchmark_confusion_matrix():
    rng = np.random.default_rng(0)
    actual = rng.integers(0, 2, size=(100, 1000))
    predicted = rng.integers(0, 2, size=actual.shape)

    def loop():
        matrix = [[0, 0], [0, 0]]
        for true_chunk, pred_chunk in zip(actual, predicted):
            for true, pred in zip(true_chunk, pred_chunk):
                matrix[true][pred] += 1
        return np.asarray(matrix)

    def numpy_updates():
        metrics = StreamingMetrics()
        for true_chunk, pred_chunk in zip(actual, predicted):
            metrics.update(true_chunk, pred_chunk)
        return metrics.result()["confusion_matrix"]

    np.testing.assert_array_equal(loop(), numpy_updates())
    for name, run in [("Python loop", loop), ("NumPy updates", numpy_updates)]:
        seconds = min(repeat(run, number=1, repeat=3))
        print(f"{name}: {seconds * 1000:.2f} ms")


def benchmark_buffer_sizes():
    csv_path = Path(__file__).resolve().parents[1] / "demo" / "sample_data.csv"
    chunks = list(iter_csv_chunks(
        csv_path, chunksize=30, skip_header=1, usecols=(0, 1, 4)
    ))

    for buffer_size in (30, 60):
        pipe = Pipeline([
            ("impute", Imputer()),
            ("tree", DecisionTreeClassifier(max_depth=3, buffer_size=buffer_size)),
        ])
        correct = scored_rows = 0
        retrain_times = []

        for index, chunk in enumerate(chunks):
            X, y = chunk[:, :2], chunk[:, 2]

            if index >= 2:
                correct += np.count_nonzero(pipe.predict(X) == y)
                scored_rows += len(y)

            previous_root = pipe.named_steps["tree"].root_
            start = perf_counter()
            pipe.partial_fit(X, y)
            elapsed = perf_counter() - start

            if pipe.named_steps["tree"].root_ is not previous_root:
                retrain_times.append(elapsed)

        print(
            f"Buffer {buffer_size}: {correct / scored_rows:.1%} accuracy "
            f"on {scored_rows} rows, {len(retrain_times)} retrains, "
            f"{1000 * np.mean(retrain_times):.2f} ms per retraining update"
        )

def benchmark_split_search():
    """Loop vs vectorised split search on one feature."""
    print("Split search on one feature (Gini):")
    for n in (500, 2000, 8000):
        rng = np.random.default_rng(n)
        x = rng.normal(size=n)
        y = (x + rng.normal(scale=0.5, size=n) > 0).astype(int)
        assert np.allclose(best_numeric_split(x, y, _loop_gini), best_numeric_split(x, y, gini))

        loop_time = _best_time(lambda: best_numeric_split(x, y, _loop_gini))
        fast_time = _best_time(lambda: best_numeric_split(x, y, gini))
        print(
            f"  {n:>5} rows: loop {loop_time * 1000:.2f} ms, "
            f"vectorised {fast_time * 1000:.2f} ms, {loop_time / fast_time:.0f}x faster"
        )

def benchmark_tree_fit():
    """Fit time for the same tree with loop and vectorised splits."""
    print("Fitting a depth-5 tree on 5 features:")
    for n in (500, 2000):
        rng = np.random.default_rng(n)
        X = rng.normal(size=(n, 5))
        y = (X[:, 0] + X[:, 1] ** 2 + rng.normal(scale=0.3, size=n) > 1).astype(int)

        def loop_fit():
            return DecisionTreeClassifier(max_depth=5, impurity=_loop_gini).fit(X, y)

        def fast_fit():
            return DecisionTreeClassifier(max_depth=5).fit(X, y)

        assert np.array_equal(loop_fit().predict(X), fast_fit().predict(X))
        loop_time = _best_time(loop_fit, repeats=1)
        fast_time = _best_time(fast_fit)
        print(
            f"  {n:>5} rows: loop {loop_time * 1000:.1f} ms, "
            f"vectorised {fast_time * 1000:.1f} ms, {loop_time / fast_time:.0f}x faster"
        )


def _drifting_stream(n_rows=10_000, chunk_size=100, seed=0):
    """Two-feature chunks whose target rule changes halfway, with 5% label noise."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_rows, 2))
    first_half = np.arange(n_rows) < n_rows // 2
    y = np.where(first_half, X[:, 0] + X[:, 1] > 0, X[:, 0] - X[:, 1] > 0).astype(int)
    noise = rng.random(n_rows) < 0.05
    y[noise] = 1 - y[noise]
    return [(X[i:i + chunk_size], y[i:i + chunk_size]) for i in range(0, n_rows, chunk_size)]


def benchmark_buffer_sizes_drift():
    """Accuracy and retraining time for several buffer sizes on a changing stream."""
    chunks = _drifting_stream()
    first_scored = 20
    print("Synthetic stream: 10,000 rows, rule changes at row 5,000, scored from row 2,001:")
    for buffer_size in (100, 250, 500, 1000, 2000):
        tree = DecisionTreeClassifier(max_depth=6, buffer_size=buffer_size)
        correct = scored_rows = 0
        retrain_times = []

        for index, (X, y) in enumerate(chunks):
            if index >= first_scored:
                correct += np.count_nonzero(tree.predict(X) == y)
                scored_rows += len(y)

            previous_root = tree.root_
            start = perf_counter()
            tree.partial_fit(X, y)
            elapsed = perf_counter() - start
            if tree.root_ is not previous_root:
                retrain_times.append(elapsed)

        print(
            f"  Buffer {buffer_size:>4}: {correct / scored_rows:.1%} accuracy, "
            f"{len(retrain_times)} retrains, "
            f"{1000 * np.mean(retrain_times):.2f} ms per retraining update"
        )


if __name__ == "__main__":
    benchmark_confusion_matrix()
    benchmark_split_search()
    benchmark_tree_fit()
    benchmark_buffer_sizes()
    benchmark_buffer_sizes_drift()