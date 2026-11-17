from pathlib import Path
from time import perf_counter
from timeit import repeat

import numpy as np

from numcompute.io import iter_csv_chunks
from numcompute_stream.metrics import StreamingMetrics
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer
from numcompute_stream.tree import DecisionTreeClassifier


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


if __name__ == "__main__":
    benchmark_confusion_matrix()
    benchmark_buffer_sizes()