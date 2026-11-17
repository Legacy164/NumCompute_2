import numpy as np

class StreamTrainer:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.history = []
        self._correct = 0
        self._scored_rows = 0

    def score_chunk(self, X, y):
        """Return the correct prediction count, or None during warm-up."""
        tree = self.pipeline.steps[-1][1]
        if tree.root_ is None:
            return None

        predicted = self.pipeline.predict(X)
        return int(np.count_nonzero(predicted == y))

    def fit_chunk(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a nonempty 2-D array")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must have one label per row of X")

        correct = self.score_chunk(X, y)
        self.pipeline.partial_fit(X, y)

        if correct is not None:
            self._correct += correct
            self._scored_rows += len(y)

        entry = {
            "chunk": len(self.history) + 1,
            "rows": len(y),
            "accuracy": None if correct is None else correct / len(y),
            "cumulative_accuracy": (
                self._correct / self._scored_rows if self._scored_rows else None
            ),
            "chunk_bytes": X.nbytes + y.nbytes,
            "buffer_bytes": sum(
                features.nbytes + labels.nbytes
                for features, labels in self.pipeline._chunks
            ),
        }
        self.history.append(entry)
        return entry