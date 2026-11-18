import numpy as np

from numcompute.metrics import confusion_matrix


def _final_step(model):
    """Return a pipeline's last step, or the model itself."""
    steps = getattr(model, "steps", None)
    return steps[-1][1] if steps else model

def _macro_scores(y_true, y_pred):
    """Macro precision, recall and F1 for one chunk."""
    labels = np.union1d(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    correct = matrix.diagonal().astype(float)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    both = predicted + actual
    zeros = np.zeros_like(correct)
    precision = np.divide(correct, predicted, out=zeros.copy(), where=predicted != 0)
    recall = np.divide(correct, actual, out=zeros.copy(), where=actual != 0)
    f1 = np.divide(2 * correct, both, out=zeros.copy(), where=both != 0)
    return {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }

class StreamTrainer:
    """Score each chunk with the current model, then train on it, and log the results."""

    def __init__(self, pipeline):
        """Accept a Pipeline or a bare model with partial_fit and predict."""
        self.pipeline = pipeline
        self.history = []
        self._correct = 0
        self._scored_rows = 0

    def _ready(self):
        """True once the final step has a trained tree."""
        return getattr(_final_step(self.pipeline), "root_", None) is not None

    def score_chunk(self, X, y):
        """Accuracy on X (n, f) and y (n,) without training, or None before the first tree."""
        if not self._ready():
            return None
        predicted = self.pipeline.predict(X)
        return float(np.mean(predicted == np.asarray(y)))

    def fit_chunk(self, X, y):
        """Score then train on one chunk; return and store the log entry."""
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a nonempty 2-D array")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must have one label per row of X")

        scores = {"accuracy": None, "precision": None, "recall": None, "f1": None}
        if self._ready():
            predicted = self.pipeline.predict(X)
            correct = int(np.count_nonzero(predicted == y))
            self._correct += correct
            self._scored_rows += len(y)
            scores = {"accuracy": correct / len(y), **_macro_scores(y, predicted)}

        self.pipeline.partial_fit(X, y)

        entry = {
            "chunk": len(self.history) + 1,
            "rows": len(y),
            **scores,
            "cumulative_accuracy": (
                self._correct / self._scored_rows if self._scored_rows else None
            ),
            "chunk_bytes": X.nbytes + y.nbytes,
            "buffer_bytes": sum(
                features.nbytes + labels.nbytes
                for features, labels in getattr(self.pipeline, "_chunks", [])
            ),
        }
        self.history.append(entry)
        return entry