import numpy as np

from numcompute.metrics import confusion_matrix, roc_auc_score


class StreamingMetrics:
    """Track accuracy and per-class precision, recall, and F1."""

    def __init__(self, labels=(0, 1), window_size=None):
        self.labels = np.asarray(labels)
        if self.labels.ndim != 1 or self.labels.size == 0:
            raise ValueError("labels must be a nonempty 1-D sequence")
        if np.unique(self.labels).size != self.labels.size:
            raise ValueError("labels must be unique")
        if window_size is not None and (
            isinstance(window_size, bool)
            or not isinstance(window_size, (int, np.integer))
            or window_size < 1
        ):
            raise ValueError("window_size must be a positive integer")

        self.window_size = window_size
        self.reset()

    def reset(self):
        size = len(self.labels)
        self._matrix = np.zeros((size, size), dtype=int)
        self._recent_true = np.empty(0, dtype=self.labels.dtype)
        self._recent_pred = np.empty(0, dtype=self.labels.dtype)

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim != 1 or y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must be matching 1-D arrays")
        if not (np.isin(y_true, self.labels).all() and np.isin(y_pred, self.labels).all()):
            raise ValueError("Each label must be listed in labels")
        if y_true.size == 0:
            return self

        if self.window_size is None:
            self._matrix += confusion_matrix(y_true, y_pred, labels=self.labels)
        else:
            n = self.window_size
            self._recent_true = np.concatenate((self._recent_true, y_true[-n:]))[-n:].copy()
            self._recent_pred = np.concatenate((self._recent_pred, y_pred[-n:]))[-n:].copy()
            self._matrix = confusion_matrix(
                self._recent_true, self._recent_pred, labels=self.labels
            )
        return self

    def result(self):
        correct = self._matrix.diagonal()
        predicted = self._matrix.sum(axis=0)
        actual = self._matrix.sum(axis=1)

        return {
            "accuracy": correct.sum() / max(actual.sum(), 1),
            "precision": self._divide(correct, predicted),
            "recall": self._divide(correct, actual),
            "f1": self._divide(2 * correct, predicted + actual),
            "confusion_matrix": self._matrix.copy(),
        }

    @staticmethod
    def _divide(numerator, denominator):
        return np.divide(
            numerator,
            denominator,
            out=np.zeros(numerator.shape, dtype=float),
            where=denominator != 0,
        )

class StreamingAUC:
    """Track binary ROC AUC across all rows or the latest window_size rows."""

    def __init__(self, pos_label=1, window_size=None):
        if window_size is not None and (
            isinstance(window_size, bool)
            or not isinstance(window_size, (int, np.integer))
            or window_size < 1
        ):
            raise ValueError("window_size must be a positive integer")

        self.pos_label = pos_label
        self.window_size = window_size
        self.reset()

    def reset(self):
        self._chunks = []

    def update(self, y_true, y_score):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score, dtype=float)
        if y_true.ndim != 1 or y_true.shape != y_score.shape:
            raise ValueError("y_true and y_score must be matching 1-D arrays")
        if not np.isfinite(y_score).all():
            raise ValueError("y_score must contain only finite values")
        if y_true.size == 0:
            return self

        positive = y_true == self.pos_label
        if self.window_size is None:
            self._chunks.append((positive, y_score.copy()))
        else:
            n = self.window_size
            if self._chunks:
                old_positive, old_score = self._chunks[0]
                positive = np.concatenate((old_positive, positive[-n:]))
                y_score = np.concatenate((old_score, y_score[-n:]))
            self._chunks = [(positive[-n:].copy(), y_score[-n:].copy())]
        return self

    def result(self):
        if not self._chunks:
            return float("nan")

        labels, scores = zip(*self._chunks)
        positive = np.concatenate(labels)
        if positive.all() or not positive.any():
            return float("nan")

        return roc_auc_score(positive.astype(int), np.concatenate(scores))