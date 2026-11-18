from copy import deepcopy

import numpy as np

from numcompute.pipeline import Pipeline as BatchPipeline


class Pipeline(BatchPipeline):
    """Transformers plus a buffered tree, all updated together at each retrain."""

    def __init__(self, steps):
        """Store the steps; the last one must have a buffer_size."""
        super().__init__(steps)
        self._chunks = []
        self._buffered_rows = 0

    def fit(self, X, y=None):
        """Fit every step from scratch and clear the buffer."""
        super().fit(X, y)
        self.n_features_in_ = np.asarray(X).shape[1]
        self._chunks.clear()
        self._buffered_rows = 0
        return self

    def partial_fit(self, X, y):
        """Buffer raw rows of X (n, f); update all steps once the buffer is full."""
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be 2-D with at least one feature")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1-D with one label per row of X")
        if len(X) == 0:
            return self

        if self._chunks:
            expected = self._chunks[0][0].shape[1]
        else:
            expected = getattr(self, "n_features_in_", X.shape[1])
        if X.shape[1] != expected:
            raise ValueError(f"X must have {expected} features, got {X.shape[1]}")

        buffer_size = self.steps[-1][1].buffer_size
        if self._buffered_rows + len(X) < buffer_size:
            self._chunks.append((X.copy(), y.copy()))
            self._buffered_rows += len(X)
            return self

        features, labels = zip(*self._chunks, (X, y))
        Xt = np.concatenate(features)
        y_batch = np.concatenate(labels)

        steps = deepcopy(self.steps)
        for _, transformer in steps[:-1]:
            update = getattr(transformer, "partial_fit", transformer.fit)
            update(Xt, y_batch)
            Xt = transformer.transform(Xt)

        steps[-1][1].fit(Xt, y_batch)
        self.steps = steps
        self.n_features_in_ = X.shape[1]
        self._chunks.clear()
        self._buffered_rows = 0
        return self