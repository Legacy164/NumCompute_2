import numpy as np

from numcompute.stats import RunningStats, quantile
from numcompute.utils import check_2d


class StreamStats:
    """Track statistics per column using shared histogram bin edges."""

    def __init__(self, bin_edges):
        edges = np.asarray(bin_edges, dtype=float)
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.isfinite(edges).all()
            or not np.all(edges[1:] > edges[:-1])
        ):
            raise ValueError("bin_edges must be finite and strictly increasing")

        self.bin_edges = edges.copy()
        self._moments = RunningStats()
        self._chunks = []
        self._histogram = None

    def update_stats(self, X_chunk):
        X = check_2d(X_chunk, "X_chunk", allow_empty=False)
        if np.isinf(X).any():
            raise ValueError("X_chunk must not contain infinity")

        self._moments.update(X)
        counts = np.stack([
            np.histogram(column[~np.isnan(column)], bins=self.bin_edges)[0]
            for column in X.T
        ])
        if self._histogram is None:
            self._histogram = counts
        else:
            self._histogram += counts

        self._chunks.append(X.copy())
        return self

    @property
    def mean(self):
        return self._moments.mean

    def var(self, ddof=0):
        return self._moments.var(ddof)

    def histogram(self):
        if self._histogram is None:
            raise ValueError("Call update_stats before histogram")
        return self._histogram.copy()

    def quantile(self, q):
        if not self._chunks:
            raise ValueError("Call update_stats before quantile")
        return quantile(np.concatenate(self._chunks), q, axis=0, nan_policy="omit")