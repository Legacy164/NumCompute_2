import numpy as np

from numcompute.preprocessing import StandardScaler as BatchStandardScaler
from numcompute.utils import check_2d


class StandardScaler(BatchStandardScaler):
    def fit(self, X, y=None):
        return self._update(X, reset=True)

    def partial_fit(self, X, y=None):
        return self._update(X, reset=False)

    def _update(self, X, reset):
        X = check_2d(X, "X", allow_empty=False)
        if np.isinf(X).any():
            raise ValueError("X must not contain infinity")

        n_features = X.shape[1]
        if reset or not hasattr(self, "n_features_in_"):
            count = mean = m2 = magnitude = np.zeros(n_features)
        else:
            if n_features != self.n_features_in_:
                raise ValueError(f"X must have {self.n_features_in_} features")
            count = self._count
            mean = self._mean
            m2 = self._m2
            magnitude = self._magnitude

        valid = ~np.isnan(X)
        filled = np.where(valid, X, 0.0)
        new_magnitude = np.maximum(magnitude, np.max(np.abs(filled), axis=0))
        divisor = np.where(new_magnitude > 0, new_magnitude, 1.0)

        ratio = magnitude / divisor
        mean = mean * ratio
        m2 = m2 * ratio ** 2

        scaled = filled / divisor
        chunk_count = valid.sum(axis=0)
        chunk_mean = scaled.sum(axis=0) / np.maximum(chunk_count, 1)
        deviations = np.where(valid, scaled - chunk_mean, 0.0)
        chunk_m2 = np.sum(deviations ** 2, axis=0)

        total = count + chunk_count
        weight = chunk_count / np.maximum(total, 1)
        delta = chunk_mean - mean
        mean = mean + delta * weight
        m2 = m2 + chunk_m2 + delta ** 2 * count * weight

        std = new_magnitude * np.sqrt(m2 / np.maximum(total, 1))
        self.mean_ = mean * new_magnitude
        self.var_ = std ** 2
        self.scale_ = np.where(std > 0, std, 1.0)
        self.n_features_in_ = n_features
        self._count, self._mean, self._m2 = total, mean, m2
        self._magnitude = new_magnitude
        return self