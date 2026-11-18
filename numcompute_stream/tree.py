import numpy as np
from numcompute.utils import NotFittedError
def gini(y):
    if len(y) == 0:
        return 0.0
    labels, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))


def entropy(y):
    if len(y) == 0:
        return 0.0
    labels, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))

def best_numeric_split(x, y, impurity=gini):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be matching 1-D arrays")
    if not np.isfinite(x).all():
        raise ValueError("x must contain only finite values")

    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    boundaries = np.flatnonzero(sorted_x[:-1] < sorted_x[1:])

    if boundaries.size == 0:
        return None

    lower = sorted_x[boundaries]
    upper = sorted_x[boundaries + 1]
    thresholds = lower / 2.0 + upper / 2.0
    thresholds = np.where(
        (thresholds < lower) | (thresholds >= upper),
        lower,
        thresholds,
    )

    n = len(y)

    if impurity is not gini and impurity is not entropy:
        scores = []
        for threshold in thresholds:
            mask = x <= threshold
            left, right = y[mask], y[~mask]
            scores.append(
                (len(left) * impurity(left)
                 + len(right) * impurity(right)) / n
            )
    else:
        labels, encoded = np.unique(y[order], return_inverse=True)
        counts = np.zeros((n, len(labels)), dtype=np.int64)
        counts[np.arange(n), encoded] = 1
        counts = np.cumsum(counts, axis=0)

        left_counts = counts[boundaries]
        right_counts = counts[-1] - left_counts
        left_size = boundaries + 1
        right_size = n - left_size

        def impurities(class_counts, size):
            p = class_counts / size[:, None]

            if impurity is gini:
                return 1.0 - np.sum(p ** 2, axis=1)

            log_p = np.zeros_like(p)
            np.log2(p, out=log_p, where=p > 0)
            return -np.sum(p * log_p, axis=1)

        scores = (
            left_size * impurities(left_counts, left_size)
            + right_size * impurities(right_counts, right_size)
        ) / n

    best = int(np.argmin(scores))
    return float(thresholds[best]), float(scores[best])

def _check_int(value, name, minimum, allow_none=False):
    """Raise TypeError for a non-integer and ValueError below the minimum."""
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")

class DecisionTreeClassifier:
    """Depth-limited Gini/entropy tree; partial_fit retrains every buffer_size rows."""

    def __init__(self, max_depth=None, min_samples_split=2, impurity=gini,
                 buffer_size=500, max_features=None, random_state=None):
        """Validate and store the hyperparameters."""
        _check_int(max_depth, "max_depth", 0, allow_none=True)
        _check_int(min_samples_split, "min_samples_split", 2)
        _check_int(buffer_size, "buffer_size", 1)
        _check_int(max_features, "max_features", 1, allow_none=True)
        if not callable(impurity):
            raise TypeError("impurity must be a function such as gini or entropy")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity = impurity
        self.buffer_size = buffer_size
        self.max_features = max_features
        self.random_state = random_state
        self.root_ = None
        self._chunks = []
        self._buffered_rows = 0
        if max_features is not None and (
            isinstance(max_features, bool)
            or not isinstance(max_features, (int, np.integer))
            or max_features < 1):
            raise ValueError("max_features must be None or a positive integer")

        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty 2-D array")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1-D with one label per row of X")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        self._rng = np.random.default_rng(self.random_state)
        self.root_ = self._build(X, y, depth=0)
        self.n_features_in_ = X.shape[1]
        self._chunks.clear()
        self._buffered_rows = 0
        return self

    def partial_fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be 2-D with at least one feature")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1-D with one label per row of X")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        if len(X) == 0:
            return self

        expected_features = getattr(self, "n_features_in_", X.shape[1])
        if self._chunks:
            expected_features = self._chunks[0][0].shape[1]
        if X.shape[1] != expected_features:
            raise ValueError(f"X must have {expected_features} features")

        self._chunks.append((X.copy(), y.copy()))
        self._buffered_rows += len(X)

        if self._buffered_rows >= self.buffer_size:
            features, labels = zip(*self._chunks)
            self.fit(np.concatenate(features), np.concatenate(labels))

        return self

    def _build(self, X, y, depth):
        labels, counts = np.unique(y, return_counts=True)
        majority = labels[np.argmax(counts)]
        leaf = {"leaf": True, "label": majority}

        if len(labels) == 1:
            return leaf
        if self.max_depth is not None and depth >= self.max_depth:
            return leaf
        if len(y) < self.min_samples_split:
            return leaf

        best_feature, best_threshold, best_score = None, None, None
        feature_indices = range(X.shape[1])
        if self.max_features is not None and self.max_features < X.shape[1]:
            feature_indices = np.sort(
                self._rng.choice(X.shape[1], size=self.max_features, replace=False))
        for feature in feature_indices:
            result = best_numeric_split(X[:, feature], y, self.impurity)
            if result is None:
                continue
            threshold, score = result
            if best_score is None or score < best_score:
                best_feature, best_threshold, best_score = feature, threshold, score

        if best_score is None or best_score >= self.impurity(y) - 1e-12:
            return leaf

        mask = X[:, best_feature] <= best_threshold
        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build(X[mask], y[mask], depth + 1),
            "right": self._build(X[~mask], y[~mask], depth + 1),
        }

    def predict(self, X):
        if self.root_ is None:
            raise NotFittedError("Call fit or partial_fit before predict")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must be 2-D with {self.n_features_in_} features"
            )
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        if len(X) == 0:
            return np.empty(0)

        pending = [(self.root_, np.arange(len(X)))]
        leaves = []

        while pending:
            node, rows = pending.pop()

            if rows.size == 0:
                continue

            if node["leaf"]:
                leaves.append((rows, node["label"]))
            else:
                mask = X[rows, node["feature"]] <= node["threshold"]
                pending.append((node["right"], rows[~mask]))
                pending.append((node["left"], rows[mask]))

        dtype = np.asarray([label for _, label in leaves]).dtype
        predictions = np.empty(len(X), dtype=dtype)

        for rows, label in leaves:
            predictions[rows] = label

        return predictions