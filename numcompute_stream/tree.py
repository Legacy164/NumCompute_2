import numpy as np

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

def best_numeric_split(x,y,impurity=gini):
    x=np.array(x)
    y=np.array(y)

    values=np.unique(x)
    if len(values)<2:
        return None

    lower, upper = values[:-1], values[1:]
    thresholds = lower / 2.0 + upper / 2.0
    thresholds = np.where((thresholds < lower) | (thresholds >= upper), lower, thresholds)
    n=len(y)
    best=None
    for t in thresholds:
        mask=x<=t
        left, right=y[mask],y[~mask]
        score=(len(left)*impurity(left)+len(right)*impurity(right))/n
        if best is None or score<best[1]:
            best=(float(t),float(score))
    return best

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, impurity=gini, buffer_size=500):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity = impurity
        self.root_ = None
        if isinstance(buffer_size, bool) or not isinstance(buffer_size, (int, np.integer)) or buffer_size < 1:
            raise ValueError("buffer_size must be a positive integer")

        self.buffer_size = buffer_size
        self._chunks = []
        self._buffered_rows = 0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty 2-D array")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1-D with one label per row of X")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")

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
        for feature in range(X.shape[1]):
            result = best_numeric_split(X[:, feature], y, self.impurity)
            if result is None:
                continue
            threshold, score = result
            if best_score is None or score < best_score:
                best_feature, best_threshold, best_score = feature, threshold, score

        if best_score is None or best_score >= self.impurity(y):
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
            raise ValueError("Call fit before predict")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must be 2-D with {self.n_features_in_} features")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        predictions = []
        for row in X:
            node = self.root_
            while not node["leaf"]:
                if row[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            predictions.append(node["label"])
        return np.array(predictions)