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

    thresholds=(values[:-1]+values[1:])/2
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
    def __init__(self, max_depth=None, min_samples_split=2, impurity=gini):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity = impurity
        self.root_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root_ = self._build(X, y, depth=0)
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
        X = np.asarray(X)
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