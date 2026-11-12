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