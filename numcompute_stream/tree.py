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

