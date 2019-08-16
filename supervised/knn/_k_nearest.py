import numpy as np
import scipy.stats
from .._base import SupervisedModel


class KNearest(SupervisedModel):
    """K-Nearest-Neighbor model, multi-class classifier."""

    def __init__(self):
        self._data = None
        self._label = None

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> np.int:
        # Lazy training for knn, no computations
        self._data = x
        self._label = label
        class_count = len(np.unique(label))
        return class_count

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        assert self._data is not None and self._label is not None
        assert 'k' in kwargs and kwargs['k'] < self._data.shape[0]
        k = kwargs['k']
        pred_label = np.zeros(x.shape[0])
        for i, xi in enumerate(x):
            dist = np.power(self._data - xi, 2).sum(axis=1)
            top_idx = np.argsort(dist)[: k]
            top_label = self._label[top_idx]
            pred_label[i] = scipy.stats.mode(top_label)[0][0]
        return pred_label

    def evaluate(self, x: np.ndarray, label: np.ndarray, **kwargs) -> tuple:
        pred_label = self.predict(x, **kwargs)
        # Use 0-1 loss
        loss = np.count_nonzero(pred_label != label)
        precision = 1 - loss / x.shape[0]
        return precision, loss
