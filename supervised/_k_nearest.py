import numpy as np
import scipy.stats
from ._base import SupervisedModel


class KNearest(SupervisedModel):
    """K-Nearest-Neighbor model, multi-class classifier."""

    def __init__(self):
        self.__data = None
        self.__label = None

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> np.int:
        # Lazy training for knn, no computations
        self.__data = x
        self.__label = label
        class_count = len(np.unique(label))
        return class_count

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        assert self.__data is not None and self.__label is not None
        assert 'k' in kwargs and kwargs['k'] < self.__data.shape[0]
        k = kwargs['k']
        pred_label = np.zeros(x.shape[0])
        for i, xi in enumerate(x):
            dist = np.power(self.__data - xi, 2).sum(axis=1)
            top_idx = np.argsort(dist)[: k]
            top_label = self.__label[top_idx]
            pred_label[i] = scipy.stats.mode(top_label)[0][0]
        return pred_label

    def evaluate(self, x: np.ndarray, label: np.ndarray, **kwargs) -> tuple:
        pred_label = self.predict(x, **kwargs)
        precision = 1 - np.count_nonzero(pred_label - label) / x.shape[0]
        # There should be no such loss for knn,
        # so I simply use error rate to represent loss instead.
        loss = 1 - precision
        return precision, loss
