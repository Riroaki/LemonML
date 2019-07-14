import numpy as np
from ._base import SupervisedModel


# TODO
class CARTNode(object):
    """Node of Classication And Regression Tree."""

    def __init__(self):
        pass


class CART(SupervisedModel):
    """Classication And Regression Tree."""

    def __init__(self):
        self.__tree = None
        self.__thres_count = 2  # Lower bound of sample count for split.
        self.__gini_threshold = 0.5  # Lower bound of gini gain for split.

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> np.float:
        n, p = x.shape
        assert n == label.shape[0]
        is_discrete = np.zeros(p)
        if 'discrete_attrs' in kwargs:
            assert isinstance(kwargs['discrete_attrs'], list)
            is_discrete[kwargs['discrete_attrs']] = 1
        self.__tree = self.__build_tree(x, label, is_discrete)

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def evaluate(self, x: np.ndarray, label: np.ndarray, **kwargs) -> tuple:
        pass

    def __build_tree(self, x: np.ndarray, label: np.ndarray,
                     is_used: np.ndarray) -> CARTNode:
        if not np.isin(False, is_used) or label.shape[0] <= self.__count_threshold:
            leaf = CARTNode()

    @staticmethod
    def __gini(label: np.ndarray) -> float:
        """Calculate gini coefficient."""
        cls, count = np.unique(label, return_counts=True)
        count /= label.shape[0]
        gini = float(1 - np.sum(np.power(count, 2)))
        return gini
