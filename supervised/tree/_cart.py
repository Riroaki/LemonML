import numpy as np
from .._base import SupervisedModel


class CARTNode(object):
    """Node of Classication And Regression Tree."""

    def __init__(self, col_index: int, is_discrete: bool, value: any):
        self.col_index = col_index
        self.is_discrete = is_discrete
        self.value = value
        self.left_child = None
        self.right_child = None


class CART(SupervisedModel):
    """Classication And Regression Tree (classification version)."""

    def __init__(self):
        self.__tree = None
        # self.__thres_count = 2  # Lower bound of sample count for split.
        # self.__gini_threshold = 0.5  # Lower bound of gini gain for split.

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> np.float:
        n, p = x.shape
        assert n == label.shape[0]
        is_discrete = np.zeros(p)
        if 'discrete_attrs' in kwargs:
            assert isinstance(kwargs['discrete_attrs'], list)
            is_discrete[kwargs['discrete_attrs']] = 1
        # Indices of attributes
        attributes_index = list(range(p))
        while len(attributes_index) > 0:
            pass

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def evaluate(self, x: np.ndarray, label: np.ndarray, **kwargs) -> tuple:
        pass

    @staticmethod
    def __gini(label: np.ndarray) -> float:
        """Calculate gini coefficient."""
        cls, count = np.unique(label, return_counts=True)
        count /= label.shape[0]
        gini = float(1 - np.sum(np.power(count, 2)))
        return gini
