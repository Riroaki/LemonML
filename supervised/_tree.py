import numpy as np
from supervised._base import SupervisedModel


class TreeNode(object):
    def __init__(self, index: int, value_list: int):
        self._index = index
        self._value_list = value_list
        self._child = {}


class Tree(object):
    def __init__(self, x: np.ndarray, label: np.ndarray,
                 available_colunm: np.ndarray):
        pass


class DecisionTree(SupervisedModel):
    def __init__(self):
        self._tree = None

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> np.float:
        assert x.shape[0] == label.shape[0]
        _, p = x.shape
        all_coclumns = np.arange(p)
        self._tree = Tree(x, label, all_coclumns)

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def evaluate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
        pass
