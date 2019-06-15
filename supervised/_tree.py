import numpy as np
from supervised._basics import Model


class TreeNode(object):
    def __init__(self, index: int, value_list: int):
        self._index = index
        self._value_list = value_list
        self._child = {}


class DecisionTree(Model):
    def __init__(self):
        pass

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.float:
        pass

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def evaluate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
        pass
