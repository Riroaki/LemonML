import numpy as np
from .._decision_tree import DecisionTree
from .._base import SupervisedModel


class RandomForest(SupervisedModel):
    def __init__(self):
        pass

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.float:
        pass

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def evaluate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
        pass
