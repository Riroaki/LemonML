import numpy as np
from supervised._base import SupervisedModel


# TODO
class Adaboost(SupervisedModel):
    def __init__(self):
        pass

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        pass

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def evaluate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
        pass
