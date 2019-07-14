import numpy as np
from ._base import Criterion


class MSECriterion(Criterion):
    """Mean Square Error Criterion."""

    def forward(self, pred: np.ndarray, y: np.ndarray) -> float:
        loss = 0.5 * np.power(y - pred, 2)
        return loss

    def backward(self, pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        grad = (pred - y) * y
        return grad
