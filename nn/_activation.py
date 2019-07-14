import numpy as np
from ._base import Activation


class SigmoidActivation(Activation):
    """Sigmoid activation layer."""

    def forward(self, in_arr: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-in_arr))
        return out

    def backward(self, out: np.ndarray) -> np.ndarray:
        grad = out * (1 - out)
        return grad


class ReLUActivation(Activation):
    """ReLU activation layer."""

    def forward(self, in_arr: np.ndarray) -> np.ndarray:
        out = np.max(in_arr, 0)
        return out

    def backward(self, out: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(out)
        grad[out > 0] = 1
        return grad


class TanhActivation(Activation):
    """Tanh activation layer."""

    def forward(self, in_arr: np.ndarray) -> np.ndarray:
        out = np.tanh(in_arr)
        return out

    def backward(self, out: np.ndarray) -> np.ndarray:
        grad = 1.0 - np.power(out, 2)
        return grad
