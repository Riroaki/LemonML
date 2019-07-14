import numpy as np
from ._base import Layer, Activation


class FullyConnectedLayer(Layer):
    """Fully connected layer."""

    def __init__(self, in_dim: int, out_dim: int, activator: Activation):
        # Basic parameters
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._activator = activator
        # Weights
        self._w = np.random.uniform(-0.1, 0.1, (out_dim, in_dim))
        self._b = np.zeros(out_dim)
        # Input, output and gradients
        self._in = None
        self._out = None
        self._grad_w = None
        self._grad_b = None

    def forward(self, in_arr: np.ndarray) -> np.ndarray:
        assert self._in_dim == in_arr.shape[0]
        self._in = in_arr
        self._out = self._activator.forward(
            np.dot(self._w, in_arr) + self._b)
        return self._out

    def backward(self, sensitivity: np.ndarray) -> np.ndarray:
        assert self._out_dim == sensitivity.shape[0]
        self._grad_b = sensitivity * self._activator.backward(self._out)
        self._grad_w = np.matmul(self._grad_b, self._in)
        sense = np.matmul(self._grad_b, self._w)
        return sense

    def update(self, learn_rate: float) -> None:
        self._w -= learn_rate * self._grad_w
        self._b -= learn_rate * self._grad_b
