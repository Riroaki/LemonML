import numpy as np
from nn._activation import SigmoidActivation
from nn._base import Module, Criterion
from nn._fully_connect import FullyConnectedLayer


class FullyConnectNN(Module):
    """Basic a simple fully connect neural network model."""

    def __init__(self, dim_list: list):
        # Initialize layers
        layers = []
        for i in range(len(dim_list) - 1):
            in_dim = dim_list[i]
            out_dim = dim_list[i + 1]
            layers.append(
                FullyConnectedLayer(in_dim, out_dim, SigmoidActivation()))
        self._layers = layers
        self._dims = dim_list
        self._lr = 0.03

    def train(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self._dims[0] and y.shape[1] == self._dims[-1]
        assert 'criterion' in kwargs and isinstance(kwargs['criterion'],
                                                    Criterion)
        assert 'epoch' in kwargs and isinstance(kwargs['epoch'], int)
        if 'lr' in kwargs and isinstance(kwargs['lr'], float):
            self._lr = kwargs['lr']
        criterion = kwargs['criterion']
        epoch = kwargs['epoch']
        loss = 0.
        for _ in range(epoch):
            for i, row in enumerate(x):
                pred = self._forward(row)
                loss += criterion.forward(pred, y[i])
                grad = criterion.backward(pred, y[i])
                self._backward(grad)
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred = []
        for row in x:
            pred.append(self._forward(row))
        pred = np.array(pred)
        return pred

    def _forward(self, in_arr: np.ndarray) -> np.ndarray:
        for layer in self._layers:
            in_arr = layer.forward(in_arr)
        return in_arr

    def _backward(self, sense: np.ndarray) -> None:
        for layer in self._layers[::-1]:
            layer.backward(sense)
            layer.update(self._lr)
