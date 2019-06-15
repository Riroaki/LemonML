import numpy as np
from supervised._basics import LinearModel
from utils import batch


class Perceptron(LinearModel):
    """Perceptron model, binary classifier."""

    def __init__(self):
        super().__init__()

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> np.float:
        assert np.array_equal(np.unique(label), np.array([-1, 1]))
        assert x.shape[0] == label.shape[0]
        n, p = x.shape
        if self._w is None or self._b is None or self._w.shape[0] != p:
            # Initialize weights using random values
            self._init_model(p)
        if kwargs is not None:
            # Update parameters of training
            self._update_params(kwargs)
        iters, loss = 0, 0.
        # Iterates till converge or iterating times exceed bound
        while iters < self._iter_bound:
            iters += 1
            # Update weights using mini-batch gradient desent
            for batch_x, batch_label in batch(x, label, self._batch_size):
                pred_val = self._predict_value(batch_x, self._w, self._b)
                loss += self._loss(pred_val, batch_label) * batch_x.shape[0]
                pred_label = self._predict_label(pred_val)
                grad_w, grad_b = self._grad(batch_x, pred_label, batch_label)
                self._w -= grad_w
                self._b -= grad_b
            loss /= x.shape[0]
            # Break if model converges.
            if loss <= self._loss_tol:
                break
        self._update_model(loss)
        return loss

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        assert not np.isinf(self._optimum['loss'])
        assert self._optimum['w'].shape[0] == x.shape[1]
        pred_val = self._predict_value(x, self._optimum['w'],
                                       self._optimum['b'])
        pred_label = self._predict_label(pred_val)
        return pred_label

    def evaluate(self, x: np.ndarray, label: np.ndarray, **kwargs) -> tuple:
        assert x.shape[0] == label.shape[0]
        assert not np.isinf(self._optimum['loss'])
        assert self._optimum['w'].shape[0] == x.shape[1]
        pred_val = self._predict_value(x, self._optimum['w'],
                                       self._optimum['b'])
        pred_label = self._predict_label(pred_val)
        precision = 1 - np.count_nonzero(pred_label - label) / x.shape[0]
        loss = self._loss(pred_val, label)
        return precision, loss

    @staticmethod
    def _predict_value(x: np.ndarray, w: np.ndarray,
                       b: np.float) -> np.ndarray:
        pred_val = np.matmul(x, w) + b
        return pred_val

    @staticmethod
    def _predict_label(pred_val: np.ndarray) -> np.ndarray:
        pred_label = np.sign(pred_val)
        pred_label[pred_label == 0] = 1
        return pred_label

    @staticmethod
    def _loss(pred_val: np.ndarray, true_label: np.ndarray) -> np.float:
        loss = -np.float(np.sum(pred_val * true_label)) / true_label.shape[0]
        return loss

    def _grad(self, x: np.ndarray, pred_label: np.ndarray,
              true_label: np.ndarray) -> tuple:
        grad_w = -(true_label.reshape((-1, 1)) * x)[pred_label != true_label]
        grad_b = -true_label[pred_label != true_label]
        grad_w = grad_w.sum(axis=0) / x.shape[0]
        grad_b = grad_b.sum() / x.shape[0]
        return grad_w, grad_b
