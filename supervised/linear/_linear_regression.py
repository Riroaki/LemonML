import numpy as np
from ._base import LinearModel
from ._regularization import REGULARIZE, Regularizer
from utils import batch


class LinearRegression(LinearModel):
    """Linear regression model."""

    def __init__(self, regular: REGULARIZE = None):
        super().__init__()
        if REGULARIZE is not None:
            self._regular = Regularizer(regular)
        else:
            self._regular = None

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.float:
        assert x.shape[0] == y.shape[0]
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
            for batch_x, batch_y in batch(x, y, self._batch_size):
                pred_val = self._predict_value(batch_x, self._w, self._b)
                loss += self._loss(pred_val, batch_y) * batch_x.shape[0]
                grad_w, grad_b = self._grad(batch_x, pred_val, batch_y)
                self._w -= grad_w
                self._b -= grad_b
            loss /= n
            # Break if model converges.
            if loss <= self._loss_tol:
                break
        # Update model with current weight and bias
        self._update_model(loss)
        return loss

    def fit_norm_eq(self, x: np.ndarray, y: np.ndarray) -> np.float:
        # Fit x using normal equation
        assert x.shape[0] == y.shape[0]
        n, p = x.shape
        if self._w is None or self._b is None or self._w.shape[0] != p:
            # Initialize weights using random values
            self._init_model(p)
        x_ext = np.hstack((np.ones((n, 1)), x))
        w_ext = np.linalg.pinv(np.matmul(x_ext.T, x_ext))
        w_ext = np.matmul(np.matmul(w_ext, x_ext.T), y)
        self._w, self._b = w_ext[1:], w_ext[0]
        # Calculate training loss
        pred_val = self._predict_value(x, self._w, self._b)
        loss = self._loss(pred_val, y)
        self._update_model(loss)
        return loss

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        assert not np.isinf(self._optimum['loss'])
        assert self._optimum['w'].shape[0] == x.shape[1]
        pred_val = self._predict_value(x, self._optimum['w'],
                                       self._optimum['b'])
        return pred_val

    def evaluate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
        assert x.shape[0] == y.shape[0]
        assert not np.isinf(self._optimum['loss'])
        assert self._optimum['w'].shape[0] == x.shape[1]
        pred_val = self._predict_value(x, self._optimum['w'],
                                       self._optimum['b'])
        # The precision part of regression is None
        precision = None
        loss = self._loss(pred_val, y)
        return precision, loss

    @staticmethod
    def _predict_value(x: np.ndarray, w: np.ndarray,
                       b: np.float) -> np.ndarray:
        pred_val = np.matmul(x, w) + b
        return pred_val

    @staticmethod
    def _predict_label(pred_val: np.ndarray) -> np.ndarray:
        # NO labeling in regression.
        pass

    def _loss(self, pred_val: np.ndarray, true_val: np.ndarray) -> np.float:
        # Use MSE loss
        loss = float(np.sum(np.power(pred_val - true_val, 2)))
        loss /= 2 * true_val.shape[0]
        # Add regularized loss
        if self._regular is not None:
            loss += self._regular[self._w]
        return loss

    def _grad(self, x: np.ndarray, pred_val: np.ndarray,
              true_val: np.ndarray) -> tuple:
        # Use MSE loss
        grad_w = (x * (pred_val - true_val).reshape((-1, 1))).mean(axis=0)
        grad_b = (pred_val - true_val).mean()
        # Use simple gradient by multiplying learning rate and grad.
        grad_w *= self._learn_rate
        grad_b *= self._learn_rate
        # Add regularized grad
        if self._regular is not None:
            grad_w += self._regular.grad(self._w)
        return grad_w, grad_b
