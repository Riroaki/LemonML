import numpy as np
from ._base import LinearModel
from ._regularization import Regularizer, REGULARIZE
from utils import batch


class LogisticRegression(LinearModel):
    """Logistic regression model, binary classifier."""

    def __init__(self, regular: REGULARIZE = None):
        super().__init__()
        if REGULARIZE is not None:
            self._regular = Regularizer(regular)
        else:
            self._regular = None

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> np.float:
        # Check labels: only containing 1 and 0
        assert np.array_equal(np.unique(label), np.array([0, 1]))
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
                grad_w, grad_b = self._grad(batch_x, pred_val, batch_label)
                self._w -= grad_w
                self._b -= grad_b
            loss /= n
            # Break if model converges.
            if loss <= self._loss_tol:
                break
        self._update_model(loss)
        return loss

    def predict(self, x: np.ndarray, **kwargs):
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
        def __sigmoid(raw: np.ndarray) -> np.ndarray:
            res = 1 / (1 + np.exp(-raw))
            return res

        prob = np.matmul(x, w) + b
        pred_val = __sigmoid(prob)
        return pred_val

    @staticmethod
    def _predict_label(pred_val: np.ndarray) -> np.ndarray:
        pred_label = np.sign(pred_val - 0.5)
        pred_label[pred_label == 0] = 1
        pred_label[pred_label < 0] = 0
        return pred_label

    # @staticmethod
    def _loss(self, pred_val: np.ndarray, true_label: np.ndarray) -> np.float:
        # Use maximum likelihood (log-likelihood loss)
        # loss = 1 / n * (-y * log(wx + b) - (1 - y) * log(wx + b))
        # Here we need to care about the log zero and overflow warning...
        mask_val = pred_val.copy()
        mask_val[mask_val == 0] = 1e-6
        mask_val[mask_val == 1] = 1 - 1e-6
        class1_loss = -true_label * np.log(mask_val)
        class0_loss = (1 - true_label) * np.log(1 - mask_val)
        loss = np.sum(class0_loss + class1_loss) / true_label.shape[0]
        # Add regularized loss
        if self._regular is not None:
            loss += self._regular[self._w]
        return loss

    def _grad(self, x: np.ndarray, pred_val: np.ndarray,
              true_label: np.ndarray) -> tuple:
        #  dc / dw = x * (pred_val - true_label)
        grad_w = (x * (pred_val - true_label).reshape((-1, 1))).mean(axis=0)
        grad_b = (pred_val - true_label).mean()
        # Use simple gradient by multiplying learning rate and grad.
        grad_w *= self._learn_rate
        grad_b *= self._learn_rate
        # Add regularized grad
        if self._regular is not None:
            grad_w += self._regular.grad(self._w)
        return grad_w, grad_b
