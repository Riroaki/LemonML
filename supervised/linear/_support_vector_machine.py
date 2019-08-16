import numpy as np
from ._base import LinearModel
from scipy.optimize import minimize


class SVM(LinearModel):
    """Support vector machine model, binary classifier."""

    def __init__(self):
        super().__init__()

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> np.float:
        # Target and constraint functions
        def target(w):
            return w[1:].dot(w[1:])

        def get_func(i):
            return lambda w: w.dot(x_ext[i]) * label[i] - 1

        # Target and constraint functions with slack variables
        def target_slack(w_e):
            w = w_e[: (p + 1)]
            eps = w_e[(p + 1):]
            return 0.5 * w[1:].dot(w[1:]) + c * np.sum(eps)

        def get_func_slack_w(i):
            return lambda w_e: w_e[: (p + 1)].dot(x_ext[:, i]) \
                               * label[0][i] - 1 + w_e[p + i]

        def get_func_slack_e(i):
            return lambda w_e: w_e[p + i]

        assert np.array_equal(np.unique(label), np.array([-1, 1]))
        assert x.shape[0] == label.shape[0]
        n, p = x.shape
        if self._w is None or self._b is None or self._w.shape[0] != p:
            # Initialize weights using random values
            self._init_model(p)
        # No slack parameters unless explicitly stated
        slack = False
        if kwargs is not None:
            # Update parameters of training
            self._update_params(kwargs)
            # Whether to use slack variables
            if 'slack' in kwargs:
                assert isinstance(kwargs['slack'], bool)
                slack = kwargs['slack']
        w_ext = np.hstack((self._w, self._b))
        x_ext = np.hstack((x, np.ones((n, 1))))

        # Find optimum w and b for both condition
        if not slack:
            # SVM without slack
            # Optimize 1/2 w^T * w
            # s.t. yi * (w^T * xi + b) - 1 >= 0
            cons = [{'type': 'ineq', 'fun': get_func(i)} for i in range(n)]
            # Find optimized w
            w_ext = minimize(target, w_ext, constraints=cons).x
        else:
            # SVM with slack
            # Optimize 1/2 w^T * w + C * sum(eps_i)
            # s.t. yi * (w^T * xi + b) - 1 + eps_i >= 0, eps_i >= 0
            c, w_and_eps = 1000, np.hstack((w_ext, np.random.randn(n)))
            cons = []
            for idx in range(n):
                cons.append({'type': 'ineq', 'fun': get_func_slack_w(idx)})
                cons.append({'type': 'ineq', 'fun': get_func_slack_e(idx)})
            cons = tuple(cons)
            w_and_eps = minimize(target_slack, w_and_eps, constraints=cons).x
            w_ext = w_and_eps[: (p + 1)]
        # Update and save optimal weights & bias
        self._w = w_ext[:-1]
        self._b = w_ext[-1]
        # Calculate loss
        pred_val = self._predict_value(x, self._w, self._b)
        loss = self._loss(pred_val, label)
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
        # Hinge loss
        loss = 1 - pred_val * true_label
        loss[loss < 0] = 0
        loss = loss.mean()
        return loss

    def _grad(self, x: np.ndarray, pred_val: np.ndarray,
              true_val: np.ndarray) -> None:
        # Use scipy.optmize to find best w and b
        # Not grad-base method
        return
