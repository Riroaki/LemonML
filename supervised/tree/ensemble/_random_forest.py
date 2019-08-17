import numpy as np
from supervised.tree import CART
from supervised._base import SupervisedModel


class RandomForest(SupervisedModel):
    """Random Forest model, based on CART decision tree."""

    def __init__(self):
        self._trees = []
        self._labels = None

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> float:
        n, p = x.shape
        # Use sqrt(p) as number of attributes of each tree
        k = int(np.sqrt(p))
        self._labels = np.unique(label)
        for i in range(k):
            pass

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        n = x.shape[0]
        c = len(self._labels)
        vote_bin = np.zeros(n, c)
        for tree in self._trees:
            label_pred = tree.predict(x)
            vote_bin[:, label_pred] += 1

    def evaluate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
        pass

    def _bootstrap_sampling(self, x: np.ndarray, label: np.ndarray):
        # Sample with replacement
        pass
