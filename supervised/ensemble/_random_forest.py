import numpy as np
from .._base import SupervisedModel


# TODO
class RandomForest(SupervisedModel):
    def __init__(self):
        self.__trees = []
        self.__labels = None

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> np.float:
        _, p = x.shape
        # Use sqrt(p) as number of trees
        k = np.sqrt(p)
        self.__labels = np.unique(label)
        for i in range(k):
            pass

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        n = x.shape[0]
        c = len(self.__labels)
        vote_bin = np.zeros(n, c)
        for tree in self.__trees:
            label_pred = tree.predict(x)
            vote_bin[:, label_pred] += 1

    def evaluate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
        pass
