import random
import numpy as np
from scipy import stats
from supervised.tree import CART
from supervised._base import SupervisedModel


class RandomForest(SupervisedModel):
    """Random Forest model, based on CART decision tree."""

    def __init__(self):
        self._trees = []
        # Default number of trees, number of samples
        self._num_trees = 10

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> float:
        assert x.shape[0] == label.shape[0]
        n, p = x.shape
        # Number of trees and number of samples in training each tree
        self._num_trees = kwargs.get('num_trees', self._num_trees)
        num_samples = kwargs.get('num_samples', n // 2)
        # Number of attributes: sqrt(p)
        num_attrs = int(np.sqrt(p))
        # Indices for sampling
        row_index = list(range(n))
        attr_index = list(range(p))
        # Record loss
        loss = 0
        for i in range(self._num_trees):
            # Sample with replacement
            sample_index = np.array(random.choices(row_index, k=num_samples))
            sample_x, sample_label = x[sample_index], label[sample_index]
            # Select certain attributes
            attrs = random.sample(attr_index, k=num_attrs)
            sample_x = x[:, attrs]
            # Build a tree based on selected data
            tree = CART()
            loss += tree.fit(sample_x, sample_label)
            self._trees.append(tree)
        return loss

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        assert len(self._trees) > 0
        n = x.shape[0]
        # Record vote results for all rows in x
        vote_results = []
        for tree in self._trees:
            vote_results.append(tree.predict(x, **kwargs))
        vote_results = np.array(vote_results).T
        # Aggregate major votes
        pred_label = []
        for index in range(n):
            major = stats.mode(vote_results[index])
            pred_label.append(major)
        return np.array(pred_label)

    def evaluate(self, x: np.ndarray, label: np.ndarray, **kwargs) -> tuple:
        assert x.shape[0] == label.shape[0]
        # Predict labels
        pred_label = self.predict(x, **kwargs)
        # Use 0-1 loss
        loss = np.count_nonzero(pred_label != label)
        precision = 1 - loss / x.shape[0]
        return precision, loss
