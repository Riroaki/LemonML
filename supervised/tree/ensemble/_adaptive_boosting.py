import numpy as np
from .._cart import CART
from supervised._base import SupervisedModel


class Adaboost(SupervisedModel):
    """Adaptive Boosting model, based on CART decision tree."""

    def __init__(self):
        self._trees = []
        # Default number of trees
        self._num_trees = 10
        # Default depth of decision tree
        self._tree_depth = 10
        # Weights of tree in decision
        self._weight_tree = None

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> float:
        assert x.shape[0] == label.shape[0]
        n, p = x.shape
        # Number of trees and depth of tree
        self._num_trees = kwargs.get('num_trees', self._num_trees)
        self._tree_depth = kwargs.get('tree_depth', self._tree_depth)
        # Initialize weights of samples
        weight_x = np.ones(n) / n
        weight_tree = np.ones(self._num_trees)
        # Record loss
        loss = 0.
        # Build trees
        for index in range(self._num_trees):
            tree = CART()
            # Train weighted data on tree
            weighted_x, weighted_label = self._make_weighted_sample(x, label,
                                                                    weight_x)
            loss += tree.fit(weighted_x, weighted_label, depth=self._tree_depth)
            # Calculate error factor e_m
            pred_label = tree.predict(x)
            e_m = np.sum(weight_x * (pred_label == label))
            # Update tree weight a_m
            a_m = 0.5 * np.log((1 - e_m) / e_m)
            weight_tree[index] = a_m
            # Update sample weights w_m
            error_factor = np.array(
                [1 if pred_label[i] == label[i] else -1 for i in
                 range(n)])
            prob_factor = weight_x * np.exp(-a_m * error_factor)
            weight_x = prob_factor / np.sum(prob_factor)
            self._trees.append(tree)
        self._weight_tree = weight_tree
        return loss

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        assert len(self._trees) > 0
        n = x.shape[0]
        # Record vote reslts for all rows in x
        vote_result = []
        for tree in self._trees:
            vote_result.append(tree.predict(x, **kwargs))
        vote_result = np.array(vote_result).T
        # Aggregate predictions according to trees' weights
        pred_label = []
        for x_index in range(n):
            # Weights stores sum of trees that predict one label
            weights = {label: 0. for label in np.unique(vote_result[x_index])}
            for tree_index, label in enumerate(vote_result[x_index]):
                weights[label] += self._weight_tree[tree_index]
            # Use label with maximum tree weight sum
            major = sorted(weights.items(), key=lambda pair: -pair[1])[0][0]
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

    @staticmethod
    def _make_weighted_sample(x: np.ndarray, label: np.ndarray,
                              weights: np.ndarray) -> tuple:
        # Produce a new sample set with weights by repeating samples
        weighted_x, weighted_label = [], []
        # Repeat time is determined by the index of sorted weights:
        # e.g., weight = [0.3, 1.5, 0.7, 0.2], repeat time = [2, 4, 3, 1]
        weight_repeat = {weight: index + 1 for index, weight in
                         enumerate(sorted(np.unique(weights)))}
        n = x.shape[0]
        for i in range(n):
            x_i, label_i, weight_i = x[i], label[i], weights[i]
            repeat = weight_repeat[weight_i]
            # Append repeated sample
            weighted_x.extend([x_i for _ in range(repeat)])
            weighted_label.extend([label_i for _ in range(repeat)])
        return np.array(weighted_x), np.array(weighted_label)
