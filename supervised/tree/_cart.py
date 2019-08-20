import sys
from collections import Counter
from scipy import stats
import numpy as np
from .._base import SupervisedModel


class CARTNode(object):
    """Node of Classication And Regression Tree."""

    def __init__(self, col_index: int, is_discrete: bool, value: object):
        self.col_index = col_index
        self.is_discrete = is_discrete
        self.value = value
        # Children
        self.left_child = None
        self.right_child = None
        # Leaf node: label info
        self.is_leaf = False
        self.cls = None
        self.labels = None


class CART(SupervisedModel):
    """Classication And Regression Tree (classification version)."""

    def __init__(self):
        self._tree = None
        self._is_discrete = None
        # Minimum count for splitting node
        self._thres_count = 2
        # Alpha in calculating loss of tree
        self._alpha = 10.
        # Default depth of tree: unlimited
        self._depth = sys.maxsize

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> float:
        n, p = x.shape
        assert n == label.shape[0]
        # Record whether each attributes is discrete
        if 'is_discrete' in kwargs:
            # Assign using custom parameter
            is_discrete = kwargs['is_discrete']
        else:
            # Assign whether discrete according to the type of columns
            is_discrete = np.full(p, False)
            for index, value in enumerate(x[0]):
                # Check whether column is discrete type
                if not isinstance(value, (int, float)):
                    is_discrete[index] = True
        # Build a tree with certain level
        self._depth = kwargs.get('depth', self._depth)

        self._is_discrete = is_discrete
        self._tree = self._build_tree(x, label, current_depth=0)
        # Whether do pruning
        if kwargs.get('pruning', False):
            self._pruning(x, label)
        # Calculate loss of current tree
        loss = self._loss()
        return loss

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # Predict class for input data
        pred_label = []
        for data in x:
            node: CARTNode = self._tree
            while not node.is_leaf:
                col = node.col_index
                if self._is_discrete[col]:
                    # Discrete value column
                    if data[col] == node.value:
                        node = node.left_child
                    else:
                        node = node.right_child
                else:
                    # Continuous value column
                    if data[col] <= node.value:
                        node = node.left_child
                        node = node.right_child
            # Add predicting result
            pred_label.append(node.cls)
        return np.array(pred_label)

    def evaluate(self, x: np.ndarray, label: np.ndarray, **kwargs) -> tuple:
        assert x.shape[0] == label.shape[0]
        # Predict labels
        pred_label = self.predict(x, **kwargs)
        # Use 0-1 loss
        loss = np.count_nonzero(pred_label != label)
        precision = 1 - loss / x.shape[0]
        return precision, loss

    def _build_tree(self, x: np.ndarray, label: np.ndarray,
                    current_depth: int) -> CARTNode:
        # Find best node for seperating
        n, p = x.shape
        # Build CART tree for classification
        if len(np.unique(label)) == 1 or n < self._thres_count \
                or current_depth > self._depth:
            # Too few data points, or depth of tree exceeds requirement,
            # just build leaf node and stop recursion.
            root = CARTNode(-1, False, None)
            root.is_leaf = True
            root.cls = stats.mode(label)
            # Record classes
            root.labels = label
        else:
            # Record best attribute with maximum gini-index
            max_gini, best_col, best_value = -1, -1, None
            for col in range(p):
                # Check if this attribute has only one value
                values = np.unique(x[:, col])
                if len(values) > 1:
                    # Calculate gini-index for values of current attribute
                    gini_values = {}
                    for value in values:
                        # Split data
                        rows1, rows2 = self._split_data(x, col, value)
                        # Calculate gini-index for this value in this attribute
                        n1, n2 = len(rows1), len(rows2)
                        gini1 = self._gini(label[rows1])
                        gini2 = self._gini(label[rows2])
                        gini = n1 / n * gini1 + n2 / n * gini2
                        gini_values[value] = gini
                    # Choose maximum gini-index for this attribute
                    value = max(gini_values, key=gini_values.get)
                    gini = gini_values[value]
                    # Update maximum gini-index and best attribute
                    if gini > max_gini:
                        max_gini = gini
                        best_col = col
                        best_value = value
            # Create root node
            root = CARTNode(best_col, self._is_discrete[best_col], best_value)
            # Recursively build subtrees
            depth = current_depth + 1
            rows1, rows2 = self._split_data(x, best_col, best_value)
            root.left_child = self._build_tree(x[rows1], label[rows1], depth)
            root.right_child = self._build_tree(x[rows2], label[rows2], depth)
        return root

    def _pruning(self, x: np.ndarray, label: np.ndarray) -> None:
        # TODO: pruning on current tree using dynamic programming.
        pass

    @staticmethod
    def _gini(label: np.ndarray) -> float:
        # Calculate gini-index.
        cls, count = np.unique(label, return_counts=True)
        count /= label.shape[0]
        gini = float(1 - np.sum(np.power(count, 2)))
        return gini

    def _split_data(self, x: np.ndarray, col: int, value: object) -> list:
        n = len(x)
        res = []
        if self._is_discrete[col]:
            # Split by discrete value
            res.append(list(
                filter(lambda row: x[row, col] == value, range(n))))
            res.append(list(
                filter(lambda row: x[row, col] != value, range(n))))
        else:
            # Split by continuous value
            res.append(list(
                filter(lambda row: x[row, col] <= value, range(n))))
            res.append(list(
                filter(lambda row: x[row, col] > value, range(n))))
        return res

    def _loss(self) -> float:
        # Calculate loss of current tree
        loss = 0.
        num_leaf = 0
        nodes = [self._tree]
        while len(nodes) > 0:
            tmp = []
            for node in nodes:
                if node.is_leaf:
                    # Calculate experience entropy of leaf node
                    total = len(node.labels)
                    counter = Counter(node.labels)
                    entropy = 0.
                    for count in counter.values():
                        entropy -= count * np.log(count / total)
                    loss += entropy
                    num_leaf += 1
                else:
                    tmp.append(node.left_child)
                    tmp.append(node.right_child)
        loss += self._alpha * num_leaf
        return loss
