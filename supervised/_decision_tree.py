import numpy as np
from scipy import stats
from ._base import SupervisedModel


class TreeNode(object):
    def __init__(self, attr_idx: int):
        self.index = attr_idx
        self.child = {}
        self.is_leaf = False
        self.cls = None


class DecisionTree(SupervisedModel):
    """Decision Tree classifier model using ID3 algorithm."""

    def __init__(self):
        self._tree = None

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> None:
        assert x.shape[0] == label.shape[0]
        _, p = x.shape
        is_used = [False for _ in range(p)]
        self._tree = self.__build_tree(x, label, is_used)

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # Predict each row's classifying result
        results = []
        for row in x:
            curr_node = self._tree
            while not curr_node.is_leaf:
                value = row[curr_node.index]
                curr_node = curr_node.child[value]
            pred = curr_node.cls
            results.append(pred)
        results = np.array(results)
        return results

    def evaluate(self, x: np.ndarray, label: np.ndarray, **kwargs) -> tuple:
        assert x.shape[0] == label.shape[0]
        label_pred = self.predict(x)
        precision = np.count_nonzero(label == label_pred)
        # TODO: loss of DT
        loss = 0.
        return precision, loss

    def __build_tree(self, x: np.ndarray, label: np.ndarray,
                     is_used: list) -> TreeNode:
        # Build a DT
        # End splitting if labels are purified or no attributes to choose from
        if np.count_nonzero(is_used) == len(is_used) or len(
                np.unique(label)) == 1:
            leaf = TreeNode(-1)
            leaf.is_leaf = True
            leaf.cls = stats.mode(label)
            return leaf
        # Calculate the information gain of left attributes
        info_gain = self.__info_gain(x, label, is_used)
        # Choose the one with maximum info gain as next node
        attr_idx = int(np.argmax(info_gain))
        root = TreeNode(attr_idx)
        # Mark this attribute to be used
        is_used[attr_idx] = True
        attr = x[:, attr_idx]
        for val in np.unique(attr):
            # Build tree recursively
            root.child[val] = self.__build_tree(x[attr == val],
                                                label[attr == val], is_used)
        return root

    @staticmethod
    def __entropy(label: np.ndarray) -> np.float:
        """Calculate information entropy for a sequence."""
        cls, count = np.unique(label, return_counts=True)
        count /= label.shape[0]
        entropy = np.float(-np.sum(count * np.log(count) / np.log(2)))
        return entropy

    def __info_gain(self, data: np.ndarray, label: np.ndarray,
                    is_used: list) -> np.ndarray:
        """Calculate information gain for each attributes."""
        n, p = data.shape
        entropy_total = self.__entropy(label)
        entropy_cond = np.zeros(p)
        for idx in range(p):
            # Check whether this attribute has been used to create a node
            if not is_used[idx]:
                attr = data[:, idx]
                # Calculate conditional entropy for current attributes
                for val, count in np.unique(attr, return_counts=True):
                    entropy_cond[idx] -= count / n * self.__entropy(
                        label[attr == val])
        info_gain = entropy_total - entropy_cond
        return info_gain
