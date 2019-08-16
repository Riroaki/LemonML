import numpy as np
from scipy import stats
from supervised._base import SupervisedModel


class ID3Node(object):
    """Iterative Dichotomiser 3 decision tree node."""

    def __init__(self, attr_idx: int, count: int, alpha: float,
                 is_leaf: bool = False, cls: any = None):
        # Index of attribute and number of samples
        self.index = attr_idx
        self.count = count
        # Leaf node: just a notion of class
        # actually it's not a node in tree
        self.is_leaf = is_leaf
        self.cls = cls
        # Loss function param
        self.alpha = alpha
        # Related nodes
        self.parent = None
        self.child = {}
        # Calculation results
        self.__entropy = 0.
        self.__loss = 0.

    @property
    def entropy(self) -> float:
        # Calculate experience entropy for this node.
        if self.__entropy != 0:
            return self.__entropy
        # Calculate if necessary.
        entropy = 0.
        for node in self.child.values():
            ratio = node.count / self.count
            entropy -= np.log(ratio) * ratio
        self.__entropy = entropy
        return entropy

    @property
    def loss(self) -> float:
        # Calculate loss of tree rooted this node.
        # Recursively call subtrees.
        if self.is_leaf or self.__loss != 0:
            return self.__loss
        # Calculate if necessary.
        loss = self.alpha + self.entropy + self.count
        for node in self.child.values():
            loss += node.loss
        self.__loss = loss
        return loss


class ID3(SupervisedModel):
    """Iterative Dichotomiser 3 decision tree."""

    def __init__(self):
        self._tree = None
        self._thres_gain = 0.1  # Lower bound of info gain for split
        self._thres_count = 2  # Lower bound of sample count for split
        self._alpha = 2.  # Loss parameter, large alpha for simple model

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> float:
        assert x.shape[0] == label.shape[0]
        # Choose alpha for tree
        if 'alpha' in kwargs:
            assert isinstance(kwargs['alpha'], (int, float))
            assert kwargs['alpha'] > 0
            self._alpha = kwargs['alpha']
        p = x.shape[1]
        is_used = [False for _ in range(p)]
        self._tree = self._build_tree(x, label, is_used)
        # Pruning
        if 'pruning' in kwargs and kwargs['pruning'] is True:
            self.prune()
        # Calculate train loss
        loss = self._tree.loss
        return loss

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
        loss = self._tree.loss
        precision = np.count_nonzero(label == label_pred) / label.shape[0]
        return precision, loss

    def prune(self) -> float:
        # Use post pruning, prune after tree is build.
        # Tend to choose simple model with large alpha
        assert self._tree is not None
        # TODO: needs to update loss each time we prune
        return self._tree.loss

    def _build_tree(self, x: np.ndarray, label: np.ndarray,
                    is_used: list) -> ID3Node:
        # Build a DT
        # Calculate the information gain of left attributes
        info_gain = self._info_gain(x, label, is_used)
        # End splitting if labels are purified
        # or no attributes to choose from
        # or no enough samples to split
        # or if info gain is less than lower bound
        if not np.isin(False, is_used) or \
                len(np.unique(label)) == 1 or \
                x.shape[0] <= self._thres_count or \
                info_gain < self._thres_gain:
            is_leaf = True
            cls = stats.mode(label)
            count = x.shape[0]
            leaf = ID3Node(-1, count, self._alpha, is_leaf, cls)
            return leaf
        # Choose the one with maximum info gain as next node
        attr_idx = int(np.argmax(info_gain))
        count = x.shape[0]
        root = ID3Node(attr_idx, count, self._alpha)
        # Mark this attribute to be used
        is_used[attr_idx] = True
        attr = x[:, attr_idx]
        for val in np.unique(attr):
            # Build tree recursively
            node = self._build_tree(x[attr == val], label[attr == val],
                                    is_used)
            # Register node's relations
            node.parent = root
            root.child[val] = node
        return root

    @staticmethod
    def _entropy(label: np.ndarray) -> float:
        """Calculate information entropy for a sequence."""
        cls, count = np.unique(label, return_counts=True)
        count /= label.shape[0]
        entropy = np.float(-np.sum(count * np.log(count) / np.log(2)))
        return entropy

    def _info_gain(self, data: np.ndarray, label: np.ndarray,
                   is_used: list) -> np.ndarray:
        """Calculate information gain for each attributes."""
        n, p = data.shape
        entropy_total = self._entropy(label)
        entropy_cond = np.zeros(p)
        for idx in range(p):
            # Check whether this attribute has been used to create a node
            if not is_used[idx]:
                attr = data[:, idx]
                # Calculate conditional entropy for current attributes
                for val, count in np.unique(attr, return_counts=True):
                    entropy_cond[idx] -= count / n * self._entropy(
                        label[attr == val])
        info_gain = entropy_total - entropy_cond
        return info_gain
