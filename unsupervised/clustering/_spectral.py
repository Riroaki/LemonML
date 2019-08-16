import numpy as np
from ._kmeans import KMeans
from ._base import ClusteringModel


class Spectral(ClusteringModel):
    def __init__(self):
        self._threshold = 0.1
        self._k_adjacency = 10

    def clustering(self, x: np.ndarray, k: int, **kwargs) -> np.ndarray:
        n, p = x.shape
        assert 0 < k < n
        if 'k_adj' in kwargs:
            assert 0 < kwargs['k_adj'] < n
            self._k_adjacency = kwargs['k_adj']
        if 'threshold' in kwargs:
            assert 0 < kwargs['threshold']
            self._threshold = kwargs['threshold']
        # Build adjacency matrix using k nearest
        w = self.__calc_adjacency(x)
        d = np.diag(w.sum(axis=1))
        l_ = d - w
        # Calculate eigen values and vectors
        values, vectors = np.linalg.eig(l_)
        # Use minimim k eigen values
        min_k_idx = values.argsort()[: k]
        x_ = vectors[:, min_k_idx]
        # Assign labels
        model = KMeans()
        assign, *_ = model.clustering(x_, k)
        return assign

    def __calc_adjacency(self, x: np.ndarray) -> np.ndarray:
        """Calculate adjacency matrix of x."""
        n, p = x.shape
        w = np.zeros((n, n))
        dists = np.zeros((n, n))
        # Calculate distances between each point
        for i in range(n):
            dists[i] = np.power(x[i] - x, 2).sum(axis=1)
        dists = np.sqrt(dists)
        # Build graph
        k = self._k_adjacency
        threshold = self._threshold
        for i in range(n):
            closest_idx = np.argsort(dists[i])[: k]
            for j in closest_idx:
                if dists[i][j] <= threshold:
                    w[i][j] = 1
        return w
